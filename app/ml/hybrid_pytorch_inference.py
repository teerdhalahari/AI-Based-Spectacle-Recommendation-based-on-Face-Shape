"""
hybrid_pytorch_inference.py

Production-style inference utilities for the PyTorch hybrid face-shape model:
- MediaPipe FaceLandmarker (478 landmarks)
- Face crop from landmark bounding box with padding
- 224x224 preprocessing (ImageNet normalization)
- 11 geometric facial metrics
- EfficientNet-B0 backbone (classifier removed) + FC head -> 5 classes

This module contains *no training code* and is designed to be imported by:
- predict_image.py
- realtime_face_shape.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision import transforms

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


CLASSES: List[str] = ["Heart", "Oblong", "Oval", "Round", "Square"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METRICS_DIM = 11
CNN_DIM = 1280
LANDMARKER_INPUT_SIZE = (640, 640)  # training-time MediaPipe preprocessing


class InferenceError(RuntimeError):
    """Base class for inference-time errors."""


class InvalidImageError(InferenceError):
    """Raised when an input image is invalid/unreadable."""


class NoFaceDetectedError(InferenceError):
    """Raised when no face is detected."""


class MultipleFacesDetectedError(InferenceError):
    """Raised when more than one face is detected."""


@dataclass(frozen=True)
class FaceDetection:
    landmarks: np.ndarray  # (478, 2) normalized [0..1] relative to the input image
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels on the input image


def _safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    if abs(den) < eps:
        return float(num / eps)
    return float(num / den)


def compute_geometric_metrics(landmarks: np.ndarray) -> torch.Tensor:
    """
    Compute the same 11 geometric metrics used during training from (478, 2) landmarks.
    Landmarks are expected to be normalized to the image space (0..1).
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (478, 2):
        raise ValueError(f"Expected (478, 2), got {landmarks.shape}")

    min_x = float(landmarks[:, 0].min())
    max_x = float(landmarks[:, 0].max())
    face_width = max_x - min_x
    if face_width == 0.0:
        face_width = 1e-8

    def x_w(i: int, j: int) -> float:
        return float(abs(landmarks[i, 0] - landmarks[j, 0]) / face_width)

    def y_h(i: int, j: int) -> float:
        return float(abs(landmarks[i, 1] - landmarks[j, 1]) / face_width)

    TOP_FOREHEAD, CHIN = 10, 152
    LEFT_CHEEK, RIGHT_CHEEK = 234, 454
    LEFT_JAW, RIGHT_JAW = 172, 397
    LEFT_FOREHEAD, RIGHT_FOREHEAD = 127, 356
    LEFT_CHEEKBONE, RIGHT_CHEEKBONE = 93, 323
    LOWER_LIP = 14

    face_height = y_h(TOP_FOREHEAD, CHIN)
    face_width_lr = x_w(LEFT_CHEEK, RIGHT_CHEEK)
    jaw_width = x_w(LEFT_JAW, RIGHT_JAW)
    forehead_width = x_w(LEFT_FOREHEAD, RIGHT_FOREHEAD)
    cheekbone_width = x_w(LEFT_CHEEKBONE, RIGHT_CHEEKBONE)
    chin_length = y_h(LOWER_LIP, CHIN)

    r_hw = _safe_div(face_height, face_width_lr)
    r_jc = _safe_div(jaw_width, cheekbone_width)
    r_fj = _safe_div(forehead_width, jaw_width)
    r_ch = _safe_div(chin_length, face_height)
    r_cw = _safe_div(cheekbone_width, face_width_lr)

    metrics = np.array(
        [
            face_height,
            face_width_lr,
            jaw_width,
            forehead_width,
            cheekbone_width,
            chin_length,
            r_hw,
            r_jc,
            r_fj,
            r_ch,
            r_cw,
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(metrics)  # (11,)


def crop_face_from_landmarks(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    padding: float = 0.10,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop face using landmark-derived bounding box with padding.

    Returns:
        (crop_bgr, (x1, y1, x2, y2))
    """
    h, w = image_bgr.shape[:2]

    xs = landmarks[:, 0] * w
    ys = landmarks[:, 1] * h
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    bw = max_x - min_x
    bh = max_y - min_y
    pad_x = bw * padding
    pad_y = bh * padding

    x1 = max(0, int(min_x - pad_x))
    y1 = max(0, int(min_y - pad_y))
    x2 = min(w, int(max_x + pad_x))
    y2 = min(h, int(max_y + pad_y))

    if x2 <= x1 or y2 <= y1:
        raise NoFaceDetectedError("Computed an invalid face crop.")

    return image_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


class MediaPipeFaceLandmarker:
    """
    MediaPipe FaceLandmarker wrapper.

    - Runs on CPU
    - Returns 478 landmarks per detected face (includes iris landmarks)
    - Enforces exactly one face for production predictability
    """

    def __init__(self, model_path: str, max_faces: int = 5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _preprocess_bgr_for_landmarks(bgr: np.ndarray) -> np.ndarray:
        """
        Match training-time preprocessing from `src/landmark_extractor.py`:
        resize to 640x640 (may distort aspect ratio) then convert BGR -> RGB.
        """
        resized = cv2.resize(bgr, LANDMARKER_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _detect(self, bgr: np.ndarray) -> List[FaceDetection]:
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
            raise InvalidImageError("Invalid image array.")

        rgb = self._preprocess_bgr_for_landmarks(bgr)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        faces = result.face_landmarks or []
        detections: List[FaceDetection] = []
        h, w = bgr.shape[:2]
        for lm in faces:
            xy = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
            if xy.shape != (478, 2):
                # Skip faces with unexpected landmark counts rather than crashing.
                continue
            xs = xy[:, 0] * w
            ys = xy[:, 1] * h
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            detections.append(FaceDetection(landmarks=xy, bbox_xyxy=(x1, y1, x2, y2)))

        return detections

    def detect_single(self, bgr: np.ndarray) -> FaceDetection:
        detections = self._detect(bgr)
        if not detections:
            raise NoFaceDetectedError("No face detected.")
        if len(detections) > 1:
            raise MultipleFacesDetectedError(f"Multiple faces detected: {len(detections)}")
        return detections[0]

    def detect_all(self, bgr: np.ndarray) -> List[FaceDetection]:
        """
        Detect up to `max_faces` faces. Returns an empty list if no faces are detected.
        """
        return self._detect(bgr)


class HybridModel(nn.Module):
    """
    EfficientNet-B0 (classifier=Identity) + head:
    Linear(1280+11→512)→BN→ReLU→Dropout(0.5)→Linear(512→256)→ReLU→Dropout(0.3)→Linear(256→5).
    This is intentionally identical to `HybridModel` in `src/train_hybrid_advanced.py`.
    """

    def __init__(
        self,
        num_classes: int = 5,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            blocks = list(self.backbone.features)
            for block in blocks[-unfreeze_last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(CNN_DIM + METRICS_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(images)
        combined = torch.cat([feat, metrics], dim=1)
        return self.head(combined)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float  # 0..1
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass(frozen=True)
class FacePrediction:
    """
    Per-face prediction used for multi-face webcam inference.
    """

    label: str
    confidence: float  # 0..1
    bbox_xyxy: Tuple[int, int, int, int]
    landmarks: np.ndarray  # (478, 2) normalized
    stability: float = 0.0  # Part 3: stability score (0..1)


class FaceShapeHybridPredictor:
    """
    End-to-end CPU predictor: BGR image -> landmarks -> crop -> metrics -> model -> label.
    """

    def __init__(
        self,
        mp_task_model_path: str,
        model_path: str,
        device: str = "cpu",
        temperature: float = 1.2,
    ):
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("This inference pipeline is CPU-only by requirement.")

        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        self.temperature = float(temperature)

        self.landmarker = MediaPipeFaceLandmarker(model_path=mp_task_model_path, max_faces=5)
        # Architecture identical to training-time HybridModel; weights are loaded from checkpoint.
        self.model = HybridModel(num_classes=len(CLASSES), freeze_backbone=False, unfreeze_last_n_blocks=0).to(
            self.device
        )
        self._load_weights(model_path)
        self.model.eval()

        # EXACT validation transform requested:
        # Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)
        self._val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def close(self) -> None:
        self.landmarker.close()

    def _load_weights(self, model_path: str) -> None:
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        # Handle possible DataParallel-style prefixes.
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            raise InferenceError(
                "Checkpoint does not match the expected hybrid EfficientNet-B0 architecture. "
                "Verify you are loading the correct `.pth` for this inference code."
            ) from e

    def _preprocess_crop_val(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """
        Apply the *exact* validation transform used during training.
        Input is a BGR crop; transform expects a PIL RGB image.
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        img = self._val_transform(pil)  # (3, 224, 224)
        return img.unsqueeze(0)  # (1, 3, 224, 224)

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """
        Gamma correction for brightness enhancement.
        Only applied to landmark detection input, NOT CNN input.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def _prepare_landmark_input(bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        Part 1: Lighting-safe dual pipeline.
        - CNN uses original image (no modification)
        - Landmark detection may use gamma-corrected version if dark
        Returns (landmark_bgr, lighting_info_dict)
        """
        original_bgr = bgr.copy()
        landmark_bgr = original_bgr.copy()
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(gray.mean())
        correction_applied = False

        if mean_intensity < 60.0:
            landmark_bgr = FaceShapeHybridPredictor.gamma_correction(landmark_bgr, gamma=1.5)
            correction_applied = True

        lighting_info = {
            "mean_intensity": mean_intensity,
            "correction_applied": correction_applied,
            "cnn_input": "original",
            "landmark_input": "adjusted" if correction_applied else "original",
        }
        return landmark_bgr, lighting_info

    @staticmethod
    def _compute_stability_score(
        prob: torch.Tensor,
        predicted_class: str,
        ratios: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Part 3: Advanced stability scoring combining entropy, margin, and geometric alignment.
        Returns (stability_score, components_dict)
        """
        import math

        prob_np = prob.detach().cpu().numpy()
        num_classes = len(prob_np)

        # 1) Softmax Entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in prob_np if p > 0)
        max_entropy = math.log(num_classes)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_component = 1.0 - normalized_entropy

        # 2) Top-1 vs Top-2 Margin
        topk = torch.topk(prob, k=2)
        margin = float(topk.values[0].item() - topk.values[1].item())

        # 3) Geometric Alignment Score
        r_hw = ratios.get("height_width_ratio", 0.0)
        r_jc = ratios.get("jaw_cheekbone_ratio", 0.0)
        r_fj = ratios.get("forehead_jaw_ratio", 0.0)

        alignment = 0.0
        if predicted_class == "Round":
            if r_hw < 1.1:
                alignment = 1.0
            else:
                alignment = max(0.0, 1.0 - abs(r_hw - 1.1) * 2.0)
        elif predicted_class == "Oval":
            if 1.1 <= r_hw <= 1.35:
                alignment = 1.0
            else:
                alignment = max(0.0, 1.0 - abs(r_hw - 1.225) * 1.5)
        elif predicted_class == "Oblong":
            if r_hw > 1.35:
                alignment = 1.0
            else:
                alignment = max(0.0, (r_hw - 1.0) / 0.35)
        elif predicted_class == "Square":
            if r_jc > 0.85:
                alignment = 1.0
            else:
                alignment = max(0.0, r_jc / 0.85)
        elif predicted_class == "Heart":
            if r_fj > 1.15:
                alignment = 1.0
            else:
                alignment = max(0.0, r_fj / 1.15)

        # Final stability formula
        stability = 0.4 * entropy_component + 0.4 * margin + 0.2 * alignment
        stability = max(0.0, min(1.0, stability))

        components = {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "entropy_component": entropy_component,
            "margin": margin,
            "geometric_alignment": alignment,
        }
        return stability, components

    @staticmethod
    def _detect_confusion(
        top2: List[Dict[str, float]],
        ratios: Dict[str, float],
    ) -> Optional[str]:
        """
        Part 2: Detect and explain borderline cases between shape pairs.
        Returns confusion explanation string or None.
        """
        if len(top2) < 2:
            return None

        margin = top2[0]["prob"] - top2[1]["prob"]
        if margin >= 0.15:
            return None

        label1, label2 = top2[0]["label"], top2[1]["label"]
        r_hw = ratios.get("height_width_ratio", 0.0)
        r_jc = ratios.get("jaw_cheekbone_ratio", 0.0)
        r_fj = ratios.get("forehead_jaw_ratio", 0.0)

        # Round vs Oval
        if {label1, label2} == {"Round", "Oval"}:
            if 1.05 <= r_hw <= 1.25:
                return "Borderline between Round and Oval — slight elongation detected."

        # Square vs Oblong
        if {label1, label2} == {"Square", "Oblong"}:
            if r_hw > 1.35 and r_jc > 0.85:
                return "Face length is high but strong jaw creates ambiguity between Square and Oblong."

        # Heart vs Oval
        if {label1, label2} == {"Heart", "Oval"}:
            if r_fj > 1.15:
                return "Forehead dominance suggests Heart, but overall symmetry leans Oval."

        # Round vs Square
        if {label1, label2} == {"Round", "Square"}:
            if abs(r_hw - 1.0) < 0.15:
                return "Jaw structure borderline between curved and angular."

        return None

    @torch.no_grad()
    def predict_from_bgr(
        self,
        bgr: np.ndarray,
        *,
        return_debug: bool = False,
    ) -> Prediction | Tuple[Prediction, Dict[str, object]]:
        # Part 1: Lighting-safe dual pipeline
        original_bgr = bgr.copy()
        landmark_bgr, lighting_info = self._prepare_landmark_input(bgr)

        det = self.landmarker.detect_single(landmark_bgr)
        # Crop from ORIGINAL image (not brightness-adjusted) for CNN
        crop_bgr, bbox = crop_face_from_landmarks(original_bgr, det.landmarks, padding=0.10)
        img_t = self._preprocess_crop_val(crop_bgr).to(self.device)
        metrics_vec = compute_geometric_metrics(det.landmarks)  # (11,)
        metrics_t = metrics_vec.unsqueeze(0).to(self.device)  # (1, 11)

        logits = self.model(img_t, metrics_t)  # (1, 5)
        # Adaptive temperature
        prob_base = torch.softmax(logits / self.temperature, dim=1)[0]
        topk_base = torch.topk(prob_base, k=2)
        margin = float(topk_base.values[0].item() - topk_base.values[1].item())
        effective_temp = 1.1 if margin < 0.20 else max(1.0, self.temperature)
        prob = torch.softmax(logits / effective_temp, dim=1)[0]
        idx = int(torch.argmax(prob).item())
        conf = float(prob[idx].item())
        pred = Prediction(label=CLASSES[idx], confidence=conf, bbox_xyxy=bbox)

        if not return_debug:
            return pred

        # Extract ratios for stability and confusion detection
        mv = metrics_vec.detach().cpu().numpy().reshape(-1)
        ratios = {
            "height_width_ratio": float(mv[6]),
            "jaw_cheekbone_ratio": float(mv[7]),
            "forehead_jaw_ratio": float(mv[8]),
        }

        # Top-2 predictions
        topk = torch.topk(prob, k=min(2, prob.shape[0]))
        top2 = [
            {"label": CLASSES[int(i.item())], "prob": float(p.item())}
            for p, i in zip(topk.values, topk.indices)
        ]

        # Part 3: Stability scoring
        stability, stability_components = self._compute_stability_score(prob, CLASSES[idx], ratios)

        # Part 2: Confusion detection
        confusion_msg = self._detect_confusion(top2, ratios)

        debug: Dict[str, object] = {
            "softmax": {name: float(prob[i].item()) for i, name in enumerate(CLASSES)},
            "bbox_xyxy": bbox,
            "crop_bgr": crop_bgr,
            "landmarks_shape": tuple(det.landmarks.shape),
            "top2": top2,
            "temperature": self.temperature,
            "effective_temperature": effective_temp,
            "margin": margin,
            "landmarks": det.landmarks.copy(),
            "ratios": ratios,
            "stability": stability,
            "stability_components": stability_components,
            "confusion_message": confusion_msg,
            "lighting_info": lighting_info,
        }
        return pred, debug

    def predict_from_path(self, image_path: str) -> Prediction:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise InvalidImageError(f"Could not read image: {image_path}")
        return self.predict_from_bgr(bgr)

    @torch.no_grad()
    def predict_faces_from_bgr(self, bgr: np.ndarray) -> List[FacePrediction]:
        """
        Multi-face prediction for a single BGR frame.

        Returns:
            List[FacePrediction], one per detected face. Empty list if no faces.
        """
        original_bgr = bgr.copy()
        landmark_bgr, _ = self._prepare_landmark_input(bgr)
        detections = self.landmarker.detect_all(landmark_bgr)
        faces: List[FacePrediction] = []
        for det in detections:
            try:
                # Crop from ORIGINAL image (not brightness-adjusted) for CNN
                crop_bgr, bbox = crop_face_from_landmarks(original_bgr, det.landmarks, padding=0.10)
            except NoFaceDetectedError:
                # Skip faces with degenerate crops.
                continue

            img_t = self._preprocess_crop_val(crop_bgr).to(self.device)
            metrics_vec = compute_geometric_metrics(det.landmarks)  # (11,)
            metrics_t = metrics_vec.unsqueeze(0).to(self.device)

            logits = self.model(img_t, metrics_t)
            # Adaptive temperature
            prob_base = torch.softmax(logits / self.temperature, dim=1)[0]
            topk_b = torch.topk(prob_base, k=2)
            margin = float(topk_b.values[0].item() - topk_b.values[1].item())
            eff_t = 1.1 if margin < 0.20 else max(1.0, self.temperature)
            prob = torch.softmax(logits / eff_t, dim=1)[0]
            idx = int(torch.argmax(prob).item())
            conf = float(prob[idx].item())

            # Part 3: Compute stability score
            mv = metrics_vec.detach().cpu().numpy().reshape(-1)
            ratios = {
                "height_width_ratio": float(mv[6]),
                "jaw_cheekbone_ratio": float(mv[7]),
                "forehead_jaw_ratio": float(mv[8]),
            }
            stability, _ = self._compute_stability_score(prob, CLASSES[idx], ratios)

            faces.append(
                FacePrediction(
                    label=CLASSES[idx],
                    confidence=conf,
                    bbox_xyxy=bbox,
                    landmarks=det.landmarks,
                    stability=stability,
                )
            )

        return faces

