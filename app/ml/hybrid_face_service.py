"""
Hybrid Face Shape Service.

Flask-compatible wrapper around FaceShapeHybridPredictor.
Provides a clean interface for face shape prediction using the PyTorch hybrid model.
Also includes skin tone detection using LAB color space analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import logging

import cv2
import numpy as np

from app.ml.hybrid_pytorch_inference import (
    FaceShapeHybridPredictor,
    InvalidImageError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    InferenceError,
    CLASSES,
)


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Skin Tone Detection
# -----------------------------------------------------------------------------

# Skin tone categories and L-channel thresholds (LAB color space)
# L channel ranges from 0 (black) to 100 (white)
SKIN_TONE_CATEGORIES = ["fair", "light", "medium", "brown", "dark"]

@dataclass
class SkinToneThresholds:
    """Configurable thresholds for skin tone classification based on LAB L-channel."""
    fair_min: float = 75.0      # L >= 75 = Fair
    light_min: float = 65.0     # L >= 65 = Light
    medium_min: float = 50.0    # L >= 50 = Medium
    brown_min: float = 35.0     # L >= 35 = Brown
    # L < 35 = Dark


def extract_cheek_region(
    crop_bgr: np.ndarray,
    center_ratio: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Extract the central cheek region from a face crop.
    
    The central region typically contains cheek skin with less variation
    from shadows, hair, or background.
    
    Args:
        crop_bgr: BGR face crop image
        center_ratio: Ratio of the image to use as the center region (0.3 = middle 30%)
        
    Returns:
        BGR image of the central cheek region, or None if extraction fails
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    
    try:
        h, w = crop_bgr.shape[:2]
        
        # Calculate center region bounds
        margin_x = int(w * (1 - center_ratio) / 2)
        margin_y = int(h * (1 - center_ratio) / 2)
        
        # Extract slightly below center (cheek area, not forehead)
        y_offset = int(h * 0.1)  # Shift down 10% to focus on cheeks
        
        x1 = margin_x
        x2 = w - margin_x
        y1 = margin_y + y_offset
        y2 = h - margin_y
        
        # Ensure valid bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return crop_bgr[y1:y2, x1:x2].copy()
    
    except Exception as e:
        logger.warning(f"Failed to extract cheek region: {e}")
        return None


def compute_mean_l_channel(bgr_image: np.ndarray) -> Optional[float]:
    """
    Compute the mean L-channel value from a BGR image using LAB color space.
    
    LAB color space separates luminance (L) from color (A, B), making it
    ideal for skin tone analysis independent of color cast.
    
    Args:
        bgr_image: BGR image (numpy array)
        
    Returns:
        Mean L-channel value (0-100 scale), or None if computation fails
    """
    if bgr_image is None or bgr_image.size == 0:
        return None
    
    try:
        # Convert BGR to LAB
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        
        # Extract L channel (index 0)
        l_channel = lab[:, :, 0]
        
        # OpenCV LAB L-channel is 0-255, convert to 0-100 scale
        mean_l = float(np.mean(l_channel)) * (100.0 / 255.0)
        
        return mean_l
    
    except Exception as e:
        logger.warning(f"Failed to compute L-channel: {e}")
        return None


def classify_skin_tone(
    mean_l: float,
    thresholds: Optional[SkinToneThresholds] = None,
) -> str:
    """
    Classify skin tone based on mean L-channel value.
    
    Args:
        mean_l: Mean L-channel value (0-100 scale)
        thresholds: Custom thresholds, or use defaults
        
    Returns:
        Skin tone category: "fair", "light", "medium", "brown", or "dark"
    """
    if thresholds is None:
        thresholds = SkinToneThresholds()
    
    if mean_l >= thresholds.fair_min:
        return "fair"
    elif mean_l >= thresholds.light_min:
        return "light"
    elif mean_l >= thresholds.medium_min:
        return "medium"
    elif mean_l >= thresholds.brown_min:
        return "brown"
    else:
        return "dark"


def detect_skin_tone(
    crop_bgr: np.ndarray,
    thresholds: Optional[SkinToneThresholds] = None,
) -> Tuple[str, Optional[float]]:
    """
    Detect skin tone from a face crop using LAB color space analysis.
    
    Extracts the central cheek region to avoid shadows and hair,
    then analyzes the L-channel (luminance) to classify skin tone.
    
    Args:
        crop_bgr: BGR face crop image
        thresholds: Optional custom classification thresholds
        
    Returns:
        Tuple of (skin_tone_category, mean_l_value)
        skin_tone_category is one of: "fair", "light", "medium", "brown", "dark", "unknown"
        mean_l_value is the computed L-channel mean, or None if detection failed
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", None
    
    try:
        # Extract central cheek region
        cheek_region = extract_cheek_region(crop_bgr, center_ratio=0.35)
        
        if cheek_region is None or cheek_region.size == 0:
            # Fallback: use entire crop if region extraction fails
            cheek_region = crop_bgr
        
        # Compute mean L-channel
        mean_l = compute_mean_l_channel(cheek_region)
        
        if mean_l is None:
            return "unknown", None
        
        # Classify based on thresholds
        skin_tone = classify_skin_tone(mean_l, thresholds)
        
        return skin_tone, mean_l
    
    except Exception as e:
        logger.warning(f"Skin tone detection failed: {e}")
        return "unknown", None


# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_shape_hybrid_final.pth"
DEFAULT_LANDMARKER_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"


class HybridFaceServiceError(Exception):
    """Base exception for HybridFaceService errors."""
    pass


class HybridFaceService:
    """
    Flask-compatible face shape prediction service.
    
    Uses the PyTorch hybrid model (EfficientNet-B0 + geometric metrics)
    with MediaPipe face landmarks.
    
    Usage:
        service = HybridFaceService()
        result = service.predict_from_bgr(bgr_image)
        
        if result["success"]:
            print(f"Face shape: {result['face_shape']}")
            print(f"Confidence: {result['confidence']:.2%}")
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        landmarker_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        temperature: float = 1.2,
    ):
        """
        Initialize the hybrid face service.
        
        Args:
            model_path: Path to face_shape_hybrid_final.pth
            landmarker_path: Path to face_landmarker.task
            device: Device to run inference on (only "cpu" supported)
            temperature: Softmax temperature for calibration
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.landmarker_path = Path(landmarker_path) if landmarker_path else DEFAULT_LANDMARKER_PATH
        self.device = device
        self.temperature = temperature
        
        self._predictor: Optional[FaceShapeHybridPredictor] = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of the predictor."""
        if self._initialized:
            return
        
        # Validate paths
        if not self.model_path.exists():
            raise HybridFaceServiceError(
                f"Model file not found: {self.model_path}\n"
                f"Please place 'face_shape_hybrid_final.pth' in the models/ directory."
            )
        
        if not self.landmarker_path.exists():
            raise HybridFaceServiceError(
                f"Landmarker file not found: {self.landmarker_path}\n"
                f"Please place 'face_landmarker.task' in the models/ directory."
            )
        
        try:
            self._predictor = FaceShapeHybridPredictor(
                mp_task_model_path=str(self.landmarker_path),
                model_path=str(self.model_path),
                device=self.device,
                temperature=self.temperature,
            )
            self._initialized = True
            logger.info("HybridFaceService initialized successfully")
        except Exception as e:
            raise HybridFaceServiceError(f"Failed to initialize predictor: {e}") from e
    
    def predict_from_bgr(
        self,
        bgr_image: np.ndarray,
        return_debug: bool = True,
        skin_tone_thresholds: Optional[SkinToneThresholds] = None,
    ) -> Dict:
        """
        Predict face shape and skin tone from a BGR image.
        
        Args:
            bgr_image: OpenCV BGR image (numpy array)
            return_debug: Whether to include debug info (stability, etc.)
            skin_tone_thresholds: Optional custom thresholds for skin tone classification
        
        Returns:
            Dictionary with:
                - success: bool
                - face_shape: str (if success)
                - confidence: float (if success)
                - stability: float (if success)
                - skin_tone: str (if success) - "fair", "light", "medium", "brown", "dark", or "unknown"
                - bbox: tuple (if success)
                - error: str (if not success)
                - error_type: str (if not success)
        """
        self._ensure_initialized()
        
        # Validate input
        if bgr_image is None or not isinstance(bgr_image, np.ndarray):
            return {
                "success": False,
                "error": "Invalid image: None or not a numpy array",
                "error_type": "invalid_image",
            }
        
        if bgr_image.size == 0:
            return {
                "success": False,
                "error": "Invalid image: empty array",
                "error_type": "invalid_image",
            }
        
        if len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
            return {
                "success": False,
                "error": f"Invalid image shape: expected (H, W, 3), got {bgr_image.shape}",
                "error_type": "invalid_image",
            }
        
        try:
            # Always get debug info for skin tone detection (need crop_bgr)
            pred, debug = self._predictor.predict_from_bgr(bgr_image, return_debug=True)
            
            # Detect skin tone from the face crop
            crop_bgr = debug.get("crop_bgr")
            skin_tone, mean_l = detect_skin_tone(crop_bgr, skin_tone_thresholds)
            
            logger.debug(f"Skin tone detected: {skin_tone} (L={mean_l:.1f})" if mean_l else f"Skin tone: {skin_tone}")
            
            # Build response
            result = {
                "success": True,
                "face_shape": pred.label,
                "confidence": float(pred.confidence),
                "stability": float(debug.get("stability", 0.0)),
                "skin_tone": skin_tone,
                "bbox": pred.bbox_xyxy,
            }
            
            # Add extended debug info if requested
            if return_debug:
                result.update({
                    "softmax_scores": debug.get("softmax", {}),
                    "top2_predictions": debug.get("top2", []),
                    "confusion_message": debug.get("confusion_message"),
                    "ratios": debug.get("ratios", {}),
                    "skin_tone_l_value": mean_l,
                })
            
            return result
        
        except NoFaceDetectedError as e:
            logger.warning(f"No face detected: {e}")
            return {
                "success": False,
                "error": "No face detected in the image. Please upload a clear front-facing photo.",
                "error_type": "no_face",
            }
        
        except MultipleFacesDetectedError as e:
            logger.warning(f"Multiple faces detected: {e}")
            return {
                "success": False,
                "error": "Multiple faces detected. Please upload a photo with only one face.",
                "error_type": "multiple_faces",
            }
        
        except InvalidImageError as e:
            logger.error(f"Invalid image: {e}")
            return {
                "success": False,
                "error": f"Invalid image: {e}",
                "error_type": "invalid_image",
            }
        
        except InferenceError as e:
            logger.error(f"Inference error: {e}")
            return {
                "success": False,
                "error": f"Face analysis failed: {e}",
                "error_type": "inference_error",
            }
        
        except Exception as e:
            logger.exception("Unexpected error during prediction")
            return {
                "success": False,
                "error": f"Unexpected error: {e}",
                "error_type": "unknown",
            }
    
    def predict_from_path(
        self,
        image_path: Union[str, Path],
        return_debug: bool = True,
        skin_tone_thresholds: Optional[SkinToneThresholds] = None,
    ) -> Dict:
        """
        Predict face shape and skin tone from an image file path.
        
        Args:
            image_path: Path to the image file
            return_debug: Whether to include debug info
            skin_tone_thresholds: Optional custom thresholds for skin tone
            
        Returns:
            Same dictionary as predict_from_bgr
        """
        path = Path(image_path)
        
        if not path.exists():
            return {
                "success": False,
                "error": f"Image file not found: {path}",
                "error_type": "file_not_found",
            }
        
        bgr_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        
        if bgr_image is None:
            return {
                "success": False,
                "error": f"Failed to load image: {path}",
                "error_type": "invalid_image",
            }
        
        return self.predict_from_bgr(
            bgr_image,
            return_debug=return_debug,
            skin_tone_thresholds=skin_tone_thresholds,
        )
    
    @property
    def available_classes(self) -> list:
        """Return list of face shape classes."""
        return CLASSES.copy()
    
    @property
    def available_skin_tones(self) -> list:
        """Return list of skin tone categories."""
        return SKIN_TONE_CATEGORIES.copy()
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized
    
    def close(self) -> None:
        """Release resources."""
        if self._predictor is not None:
            self._predictor.close()
            self._predictor = None
            self._initialized = False
            logger.info("HybridFaceService closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Global service instance for Flask (initialized lazily)
_global_service: Optional[HybridFaceService] = None


def get_hybrid_face_service(
    model_path: Optional[Union[str, Path]] = None,
    landmarker_path: Optional[Union[str, Path]] = None,
) -> HybridFaceService:
    """
    Get or create a global HybridFaceService instance.
    
    This is useful for Flask applications where you want to initialize
    the service once and reuse it across requests.
    
    Args:
        model_path: Optional custom model path
        landmarker_path: Optional custom landmarker path
        
    Returns:
        HybridFaceService instance
    """
    global _global_service
    
    if _global_service is None:
        _global_service = HybridFaceService(
            model_path=model_path,
            landmarker_path=landmarker_path,
        )
    
    return _global_service


def predict_face_shape(
    image: Union[str, Path, np.ndarray],
    model_path: Optional[Union[str, Path]] = None,
    landmarker_path: Optional[Union[str, Path]] = None,
    return_debug: bool = True,
    skin_tone_thresholds: Optional[SkinToneThresholds] = None,
) -> Dict:
    """
    Convenience function to predict face shape and skin tone.
    
    Args:
        image: Image path or BGR numpy array
        model_path: Optional custom model path
        landmarker_path: Optional custom landmarker path
        return_debug: Whether to include debug info
        skin_tone_thresholds: Optional custom thresholds for skin tone
        
    Returns:
        Prediction result dictionary including face_shape and skin_tone
    """
    service = get_hybrid_face_service(model_path, landmarker_path)
    
    if isinstance(image, np.ndarray):
        return service.predict_from_bgr(
            image,
            return_debug=return_debug,
            skin_tone_thresholds=skin_tone_thresholds,
        )
    else:
        return service.predict_from_path(
            image,
            return_debug=return_debug,
            skin_tone_thresholds=skin_tone_thresholds,
        )
