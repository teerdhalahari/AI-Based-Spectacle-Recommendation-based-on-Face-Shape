"""
Face Feature Extraction Module using MediaPipe Face Mesh.

Extracts facial landmarks, computes measurements, ratios, and detects skin tone.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
import math

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


# -----------------------------------------------------------------------------
# Constants: MediaPipe Face Mesh Landmark Indices
# -----------------------------------------------------------------------------

# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

class FaceLandmarks:
    """Key landmark indices for facial measurements."""
    # Vertical reference points
    CHIN = 152
    FOREHEAD_TOP = 10
    NOSE_TIP = 4
    
    # Jaw width (angle of mandible)
    JAW_LEFT = 234
    JAW_RIGHT = 454
    
    # Cheekbone width (zygomatic arch)
    CHEEKBONE_LEFT = 93
    CHEEKBONE_RIGHT = 323
    
    # Forehead width
    FOREHEAD_LEFT = 70
    FOREHEAD_RIGHT = 300
    
    # Cheek sampling points for skin tone (inner cheek area)
    CHEEK_LEFT_CENTER = 50
    CHEEK_RIGHT_CENTER = 280
    
    # Additional cheek region landmarks for skin sampling
    CHEEK_LEFT_REGION = [50, 101, 118, 119, 100]
    CHEEK_RIGHT_REGION = [280, 330, 347, 348, 329]


# -----------------------------------------------------------------------------
# Skin Tone Classification
# -----------------------------------------------------------------------------

class SkinTone(Enum):
    """Skin tone classification."""
    FAIR = "fair"
    MEDIUM = "medium"
    DUSKY = "dusky"
    DARK = "dark"
    UNKNOWN = "unknown"


@dataclass
class SkinToneThresholds:
    """
    Configurable thresholds for skin tone classification.
    Based on V (Value/Brightness) channel in HSV color space.
    
    V channel range: 0-255
    - Higher V = lighter skin
    - Lower V = darker skin
    """
    fair_min: int = 180      # V >= 180 = Fair
    medium_min: int = 130    # V >= 130 = Medium
    dusky_min: int = 80      # V >= 80 = Dusky
    # V < 80 = Dark
    
    def __post_init__(self):
        if not (self.dusky_min < self.medium_min < self.fair_min <= 255):
            raise ValueError(
                f"Invalid thresholds. Must satisfy: dusky < medium < fair <= 255. "
                f"Got: dusky={self.dusky_min}, medium={self.medium_min}, fair={self.fair_min}"
            )


# -----------------------------------------------------------------------------
# Data Classes for Output
# -----------------------------------------------------------------------------

@dataclass
class FacialMetrics:
    """Computed facial measurements and ratios."""
    jaw_width: float
    cheekbone_width: float
    forehead_width: float
    face_length: float
    
    # Normalized ratios (relative to face_length)
    jaw_width_normalized: float = 0.0
    cheekbone_width_normalized: float = 0.0
    forehead_width_normalized: float = 0.0
    
    # Width-to-length ratios
    jaw_to_length_ratio: float = 0.0
    cheek_to_length_ratio: float = 0.0
    forehead_to_length_ratio: float = 0.0
    
    # Width comparisons
    jaw_to_cheek_ratio: float = 0.0
    forehead_to_cheek_ratio: float = 0.0
    
    def __post_init__(self):
        """Compute normalized values and ratios."""
        if self.face_length > 0:
            self.jaw_width_normalized = self.jaw_width / self.face_length
            self.cheekbone_width_normalized = self.cheekbone_width / self.face_length
            self.forehead_width_normalized = self.forehead_width / self.face_length
            
            self.jaw_to_length_ratio = self.jaw_width / self.face_length
            self.cheek_to_length_ratio = self.cheekbone_width / self.face_length
            self.forehead_to_length_ratio = self.forehead_width / self.face_length
        
        if self.cheekbone_width > 0:
            self.jaw_to_cheek_ratio = self.jaw_width / self.cheekbone_width
            self.forehead_to_cheek_ratio = self.forehead_width / self.cheekbone_width
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "jaw_width": round(self.jaw_width, 2),
            "cheekbone_width": round(self.cheekbone_width, 2),
            "forehead_width": round(self.forehead_width, 2),
            "face_length": round(self.face_length, 2),
            "jaw_width_normalized": round(self.jaw_width_normalized, 4),
            "cheekbone_width_normalized": round(self.cheekbone_width_normalized, 4),
            "forehead_width_normalized": round(self.forehead_width_normalized, 4),
            "jaw_to_length_ratio": round(self.jaw_to_length_ratio, 4),
            "cheek_to_length_ratio": round(self.cheek_to_length_ratio, 4),
            "forehead_to_length_ratio": round(self.forehead_to_length_ratio, 4),
            "jaw_to_cheek_ratio": round(self.jaw_to_cheek_ratio, 4),
            "forehead_to_cheek_ratio": round(self.forehead_to_cheek_ratio, 4),
        }


@dataclass
class SkinToneResult:
    """Skin tone detection result."""
    classification: SkinTone
    hsv_values: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "classification": self.classification.value,
            "hsv_values": {
                k: round(v, 2) for k, v in self.hsv_values.items()
            }
        }


@dataclass
class FaceFeatureResult:
    """Complete face feature extraction result."""
    face_detected: bool
    facial_metrics: Optional[FacialMetrics] = None
    skin_tone: Optional[SkinToneResult] = None
    landmarks_count: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        result = {
            "face_detected": self.face_detected,
            "landmarks_count": self.landmarks_count,
        }
        
        if self.facial_metrics:
            result["facial_metrics"] = self.facial_metrics.to_dict()
        
        if self.skin_tone:
            result["skin_tone"] = self.skin_tone.classification.value
            result["skin_tone_details"] = self.skin_tone.to_dict()
        
        if self.error_message:
            result["error"] = self.error_message
        
        return result


# -----------------------------------------------------------------------------
# Face Feature Extractor
# -----------------------------------------------------------------------------

class FaceFeatureExtractor:
    """
    Extracts facial features using MediaPipe Face Mesh.
    
    Usage:
        extractor = FaceFeatureExtractor()
        result = extractor.extract("face.jpg")
        print(result.to_dict())
    """
    
    def __init__(
        self,
        skin_tone_thresholds: Optional[SkinToneThresholds] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the face feature extractor.
        
        Args:
            skin_tone_thresholds: Custom thresholds for skin tone classification.
            min_detection_confidence: Minimum confidence for face detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required. Install with: pip install mediapipe"
            )
        
        self.skin_thresholds = skin_tone_thresholds or SkinToneThresholds()
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    
    def extract(
        self,
        image: Union[np.ndarray, str, Path],
    ) -> FaceFeatureResult:
        """
        Extract facial features from an image.
        
        Args:
            image: Image as numpy array (BGR) or path to image file.
            
        Returns:
            FaceFeatureResult with metrics, skin tone, and detection status.
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image = self._load_image(image)
            
            if image is None or image.size == 0:
                return FaceFeatureResult(
                    face_detected=False,
                    error_message="Invalid image: empty or None"
                )
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with Face Mesh
            results = self.face_mesh.process(image_rgb)
            
            # Check if face detected
            if not results.multi_face_landmarks:
                return FaceFeatureResult(
                    face_detected=False,
                    error_message="No face detected in image"
                )
            
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Convert to pixel coordinates
            landmarks = self._landmarks_to_pixels(face_landmarks, w, h)
            
            # Compute facial metrics
            metrics = self._compute_metrics(landmarks)
            
            # Detect skin tone
            skin_tone = self._detect_skin_tone(image, landmarks)
            
            return FaceFeatureResult(
                face_detected=True,
                facial_metrics=metrics,
                skin_tone=skin_tone,
                landmarks_count=len(face_landmarks.landmark),
            )
            
        except Exception as e:
            return FaceFeatureResult(
                face_detected=False,
                error_message=str(e)
            )
    
    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        
        return image
    
    def _landmarks_to_pixels(
        self,
        face_landmarks,
        width: int,
        height: int,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Convert normalized MediaPipe landmarks to pixel coordinates.
        
        Returns:
            Dictionary mapping landmark index to (x, y) pixel coordinates.
        """
        landmarks = {}
        for idx, lm in enumerate(face_landmarks.landmark):
            landmarks[idx] = (lm.x * width, lm.y * height)
        return landmarks
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
    def _compute_metrics(self, landmarks: Dict[int, Tuple[float, float]]) -> FacialMetrics:
        """
        Compute facial measurements from landmarks.
        
        Args:
            landmarks: Dictionary of landmark index -> (x, y) pixel coordinates.
            
        Returns:
            FacialMetrics with all measurements and ratios.
        """
        # Jaw width: distance between jaw angles
        jaw_width = self._distance(
            landmarks[FaceLandmarks.JAW_LEFT],
            landmarks[FaceLandmarks.JAW_RIGHT]
        )
        
        # Cheekbone width: distance between zygomatic points
        cheekbone_width = self._distance(
            landmarks[FaceLandmarks.CHEEKBONE_LEFT],
            landmarks[FaceLandmarks.CHEEKBONE_RIGHT]
        )
        
        # Forehead width
        forehead_width = self._distance(
            landmarks[FaceLandmarks.FOREHEAD_LEFT],
            landmarks[FaceLandmarks.FOREHEAD_RIGHT]
        )
        
        # Face length: chin to forehead
        face_length = self._distance(
            landmarks[FaceLandmarks.CHIN],
            landmarks[FaceLandmarks.FOREHEAD_TOP]
        )
        
        return FacialMetrics(
            jaw_width=jaw_width,
            cheekbone_width=cheekbone_width,
            forehead_width=forehead_width,
            face_length=face_length,
        )
    
    def _detect_skin_tone(
        self,
        image: np.ndarray,
        landmarks: Dict[int, Tuple[float, float]],
    ) -> SkinToneResult:
        """
        Detect skin tone by sampling the cheek regions.
        
        Args:
            image: BGR image.
            landmarks: Landmark coordinates.
            
        Returns:
            SkinToneResult with classification and HSV values.
        """
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Sample pixels from cheek regions
        sampled_hsv = []
        
        # Left cheek region
        for idx in FaceLandmarks.CHEEK_LEFT_REGION:
            if idx in landmarks:
                x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    sampled_hsv.append(hsv_image[y, x])
        
        # Right cheek region
        for idx in FaceLandmarks.CHEEK_RIGHT_REGION:
            if idx in landmarks:
                x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    sampled_hsv.append(hsv_image[y, x])
        
        if not sampled_hsv:
            return SkinToneResult(
                classification=SkinTone.UNKNOWN,
                hsv_values={"h": 0, "s": 0, "v": 0}
            )
        
        # Compute average HSV
        sampled_hsv = np.array(sampled_hsv)
        avg_h = float(np.mean(sampled_hsv[:, 0]))
        avg_s = float(np.mean(sampled_hsv[:, 1]))
        avg_v = float(np.mean(sampled_hsv[:, 2]))
        
        # Classify based on V (brightness) channel
        classification = self._classify_skin_tone(avg_v)
        
        return SkinToneResult(
            classification=classification,
            hsv_values={"h": avg_h, "s": avg_s, "v": avg_v}
        )
    
    def _classify_skin_tone(self, brightness: float) -> SkinTone:
        """
        Classify skin tone based on brightness value.
        
        Args:
            brightness: Average V channel value (0-255).
            
        Returns:
            SkinTone classification.
        """
        if brightness >= self.skin_thresholds.fair_min:
            return SkinTone.FAIR
        elif brightness >= self.skin_thresholds.medium_min:
            return SkinTone.MEDIUM
        elif brightness >= self.skin_thresholds.dusky_min:
            return SkinTone.DUSKY
        else:
            return SkinTone.DARK
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def extract_face_features(
    image: Union[np.ndarray, str, Path],
    skin_tone_thresholds: Optional[SkinToneThresholds] = None,
) -> FaceFeatureResult:
    """
    Convenience function to extract face features.
    
    Args:
        image: Image as numpy array (BGR) or path to image file.
        skin_tone_thresholds: Optional custom skin tone thresholds.
        
    Returns:
        FaceFeatureResult with metrics, skin tone, and detection status.
        
    Example:
        result = extract_face_features("face.jpg")
        if result.face_detected:
            print(f"Face length: {result.facial_metrics.face_length}")
            print(f"Skin tone: {result.skin_tone.classification.value}")
    """
    with FaceFeatureExtractor(skin_tone_thresholds=skin_tone_thresholds) as extractor:
        return extractor.extract(image)


def get_face_metrics_json(image: Union[np.ndarray, str, Path]) -> dict:
    """
    Extract face features and return as JSON-compatible dictionary.
    
    Args:
        image: Image as numpy array (BGR) or path to image file.
        
    Returns:
        Dictionary with all face feature data.
        
    Example:
        data = get_face_metrics_json("face.jpg")
        print(json.dumps(data, indent=2))
    """
    result = extract_face_features(image)
    return result.to_dict()


# -----------------------------------------------------------------------------
# CLI for Testing
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python face_feature_extractor.py <image_path>")
        print("Example: python face_feature_extractor.py face.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        result = extract_face_features(image_path)
        print(json.dumps(result.to_dict(), indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
