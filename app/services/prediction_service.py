"""
Unified Prediction Service.

Orchestrates the complete Face2Frame pipeline:
1. Image brightness optimization (handled by hybrid model internally)
2. Face shape prediction (PyTorch Hybrid: EfficientNet-B0 + geometric metrics)
3. Face feature extraction (MediaPipe landmarks + skin tone)
4. Spectacle recommendation

Returns a single structured response with all results.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Dict, List, Any
import logging

import cv2
import numpy as np

# Internal modules
from app.utils.image_brightness import (
    ImageBrightnessOptimizer,
    BrightnessMetadata,
    BrightnessStatus,
)
from app.ml.face_feature_extractor import (
    FaceFeatureExtractor,
    FaceFeatureResult,
    SkinToneThresholds,
)
from app.ml.hybrid_face_service import (
    HybridFaceService,
    HybridFaceServiceError,
)
from app.ml.spectacle_recommender import (
    SpectacleRecommender,
    RecommendationResult,
)


# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Complete prediction result from the unified service."""
    success: bool
    face_shape: Optional[str] = None
    face_shape_confidence: Optional[Dict[str, float]] = None
    skin_tone: Optional[str] = None
    facial_metrics: Optional[Dict[str, float]] = None
    recommendations: List[Dict] = field(default_factory=list)
    brightness_info: Optional[Dict] = None
    landmarks_detected: bool = False
    stability: Optional[float] = None
    error_message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        result = {
            "success": self.success,
            "face_shape": self.face_shape,
            "skin_tone": self.skin_tone,
            "facial_metrics": self.facial_metrics,
            "recommendations": self.recommendations,
        }
        
        # Optional fields
        if self.face_shape_confidence:
            result["face_shape_confidence"] = self.face_shape_confidence
        
        if self.brightness_info:
            result["brightness_info"] = self.brightness_info
        
        if self.stability is not None:
            result["stability"] = self.stability
        
        result["landmarks_detected"] = self.landmarks_detected
        
        if self.error_message:
            result["error"] = self.error_message
        
        if self.errors:
            result["warnings"] = self.errors
        
        return result


# -----------------------------------------------------------------------------
# Prediction Service
# -----------------------------------------------------------------------------

class PredictionService:
    """
    Unified prediction service that orchestrates the Face2Frame pipeline.
    
    Uses PyTorch Hybrid model (EfficientNet-B0 + geometric metrics) for face shape.
    
    Usage:
        service = PredictionService()
        result = service.predict("path/to/face.jpg")
        print(result.to_dict())
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        landmarker_path: Optional[Union[str, Path]] = None,
        catalog_path: Optional[Union[str, Path]] = None,
        enable_brightness_optimization: bool = True,
        skin_tone_thresholds: Optional[SkinToneThresholds] = None,
    ):
        """
        Initialize the prediction service.
        
        Args:
            model_path: Path to face_shape_hybrid_final.pth
            landmarker_path: Path to face_landmarker.task
            catalog_path: Path to glasses_metadata.xlsx or glasses_metadata.csv
            enable_brightness_optimization: Whether to optimize image brightness.
            skin_tone_thresholds: Custom skin tone classification thresholds.
        """
        self.model_path = Path(model_path) if model_path else None
        self.landmarker_path = Path(landmarker_path) if landmarker_path else None
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self.enable_brightness_optimization = enable_brightness_optimization
        self.skin_tone_thresholds = skin_tone_thresholds
        
        # Lazy-loaded components
        self._hybrid_face_service = None
        self._brightness_optimizer = None
        self._feature_extractor = None
        self._recommender = None
    
    @property
    def hybrid_face_service(self) -> HybridFaceService:
        """Lazy-load hybrid face service."""
        if self._hybrid_face_service is None:
            self._hybrid_face_service = HybridFaceService(
                model_path=self.model_path,
                landmarker_path=self.landmarker_path,
            )
        return self._hybrid_face_service
    
    @property
    def brightness_optimizer(self) -> ImageBrightnessOptimizer:
        """Lazy-load brightness optimizer."""
        if self._brightness_optimizer is None:
            self._brightness_optimizer = ImageBrightnessOptimizer()
        return self._brightness_optimizer
    
    @property
    def feature_extractor(self) -> FaceFeatureExtractor:
        """Lazy-load face feature extractor."""
        if self._feature_extractor is None:
            self._feature_extractor = FaceFeatureExtractor(
                skin_tone_thresholds=self.skin_tone_thresholds
            )
        return self._feature_extractor
    
    @property
    def recommender(self) -> SpectacleRecommender:
        """Lazy-load spectacle recommender and load catalog from Excel/CSV."""
        if self._recommender is None:
            self._recommender = SpectacleRecommender()
            if not self.catalog_path:
                raise ValueError("Catalog path not configured. Set catalog_path to data/glasses_metadata.xlsx")
            if not self.catalog_path.exists():
                raise FileNotFoundError(
                    f"Glasses catalog not found: {self.catalog_path}. "
                    "Add data/glasses_metadata.xlsx with columns: frame_id, frame_shape, frame_color, rim_type, frame_width_category, image_path"
                )
            count = self._recommender.load_catalog(self.catalog_path)
            logger.info("Catalog loaded from %s: %d frames", self.catalog_path, count)
        return self._recommender
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        
        return image
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        top_n_recommendations: int = 10,
    ) -> PredictionResult:
        """
        Run the complete prediction pipeline.
        
        Args:
            image: Image file path or numpy array (BGR).
            top_n_recommendations: Number of recommendations to return.
            
        Returns:
            PredictionResult with all predictions and recommendations.
        """
        errors: List[str] = []
        
        # Step 0: Load image if path provided
        try:
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                original_image = self._load_image(image_path)
            else:
                original_image = image.copy()
                image_path = None
        except FileNotFoundError as e:
            return PredictionResult(
                success=False,
                error_message=str(e),
            )
        except ValueError as e:
            return PredictionResult(
                success=False,
                error_message=str(e),
            )
        
        # Step 1: Face shape + skin tone prediction using Hybrid PyTorch model
        # (Hybrid model handles brightness internally via dual pipeline)
        # Skin tone is now detected using LAB color space on face crop
        face_shape = None
        face_shape_confidence = None
        stability = None
        skin_tone = None
        brightness_info = None
        
        try:
            shape_result = self.hybrid_face_service.predict_from_bgr(original_image, return_debug=True)
            
            if shape_result["success"]:
                face_shape = shape_result["face_shape"].lower()
                face_shape_confidence = shape_result.get("softmax_scores", {})
                stability = shape_result.get("stability", 0.0)
                
                # Get skin tone from hybrid service (LAB color space based)
                skin_tone = shape_result.get("skin_tone")
                skin_tone_l = shape_result.get("skin_tone_l_value")
                
                if skin_tone_l is not None:
                    logger.info(f"Skin tone detected: {skin_tone} (L={skin_tone_l:.1f})")
                
                # Check for confusion message
                confusion_msg = shape_result.get("confusion_message")
                if confusion_msg:
                    errors.append(f"Note: {confusion_msg}")
                
                logger.info(f"Face shape predicted: {face_shape} "
                           f"(confidence={shape_result['confidence']:.2%}, stability={stability:.2f})")
            else:
                error_type = shape_result.get("error_type", "unknown")
                error_msg = shape_result.get("error", "Face shape prediction failed")
                
                if error_type == "no_face":
                    return PredictionResult(
                        success=False,
                        error_message=error_msg,
                    )
                elif error_type == "multiple_faces":
                    return PredictionResult(
                        success=False,
                        error_message=error_msg,
                    )
                else:
                    errors.append(f"Face shape prediction: {error_msg}")
        
        except HybridFaceServiceError as e:
            errors.append(f"Face shape service error: {e}")
        except Exception as e:
            errors.append(f"Face shape prediction failed: {e}")
        
        # Step 2: Face feature extraction (facial metrics from landmarks)
        # Note: skin_tone is already detected in Step 1 using LAB color space
        facial_metrics = None
        landmarks_detected = False
        
        try:
            feature_result = self.feature_extractor.extract(original_image)
            
            if feature_result.face_detected:
                landmarks_detected = True
                
                if feature_result.facial_metrics:
                    facial_metrics = feature_result.facial_metrics.to_dict()
                
                # Fallback: use feature extractor skin tone only if hybrid service didn't provide one
                if not skin_tone and feature_result.skin_tone:
                    skin_tone = feature_result.skin_tone.classification.value
                    logger.info(f"Using fallback skin tone from feature extractor: {skin_tone}")
            else:
                # Only add as warning if face shape was detected (inconsistent state)
                if face_shape:
                    errors.append("Feature extraction: No face detected by landmark extractor")
        
        except Exception as e:
            errors.append(f"Face feature extraction failed: {e}")
        
        # Step 3: Brightness info (for display purposes)
        if self.enable_brightness_optimization:
            try:
                _, brightness_meta = self.brightness_optimizer.optimize(original_image)
                brightness_info = brightness_meta.to_dict()
            except Exception:
                pass  # Non-critical
        
        # Step 4: Spectacle recommendations
        recommendations = []
        
        if face_shape and skin_tone:
            try:
                rec_result = self.recommender.recommend(
                    face_shape=face_shape,
                    skin_tone=skin_tone,
                    facial_metrics=facial_metrics,
                    top_n=top_n_recommendations,
                )
                
                if rec_result.success:
                    recommendations = [r.to_dict() for r in rec_result.recommendations]
                else:
                    errors.append(f"Recommendation failed: {rec_result.error_message}")
            
            except Exception as e:
                errors.append(f"Recommendation engine failed: {e}")
        else:
            missing = []
            if not face_shape:
                missing.append("face_shape")
            if not skin_tone:
                missing.append("skin_tone")
            errors.append(f"Cannot generate recommendations: missing {', '.join(missing)}")
        
        # Determine overall success
        # Success if we have at least face_shape and skin_tone
        success = bool(face_shape and skin_tone)
        
        return PredictionResult(
            success=success,
            face_shape=face_shape,
            face_shape_confidence=face_shape_confidence,
            skin_tone=skin_tone,
            facial_metrics=facial_metrics,
            recommendations=recommendations,
            brightness_info=brightness_info,
            landmarks_detected=landmarks_detected,
            stability=stability,
            error_message=None if success else "Prediction incomplete",
            errors=errors if errors else [],
        )
    
    def _infer_shape_from_metrics(self, metrics: Dict[str, float]) -> Optional[str]:
        """
        Fallback: Infer face shape from facial metrics ratios.
        
        This is a simplified heuristic when the ML model is not available.
        """
        jaw_to_cheek = metrics.get("jaw_to_cheek_ratio", 0)
        forehead_to_cheek = metrics.get("forehead_to_cheek_ratio", 0)
        cheek_to_length = metrics.get("cheek_to_length_ratio", 0)
        
        if not (jaw_to_cheek and forehead_to_cheek and cheek_to_length):
            return None
        
        # Simplified rules based on proportions
        if cheek_to_length > 0.85:
            # Wide face
            if jaw_to_cheek > 0.9:
                return "square"
            else:
                return "round"
        elif cheek_to_length < 0.70:
            # Long face
            return "oblong"
        else:
            # Medium proportions
            if forehead_to_cheek > 0.9 and jaw_to_cheek < 0.8:
                return "heart"
            elif abs(jaw_to_cheek - forehead_to_cheek) < 0.1:
                return "oval"
            else:
                return "oval"  # Default to oval
    
    def close(self):
        """Release resources."""
        if self._hybrid_face_service:
            self._hybrid_face_service.close()
        if self._feature_extractor:
            self._feature_extractor.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -----------------------------------------------------------------------------
# Convenience Function
# -----------------------------------------------------------------------------


def predict_and_recommend(
    image: Union[str, Path, np.ndarray],
    model_path: Optional[Union[str, Path]] = None,
    landmarker_path: Optional[Union[str, Path]] = None,
    catalog_path: Optional[Union[str, Path]] = None,
    top_n: int = 10,
) -> PredictionResult:
    """
    Convenience function to run the complete prediction pipeline.
    
    Args:
        image: Image file path or numpy array (BGR).
        model_path: Path to face_shape_hybrid_final.pth
        landmarker_path: Path to face_landmarker.task
        catalog_path: Path to glasses_metadata.csv.
        top_n: Number of recommendations to return.
        
    Returns:
        PredictionResult with all predictions and recommendations.
        
    Example:
        result = predict_and_recommend("face.jpg")
        
        if result.success:
            print(f"Face shape: {result.face_shape}")
            print(f"Skin tone: {result.skin_tone}")
            for rec in result.recommendations:
                print(f"  - {rec['frame_id']}: {rec['explanation']}")
    """
    with PredictionService(
        model_path=model_path,
        landmarker_path=landmarker_path,
        catalog_path=catalog_path,
    ) as service:
        return service.predict(image, top_n_recommendations=top_n)


# -----------------------------------------------------------------------------
# CLI for Testing
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import json
    
    print("=== Face2Frame Prediction Service ===\n")
    
    if len(sys.argv) < 2:
        print("Usage: python prediction_service.py <image_path> [model_path]")
        print("Example: python prediction_service.py face.jpg models/face_shape_mobilenetv2.h5")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = predict_and_recommend(
            image_path,
            model_path=model_path,
            top_n=5,
        )
        
        print(json.dumps(result.to_dict(), indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
