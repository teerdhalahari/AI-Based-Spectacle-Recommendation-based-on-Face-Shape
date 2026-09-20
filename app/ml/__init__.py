# ML module: face detection, face shape classification, feature extraction, recommendation.

from app.ml.face_shape_classifier import (
    FACE_SHAPE_CLASSES,
    build_model,
    load_model,
    predict_face_shape,
)

from app.ml.face_feature_extractor import (
    FaceFeatureExtractor,
    FaceFeatureResult,
    FacialMetrics,
    SkinTone,
    SkinToneThresholds,
    SkinToneResult,
    extract_face_features,
    get_face_metrics_json,
)

from app.ml.hybrid_face_service import (
    HybridFaceService,
    HybridFaceServiceError,
    get_hybrid_face_service,
    predict_face_shape as predict_face_shape_hybrid,
    # Skin tone detection
    SKIN_TONE_CATEGORIES,
    SkinToneThresholds as HybridSkinToneThresholds,
    detect_skin_tone,
    extract_cheek_region,
    compute_mean_l_channel,
    classify_skin_tone,
)

from app.ml.spectacle_recommender import (
    SpectacleRecommender,
    GlassesFrame,
    Recommendation,
    RecommendationResult,
    ScoringWeights,
    recommend_spectacles,
)

__all__ = [
    # Face shape classifier (TensorFlow - legacy)
    "FACE_SHAPE_CLASSES",
    "build_model",
    "load_model",
    "predict_face_shape",
    # Face feature extractor
    "FaceFeatureExtractor",
    "FaceFeatureResult",
    "FacialMetrics",
    "SkinTone",
    "SkinToneThresholds",
    "SkinToneResult",
    "extract_face_features",
    "get_face_metrics_json",
    # Hybrid face service (PyTorch - primary)
    "HybridFaceService",
    "HybridFaceServiceError",
    "get_hybrid_face_service",
    "predict_face_shape_hybrid",
    # Skin tone detection (LAB-based)
    "SKIN_TONE_CATEGORIES",
    "HybridSkinToneThresholds",
    "detect_skin_tone",
    "extract_cheek_region",
    "compute_mean_l_channel",
    "classify_skin_tone",
    # Spectacle recommender
    "SpectacleRecommender",
    "GlassesFrame",
    "Recommendation",
    "RecommendationResult",
    "ScoringWeights",
    "recommend_spectacles",
]
