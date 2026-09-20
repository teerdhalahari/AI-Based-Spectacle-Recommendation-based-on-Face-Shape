# Service modules for Face2Frame

from app.services.prediction_service import (
    PredictionService,
    PredictionResult,
    predict_and_recommend,
)

__all__ = [
    "PredictionService",
    "PredictionResult",
    "predict_and_recommend",
]
