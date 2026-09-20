"""
Rule-Based Spectacle Recommendation Engine.

Recommends spectacles based on:
- Face shape compatibility
- Skin tone color matching
- Facial metrics (width ratios)

Fully explainable with weighted scoring system.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import csv
import math

import pandas as pd


# -----------------------------------------------------------------------------
# Enums and Constants
# -----------------------------------------------------------------------------

class FaceShape(Enum):
    """Supported face shapes."""
    ROUND = "round"
    OVAL = "oval"
    SQUARE = "square"
    HEART = "heart"
    OBLONG = "oblong"


class SkinTone(Enum):
    """Supported skin tones (LAB L-channel based classification)."""
    FAIR = "fair"       # L >= 75
    LIGHT = "light"     # L >= 65
    MEDIUM = "medium"   # L >= 50
    BROWN = "brown"     # L >= 35
    DARK = "dark"       # L < 35
    # Legacy aliases for backward compatibility
    DUSKY = "dusky"     # Maps to brown in rules


class FrameShape(Enum):
    """Frame shape categories."""
    RECTANGULAR = "rectangular"
    SQUARE = "square"
    ROUND = "round"
    OVAL = "oval"
    CAT_EYE = "cat-eye"
    AVIATOR = "aviator"
    WAYFARER = "wayfarer"
    BROWLINE = "browline"
    GEOMETRIC = "geometric"


class FrameColor(Enum):
    """Frame color categories."""
    BLACK = "black"
    BROWN = "brown"
    GOLD = "gold"
    SILVER = "silver"
    TORTOISE = "tortoise"
    TRANSPARENT = "transparent"
    BLUE = "blue"
    RED = "red"
    PINK = "pink"
    GREEN = "green"
    WHITE = "white"


class FrameWidth(Enum):
    """Frame width categories."""
    NARROW = "narrow"
    MEDIUM = "medium"
    WIDE = "wide"


# -----------------------------------------------------------------------------
# Scoring Weights
# -----------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """Configurable weights for scoring components."""
    shape_match: float = 0.50    # Highest priority
    color_match: float = 0.30    # Second priority
    width_match: float = 0.20    # Third priority
    
    def __post_init__(self):
        total = self.shape_match + self.color_match + self.width_match
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# -----------------------------------------------------------------------------
# Rule Definitions
# -----------------------------------------------------------------------------

# Face Shape → Recommended Frame Shapes (with compatibility scores 0-1)
FACE_SHAPE_RULES: Dict[str, Dict[str, float]] = {
    "round": {
        # Angular frames to add definition
        "rectangular": 1.0,
        "square": 0.9,
        "wayfarer": 0.85,
        "browline": 0.8,
        "geometric": 0.75,
        "cat-eye": 0.6,
        "aviator": 0.5,
        # Avoid round frames (too similar)
        "round": 0.2,
        "oval": 0.3,
    },
    "square": {
        # Round/oval frames to soften angles
        "round": 1.0,
        "oval": 0.95,
        "aviator": 0.85,
        "cat-eye": 0.7,
        "browline": 0.6,
        # Avoid angular frames
        "rectangular": 0.3,
        "square": 0.2,
        "geometric": 0.4,
        "wayfarer": 0.5,
    },
    "oval": {
        # Versatile - most frames work
        "rectangular": 0.9,
        "square": 0.85,
        "round": 0.85,
        "oval": 0.8,
        "wayfarer": 0.9,
        "aviator": 0.85,
        "cat-eye": 0.8,
        "browline": 0.85,
        "geometric": 0.8,
    },
    "heart": {
        # Bottom-heavy frames to balance wide forehead
        "aviator": 1.0,
        "round": 0.85,
        "oval": 0.8,
        "wayfarer": 0.75,
        "cat-eye": 0.5,  # Top-heavy, less ideal
        "browline": 0.4,  # Top-heavy
        "rectangular": 0.6,
        "square": 0.5,
        "geometric": 0.6,
    },
    "oblong": {
        # Taller/deeper frames to shorten face visually
        "aviator": 0.95,
        "round": 0.9,
        "oval": 0.85,
        "wayfarer": 1.0,
        "browline": 0.8,
        "cat-eye": 0.75,
        # Narrow frames make face look longer
        "rectangular": 0.5,
        "square": 0.55,
        "geometric": 0.7,
    },
}

# Skin Tone → Recommended Frame Colors (with compatibility scores 0-1)
SKIN_TONE_COLOR_RULES: Dict[str, Dict[str, float]] = {
    "fair": {
        # Soft, cool colors work best
        "black": 0.7,
        "brown": 0.6,
        "tortoise": 0.85,
        "gold": 0.5,
        "silver": 0.95,
        "transparent": 0.9,
        "blue": 0.85,
        "pink": 0.8,
        "red": 0.6,
        "green": 0.7,
        "white": 0.5,
    },
    "light": {
        # Balanced colors, slightly warmer than fair
        "black": 0.75,
        "brown": 0.75,
        "tortoise": 0.9,
        "gold": 0.7,
        "silver": 0.9,
        "transparent": 0.85,
        "blue": 0.8,
        "pink": 0.75,
        "red": 0.65,
        "green": 0.75,
        "white": 0.55,
    },
    "medium": {
        # Warm earth tones and metals
        "black": 0.85,
        "brown": 0.9,
        "tortoise": 1.0,
        "gold": 0.9,
        "silver": 0.7,
        "transparent": 0.75,
        "blue": 0.7,
        "pink": 0.6,
        "red": 0.75,
        "green": 0.8,
        "white": 0.6,
    },
    "brown": {
        # Rich, warm colors (similar to dusky)
        "black": 0.9,
        "brown": 0.95,
        "tortoise": 0.9,
        "gold": 1.0,
        "silver": 0.6,
        "transparent": 0.5,
        "blue": 0.65,
        "pink": 0.5,
        "red": 0.8,
        "green": 0.85,
        "white": 0.55,
    },
    "dusky": {
        # Legacy alias - same as brown
        "black": 0.9,
        "brown": 0.95,
        "tortoise": 0.9,
        "gold": 1.0,
        "silver": 0.6,
        "transparent": 0.5,
        "blue": 0.65,
        "pink": 0.5,
        "red": 0.8,
        "green": 0.85,
        "white": 0.55,
    },
    "dark": {
        # Bold, high-contrast colors
        "black": 0.85,
        "brown": 0.8,
        "tortoise": 0.75,
        "gold": 1.0,
        "silver": 0.7,
        "transparent": 0.6,
        "blue": 0.7,
        "pink": 0.6,
        "red": 0.85,
        "green": 0.8,
        "white": 0.9,
    },
}

# Facial Width Ratio → Frame Width Compatibility
# cheek_to_length_ratio thresholds
WIDTH_RATIO_RULES = {
    "narrow": (0.0, 0.70),    # Narrow face
    "medium": (0.70, 0.85),   # Medium face
    "wide": (0.85, 1.5),      # Wide face
}


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class GlassesFrame:
    """Represents a glasses frame from the catalog."""
    frame_id: str
    frame_shape: str
    frame_color: str
    rim_type: str
    frame_width_category: str
    image_path: str
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> "GlassesFrame":
        """Create from CSV row dictionary. Handles pandas NaN via _str()."""
        def _str(v: Any) -> str:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            return str(v).strip()

        image_path = _str(row.get("image_path")) or "/static/glasses/placeholder.jpg"
        if image_path and not image_path.startswith("/"):
            image_path = "/static/glasses/" + image_path.lstrip("/")
        return cls(
            frame_id=_str(row.get("frame_id")),
            frame_shape=_str(row.get("frame_shape")).lower(),
            frame_color=_str(row.get("frame_color")).lower(),
            rim_type=_str(row.get("rim_type")).lower(),
            frame_width_category=_str(row.get("frame_width_category")) or "medium",
            image_path=image_path,
        )


@dataclass
class ScoreBreakdown:
    """Detailed scoring breakdown for explainability."""
    shape_score: float = 0.0
    shape_reason: str = ""
    color_score: float = 0.0
    color_reason: str = ""
    width_score: float = 0.0
    width_reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            "shape": {"score": round(self.shape_score, 3), "reason": self.shape_reason},
            "color": {"score": round(self.color_score, 3), "reason": self.color_reason},
            "width": {"score": round(self.width_score, 3), "reason": self.width_reason},
        }


@dataclass
class Recommendation:
    """A single recommendation with score and explanation."""
    frame: GlassesFrame
    total_score: float
    breakdown: ScoreBreakdown
    explanation: str
    rank: int = 0
    
    def to_dict(self) -> dict:
        score = round(self.total_score, 3)
        return {
            "rank": self.rank,
            "frame_id": self.frame.frame_id,
            "frame_shape": self.frame.frame_shape,
            "frame_color": self.frame.frame_color,
            "rim_type": self.frame.rim_type,
            "frame_width": self.frame.frame_width_category,
            "image_path": self.frame.image_path,
            "score": score,
            "total_score": score,
            "score_breakdown": self.breakdown.to_dict(),
            "explanation": self.explanation,
        }


@dataclass
class RecommendationResult:
    """Complete recommendation result."""
    success: bool
    recommendations: List[Recommendation] = field(default_factory=list)
    input_summary: Dict = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "input_summary": self.input_summary,
            "recommendations_count": len(self.recommendations),
            "recommendations": [r.to_dict() for r in self.recommendations],
        }
        if self.error_message:
            result["error"] = self.error_message
        return result


# -----------------------------------------------------------------------------
# Recommendation Engine
# -----------------------------------------------------------------------------

class SpectacleRecommender:
    """
    Rule-based spectacle recommendation engine.
    
    Usage:
        recommender = SpectacleRecommender()
        recommender.load_catalog("glasses_metadata.csv")
        
        result = recommender.recommend(
            face_shape="round",
            skin_tone="medium",
            facial_metrics={"cheek_to_length_ratio": 0.75}
        )
        
        for rec in result.recommendations:
            print(f"{rec.rank}. {rec.frame.frame_id} - Score: {rec.total_score}")
            print(f"   {rec.explanation}")
    """
    
    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        default_width_score: float = 0.5,
    ):
        """
        Initialize the recommender.
        
        Args:
            weights: Custom scoring weights. Uses defaults if None.
            default_width_score: Score when width matching is not possible.
        """
        self.weights = weights or ScoringWeights()
        self.default_width_score = default_width_score
        self.catalog: List[GlassesFrame] = []
    
    def load_catalog(self, catalog_path: Union[str, Path]) -> int:
        """
        Load glasses catalog from CSV or Excel file.
        
        Supports:
            - .xlsx, .xls via pandas (columns: frame_id, frame_shape, frame_color,
              rim_type, frame_width_category, image_path)
            - .csv via csv.DictReader (same columns)
        
        Args:
            catalog_path: Path to glasses_metadata.csv or glasses_metadata.xlsx
            
        Returns:
            Number of frames loaded.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        catalog_path = Path(catalog_path)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")
        
        self.catalog = []
        suffix = catalog_path.suffix.lower()

        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(catalog_path)
            for _, row in df.iterrows():
                try:
                    d = row.to_dict()
                    frame = GlassesFrame.from_csv_row(d)
                    if frame.frame_id:
                        self.catalog.append(frame)
                except Exception:
                    continue
        else:
            with open(catalog_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        frame = GlassesFrame.from_csv_row(row)
                        if frame.frame_id:
                            self.catalog.append(frame)
                    except Exception:
                        continue

        return len(self.catalog)
    
    def set_catalog(self, frames: List[GlassesFrame]) -> None:
        """Set catalog directly from list of frames."""
        self.catalog = frames
    
    def recommend(
        self,
        face_shape: str,
        skin_tone: str,
        facial_metrics: Optional[Dict[str, float]] = None,
        top_n: int = 5,
    ) -> RecommendationResult:
        """
        Generate spectacle recommendations.
        
        Args:
            face_shape: Face shape (round, oval, square, heart, oblong).
            skin_tone: Skin tone (fair, medium, dusky, dark).
            facial_metrics: Dictionary with ratios like 'cheek_to_length_ratio'.
            top_n: Number of recommendations to return.
            
        Returns:
            RecommendationResult with scored and ranked recommendations.
        """
        # Validate inputs
        face_shape = face_shape.lower().strip()
        skin_tone = skin_tone.lower().strip()
        facial_metrics = facial_metrics or {}
        
        input_summary = {
            "face_shape": face_shape,
            "skin_tone": skin_tone,
            "facial_metrics": facial_metrics,
        }
        
        # Check catalog
        if not self.catalog:
            return RecommendationResult(
                success=False,
                input_summary=input_summary,
                error_message="No glasses catalog loaded. Call load_catalog() first."
            )
        
        # Validate face shape
        if face_shape not in FACE_SHAPE_RULES:
            return RecommendationResult(
                success=False,
                input_summary=input_summary,
                error_message=f"Unknown face shape: {face_shape}. "
                              f"Supported: {list(FACE_SHAPE_RULES.keys())}"
            )
        
        # Validate skin tone
        if skin_tone not in SKIN_TONE_COLOR_RULES:
            return RecommendationResult(
                success=False,
                input_summary=input_summary,
                error_message=f"Unknown skin tone: {skin_tone}. "
                              f"Supported: {list(SKIN_TONE_COLOR_RULES.keys())}"
            )
        
        # Determine face width category from metrics
        face_width_category = self._determine_width_category(facial_metrics)
        
        # Score all frames
        scored_frames: List[Tuple[float, ScoreBreakdown, GlassesFrame]] = []
        
        for frame in self.catalog:
            score, breakdown = self._score_frame(
                frame, face_shape, skin_tone, face_width_category
            )
            scored_frames.append((score, breakdown, frame))
        
        # Sort by score descending
        scored_frames.sort(key=lambda x: x[0], reverse=True)
        
        # Build recommendations
        recommendations = []
        for rank, (score, breakdown, frame) in enumerate(scored_frames[:top_n], start=1):
            explanation = self._generate_explanation(
                frame, face_shape, skin_tone, breakdown
            )
            recommendations.append(Recommendation(
                frame=frame,
                total_score=score,
                breakdown=breakdown,
                explanation=explanation,
                rank=rank,
            ))
        
        return RecommendationResult(
            success=True,
            recommendations=recommendations,
            input_summary=input_summary,
        )
    
    def _determine_width_category(self, metrics: Dict[str, float]) -> Optional[str]:
        """Determine face width category from facial metrics."""
        ratio = metrics.get("cheek_to_length_ratio")
        if ratio is None:
            return None
        
        for category, (min_val, max_val) in WIDTH_RATIO_RULES.items():
            if min_val <= ratio < max_val:
                return category
        
        return "medium"  # Default fallback
    
    def _score_frame(
        self,
        frame: GlassesFrame,
        face_shape: str,
        skin_tone: str,
        face_width_category: Optional[str],
    ) -> Tuple[float, ScoreBreakdown]:
        """
        Score a single frame against the input criteria.
        
        Returns:
            Tuple of (total_score, breakdown).
        """
        breakdown = ScoreBreakdown()
        
        # 1. Shape compatibility score
        shape_rules = FACE_SHAPE_RULES.get(face_shape, {})
        shape_score = shape_rules.get(frame.frame_shape, 0.5)  # Default 0.5 for unknown
        breakdown.shape_score = shape_score
        
        if shape_score >= 0.8:
            breakdown.shape_reason = f"'{frame.frame_shape}' is highly recommended for '{face_shape}' face"
        elif shape_score >= 0.6:
            breakdown.shape_reason = f"'{frame.frame_shape}' is a good match for '{face_shape}' face"
        elif shape_score >= 0.4:
            breakdown.shape_reason = f"'{frame.frame_shape}' is acceptable for '{face_shape}' face"
        else:
            breakdown.shape_reason = f"'{frame.frame_shape}' is not ideal for '{face_shape}' face"
        
        # 2. Color compatibility score
        color_rules = SKIN_TONE_COLOR_RULES.get(skin_tone, {})
        color_score = color_rules.get(frame.frame_color, 0.5)  # Default 0.5 for unknown
        breakdown.color_score = color_score
        
        if color_score >= 0.8:
            breakdown.color_reason = f"'{frame.frame_color}' complements '{skin_tone}' skin tone beautifully"
        elif color_score >= 0.6:
            breakdown.color_reason = f"'{frame.frame_color}' works well with '{skin_tone}' skin tone"
        else:
            breakdown.color_reason = f"'{frame.frame_color}' is less ideal for '{skin_tone}' skin tone"
        
        # 3. Width compatibility score
        if face_width_category:
            width_score = self._calculate_width_score(
                frame.frame_width_category, face_width_category
            )
            breakdown.width_score = width_score
            
            if width_score >= 0.8:
                breakdown.width_reason = f"'{frame.frame_width_category}' width matches your face proportions well"
            elif width_score >= 0.5:
                breakdown.width_reason = f"'{frame.frame_width_category}' width is acceptable for your face"
            else:
                breakdown.width_reason = f"'{frame.frame_width_category}' width may not be optimal for your face proportions"
        else:
            breakdown.width_score = self.default_width_score
            breakdown.width_reason = "Width matching not applied (no facial metrics provided)"
        
        # Calculate weighted total
        total_score = (
            self.weights.shape_match * breakdown.shape_score +
            self.weights.color_match * breakdown.color_score +
            self.weights.width_match * breakdown.width_score
        )
        
        return total_score, breakdown
    
    def _calculate_width_score(self, frame_width: str, face_width: str) -> float:
        """
        Calculate width compatibility score.
        
        Best: frame width matches face width
        Good: frame width is one category different
        Poor: frame width is two categories different
        """
        width_order = ["narrow", "medium", "wide"]
        
        try:
            frame_idx = width_order.index(frame_width.lower())
            face_idx = width_order.index(face_width.lower())
        except ValueError:
            return self.default_width_score
        
        diff = abs(frame_idx - face_idx)
        
        if diff == 0:
            return 1.0   # Perfect match
        elif diff == 1:
            return 0.6   # One category off
        else:
            return 0.3   # Two categories off
    
    def _generate_explanation(
        self,
        frame: GlassesFrame,
        face_shape: str,
        skin_tone: str,
        breakdown: ScoreBreakdown,
    ) -> str:
        """Generate human-readable explanation for the recommendation."""
        parts = []
        
        # Shape explanation
        if breakdown.shape_score >= 0.7:
            parts.append(f"The {frame.frame_shape} shape works great for your {face_shape} face")
        elif breakdown.shape_score >= 0.5:
            parts.append(f"The {frame.frame_shape} shape is suitable for your {face_shape} face")
        
        # Color explanation
        if breakdown.color_score >= 0.7:
            parts.append(f"the {frame.frame_color} color complements your {skin_tone} skin tone")
        elif breakdown.color_score >= 0.5:
            parts.append(f"the {frame.frame_color} color pairs acceptably with your skin tone")
        
        # Width explanation
        if breakdown.width_score >= 0.8:
            parts.append(f"and the {frame.frame_width_category} width fits your face proportions")
        
        if not parts:
            return "This frame is a reasonable choice based on overall compatibility."
        
        # Capitalize first part and join
        explanation = parts[0].capitalize()
        if len(parts) > 1:
            explanation += ", " + ", ".join(parts[1:])
        explanation += "."
        
        return explanation


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def recommend_spectacles(
    face_shape: str,
    skin_tone: str,
    facial_metrics: Optional[Dict[str, float]] = None,
    catalog_path: Optional[Union[str, Path]] = None,
    catalog_frames: Optional[List[GlassesFrame]] = None,
    top_n: int = 5,
) -> RecommendationResult:
    """
    Convenience function to get spectacle recommendations.
    
    Args:
        face_shape: Face shape (round, oval, square, heart, oblong).
        skin_tone: Skin tone (fair, medium, dusky, dark).
        facial_metrics: Optional dict with ratios like 'cheek_to_length_ratio'.
        catalog_path: Path to glasses_metadata.csv.
        catalog_frames: Or provide frames directly as list.
        top_n: Number of recommendations to return.
        
    Returns:
        RecommendationResult with scored recommendations.
        
    Example:
        result = recommend_spectacles(
            face_shape="round",
            skin_tone="medium",
            facial_metrics={"cheek_to_length_ratio": 0.78},
            catalog_path="data/glasses_metadata.csv",
        )
        
        for rec in result.recommendations:
            print(f"{rec.rank}. {rec.frame.frame_id} - {rec.explanation}")
    """
    recommender = SpectacleRecommender()
    
    if catalog_path:
        recommender.load_catalog(catalog_path)
    elif catalog_frames:
        recommender.set_catalog(catalog_frames)
    else:
        return RecommendationResult(
            success=False,
            error_message="Either catalog_path or catalog_frames must be provided."
        )
    
    return recommender.recommend(
        face_shape=face_shape,
        skin_tone=skin_tone,
        facial_metrics=facial_metrics,
        top_n=top_n,
    )


# -----------------------------------------------------------------------------
# CLI for Testing
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import json
    import sys
    
    # Demo with sample data if no catalog
    print("=== Spectacle Recommender Demo ===\n")
    
    # Create sample catalog
    sample_frames = [
        GlassesFrame("F001", "rectangular", "black", "full-rim", "medium", "/images/f001.jpg"),
        GlassesFrame("F002", "round", "gold", "full-rim", "narrow", "/images/f002.jpg"),
        GlassesFrame("F003", "aviator", "silver", "full-rim", "wide", "/images/f003.jpg"),
        GlassesFrame("F004", "wayfarer", "tortoise", "full-rim", "medium", "/images/f004.jpg"),
        GlassesFrame("F005", "cat-eye", "black", "full-rim", "narrow", "/images/f005.jpg"),
        GlassesFrame("F006", "oval", "brown", "semi-rimless", "medium", "/images/f006.jpg"),
        GlassesFrame("F007", "square", "blue", "full-rim", "wide", "/images/f007.jpg"),
        GlassesFrame("F008", "browline", "black", "semi-rimless", "medium", "/images/f008.jpg"),
    ]
    
    recommender = SpectacleRecommender()
    recommender.set_catalog(sample_frames)
    
    # Test recommendation
    result = recommender.recommend(
        face_shape="round",
        skin_tone="medium",
        facial_metrics={"cheek_to_length_ratio": 0.75},
        top_n=5,
    )
    
    print(json.dumps(result.to_dict(), indent=2))
