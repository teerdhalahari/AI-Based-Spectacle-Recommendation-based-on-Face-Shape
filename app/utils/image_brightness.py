"""
Image Brightness Optimization Module.

Analyzes image brightness and applies corrections:
- Histogram equalization for dark images
- Gamma correction for bright images

Uses OpenCV for image processing.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Optional

import cv2
import numpy as np


class BrightnessStatus(Enum):
    """Classification of image brightness."""
    DARK = "dark"
    NORMAL = "normal"
    BRIGHT = "bright"


@dataclass
class BrightnessMetadata:
    """Metadata about image brightness analysis and optimization."""
    original_brightness: float
    optimized_brightness: Optional[float]
    status: BrightnessStatus
    correction_applied: Optional[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_brightness": round(self.original_brightness, 2),
            "optimized_brightness": round(self.optimized_brightness, 2) if self.optimized_brightness else None,
            "status": self.status.value,
            "correction_applied": self.correction_applied,
        }


@dataclass
class BrightnessThresholds:
    """Configurable thresholds for brightness detection."""
    dark_threshold: float = 60.0      # Below this = dark (0-255 scale)
    bright_threshold: float = 190.0   # Above this = bright (0-255 scale)
    
    def __post_init__(self):
        if not (0 <= self.dark_threshold < self.bright_threshold <= 255):
            raise ValueError(
                f"Invalid thresholds: dark={self.dark_threshold}, bright={self.bright_threshold}. "
                "Must satisfy: 0 <= dark < bright <= 255"
            )


class ImageBrightnessOptimizer:
    """
    Analyzes and optimizes image brightness.
    
    Usage:
        optimizer = ImageBrightnessOptimizer()
        optimized_img, metadata = optimizer.optimize(image)
        
        # With custom thresholds
        optimizer = ImageBrightnessOptimizer(
            thresholds=BrightnessThresholds(dark_threshold=50, bright_threshold=200)
        )
    """
    
    def __init__(
        self,
        thresholds: Optional[BrightnessThresholds] = None,
        gamma_dark: float = 0.7,
        gamma_bright: float = 1.5,
    ):
        """
        Initialize the brightness optimizer.
        
        Args:
            thresholds: Custom brightness thresholds. Uses defaults if None.
            gamma_dark: Gamma value for brightening dark images (< 1 brightens).
            gamma_bright: Gamma value for darkening bright images (> 1 darkens).
        """
        self.thresholds = thresholds or BrightnessThresholds()
        self.gamma_dark = gamma_dark
        self.gamma_bright = gamma_bright
    
    def compute_brightness(self, image: np.ndarray) -> float:
        """
        Compute mean brightness/luminance of an image.
        
        Uses the V (Value) channel from HSV color space for color images,
        or direct mean for grayscale images.
        
        Args:
            image: BGR or grayscale image (numpy array).
            
        Returns:
            Mean brightness value (0-255 scale).
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: empty or None")
        
        if len(image.shape) == 2:
            # Grayscale image
            return float(np.mean(image))
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # BGR color image -> convert to HSV and use V channel
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            return float(np.mean(v_channel))
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # BGRA -> convert to BGR first
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            return float(np.mean(hsv[:, :, 2]))
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    
    def classify_brightness(self, brightness: float) -> BrightnessStatus:
        """
        Classify brightness level based on thresholds.
        
        Args:
            brightness: Mean brightness value (0-255).
            
        Returns:
            BrightnessStatus enum value.
        """
        if brightness < self.thresholds.dark_threshold:
            return BrightnessStatus.DARK
        elif brightness > self.thresholds.bright_threshold:
            return BrightnessStatus.BRIGHT
        return BrightnessStatus.NORMAL
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to brighten dark images.
        
        CLAHE is preferred over standard histogram equalization as it:
        - Limits contrast amplification to reduce noise
        - Works on small tiles for better local contrast
        
        Args:
            image: BGR or grayscale image.
            
        Returns:
            Brightness-corrected image.
        """
        if len(image.shape) == 2:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        
        # Color image: apply CLAHE to L channel in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_corrected = clahe.apply(l_channel)
        
        lab_corrected = cv2.merge([l_corrected, a_channel, b_channel])
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to adjust image brightness.
        
        - gamma < 1: brightens the image
        - gamma > 1: darkens the image
        - gamma = 1: no change
        
        Args:
            image: BGR or grayscale image.
            gamma: Gamma correction value.
            
        Returns:
            Gamma-corrected image.
        """
        # Build lookup table for efficiency
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        
        return cv2.LUT(image, table)
    
    def optimize(
        self,
        image: Union[np.ndarray, str, Path],
    ) -> Tuple[np.ndarray, BrightnessMetadata]:
        """
        Analyze image brightness and apply appropriate correction.
        
        Args:
            image: Image as numpy array (BGR), or path to image file.
            
        Returns:
            Tuple of (optimized_image, metadata).
            
        Raises:
            ValueError: If image is invalid or cannot be loaded.
            FileNotFoundError: If image path does not exist.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        if image is None or image.size == 0:
            raise ValueError("Invalid image: empty or None")
        
        # Compute original brightness
        original_brightness = self.compute_brightness(image)
        status = self.classify_brightness(original_brightness)
        
        # Apply correction based on status
        optimized_image = image.copy()
        correction_applied = None
        optimized_brightness = None
        
        if status == BrightnessStatus.DARK:
            optimized_image = self.apply_histogram_equalization(image)
            correction_applied = "histogram_equalization_clahe"
            optimized_brightness = self.compute_brightness(optimized_image)
            
        elif status == BrightnessStatus.BRIGHT:
            optimized_image = self.apply_gamma_correction(image, self.gamma_bright)
            correction_applied = f"gamma_correction_{self.gamma_bright}"
            optimized_brightness = self.compute_brightness(optimized_image)
            
        else:
            # Normal brightness, no correction needed
            optimized_brightness = original_brightness
            correction_applied = None
        
        metadata = BrightnessMetadata(
            original_brightness=original_brightness,
            optimized_brightness=optimized_brightness,
            status=status,
            correction_applied=correction_applied,
        )
        
        return optimized_image, metadata
    
    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        
        return image


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------


def optimize_brightness(
    image: Union[np.ndarray, str, Path],
    dark_threshold: float = 60.0,
    bright_threshold: float = 190.0,
) -> Tuple[np.ndarray, BrightnessMetadata]:
    """
    Convenience function to optimize image brightness.
    
    Args:
        image: Image as numpy array (BGR) or path to image file.
        dark_threshold: Brightness below this is considered dark (0-255).
        bright_threshold: Brightness above this is considered bright (0-255).
        
    Returns:
        Tuple of (optimized_image, metadata).
        
    Example:
        optimized, meta = optimize_brightness("photo.jpg")
        print(f"Status: {meta.status.value}, Original: {meta.original_brightness}")
        cv2.imwrite("photo_optimized.jpg", optimized)
    """
    optimizer = ImageBrightnessOptimizer(
        thresholds=BrightnessThresholds(
            dark_threshold=dark_threshold,
            bright_threshold=bright_threshold,
        )
    )
    return optimizer.optimize(image)


def compute_brightness(image: Union[np.ndarray, str, Path]) -> float:
    """
    Convenience function to compute image brightness.
    
    Args:
        image: Image as numpy array (BGR) or path to image file.
        
    Returns:
        Mean brightness value (0-255 scale).
    """
    optimizer = ImageBrightnessOptimizer()
    if isinstance(image, (str, Path)):
        image = optimizer._load_image(image)
    return optimizer.compute_brightness(image)


# -----------------------------------------------------------------------------
# CLI for testing
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_brightness.py <image_path> [output_path]")
        print("Example: python image_brightness.py input.jpg output.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        optimized, metadata = optimize_brightness(input_path)
        
        print(f"Original brightness: {metadata.original_brightness:.2f}")
        print(f"Status: {metadata.status.value}")
        print(f"Correction applied: {metadata.correction_applied or 'None'}")
        print(f"Optimized brightness: {metadata.optimized_brightness:.2f}")
        
        if output_path:
            cv2.imwrite(output_path, optimized)
            print(f"Saved optimized image to: {output_path}")
        else:
            print("No output path specified. Image not saved.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
