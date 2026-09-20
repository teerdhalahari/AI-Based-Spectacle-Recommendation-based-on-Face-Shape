"""
Example: load the trained face-shape model and run inference on an image.

Usage:
    python scripts/run_inference_example.py --model models/face_shape_mobilenetv2.h5 --image path/to/face.jpg
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.face_shape_classifier import predict_face_shape


def main():
    parser = argparse.ArgumentParser(description="Run face-shape inference on an image.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .h5 model.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument("--probs", action="store_true", help="Print class probabilities.")
    args = parser.parse_args()

    result = predict_face_shape(
        args.image,
        model_path=args.model,
        return_probs=args.probs,
    )
    if args.probs:
        label, probs = result
        print(f"Predicted: {label}")
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {p:.3f}")
    else:
        print(f"Predicted face shape: {result}")


if __name__ == "__main__":
    main()
