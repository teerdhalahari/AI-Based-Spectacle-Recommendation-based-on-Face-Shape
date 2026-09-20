"""
Train face-shape classifier (MobileNetV2 transfer learning).

Expects dataset layout:
    FaceShape Dataset/
        training_set/
            Heart/, Oblong/, Oval/, Round/, Square/
        testing_set/
            Heart/, Oblong/, Oval/, Round/, Square/

- Image size: 224x224
- Data augmentation on training set only; no augmentation on validation
- Early stopping
- Saves best model as .h5
"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import sys
from pathlib import Path

# Add project root so we can import app.ml
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from app.ml.face_shape_classifier import (
    FACE_SHAPE_CLASSES,
    IMG_SIZE,
    build_model,
    NUM_CLASSES,
)


# -----------------------------------------------------------------------------
# Config (overridable by CLI)
# -----------------------------------------------------------------------------

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "FaceShape Dataset" / "training_set"
DEFAULT_VAL_DIR = PROJECT_ROOT / "FaceShape Dataset" / "testing_set"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "models" / "face_shape_mobilenetv2.h5"
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 7
LEARNING_RATE = 1e-5  # Lower LR for fine-tuning unfrozen base layers
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Data: augmentation and generators
# -----------------------------------------------------------------------------


def get_train_datagen(seed: int) -> ImageDataGenerator:
    """
    Training data generator with augmentation.
    Uses MobileNetV2 preprocess_input for correct input scaling.
    """
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def get_val_datagen() -> ImageDataGenerator:
    """Validation data generator: no augmentation, only preprocessing."""
    return ImageDataGenerator(preprocessing_function=preprocess_input)


def get_generators(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = BATCH_SIZE,
    img_size: tuple = IMG_SIZE,
    seed: int = RANDOM_SEED,
):
    """
    Create train and validation generators from separate train/val directories.

    Each directory must contain one folder per class (names must match FACE_SHAPE_CLASSES).
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_datagen = get_train_datagen(seed)
    val_datagen = get_val_datagen()

    # Class order fixed so model output index matches FACE_SHAPE_CLASSES
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=FACE_SHAPE_CLASSES,
        seed=seed,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=FACE_SHAPE_CLASSES,
        shuffle=False,
    )
    return train_gen, val_gen


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(
    train_dir: Path,
    val_dir: Path,
    save_path: Path,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
) -> keras.callbacks.History:
    """
    Build model, train with early stopping, save best weights as .h5.
    """
    train_gen, val_gen = get_generators(
        train_dir,
        val_dir,
        batch_size=batch_size,
        seed=seed,
    )

    num_classes = train_gen.num_classes
    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"Dataset has {num_classes} classes; expected {NUM_CLASSES} ({FACE_SHAPE_CLASSES})"
        )

    # Fine-tune: unfreeze last 30 layers of MobileNetV2 base
    model = build_model(num_classes=num_classes, dropout_rate=0.5, trainable_base_layers=30)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # Ensure final saved model is the best (EarlyStopping restores best weights)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")
    return history


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train face-shape classifier (MobileNetV2, 5 classes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Training directory (e.g. FaceShape Dataset/training_set).",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help="Validation/testing directory (e.g. FaceShape Dataset/testing_set).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help="Path to save the best model (.h5).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, dest="learning_rate")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        save_path=args.save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
