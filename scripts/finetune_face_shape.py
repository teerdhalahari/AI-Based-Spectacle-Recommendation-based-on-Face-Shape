"""
Fine-tune the pre-trained face-shape classifier.

Phase 2 of transfer learning:
1. Load the best saved model from Phase 1 (~56% val accuracy).
2. Unfreeze the last 30 layers of MobileNetV2 base.
3. Recompile with a very low learning rate (1e-5).
4. Train for additional epochs to refine learned features.

This script assumes you have already run train_face_shape.py and have a saved .h5 model.
"""

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import sys
from pathlib import Path

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
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "FaceShape Dataset" / "training_set"
DEFAULT_VAL_DIR = PROJECT_ROOT / "FaceShape Dataset" / "testing_set"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_shape_mobilenetv2.h5"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "models" / "face_shape_mobilenetv2_finetuned.h5"

BATCH_SIZE = 32
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-5
UNFREEZE_LAYERS = 30
PATIENCE = 5
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Data generators (same as training script)
# -----------------------------------------------------------------------------


def get_train_datagen(seed: int) -> ImageDataGenerator:
    """Training data generator with augmentation."""
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
    """Validation data generator: no augmentation."""
    return ImageDataGenerator(preprocessing_function=preprocess_input)


def get_generators(train_dir: Path, val_dir: Path, batch_size: int, seed: int):
    """Create train and validation generators."""
    train_datagen = get_train_datagen(seed)
    val_datagen = get_val_datagen()

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=FACE_SHAPE_CLASSES,
        seed=seed,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=FACE_SHAPE_CLASSES,
        shuffle=False,
    )
    return train_gen, val_gen


# -----------------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------------


def unfreeze_base_layers(model: keras.Model, num_layers_to_unfreeze: int) -> keras.Model:
    """
    Unfreeze the last `num_layers_to_unfreeze` layers of the MobileNetV2 base.
    
    The model structure is:
        - Layer 0: InputLayer
        - Layer 1: MobileNetV2 (Functional model)
        - Layer 2: GlobalAveragePooling2D
        - Layer 3: Dropout
        - Layer 4: Dense (predictions)
    """
    # Find the MobileNetV2 base (it's the second layer, index 1)
    base_model = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower() or isinstance(layer, keras.Model):
            base_model = layer
            break
    
    if base_model is None:
        raise ValueError("Could not find MobileNetV2 base layer in the model.")
    
    # Make the base model trainable
    base_model.trainable = True
    
    total_layers = len(base_model.layers)
    print(f"MobileNetV2 base has {total_layers} layers.")
    print(f"Freezing first {total_layers - num_layers_to_unfreeze} layers.")
    print(f"Unfreezing last {num_layers_to_unfreeze} layers for fine-tuning.")
    
    # Freeze all layers except the last `num_layers_to_unfreeze`
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Count trainable vs frozen
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    frozen_count = total_layers - trainable_count
    print(f"Trainable layers in base: {trainable_count}")
    print(f"Frozen layers in base: {frozen_count}")
    
    return model


def finetune(
    model_path: Path,
    save_path: Path,
    train_dir: Path,
    val_dir: Path,
    batch_size: int = BATCH_SIZE,
    epochs: int = FINETUNE_EPOCHS,
    learning_rate: float = FINETUNE_LR,
    unfreeze_layers: int = UNFREEZE_LAYERS,
    patience: int = PATIENCE,
    seed: int = RANDOM_SEED,
) -> keras.callbacks.History:
    """
    Load a pre-trained model, unfreeze top layers, and fine-tune.
    """
    # Step 1: Load the saved model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(str(model_path))
    print("Model loaded successfully.")
    model.summary()
    
    # Step 2: Unfreeze the last N layers of MobileNetV2
    print(f"\n--- Unfreezing last {unfreeze_layers} layers of MobileNetV2 ---")
    model = unfreeze_base_layers(model, unfreeze_layers)
    
    # Step 3: Recompile with lower learning rate
    print(f"\n--- Recompiling with Adam optimizer, lr={learning_rate} ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Step 4: Prepare data generators
    train_gen, val_gen = get_generators(train_dir, val_dir, batch_size, seed)
    
    # Step 5: Set up callbacks
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
    
    # Step 6: Fine-tune
    print(f"\n--- Starting fine-tuning for {epochs} epochs ---")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save final model
    model.save(str(save_path))
    print(f"\nFine-tuned model saved to: {save_path}")
    
    # Print final metrics
    final_val_acc = history.history["val_accuracy"][-1]
    final_val_loss = history.history["val_loss"][-1]
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nFinal val_accuracy: {final_val_acc:.4f}")
    print(f"Best val_accuracy: {best_val_acc:.4f}")
    
    return history


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune face-shape classifier (Phase 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the pre-trained .h5 model from Phase 1.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help="Path to save the fine-tuned model.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Training directory.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help="Validation directory.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--lr", type=float, default=FINETUNE_LR, dest="learning_rate")
    parser.add_argument("--unfreeze-layers", type=int, default=UNFREEZE_LAYERS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    
    finetune(
        model_path=args.model_path,
        save_path=args.save_path,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        unfreeze_layers=args.unfreeze_layers,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
