"""
Face shape classifier using MobileNetV2 (transfer learning).

Classes: Round, Oval, Square, Heart, Oblong.
Input: 224x224 RGB images.
Output: class label and optional probabilities.
"""

from pathlib import Path
from typing import Union

import numpy as np

# Try TensorFlow; fallback message if not installed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError:
    tf = None
    keras = None
    layers = None
    MobileNetV2 = None
    preprocess_input = None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

IMG_SIZE = (224, 224)
FACE_SHAPE_CLASSES = [
    "Round",
    "Oval",
    "Square",
    "Heart",
    "Oblong",
]
NUM_CLASSES = len(FACE_SHAPE_CLASSES)


# -----------------------------------------------------------------------------
# Model build
# -----------------------------------------------------------------------------


def build_model(
    input_shape: tuple = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.5,
    trainable_base_layers: int = 0,
) -> keras.Model:
    """
    Build MobileNetV2-based classifier for face shapes.

    - Uses ImageNet-pretrained MobileNetV2 as feature extractor.
    - Adds global pooling, dropout, and a dense softmax head.
    - By default the base is frozen; set trainable_base_layers > 0 to fine-tune.

    Args:
        input_shape: (height, width, channels), default (224, 224, 3).
        num_classes: Number of output classes (default 5).
        dropout_rate: Dropout before the final dense layer.
        trainable_base_layers: Number of top layers of the base to unfreeze (0 = frozen).

    Returns:
        Compiled Keras Model.
    """
    if keras is None:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    # Base: ImageNet-pretrained MobileNetV2 (no top)
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    # Fine-tuning: unfreeze top layers of the base model
    base.trainable = True
    if trainable_base_layers > 0:
        # Freeze all layers except the last `trainable_base_layers`
        for layer in base.layers[:-trainable_base_layers]:
            layer.trainable = False
    # If trainable_base_layers == 0, all base layers remain trainable

    inputs = keras.Input(shape=input_shape)
    # training=True so BatchNorm layers update statistics during fine-tuning
    x = base(inputs, training=True)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="face_shape_mobilenetv2")
    return model


# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------


def load_model(model_path: Union[str, Path]) -> keras.Model:
    """
    Load a saved face-shape classifier (.h5 or SavedModel).

    Args:
        model_path: Path to .h5 file or SavedModel directory.

    Returns:
        Loaded Keras Model.
    """
    if keras is None:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if path.suffix == ".h5" or path.name.endswith(".h5"):
        return keras.models.load_model(str(path))
    return keras.models.load_model(str(path))


# -----------------------------------------------------------------------------
# Preprocess image for inference
# -----------------------------------------------------------------------------


def _preprocess_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Load and preprocess a single image to (1, 224, 224, 3) for the model.

    - Resizes to IMG_SIZE.
    - Applies MobileNetV2 preprocess_input (scale to [-1, 1]).
    """
    if keras is None or preprocess_input is None:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    if isinstance(image, (str, Path)):
        img = keras.utils.load_img(str(image), target_size=IMG_SIZE, color_mode="rgb")
        img = keras.utils.img_to_array(img)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        img = tf.image.resize(image, IMG_SIZE).numpy()
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
    else:
        raise TypeError("image must be path (str/Path) or numpy array")

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype(np.float32))
    return img


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def predict_face_shape(
    image: Union[str, Path, np.ndarray],
    model: Union[keras.Model, str, Path, None] = None,
    model_path: Union[str, Path, None] = None,
    return_probs: bool = False,
) -> Union[str, tuple]:
    """
    Predict face shape class for one image.

    Args:
        image: File path (str/Path) or numpy array (H, W) or (H, W, 3).
        model: Loaded Keras model, or None to load from model_path.
        model_path: Path to .h5 (or SavedModel). Used only if model is None.
        return_probs: If True, return (class_name, probabilities_dict).

    Returns:
        If return_probs is False: predicted class name (e.g. "Oval").
        If return_probs is True: (class_name, {class_name: prob, ...}).
    """
    if model is None:
        path = model_path
        if path is None:
            raise ValueError("Either model or model_path must be provided.")
        model = load_model(path)

    x = _preprocess_image(image)
    logits = model(x, training=False)
    probs = tf.nn.softmax(logits).numpy()[0]

    class_indices = range(len(FACE_SHAPE_CLASSES))
    prob_dict = {FACE_SHAPE_CLASSES[i]: float(probs[i]) for i in class_indices}
    predicted_idx = int(np.argmax(probs))
    class_name = FACE_SHAPE_CLASSES[predicted_idx]

    if return_probs:
        return class_name, prob_dict
    return class_name
