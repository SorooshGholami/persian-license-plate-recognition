"""The VGG-style CNN used for single-character recognition.

Kept deliberately small: the inputs are 64x64 crops of one glyph, not natural
images, so the full VGG-16 would be wasteful.

Uses ``tensorflow.keras`` throughout -- the old standalone ``keras.layers.*``
import paths this was originally written against no longer exist.
"""

from __future__ import annotations

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential


def build_vggnet(width: int, height: int, depth: int, classes: int) -> Sequential:
    """Build the character-recognition network.

    Args:
        width: Input image width in pixels.
        height: Input image height in pixels.
        depth: Number of channels (3 for RGB).
        classes: Number of output classes.

    Returns:
        An uncompiled Keras model.
    """
    input_shape = (height, width, depth)
    channel_axis = -1

    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
        channel_axis = 1

    model = Sequential(name="vggnet_char_ocr")

    # CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    for _ in range(2):
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 3 => POOL, twice
    for _ in range(2):
        for _ in range(3):
            model.add(Conv2D(128, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=channel_axis))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    # Fully connected head
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


class VGGNet:
    """Backwards-compatible alias for the original ``VGGNet.build`` API."""

    build = staticmethod(build_vggnet)
