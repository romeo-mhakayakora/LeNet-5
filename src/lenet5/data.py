from __future__ import annotations

import numpy as np
import tensorflow as tf


def load_mnist_padded(
    num_classes: int = 10,
    pad: int = 2,
    normalize: bool = True,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load MNIST and preprocess for LeNet-5.

    - Pads 28x28 to 32x32 by default.
    - Adds a channel dimension.
    - Normalizes to [0, 1] float32.
    - One-hot encodes labels.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if pad:
        x_train = np.pad(x_train, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
        x_test = np.pad(x_test, ((0, 0), (pad, pad), (pad, pad)), mode="constant")

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype("float32")
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype("float32")

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

