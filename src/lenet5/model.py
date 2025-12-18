from __future__ import annotations

import tensorflow as tf


def build_lenet5(
    input_shape: tuple[int, int, int] = (32, 32, 1),
    num_classes: int = 10,
) -> tf.keras.Model:
    """Classic-ish LeNet-5 style CNN (tanh + avg pooling)."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=5,
                strides=1,
                activation="tanh",
                input_shape=input_shape,
                padding="same",
            ),
            tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="valid"),
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="same",
            ),
            tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="valid"),
            tf.keras.layers.Conv2D(
                filters=120,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="same",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(84, activation="tanh"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model

