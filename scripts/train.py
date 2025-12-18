from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from lenet5.data import load_mnist_padded
from lenet5.model import build_lenet5


def lr_schedule(epoch: int, lr: float) -> float:
    # Keep lr for first 10 epochs, then decay by 10x.
    return lr if epoch < 10 else lr * 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--model-out", type=str, default="models/best_model.keras")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    (x_train, y_train), (x_test, y_test) = load_mnist_padded()

    model = build_lenet5()
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_out),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
    )

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint, lr_scheduler],
        verbose=2,
        shuffle=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

