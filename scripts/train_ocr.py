"""Train the character-recognition CNN on the Iranis dataset.

Expects one folder per class under ``data/ocr_chars/`` (0, 1, ..., Taxi, V, Y).
Writes the trained model and its label encoder to ``models/ocr/``.

Examples:
    python scripts/train_ocr.py --epochs 100
    python scripts/train_ocr.py --output-dir models/ocr_v2
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.disable(logging.WARNING)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelBinarizer  # noqa: E402
from tensorflow.keras.callbacks import (  # noqa: E402
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    TerminateOnNaN,
)
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: E402

import _bootstrap  # noqa: F401,E402  (side effect: makes `lpr` importable)
from lpr import config  # noqa: E402
from lpr.vgg import build_vggnet  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
RANDOM_SEED = 2


def load_dataset(data_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load every image under ``data_dir``, labelled by its parent folder."""
    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"No images found under {data_dir}")

    random.seed(RANDOM_SEED)
    random.shuffle(paths)

    images, labels = [], []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[warn] unreadable image: {path}")
            continue
        images.append(cv2.resize(image, (size, size)))
        labels.append(path.parent.name)

    data = np.array(images, dtype="float32") / 255.0
    return data, np.array(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=config.OCR_CHARS_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=config.MODELS_DIR / "ocr"
    )
    parser.add_argument("--log-dir", type=Path, default=config.OUTPUTS_DIR / "ocr_train_logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing trained model in the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    size = config.OCR_INPUT_SIZE

    model_path = args.output_dir / "trained_VGG_model.h5"
    labels_path = args.output_dir / "labels"

    if model_path.exists() and not args.force:
        print(
            f"{model_path} already exists.\n"
            "Pass --force to overwrite it, or --output-dir to train elsewhere."
        )
        return 1

    print(f"Loading images from {args.data_dir} ...")
    data, labels = load_dataset(args.data_dir, size)
    print(f"Loaded {len(data)} images across {len(set(labels))} classes.")

    binarizer = LabelBinarizer()
    encoded = binarizer.fit_transform(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "wb") as handle:
        pickle.dump(binarizer, handle)

    train_x, test_x, train_y, test_y = train_test_split(
        data, encoded, test_size=args.test_size, random_state=42
    )

    # No horizontal_flip: mirroring a glyph produces a character that never
    # appears on a real plate and only confuses the classifier.
    augmenter = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest",
    )

    model = build_vggnet(
        width=size, height=size, depth=3, classes=len(binarizer.classes_)
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=args.learning_rate),
        metrics=["accuracy"],
    )

    checkpoint_pattern = str(
        args.output_dir / "VGG_epoch-{epoch:02d}_val_loss-{val_loss:.4f}.h5"
    )
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_pattern,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_freq="epoch",
        ),
        TensorBoard(log_dir=str(args.log_dir), histogram_freq=0, write_graph=True),
        TerminateOnNaN(),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=10, min_lr=1e-5
        ),
    ]

    print("Training ...")
    model.fit(
        augmenter.flow(train_x, train_y, batch_size=args.batch_size),
        validation_data=(test_x, test_y),
        steps_per_epoch=len(train_x) // args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(str(model_path))
    print(f"Saved model to {model_path}")

    print("\nEvaluation:")
    predictions = model.predict(test_x, batch_size=32)
    print(
        classification_report(
            test_y.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=binarizer.classes_,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
