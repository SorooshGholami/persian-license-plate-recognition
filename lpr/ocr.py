"""Character segmentation and recognition on a cropped plate.

Segmentation is classical OpenCV (adaptive threshold + connected components);
recognition uses the VGG-style CNN trained on the Iranis dataset.

The model is loaded once per :class:`PlateOCR` instance and every character in
a plate is classified in a single batched call.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imutils
import numpy as np
from tensorflow import keras

from lpr import config

#: Bounding boxes whose tops differ by less than this are treated as one row.
_SAME_ROW_TOLERANCE = 10


@dataclass
class OcrResult:
    """Everything the OCR stage produced for one plate."""

    text: str
    """Concatenated class names, e.g. ``"12PwD35373"``."""

    annotated: np.ndarray
    """The resized plate with each character boxed and labelled."""

    confidences: list[float] = field(default_factory=list)
    """Softmax probability of each recognised character, in reading order."""

    @property
    def mean_confidence(self) -> float:
        """Average confidence, or ``0.0`` when nothing was recognised."""
        return float(np.mean(self.confidences)) if self.confidences else 0.0


def _reading_order(first: tuple, second: tuple) -> int:
    """Sort boxes top-to-bottom, then left-to-right within the same row."""
    if abs(first[1] - second[1]) > _SAME_ROW_TOLERANCE:
        return first[1] - second[1]
    return first[0] - second[0]


class PlateOCR:
    """Reads the characters off a cropped license plate."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        classes: tuple[str, ...] | None = None,
    ) -> None:
        model_path = Path(model_path or config.OCR_MODEL_PATH)
        # compile=False: we only ever run inference, and it avoids warnings
        # about the optimizer state saved with the .h5 file.
        self._model = keras.models.load_model(str(model_path), compile=False)
        self.classes = classes or config.load_ocr_classes()

    # -- segmentation -----------------------------------------------------
    def _character_mask(self, plate: np.ndarray) -> np.ndarray:
        """Isolate character-sized blobs into a binary mask."""
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            45,
            15,
        )

        _, labels = cv2.connectedComponents(thresh)
        mask = np.zeros(thresh.shape, dtype="uint8")

        total_pixels = plate.shape[0] * plate.shape[1]
        lower = total_pixels // config.SEGMENT_MIN_AREA_DIVISOR
        upper = total_pixels // config.SEGMENT_MAX_AREA_DIVISOR

        for label in np.unique(labels):
            if label == 0:  # 0 is always the background component
                continue
            component = np.zeros(thresh.shape, dtype="uint8")
            component[labels == label] = 255
            if lower < cv2.countNonZero(component) < upper:
                mask = cv2.add(mask, component)

        return mask

    def _prepare(self, mask: np.ndarray, rect: tuple) -> np.ndarray:
        """Crop one character and shape it like the training images."""
        x, y, w, h = rect
        size = config.OCR_INPUT_SIZE

        # Training data is black glyphs on white, the mask is the inverse.
        crop = cv2.bitwise_not(mask[y : y + h, x : x + w])

        rows, columns = crop.shape[:2]
        pad_y = (size - rows) // 2 if rows < size else int(0.17 * rows)
        pad_x = (size - columns) // 2 if columns < size else int(0.45 * columns)
        crop = cv2.copyMakeBorder(
            crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, None, 255
        )

        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop = cv2.resize(crop, (size, size))
        return crop.astype("float32") / 255.0

    # -- recognition ------------------------------------------------------
    def read(self, plate: np.ndarray) -> OcrResult:
        """Recognise every character on ``plate``.

        Args:
            plate: A cropped plate in BGR. It is resized internally so the
                area heuristics behave the same at any camera resolution.

        Returns:
            The recognised text plus an annotated copy of the plate.
        """
        plate = imutils.resize(plate, width=config.PLATE_RESIZE_WIDTH)
        annotated = plate.copy()

        mask = self._character_mask(plate)
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = sorted(
            (cv2.boundingRect(c) for c in contours),
            key=functools.cmp_to_key(_reading_order),
        )

        if not boxes:
            return OcrResult(text="", annotated=annotated)

        batch = np.stack([self._prepare(mask, rect) for rect in boxes])
        probabilities = self._model.predict(batch, verbose=0)

        text = ""
        confidences: list[float] = []

        for rect, probability in zip(boxes, probabilities):
            index = int(np.argmax(probability))
            character = self.classes[index]
            text += character
            confidences.append(float(probability[index]))

            x, y, w, h = rect
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated, character, (x, y + 15), 0, 0.8, (128, 0, 255), 2
            )

        return OcrResult(text=text, annotated=annotated, confidences=confidences)
