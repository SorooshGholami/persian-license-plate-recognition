"""The end-to-end pipeline: frame in, saved and parsed plates out.

Both the offline (single image) and live (camera) entry points drive this
same class, so the two modes can never drift apart again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lpr import config
from lpr.detector import Detections, PlateDetector
from lpr.ocr import OcrResult, PlateOCR
from lpr.plate_format import PlateInfo, try_parse_plate
from lpr.storage import PlateDatabase, format_timestamp, save_capture


@dataclass
class PlateResult:
    """One plate found in one frame."""

    timestamp: str
    text: str
    """Raw OCR output, e.g. ``"12PwD35373"``."""

    ocr: OcrResult
    plate: PlateInfo | None
    """Parsed plate, or ``None`` when the text failed validation."""

    error: str | None
    """Why parsing failed, or ``None`` on success."""

    capture_dir: Path | None
    """Where the images and text were written."""

    @property
    def is_valid(self) -> bool:
        return self.plate is not None

    def describe(self) -> str:
        """A one-line human-readable summary."""
        if self.plate is None:
            return f"{self.text or '<empty>'}  (invalid: {self.error})"
        parts = [self.plate.template]
        if self.plate.province:
            parts.append(self.plate.province)
        if self.plate.category:
            parts.append(self.plate.category)
        return "  |  ".join(parts)


class LicensePlatePipeline:
    """Detect plates, read them, validate them, and store the results."""

    def __init__(
        self,
        detector: PlateDetector | None = None,
        ocr: PlateOCR | None = None,
        database: PlateDatabase | None = None,
        outputs_dir: Path | None = None,
    ) -> None:
        self.detector = detector or PlateDetector()
        self.ocr = ocr or PlateOCR()
        self.database = database if database is not None else PlateDatabase()
        self.outputs_dir = Path(outputs_dir or config.OUTPUTS_DIR)

    def annotate(self, frame: np.ndarray, detections: Detections) -> np.ndarray:
        """Draw detection boxes on a copy of ``frame``."""
        return self.detector.annotate(frame, detections)

    def process(
        self,
        frame: np.ndarray,
        detections: Detections | None = None,
        annotated: np.ndarray | None = None,
        save: bool = True,
    ) -> list[PlateResult]:
        """Run the full pipeline over one frame.

        Args:
            frame: The clean BGR frame.
            detections: Reuse detections already computed for this frame.
            annotated: Reuse an annotated frame already rendered for display.
            save: Whether to write results to disk and the database.

        Returns:
            One :class:`PlateResult` per plate above the detection threshold.
        """
        if detections is None:
            detections = self.detector.detect(frame)

        # Crop from the clean frame: cropping the annotated one would bake the
        # drawn boxes into the plate and confuse character segmentation.
        crops = self.detector.crop_plates(frame, detections)
        if not crops:
            return []

        if annotated is None:
            annotated = self.annotate(frame, detections)

        results: list[PlateResult] = []

        for crop in crops:
            timestamp = format_timestamp()
            ocr_result = self.ocr.read(crop)
            plate, error = try_parse_plate(ocr_result.text)

            capture_dir = None
            if save:
                capture_dir = save_capture(
                    frame=annotated,
                    plate=crop,
                    ocr_image=ocr_result.annotated,
                    text=ocr_result.text,
                    timestamp=timestamp,
                    outputs_dir=self.outputs_dir,
                )
                if plate is not None:
                    self.database.save_plate(ocr_result.text, timestamp, plate)
                else:
                    self.database.save_error(error or "unknown", timestamp)

            results.append(
                PlateResult(
                    timestamp=timestamp,
                    text=ocr_result.text,
                    ocr=ocr_result,
                    plate=plate,
                    error=error,
                    capture_dir=capture_dir,
                )
            )

        return results

    def close(self) -> None:
        """Release the database connection."""
        self.database.close()
