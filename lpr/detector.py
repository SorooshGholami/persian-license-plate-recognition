"""License plate localisation with the TensorFlow Object Detection API.

Wraps the fine-tuned SSD MobileNet V2 FPN model: build once, then call
:meth:`PlateDetector.detect` per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils

from lpr import config

#: The label map is 1-based while the model emits 0-based class ids.
_LABEL_ID_OFFSET = 1


@dataclass(frozen=True)
class Detections:
    """Model output for a single frame."""

    boxes: np.ndarray
    """Normalised ``(ymin, xmin, ymax, xmax)`` rows, ordered by score."""

    scores: np.ndarray
    classes: np.ndarray

    def above(self, threshold: float) -> "Detections":
        """Return only the detections scoring above ``threshold``."""
        keep = self.scores > threshold
        return Detections(self.boxes[keep], self.scores[keep], self.classes[keep])

    def __len__(self) -> int:
        return int(self.scores.shape[0])


class PlateDetector:
    """Detects license plates in BGR images."""

    def __init__(
        self,
        pipeline_config: Path | str | None = None,
        checkpoint_dir: Path | str | None = None,
        checkpoint: str | None = None,
        label_map: Path | str | None = None,
    ) -> None:
        pipeline_config = Path(pipeline_config or config.PIPELINE_CONFIG)
        checkpoint_dir = Path(checkpoint_dir or config.DETECTION_MODEL_DIR)
        checkpoint = checkpoint or config.DETECTION_CHECKPOINT
        label_map = Path(label_map or config.LABEL_MAP)

        configs = config_util.get_configs_from_pipeline_file(str(pipeline_config))
        self._model = model_builder.build(
            model_config=configs["model"], is_training=False
        )

        restore_path = checkpoint_dir / checkpoint
        tf.compat.v2.train.Checkpoint(model=self._model).restore(
            str(restore_path)
        ).expect_partial()

        self.category_index = label_map_util.create_category_index_from_labelmap(
            str(label_map)
        )
        # Wrapping per instance keeps each model's graph separate.
        self._detect_fn = tf.function(self._forward)

    def _forward(self, image: tf.Tensor) -> dict:
        image, shapes = self._model.preprocess(image)
        prediction = self._model.predict(image, shapes)
        return self._model.postprocess(prediction, shapes)

    def detect(self, image: np.ndarray) -> Detections:
        """Run detection on one image.

        Args:
            image: An HxWx3 BGR array, as returned by ``cv2.imread``.

        Returns:
            All detections the model produced, unfiltered.
        """
        tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        raw = self._detect_fn(tensor)

        count = int(raw.pop("num_detections"))
        arrays = {key: value[0, :count].numpy() for key, value in raw.items()}

        return Detections(
            boxes=arrays["detection_boxes"],
            scores=arrays["detection_scores"],
            classes=arrays["detection_classes"].astype(np.int64),
        )

    def annotate(
        self,
        image: np.ndarray,
        detections: Detections,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return a copy of ``image`` with detection boxes drawn on it."""
        annotated = image.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            annotated,
            detections.boxes,
            detections.classes + _LABEL_ID_OFFSET,
            detections.scores,
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=config.MAX_BOXES_TO_DRAW,
            min_score_thresh=(
                config.VISUALIZATION_THRESHOLD if threshold is None else threshold
            ),
            agnostic_mode=False,
        )
        return annotated

    def crop_plates(
        self,
        image: np.ndarray,
        detections: Detections,
        threshold: float | None = None,
    ) -> list[np.ndarray]:
        """Cut the detected plates out of ``image``.

        Note:
            Always pass the *clean* frame here, never the annotated one --
            otherwise the drawn boxes end up inside the crop and confuse
            character segmentation.
        """
        if threshold is None:
            threshold = config.DETECTION_THRESHOLD

        height, width = image.shape[:2]
        crops: list[np.ndarray] = []

        for box in detections.above(threshold).boxes:
            ymin, xmin, ymax, xmax = box * [height, width, height, width]
            crop = image[int(ymin) : int(ymax), int(xmin) : int(xmax)]
            if crop.size:
                crops.append(crop)

        return crops
