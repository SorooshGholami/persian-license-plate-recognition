"""Build the TFRecord files the detection trainer consumes.

Class ids come from ``configs/label_map.pbtxt``, so the mapping can never
drift from what the model was configured with.

Examples:
    python scripts/generate_tfrecord.py                # train and test splits
    python scripts/generate_tfrecord.py --split train
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import namedtuple
from pathlib import Path

import _bootstrap  # noqa: F401  (side effect: makes `lpr` importable)

# `lpr` must be imported before `object_detection`: importing it puts the
# vendored Object Detection API on sys.path.
from lpr import config  # noqa: E402  (import order is deliberate)

import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from object_detection.utils import dataset_util, label_map_util  # noqa: E402
from PIL import Image  # noqa: E402

ImageGroup = namedtuple("ImageGroup", ["filename", "boxes"])


def load_class_ids(label_map_path: Path) -> dict[str, int]:
    """Read ``name -> id`` pairs out of a label map."""
    label_map = label_map_util.load_labelmap(str(label_map_path))
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True
    )
    return {category["name"]: category["id"] for category in categories}


def group_by_image(frame: pd.DataFrame) -> list[ImageGroup]:
    """Group annotation rows so each image becomes one TFRecord example."""
    grouped = frame.groupby("filename")
    return [ImageGroup(filename, grouped.get_group(filename)) for filename in grouped.groups]


def create_example(
    group: ImageGroup, images_dir: Path, class_ids: dict[str, int]
) -> tf.train.Example:
    """Turn one image and its boxes into a ``tf.train.Example``."""
    image_path = images_dir / group.filename
    with tf.io.gfile.GFile(str(image_path), "rb") as handle:
        encoded_image = handle.read()

    width, height = Image.open(io.BytesIO(encoded_image)).size
    encoded_filename = group.filename.encode("utf8")
    image_format = image_path.suffix.lstrip(".").replace("jpeg", "jpg").encode("utf8")

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    class_names, classes = [], []

    for _, row in group.boxes.iterrows():
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        class_names.append(str(row["class"]).encode("utf8"))
        classes.append(class_ids.get(str(row["class"]), 0))

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(encoded_filename),
                "image/source_id": dataset_util.bytes_feature(encoded_filename),
                "image/encoded": dataset_util.bytes_feature(encoded_image),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(class_names),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )


def build_record(split: str, images_dir: Path, data_dir: Path, records_dir: Path,
                 class_ids: dict[str, int]) -> bool:
    """Write one split's TFRecord. Returns ``False`` if inputs are missing."""
    csv_path = data_dir / f"{split}_labels.csv"
    split_images = images_dir / split

    if not csv_path.is_file():
        print(f"[warn] missing {csv_path} -- run scripts/xml_to_csv.py first")
        return False
    if not split_images.is_dir():
        print(f"[warn] missing image directory: {split_images}")
        return False

    records_dir.mkdir(parents=True, exist_ok=True)
    output_path = records_dir / f"{split}.record"

    frame = pd.read_csv(csv_path)
    written = 0

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for group in group_by_image(frame):
            writer.write(
                create_example(group, split_images, class_ids).SerializeToString()
            )
            written += 1

    print(f"{split}: {written} images -> {output_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split", choices=["train", "test", "both"], default="both",
        help="Which split to build (default: both).",
    )
    parser.add_argument("--images-dir", type=Path, default=config.IMAGES_DIR)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--records-dir", type=Path, default=config.RECORDS_DIR)
    parser.add_argument("--label-map", type=Path, default=config.LABEL_MAP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    class_ids = load_class_ids(args.label_map)
    splits = ["train", "test"] if args.split == "both" else [args.split]

    ok = True
    for split in splits:
        ok &= build_record(
            split, args.images_dir, args.data_dir, args.records_dir, class_ids
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
