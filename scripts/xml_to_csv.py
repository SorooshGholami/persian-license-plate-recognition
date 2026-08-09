"""Convert LabelImg XML annotations into the CSV that TFRecords are built from.

Examples:
    python scripts/xml_to_csv.py                     # train and test splits
    python scripts/xml_to_csv.py --split train
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401  (side effect: makes `lpr` importable)
from lpr import config

COLUMNS = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]


def xml_to_dataframe(directory: Path) -> pd.DataFrame:
    """Read every ``*.xml`` annotation in ``directory`` into a DataFrame."""
    rows = []

    for xml_file in sorted(directory.glob("*.xml")):
        root = ET.parse(xml_file).getroot()
        filename = root.findtext("filename")
        size = root.find("size")
        if filename is None or size is None:
            print(f"[warn] skipping malformed annotation: {xml_file.name}")
            continue

        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))

        for member in root.findall("object"):
            box = member.find("bndbox")
            if box is None:
                continue
            rows.append(
                (
                    filename,
                    width,
                    height,
                    member.findtext("name", ""),
                    int(box.findtext("xmin", "0")),
                    int(box.findtext("ymin", "0")),
                    int(box.findtext("xmax", "0")),
                    int(box.findtext("ymax", "0")),
                )
            )

    return pd.DataFrame(rows, columns=COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to convert (default: both).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=config.IMAGES_DIR,
        help="Directory holding the train/ and test/ image folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Where to write the CSV files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = ["train", "test"] if args.split == "both" else [args.split]
    exit_code = 0

    for split in splits:
        source = args.images_dir / split
        if not source.is_dir():
            print(f"[warn] missing directory: {source}")
            exit_code = 1
            continue

        frame = xml_to_dataframe(source)
        target = args.output_dir / f"{split}_labels.csv"
        frame.to_csv(target, index=False)
        print(f"{split}: {len(frame)} boxes -> {target}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
