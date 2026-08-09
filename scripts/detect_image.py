"""Recognise plates in still images -- the offline, no-camera mode.

Examples:
    python scripts/detect_image.py                    # every image in data/samples
    python scripts/detect_image.py photo.jpg          # one image
    python scripts/detect_image.py data/samples --show
    python scripts/detect_image.py photo.jpg --render  # also draw a clean plate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401  (side effect: makes `lpr` importable)
from lpr import config
from lpr.pipeline import LicensePlatePipeline
from lpr.plate_render import render_plate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(inputs: list[str]) -> list[Path]:
    """Expand the given files and directories into a sorted list of images."""
    if not inputs:
        inputs = [str(config.SAMPLES_DIR)]

    images: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            images.extend(
                child
                for child in sorted(path.iterdir())
                if child.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file():
            images.append(path)
        else:
            print(f"[warn] not found: {path}")

    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Image files or directories (default: data/samples).",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display each result in a window."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write results to outputs/ or the database.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render the plate onto the clean template.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = collect_images(args.inputs)

    if not images:
        print("No images to process.")
        return 1

    pipeline = LicensePlatePipeline()
    found = 0

    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"[warn] could not read {image_path}")
                continue

            results = pipeline.process(frame, save=not args.no_save)
            if not results:
                print(f"{image_path.name}: no plate detected")
                continue

            for result in results:
                found += 1
                print(f"{image_path.name}: {result.describe()}")

                if args.show:
                    cv2.imshow(f"OCR - {image_path.name}", result.ocr.annotated)

                if args.render and result.plate is not None:
                    rendered = render_plate(result.plate)
                    if result.capture_dir is not None:
                        target = result.capture_dir / f"{result.timestamp}-plate.jpg"
                        cv2.imwrite(str(target), rendered)
                    if args.show:
                        cv2.imshow(f"Plate - {image_path.name}", rendered)

            if args.show:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    finally:
        pipeline.close()

    print(f"\nDone. {found} plate(s) recognised in {len(images)} image(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
