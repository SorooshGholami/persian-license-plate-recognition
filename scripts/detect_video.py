"""Recognise plates from a live camera, a video file, or an RTSP stream.

The source comes from ``LPR_CAMERA_SOURCE`` in ``.env`` unless overridden.

Examples:
    python scripts/detect_video.py
    python scripts/detect_video.py --source rtsp://camera.example.local:554/1/1
    python scripts/detect_video.py --source 0 --headless
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

import _bootstrap  # noqa: F401  (side effect: makes `lpr` importable)
from lpr.camera import CameraStream
from lpr.pipeline import LicensePlatePipeline

#: Seconds to idle while the camera is connecting or reconnecting.
IDLE_SLEEP = 0.05
WINDOW_NAME = "License Plate Recognition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default=None,
        help="RTSP URL, video file, or webcam index (default: LPR_CAMERA_SOURCE).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a preview window; useful on a server.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write results to outputs/ or the database.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pipeline = LicensePlatePipeline()
    camera = CameraStream(source=args.source).start()

    print("Running. Press Ctrl+C to stop" + ("" if args.headless else ", or 'q' in the window") + ".")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(IDLE_SLEEP)
                continue

            detections = pipeline.detector.detect(frame)
            annotated = pipeline.annotate(frame, detections)

            for result in pipeline.process(
                frame, detections, annotated, save=not args.no_save
            ):
                print(f"[plate] {result.describe()}")

            if not args.headless:
                cv2.imshow(WINDOW_NAME, cv2.resize(annotated, (800, 600)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.stop()
        pipeline.close()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
