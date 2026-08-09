"""Fine-tune the plate detector using the Object Detection API trainer.

A thin wrapper around ``model_main_tf2.py`` that fills in this project's
paths and puts the vendored library on ``PYTHONPATH`` for the child process.

Examples:
    python scripts/train_detector.py --num-train-steps 50000
    python scripts/train_detector.py --eval          # run evaluation instead
    python scripts/train_detector.py --export        # export a SavedModel
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (side effect: makes `lpr` importable)
from lpr import config
from lpr.bootstrap import RESEARCH_DIR

TRAINER = RESEARCH_DIR / "object_detection" / "model_main_tf2.py"
EXPORTER = RESEARCH_DIR / "object_detection" / "exporter_main_v2.py"


def child_env() -> dict[str, str]:
    """Environment with the vendored packages on PYTHONPATH."""
    env = os.environ.copy()
    extra = os.pathsep.join([str(RESEARCH_DIR), str(RESEARCH_DIR / "slim")])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{extra}{os.pathsep}{existing}" if existing else extra
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir", type=Path, default=config.DETECTION_MODEL_DIR,
        help="Where checkpoints are read from and written to.",
    )
    parser.add_argument(
        "--pipeline-config", type=Path, default=config.PIPELINE_CONFIG
    )
    parser.add_argument("--num-train-steps", type=int, default=50000)
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate instead of training."
    )
    parser.add_argument(
        "--export", action="store_true", help="Export a SavedModel and exit."
    )
    parser.add_argument(
        "--export-dir", type=Path, default=config.MODELS_DIR / "exported",
        help="Destination for --export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.export:
        if not EXPORTER.is_file():
            print(f"Exporter not found: {EXPORTER}")
            return 1
        command = [
            sys.executable, str(EXPORTER),
            "--input_type=image_tensor",
            f"--pipeline_config_path={args.pipeline_config}",
            f"--trained_checkpoint_dir={args.model_dir}",
            f"--output_directory={args.export_dir}",
        ]
    else:
        if not TRAINER.is_file():
            print(f"Trainer not found: {TRAINER}")
            return 1
        command = [
            sys.executable, str(TRAINER),
            f"--model_dir={args.model_dir}",
            f"--pipeline_config_path={args.pipeline_config}",
        ]
        if args.eval:
            command.append(f"--checkpoint_dir={args.model_dir}")
        else:
            command.append(f"--num_train_steps={args.num_train_steps}")

    print("Running:\n  " + " \\\n    ".join(command) + "\n")
    # Paths inside pipeline.config are relative to the project root.
    return subprocess.call(command, cwd=str(config.ROOT_DIR), env=child_env())


if __name__ == "__main__":
    sys.exit(main())
