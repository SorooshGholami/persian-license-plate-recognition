"""Make the vendored TensorFlow Object Detection API importable.

The project depends on ``object_detection`` (and its ``slim`` helper package),
which live inside ``third_party/tensorflow_models/research``. Instead of
requiring a ``pip install`` of that repository, we simply put the two
directories on ``sys.path``. Importing :mod:`lpr` runs this automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT_DIR / "third_party" / "tensorflow_models" / "research"


def add_vendor_paths() -> None:
    """Prepend the vendored research/ and research/slim/ dirs to sys.path."""
    for path in (RESEARCH_DIR, RESEARCH_DIR / "slim"):
        entry = str(path)
        if path.is_dir() and entry not in sys.path:
            sys.path.insert(0, entry)


add_vendor_paths()
