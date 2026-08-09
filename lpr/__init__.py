"""Iranian license plate detection and recognition.

Importing this package puts the vendored TensorFlow Object Detection API on
``sys.path``, so ``from object_detection...`` works without installing it.
"""

from __future__ import annotations

from lpr import bootstrap  # noqa: F401  (side effect: extends sys.path)

__version__ = "1.0.0"
