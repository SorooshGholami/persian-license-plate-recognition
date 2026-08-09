"""Put the project root on ``sys.path`` so ``import lpr`` works.

Every script in this directory imports this module first, which lets them be
run directly (``python scripts/detect_image.py``) without installing the
package or setting PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
