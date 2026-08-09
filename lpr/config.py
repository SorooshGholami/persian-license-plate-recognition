"""Central configuration: every path and tunable lives here.

Anything environment-specific (camera address, database credentials) is read
from environment variables, optionally populated from a ``.env`` file in the
project root. Nothing sensitive is hard-coded -- see ``.env.example``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Populate os.environ from a .env file, if present.

    Deliberately dependency-free: a tiny KEY=VALUE parser is enough here and
    keeps ``python-dotenv`` optional. Existing environment variables win.
    """
    env_file = ROOT_DIR / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------
CONFIGS_DIR = ROOT_DIR / "configs"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VENDOR_DIR = ROOT_DIR / "third_party" / "tensorflow_models"

# --------------------------------------------------------------------------
# Plate detection (SSD MobileNet V2 FPN, TF Object Detection API)
# --------------------------------------------------------------------------
PIPELINE_CONFIG = CONFIGS_DIR / "pipeline.config"
LABEL_MAP = CONFIGS_DIR / "label_map.pbtxt"

DETECTION_MODEL_DIR = MODELS_DIR / "detection"
#: Which checkpoint to restore. ``ckpt-52`` is the final one from training.
DETECTION_CHECKPOINT = _env_str("LPR_DETECTION_CHECKPOINT", "ckpt-52")

#: Boxes below this score are ignored when cropping plates.
DETECTION_THRESHOLD = _env_float("LPR_DETECTION_THRESHOLD", 0.7)
#: Separate (higher) threshold for the boxes drawn on the preview image.
VISUALIZATION_THRESHOLD = _env_float("LPR_VISUALIZATION_THRESHOLD", 0.8)
MAX_BOXES_TO_DRAW = _env_int("LPR_MAX_BOXES_TO_DRAW", 5)

# --------------------------------------------------------------------------
# Character OCR (VGG-style CNN trained on the Iranis dataset)
# --------------------------------------------------------------------------
OCR_MODEL_PATH = MODELS_DIR / "ocr" / "trained_VGG_model.h5"
OCR_LABELS_PATH = MODELS_DIR / "ocr" / "labels"

#: Input size the OCR network expects (square).
OCR_INPUT_SIZE = 64
#: Plates are resized to this width before character segmentation, so the
#: pixel-area heuristics below stay meaningful across camera resolutions.
PLATE_RESIZE_WIDTH = _env_int("LPR_PLATE_RESIZE_WIDTH", 400)

#: Connected components outside ``area/DIVISOR`` bounds are not characters.
SEGMENT_MIN_AREA_DIVISOR = _env_int("LPR_SEGMENT_MIN_AREA_DIVISOR", 130)
SEGMENT_MAX_AREA_DIVISOR = _env_int("LPR_SEGMENT_MAX_AREA_DIVISOR", 30)

#: Class order used when the pickled LabelBinarizer cannot be read. It matches
#: the alphabetically sorted Iranis class folders the model was trained on.
FALLBACK_OCR_CLASSES = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "D", "Gh", "H", "J", "L", "M", "N", "P",
    "PuV", "PwD", "Sad", "Sin", "T", "Taxi", "V", "Y",
)


def load_ocr_classes() -> tuple[str, ...]:
    """Return OCR class labels in the order the model outputs them.

    Prefers the ``LabelBinarizer`` pickled during training so the labels can
    never drift from the weights; falls back to the known class order.
    """
    try:
        with open(OCR_LABELS_PATH, "rb") as handle:
            return tuple(str(cls) for cls in pickle.load(handle).classes_)
    except Exception:  # noqa: BLE001 - any failure means "use the fallback"
        return FALLBACK_OCR_CLASSES


# --------------------------------------------------------------------------
# Rendering a clean plate image
# --------------------------------------------------------------------------
PLATE_TEMPLATE = ASSETS_DIR / "template-base.png"
FONT_FARSI = ASSETS_DIR / "fonts" / "Nazaninb.ttf"
FONT_EMOJI = ASSETS_DIR / "fonts" / "OpenSansEmoji.ttf"

# --------------------------------------------------------------------------
# Data locations
# --------------------------------------------------------------------------
SAMPLES_DIR = DATA_DIR / "samples"
IMAGES_DIR = DATA_DIR / "images"
RECORDS_DIR = DATA_DIR / "records"
OCR_CHARS_DIR = DATA_DIR / "ocr_chars"

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
#: RTSP URL, a video file, or a webcam index such as ``0``.
CAMERA_SOURCE = _env_str("LPR_CAMERA_SOURCE", "0")
#: Seconds to wait before reconnecting after the stream drops.
CAMERA_RECONNECT_DELAY = _env_float("LPR_CAMERA_RECONNECT_DELAY", 2.0)

# --------------------------------------------------------------------------
# Database (optional -- results are always written to disk regardless)
# --------------------------------------------------------------------------
DB_ENABLED = _env_str("LPR_DB_ENABLED", "false").lower() in {"1", "true", "yes"}
DB_HOST = _env_str("LPR_DB_HOST", "127.0.0.1")
DB_PORT = _env_int("LPR_DB_PORT", 3306)
DB_USER = _env_str("LPR_DB_USER", "root")
DB_PASSWORD = _env_str("LPR_DB_PASSWORD", "")
DB_NAME = _env_str("LPR_DB_NAME", "License_Plate_OCR")

# --------------------------------------------------------------------------
# Web application
# --------------------------------------------------------------------------
WEB_HOST = _env_str("LPR_WEB_HOST", "127.0.0.1")
WEB_PORT = _env_int("LPR_WEB_PORT", 5000)
WEB_SECRET_KEY = _env_str("LPR_WEB_SECRET_KEY", "change-me")
#: How many recent plates the dashboard shows.
WEB_PLATE_LIMIT = _env_int("LPR_WEB_PLATE_LIMIT", 20)

#: Timestamp formats shared by the storage layer and the web app.
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S.%f"
