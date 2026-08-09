"""Persisting results: image/text files on disk, and an optional MySQL table.

On-disk layout (unchanged from the original project, so old captures still
load in the dashboard)::

    outputs/<date>/<timestamp>.jpg           full frame with detection boxes
    outputs/<date>/<timestamp>/
        <timestamp>.jpg                      cropped plate
        <timestamp>-OCR.jpg                  plate with per-character boxes
        <timestamp>.txt                      recognised plate text

The database is optional. When it is disabled or unreachable the pipeline
keeps running and still writes every result to disk.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lpr import config
from lpr.plate_format import PlateInfo

#: Database timestamps are stored as hundredths of a second since the epoch,
#: which keeps them sortable integers while preserving the capture precision.
_TIMESTAMP_SCALE = 100


def format_timestamp(moment: datetime | None = None) -> str:
    """Format ``moment`` the way capture directories are named."""
    moment = moment or datetime.now()
    return moment.strftime(config.TIMESTAMP_FORMAT)[:-4]


def timestamp_to_int(timestamp: str) -> int:
    """Convert a capture timestamp into its integer database form."""
    moment = datetime.strptime(timestamp, config.TIMESTAMP_FORMAT)
    return int(moment.timestamp() * _TIMESTAMP_SCALE)


def int_to_timestamp(value: float) -> str:
    """Inverse of :func:`timestamp_to_int`."""
    return format_timestamp(datetime.fromtimestamp(float(value) / _TIMESTAMP_SCALE))


@dataclass(frozen=True)
class Capture:
    """One saved detection, as read back from disk."""

    timestamp: str
    directory: Path
    text: str
    plate_image: str | None
    """Base64-encoded JPEG of the cropped plate, ready for an ``<img>`` tag."""
    ocr_image: str | None
    """Base64-encoded JPEG of the annotated plate."""

    @property
    def date(self) -> str:
        """Capture time without the fractional seconds."""
        return self.timestamp[:-3]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def save_capture(
    frame: np.ndarray,
    plate: np.ndarray,
    ocr_image: np.ndarray,
    text: str,
    timestamp: str | None = None,
    outputs_dir: Path | None = None,
) -> Path:
    """Write one detection to disk and return its directory.

    Args:
        frame: The full frame with detection boxes drawn on it.
        plate: The cropped plate.
        ocr_image: The plate with per-character boxes.
        text: Recognised plate text.
        timestamp: Capture time; defaults to now.
        outputs_dir: Root output directory; defaults to ``outputs/``.
    """
    timestamp = timestamp or format_timestamp()
    outputs_dir = Path(outputs_dir or config.OUTPUTS_DIR)

    day_dir = outputs_dir / timestamp.split(" ")[0]
    capture_dir = day_dir / timestamp
    capture_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(day_dir / f"{timestamp}.jpg"), frame)
    cv2.imwrite(str(capture_dir / f"{timestamp}.jpg"), plate)
    cv2.imwrite(str(capture_dir / f"{timestamp}-OCR.jpg"), ocr_image)
    (capture_dir / f"{timestamp}.txt").write_text(text, encoding="utf-8")

    return capture_dir


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def read_capture(directory: Path) -> Capture:
    """Load one capture directory into a :class:`Capture`."""
    directory = Path(directory)
    text, plate_image, ocr_image = "", None, None

    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        if entry.name.endswith("-OCR.jpg"):
            ocr_image = _encode_image(entry)
        elif entry.suffix == ".jpg":
            plate_image = _encode_image(entry)
        elif entry.suffix == ".txt":
            text = entry.read_text(encoding="utf-8")

    return Capture(
        timestamp=directory.name,
        directory=directory,
        text=text,
        plate_image=plate_image,
        ocr_image=ocr_image,
    )


def _parse_dir_names(parent: Path, pattern: str) -> list[datetime]:
    """Return sub-directory names of ``parent`` parsed with ``pattern``."""
    if not parent.is_dir():
        return []
    parsed = []
    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        try:
            parsed.append(datetime.strptime(entry.name, pattern))
        except ValueError:
            continue  # not a capture directory; ignore
    return parsed


def latest_day(outputs_dir: Path | None = None) -> Path | None:
    """Return the most recent day directory, or ``None`` if there is none."""
    outputs_dir = Path(outputs_dir or config.OUTPUTS_DIR)
    days = _parse_dir_names(outputs_dir, config.DATE_FORMAT)
    if not days:
        return None
    return outputs_dir / max(days).strftime(config.DATE_FORMAT)


def recent_captures(
    limit: int | None = None, outputs_dir: Path | None = None
) -> list[Capture]:
    """Return the newest captures from the most recent day, newest first."""
    limit = limit or config.WEB_PLATE_LIMIT
    day_dir = latest_day(outputs_dir)
    if day_dir is None:
        return []

    moments = sorted(
        _parse_dir_names(day_dir, config.TIMESTAMP_FORMAT), reverse=True
    )[:limit]

    return [
        read_capture(day_dir / moment.strftime(config.TIMESTAMP_FORMAT)[:-4])
        for moment in moments
    ]


def capture_dir_for(timestamp_value: float, outputs_dir: Path | None = None) -> Path:
    """Map a database timestamp back to its capture directory."""
    outputs_dir = Path(outputs_dir or config.OUTPUTS_DIR)
    timestamp = int_to_timestamp(timestamp_value)
    return outputs_dir / timestamp.split(" ")[0] / timestamp


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
class PlateDatabase:
    """Thin MySQL wrapper. Every method is a no-op when disabled."""

    _CREATE_PLATE = """
        CREATE TABLE IF NOT EXISTS Plate (
            id INT NOT NULL AUTO_INCREMENT,
            plate VARCHAR(25) CHARACTER SET utf8mb4
                  COLLATE utf8mb4_general_ci NOT NULL,
            plate_fa VARCHAR(40) CHARACTER SET utf8mb4
                     COLLATE utf8mb4_general_ci NULL,
            province VARCHAR(60) CHARACTER SET utf8mb4
                     COLLATE utf8mb4_general_ci NULL,
            category VARCHAR(60) CHARACTER SET utf8mb4
                     COLLATE utf8mb4_general_ci NULL,
            timestamp BIGINT(30),
            PRIMARY KEY (id)
        )
    """

    _CREATE_ERROR = """
        CREATE TABLE IF NOT EXISTS Error (
            id INT NOT NULL AUTO_INCREMENT,
            Reason VARCHAR(255) CHARACTER SET utf8mb4
                   COLLATE utf8mb4_general_ci NOT NULL,
            TimeStamp VARCHAR(40),
            PRIMARY KEY (id)
        )
    """

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = config.DB_ENABLED if enabled is None else enabled
        self._connection = None

        if self.enabled:
            try:
                self._connect()
                self._create_tables()
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                print(f"[db] disabled: {exc}")
                self.enabled = False

    def _connect(self) -> None:
        import mysql.connector as mysql  # imported lazily: optional dependency

        self._connection = mysql.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            passwd=config.DB_PASSWORD,
            database=config.DB_NAME,
        )

    def _execute(self, sql: str, values: tuple = ()) -> None:
        if not self.enabled or self._connection is None:
            return
        try:
            cursor = self._connection.cursor()
            cursor.execute(sql, values)
            self._connection.commit()
            cursor.close()
        except Exception as exc:  # noqa: BLE001 - a dead DB must not stop OCR
            print(f"[db] write failed: {exc}")

    def _create_tables(self) -> None:
        self._execute(self._CREATE_PLATE)
        self._execute(self._CREATE_ERROR)

    def save_plate(self, text: str, timestamp: str, plate: PlateInfo | None) -> None:
        """Record a recognised plate."""
        self._execute(
            "INSERT INTO Plate (plate, plate_fa, province, category, timestamp) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                text,
                plate.template if plate else None,
                plate.province if plate else None,
                plate.category if plate else None,
                timestamp_to_int(timestamp),
            ),
        )

    def save_error(self, reason: str, timestamp: str) -> None:
        """Record why a capture could not be parsed into a valid plate."""
        self._execute(
            "INSERT INTO Error (Reason, TimeStamp) VALUES (%s, %s)",
            (reason, timestamp),
        )

    def search(
        self,
        plate: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> list[dict]:
        """Query stored plates by text and/or timestamp range."""
        if not self.enabled or self._connection is None:
            return []

        clauses, values = [], []
        if plate:
            clauses.append("plate = %s")
            values.append(plate)
        if start is not None and end is not None:
            clauses.append("timestamp BETWEEN %s AND %s")
            values.extend([start, end])

        sql = "SELECT * FROM Plate"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC"

        try:
            cursor = self._connection.cursor(dictionary=True)
            cursor.execute(sql, tuple(values))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Exception as exc:  # noqa: BLE001
            print(f"[db] query failed: {exc}")
            return []

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
