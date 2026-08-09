"""Threaded video capture that survives a dropped connection.

``cv2.VideoCapture.read`` blocks, so a background thread keeps the newest
frame ready and callers never wait. If the stream dies the thread reopens it
instead of letting the pipeline crash.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from lpr import config


def _normalise_source(source: str | int) -> str | int:
    """Turn ``"0"`` into the integer webcam index ``0``, leave URLs alone."""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    return int(text) if text.isdigit() else text


class CameraStream:
    """Continuously reads frames from a camera, file, or RTSP URL."""

    def __init__(
        self,
        source: str | int | None = None,
        reconnect_delay: float | None = None,
    ) -> None:
        self.source = _normalise_source(
            source if source is not None else config.CAMERA_SOURCE
        )
        self.reconnect_delay = (
            config.CAMERA_RECONNECT_DELAY
            if reconnect_delay is None
            else reconnect_delay
        )

        self._capture: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "CameraStream":
        """Open the stream and begin buffering frames in the background."""
        if self._running:
            return self

        self._capture = cv2.VideoCapture(self.source)
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the reader thread and release the capture device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    # -- internals --------------------------------------------------------
    def _reconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
        time.sleep(self.reconnect_delay)
        if self._running:
            self._capture = cv2.VideoCapture(self.source)

    def _update(self) -> None:
        while self._running:
            if self._capture is None or not self._capture.isOpened():
                self._reconnect()
                continue

            grabbed, frame = self._capture.read()
            if not grabbed or frame is None:
                print("[camera] stream dropped, reconnecting...")
                self._reconnect()
                continue

            with self._lock:
                self._frame = frame

    # -- consumption ------------------------------------------------------
    def read(self) -> np.ndarray | None:
        """Return the most recent frame, or ``None`` if none has arrived yet.

        Callers must handle ``None``: it means the stream has not connected
        yet, or is currently reconnecting.
        """
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def is_running(self) -> bool:
        return self._running
