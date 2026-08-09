"""Flask + Socket.IO dashboard.

Serves an MJPEG preview of the camera, pushes each newly recognised plate to
connected browsers, and offers a search page backed by the database.

The streaming loop runs the recognition pipeline itself, so results are
emitted directly -- no filesystem watcher is needed.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Iterator

import cv2
from flask import Flask, Response, flash, render_template, request
from flask_socketio import SocketIO

from lpr import config
from lpr.camera import CameraStream
from lpr.pipeline import LicensePlatePipeline, PlateResult
from lpr.storage import capture_dir_for, read_capture, recent_captures

#: Seconds to idle when the camera has not produced a frame yet.
_IDLE_SLEEP = 0.05
#: Date format accepted by the search form.
_SEARCH_TIME_FORMAT = "%Y-%m-%d %H-%M"


def _capture_payload(result: PlateResult) -> dict:
    """Shape a pipeline result the way the browser expects it."""
    return {
        "date": result.timestamp[:-3],
        "plate": result.text,
        "plate_fa": result.plate.template if result.plate else "",
        "province": (result.plate.province or "") if result.plate else "",
        "category": (result.plate.category or "") if result.plate else "",
        "real": "",
        "ocr": "",
    }


def create_app(
    pipeline: LicensePlatePipeline | None = None,
    camera: CameraStream | None = None,
) -> tuple[Flask, SocketIO]:
    """Build the Flask app and its Socket.IO server.

    Args:
        pipeline: Recognition pipeline; one is created if omitted.
        camera: Video source; one is created and started if omitted.

    Returns:
        ``(app, socketio)`` -- run the app through ``socketio.run``.
    """
    app = Flask(__name__)
    app.secret_key = config.WEB_SECRET_KEY
    socketio = SocketIO(app)

    pipeline = pipeline or LicensePlatePipeline()
    camera = camera or CameraStream().start()

    # -- pages ------------------------------------------------------------
    @app.route("/")
    def index() -> str:
        return render_template(
            "index.html",
            captures=recent_captures(config.WEB_PLATE_LIMIT),
            count_limit=config.WEB_PLATE_LIMIT,
        )

    @app.route("/search")
    def search() -> str:
        plate = (request.args.get("plate") or "").strip()
        start = (request.args.get("start") or "").strip()
        end = (request.args.get("end") or "").strip()

        if not plate and not start and not end:
            return render_template("search.html", captures=[])

        if bool(start) != bool(end):
            flash("Enter both a start and an end date, or neither.")
            return render_template("search.html", captures=[])

        if not pipeline.database.enabled:
            flash("Search needs the database. Set LPR_DB_ENABLED=true in .env.")
            return render_template("search.html", captures=[])

        bounds: list[int | None] = [None, None]
        for index, (value, label) in enumerate(((start, "start"), (end, "end"))):
            if not value:
                continue
            try:
                moment = datetime.strptime(value, _SEARCH_TIME_FORMAT)
            except ValueError:
                flash(f"Invalid {label} date. Use YYYY-MM-DD HH-MM.")
                return render_template("search.html", captures=[])
            bounds[index] = int(moment.timestamp() * 100)

        rows = pipeline.database.search(
            plate=plate or None, start=bounds[0], end=bounds[1]
        )

        captures = []
        for row in rows:
            directory = capture_dir_for(row["timestamp"])
            if directory.is_dir():
                captures.append(read_capture(directory))

        return render_template("search.html", captures=captures)

    # -- live stream ------------------------------------------------------
    def frames() -> Iterator[bytes]:
        while camera.is_running:
            frame = camera.read()
            if frame is None:
                time.sleep(_IDLE_SLEEP)
                continue

            detections = pipeline.detector.detect(frame)
            annotated = pipeline.annotate(frame, detections)

            for result in pipeline.process(frame, detections, annotated):
                print(f"[plate] {result.describe()}")
                socketio.emit("create_event", _capture_payload(result))

            encoded, buffer = cv2.imencode(".jpg", annotated)
            if not encoded:
                continue

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    @app.route("/video_feed")
    def video_feed() -> Response:
        return Response(
            frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    app.config["LPR_PIPELINE"] = pipeline
    app.config["LPR_CAMERA"] = camera
    return app, socketio
