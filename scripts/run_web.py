"""Start the web dashboard: live stream, recent plates, and search.

Examples:
    python scripts/run_web.py
    python scripts/run_web.py --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (side effect: makes `lpr` importable)
from lpr import config
from lpr.web import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=config.WEB_HOST, help="Bind address.")
    parser.add_argument(
        "--port", type=int, default=config.WEB_PORT, help="Bind port."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable Flask debug mode."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app, socketio = create_app()

    print(f"Dashboard: http://{args.host}:{args.port}")
    try:
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera = app.config.get("LPR_CAMERA")
        pipeline = app.config.get("LPR_PIPELINE")
        if camera is not None:
            camera.stop()
        if pipeline is not None:
            pipeline.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
