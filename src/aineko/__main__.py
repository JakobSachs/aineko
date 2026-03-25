"""Entrypoint: `python -m aineko` or `aineko` CLI."""

import signal
import sys

import uvicorn

from aineko.app import create_app
from aineko.config import Settings


def main() -> None:
    # Ensure SIGTERM triggers clean shutdown (same as SIGINT/Ctrl+C)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
