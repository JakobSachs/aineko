"""Entrypoint: `python -m aineko` or `aineko` CLI."""

import uvicorn

from aineko.app import create_app
from aineko.config import Settings


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
