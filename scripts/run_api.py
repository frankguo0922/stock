from __future__ import annotations

import tyro
import uvicorn

from src.api.main import create_app


def main(config: str) -> None:
    app = create_app(config)
    from src.utils.config import Config

    cfg = Config.load(config)
    host = cfg.get("api", "host", default="127.0.0.1")
    port = int(cfg.get("api", "port", default=8000))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
