from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import tyro

from src.forecast.train import train_forecast, train_shared_forecast
from src.utils.config import Config
from src.utils.seed import set_seed


@dataclass
class CLIArgs:
    stock_id: Optional[str] = None
    config: str = "configs/default.yaml"
    source: str = "local"
    shared_forecast: bool = False
    stock_ids: str = ""
    forecast_id: str = "shared_v1"


def main() -> None:
    args = tyro.cli(CLIArgs)
    cfg = Config.load(args.config)
    set_seed(cfg.get("project", "seed", default=42))
    if args.shared_forecast:
        stock_ids = [s.strip() for s in args.stock_ids.split(",") if s.strip()]
        if not stock_ids:
            stock_ids = cfg.get("universe", "stock_ids", default=[])
        if not stock_ids:
            raise ValueError("No stock_ids provided for shared forecast")
        train_shared_forecast(cfg.raw, stock_ids, args.forecast_id, source=args.source)
    else:
        if not args.stock_id:
            raise ValueError("stock_id is required when shared_forecast is false")
        train_forecast(cfg.raw, args.stock_id, source=args.source)


if __name__ == "__main__":
    main()
