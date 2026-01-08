from __future__ import annotations

from dataclasses import dataclass
import tyro

from src.agent.train_rl import train_rl, train_rl_shared
from src.utils.config import Config
from src.utils.seed import set_seed


@dataclass
class CLIArgs:
    stock_id: str = ""
    stock_ids: str = ""
    shared_policy: bool = False
    source: str = "local"
    config: str = "configs/default.yaml"


def _parse_stock_ids(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    args = tyro.cli(CLIArgs)
    cfg = Config.load(args.config)
    set_seed(cfg.get("project", "seed", default=42))

    if args.shared_policy:
        stock_ids = _parse_stock_ids(args.stock_ids) or cfg.get("universe", "stock_ids", default=[])
        if not stock_ids:
            raise ValueError("No stock_ids provided for shared policy")
        train_rl_shared(cfg.raw, stock_ids, source=args.source)
    else:
        if not args.stock_id:
            raise ValueError("stock_id is required when shared_policy is false")
        train_rl(cfg.raw, args.stock_id, source=args.source)


if __name__ == "__main__":
    main()
