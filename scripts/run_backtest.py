from __future__ import annotations

import json
import os
from dataclasses import dataclass
import tyro
import pandas as pd

from src.agent.eval_rl import evaluate_rl, evaluate_rl_shared
from src.backtest.report import build_buy_hold, save_report
from src.data.features import add_features
from src.data.ingest import load_price_data
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


def _run_single(cfg: Config, stock_id: str, shared_policy: bool, source: str) -> dict:
    if shared_policy:
        equity, metrics = evaluate_rl_shared(cfg.raw, stock_id, source=source)
    else:
        equity, metrics = evaluate_rl(cfg.raw, stock_id, source=source)

    df, _ = load_price_data(
        stock_id,
        cfg.get("data", "prices_dir"),
        source=source,
        lookback_days_for_live=cfg.get("data", "lookback_days_for_live"),
        cache_min_refresh_seconds=cfg.get("data", "cache_min_refresh_seconds"),
        interval=cfg.get("provider", "yahoo", "interval", default="1d"),
    )
    df = add_features(df)
    if len(equity):
        start = equity["Date"].iloc[0]
        end = equity["Date"].iloc[-1]
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
    bh = build_buy_hold(df["Close"].values)
    buy_hold = pd.DataFrame({"Date": df["Date"].values[: len(bh)], "equity": bh})

    save_report(equity, buy_hold, cfg.raw, stock_id)
    return {"stock_id": stock_id, "metrics": metrics}


def main() -> None:
    args = tyro.cli(CLIArgs)
    cfg = Config.load(args.config)
    set_seed(cfg.get("project", "seed", default=42))

    stock_ids = _parse_stock_ids(args.stock_ids)
    if args.stock_id:
        stock_ids = [args.stock_id]
    if not stock_ids:
        stock_ids = cfg.get("universe", "stock_ids", default=[])
    if not stock_ids:
        raise ValueError("No stock_id(s) provided")

    summaries = []
    for sid in stock_ids:
        summaries.append(_run_single(cfg, sid, args.shared_policy, args.source))

    if len(summaries) > 1:
        out_dir = os.path.join(cfg.get("project", "output_dir"), "backtest")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"summaries": summaries}, f, indent=2)


if __name__ == "__main__":
    main()
