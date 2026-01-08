from __future__ import annotations

from typing import Dict, Tuple

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.backtest.metrics import compute_metrics


def build_buy_hold(prices: np.ndarray) -> np.ndarray:
    if len(prices) == 0:
        return np.array([], dtype=np.float32)
    return prices / max(prices[0], 1e-9)


def save_report(
    equity: pd.DataFrame,
    buy_hold: pd.DataFrame,
    config: Dict[str, Dict],
    stock_id: str,
) -> Dict[str, float]:
    output_dir = config["project"]["output_dir"]
    bt_dir = os.path.join(output_dir, "backtest", stock_id)
    os.makedirs(bt_dir, exist_ok=True)

    metrics = compute_metrics(equity["equity"].values)
    report = {
        "metrics": metrics,
        "start": str(equity["Date"].iloc[0]) if len(equity) else None,
        "end": str(equity["Date"].iloc[-1]) if len(equity) else None,
    }
    with open(os.path.join(bt_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.plot(equity["Date"], equity["equity"], label="RL")
    plt.plot(buy_hold["Date"], buy_hold["equity"], label="Buy&Hold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(bt_dir, "equity_curve.png"), dpi=150)
    plt.close()

    return metrics
