from __future__ import annotations

from typing import Dict

import numpy as np


def compute_cagr(equity: np.ndarray, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity[-1] / max(equity[0], 1e-9)
    years = len(equity) / periods_per_year
    return float(total_return ** (1 / max(years, 1e-9)) - 1)


def compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size == 0:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float((mean / std) * np.sqrt(periods_per_year))


def compute_mdd(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def compute_hit_rate(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float((returns > 0).mean())


def compute_metrics(equity: np.ndarray) -> Dict[str, float]:
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    return {
        "cagr": compute_cagr(equity),
        "sharpe": compute_sharpe(returns),
        "mdd": compute_mdd(equity),
        "hit_rate": compute_hit_rate(returns),
        "trades": float(len(returns)),
    }
