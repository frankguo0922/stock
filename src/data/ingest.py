from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from src.data.providers import LocalCSVProvider, ProviderMeta, YahooProvider


def load_price_data(
    stock_id: str,
    prices_dir: str,
    source: str = "local",
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback_days_for_live: int = 365,
    cache_min_refresh_seconds: int = 60,
    interval: str = "1d",
) -> Tuple[pd.DataFrame, ProviderMeta]:
    local = LocalCSVProvider(prices_dir)
    if source == "local":
        return local.get_price_df(stock_id, start, end, interval)

    yahoo = YahooProvider(
        prices_dir=prices_dir,
        lookback_days_for_live=lookback_days_for_live,
        cache_min_refresh_seconds=cache_min_refresh_seconds,
        interval=interval,
    )
    df, meta = yahoo.get_price_df(stock_id, start, end, interval)
    if df.empty:
        local_df, local_meta = local.get_price_df(stock_id, start, end, interval)
        local_meta.error = meta.error
        return local_df, local_meta
    return df, meta
