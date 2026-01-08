from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency at runtime
    yf = None


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class ProviderMeta:
    data_source_used: str
    last_data_date: Optional[str]
    cache_updated: bool
    error: Optional[str] = None


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    if "Date" not in out.columns:
        out = out.reset_index()
    cols = {c: c.capitalize() for c in out.columns}
    out = out.rename(columns=cols)
    rename_map = {
        "Adj Close": "Close",
        "Adjclose": "Close",
    }
    out = out.rename(columns=rename_map)
    keep = [c for c in REQUIRED_COLUMNS if c in out.columns]
    out = out[keep]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out


class BaseProvider:
    def get_price_df(
        self,
        stock_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Tuple[pd.DataFrame, ProviderMeta]:
        raise NotImplementedError


class LocalCSVProvider(BaseProvider):
    def __init__(self, prices_dir: str) -> None:
        self.prices_dir = prices_dir

    def get_price_df(
        self,
        stock_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Tuple[pd.DataFrame, ProviderMeta]:
        path = os.path.join(self.prices_dir, f"{stock_id}.csv")
        if not os.path.exists(path):
            meta = ProviderMeta("local", None, False, error=f"Data file not found: {path}")
            return pd.DataFrame(), meta
        df = pd.read_csv(path)
        df = _normalize_ohlcv(df)
        if start:
            df = df[df["Date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["Date"] <= pd.to_datetime(end)]
        last_date = str(df["Date"].iloc[-1]) if len(df) else None
        meta = ProviderMeta("local", last_date, False)
        return df, meta


_LAST_FETCH: dict[str, float] = {}


class YahooProvider(BaseProvider):
    def __init__(
        self,
        prices_dir: str,
        lookback_days_for_live: int = 365,
        cache_min_refresh_seconds: int = 60,
        interval: str = "1d",
    ) -> None:
        self.prices_dir = prices_dir
        self.lookback_days_for_live = lookback_days_for_live
        self.cache_min_refresh_seconds = cache_min_refresh_seconds
        self.interval = interval

    def _can_fetch(self, stock_id: str) -> bool:
        last = _LAST_FETCH.get(stock_id)
        if last is None:
            return True
        return (time.time() - last) >= self.cache_min_refresh_seconds

    def get_price_df(
        self,
        stock_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Tuple[pd.DataFrame, ProviderMeta]:
        if yf is None:
            meta = ProviderMeta("yahoo", None, False, error="yfinance not installed")
            return pd.DataFrame(), meta

        interval = interval or self.interval
        if not self._can_fetch(stock_id):
            local = LocalCSVProvider(self.prices_dir)
            df, meta = local.get_price_df(stock_id, start, end, interval)
            meta.data_source_used = "local_cache"
            return df, meta

        try:
            if start or end:
                df = yf.download(stock_id, start=start, end=end, interval=interval, progress=False)
            else:
                df = yf.download(stock_id, period=f"{self.lookback_days_for_live}d", interval=interval, progress=False)
            norm = _normalize_ohlcv(df)
            if norm.empty:
                raise ValueError("Empty data from Yahoo")
            _LAST_FETCH[stock_id] = time.time()

            path = os.path.join(self.prices_dir, f"{stock_id}.csv")
            os.makedirs(self.prices_dir, exist_ok=True)
            if os.path.exists(path):
                local_df = pd.read_csv(path)
                local_df = _normalize_ohlcv(local_df)
                merged = pd.concat([local_df, norm]).drop_duplicates(subset=["Date"], keep="last")
                merged = merged.sort_values("Date").reset_index(drop=True)
            else:
                merged = norm
            merged.to_csv(path, index=False)

            last_date = str(merged["Date"].iloc[-1]) if len(merged) else None
            meta = ProviderMeta("yahoo", last_date, True)
            return merged, meta
        except Exception as exc:
            meta = ProviderMeta("yahoo", None, False, error=str(exc))
            return pd.DataFrame(), meta
