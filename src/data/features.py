from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame({"macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["returns"] = out["Close"].pct_change()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["ma5"] = out["Close"].rolling(5).mean()
    out["ma20"] = out["Close"].rolling(20).mean()
    out["rsi14"] = compute_rsi(out["Close"], period=14)
    macd_df = compute_macd(out["Close"], 12, 26, 9)
    out = pd.concat([out, macd_df], axis=1)
    out["vol20"] = out["returns"].rolling(20).std()
    out = out.ffill().dropna().reset_index(drop=True)
    return out


def add_multimodal_features(
    df: pd.DataFrame,
    stock_id: str,
    fundamentals_path: str,
    news_sentiment_path: str,
    prefix: str = "mm_",
) -> pd.DataFrame:
    out = df.copy()
    if fundamentals_path:
        try:
            f = pd.read_csv(fundamentals_path)
            f["Date"] = pd.to_datetime(f["Date"], errors="coerce")
            f = f[f["stock_id"] == stock_id].dropna(subset=["Date"])
            f = f.drop(columns=["stock_id"]).sort_values("Date")
            f = f.add_prefix(prefix)
            f = f.rename(columns={f"{prefix}Date": "Date"})
            out = out.merge(f, on="Date", how="left")
        except Exception:
            pass

    if news_sentiment_path:
        try:
            n = pd.read_csv(news_sentiment_path)
            n["Date"] = pd.to_datetime(n["Date"], errors="coerce")
            n = n[n["stock_id"] == stock_id].dropna(subset=["Date"])
            n = n.drop(columns=["stock_id"]).sort_values("Date")
            n = n.add_prefix(prefix)
            n = n.rename(columns={f"{prefix}Date": "Date"})
            out = out.merge(n, on="Date", how="left")
        except Exception:
            pass

    out = out.ffill().dropna().reset_index(drop=True)
    return out
