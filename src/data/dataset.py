from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "returns",
    "log_return",
    "ma5",
    "ma20",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "vol20",
]

# Backward-compat alias for existing imports.
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS


def get_feature_columns(df: pd.DataFrame, multimodal_prefix: str = "mm_") -> list[str]:
    cols = BASE_FEATURE_COLUMNS.copy()
    cols.extend([c for c in df.columns if c.startswith(multimodal_prefix)])
    return cols


def build_multi_stock_dataset(
    stock_dfs: dict[str, pd.DataFrame],
    lookback: int,
    horizons: list[int],
    stock_ids: list[str],
    scaler: StandardScaler,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    id_list: list[np.ndarray] = []

    for stock_id in stock_ids:
        df = stock_dfs[stock_id]
        scaled = df.copy()
        scaled[feature_cols] = scaler.transform(df[feature_cols])
        values = scaled[feature_cols].values
        close = df["Close"].values
        for i in range(lookback, len(df) - max(horizons)):
            X_list.append(values[i - lookback : i])
            labels = []
            for h in horizons:
                labels.append(np.log(close[i + h] / close[i]))
            y_list.append(np.array(labels, dtype=np.float32))
            onehot = np.zeros(len(stock_ids), dtype=np.float32)
            onehot[stock_ids.index(stock_id)] = 1.0
            id_list.append(onehot)

    if not X_list:
        X = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
        y = np.empty((0, len(horizons)), dtype=np.float32)
        sid = np.empty((0, len(stock_ids)), dtype=np.float32)
        return X, y, sid

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    sid = np.stack(id_list).astype(np.float32)
    return X, y, sid


def train_val_test_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def build_supervised_arrays(df: pd.DataFrame, lookback: int, feature_cols: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    values = df[feature_cols].values
    y = df["returns"].shift(-1).values
    for i in range(lookback, len(df) - 1):
        X_list.append(values[i - lookback : i])
        y_list.append(y[i])
    if not X_list:
        X = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        return X, y
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols])
    return out
