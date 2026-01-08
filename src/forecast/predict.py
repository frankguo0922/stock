from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib
import torch
import os
import json

from src.data.dataset import FEATURE_COLUMNS, apply_scaler, get_feature_columns
from src.data.features import add_features, add_multimodal_features
from src.data.ingest import load_price_data
from src.forecast.models import LSTMForecast, SharedForecastModel, SharedTransformerForecast


def load_lstm(model_path: str, input_dim: int, hidden_dim: int, num_layers: int) -> LSTMForecast:
    model = LSTMForecast(input_dim, hidden_dim, num_layers)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_forecast(
    config: Dict[str, Dict],
    stock_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: str = "local",
) -> pd.DataFrame:
    prices_dir = config["data"]["prices_dir"]
    output_dir = config["project"]["output_dir"]
    lookback = config["features"]["lookback"]
    model_type = config["forecast"]["model"]

    df, _ = load_price_data(
        stock_id,
        prices_dir,
        source=source,
        start=start,
        end=end,
        lookback_days_for_live=config["data"]["lookback_days_for_live"],
        cache_min_refresh_seconds=config["data"]["cache_min_refresh_seconds"],
        interval=config["provider"]["yahoo"]["interval"],
    )
    if df.empty:
        return pd.DataFrame(columns=["Date", "mu", "sigma", "p_up"])

    df = add_features(df)
    if config.get("multimodal", {}).get("enabled"):
        df = add_multimodal_features(
            df,
            stock_id,
            fundamentals_path=config["multimodal"]["fundamentals_csv"],
            news_sentiment_path=config["multimodal"]["news_sentiment_csv"],
            prefix=config["multimodal"]["feature_prefix"],
        )

    forecast_dir = os.path.join(output_dir, "forecast", stock_id)
    scaler = joblib.load(os.path.join(forecast_dir, "scaler.joblib"))
    feature_cols = get_feature_columns(df, config.get("multimodal", {}).get("feature_prefix", "mm_"))
    scaled = apply_scaler(df, scaler, feature_cols)

    if model_type == "linear":
        model = joblib.load(os.path.join(forecast_dir, "linear_model.joblib"))
        mu = model.predict(scaled[feature_cols].values)
    else:
        model = load_lstm(
            os.path.join(forecast_dir, "lstm_model.pt"),
            input_dim=len(FEATURE_COLUMNS),
            hidden_dim=config["forecast"]["lstm_hidden"],
            num_layers=config["forecast"]["lstm_layers"],
        )
        X = scaled[feature_cols].values
        seq = []
        for i in range(lookback, len(df)):
            seq.append(X[i - lookback : i])
        if seq:
            with torch.no_grad():
                mu_seq = model(torch.tensor(np.array(seq, dtype=np.float32))).numpy()
            mu = np.concatenate([np.full(lookback, np.nan, dtype=np.float32), mu_seq])
        else:
            mu = np.full(len(df), np.nan, dtype=np.float32)

    sigma = df["returns"].rolling(20).std().bfill().values
    if len(mu) < len(sigma):
        mu = np.concatenate([mu, np.full(len(sigma) - len(mu), np.nan, dtype=np.float32)])
    p_up = 1 / (1 + np.exp(-mu / (sigma + 1e-6)))

    return pd.DataFrame({"Date": df["Date"].values, "mu": mu, "sigma": sigma, "p_up": p_up})


def predict_shared_forecast(
    config: Dict[str, Dict],
    stock_id: str,
    source: str = "yahoo",
    forecast_id: str | None = None,
) -> dict:
    prices_dir = config["data"]["prices_dir"]
    output_dir = config["project"]["output_dir"]
    lookback = config["features"]["lookback"]
    horizons = config["forecast"]["horizons"]

    df, meta = load_price_data(
        stock_id,
        prices_dir,
        source=source,
        lookback_days_for_live=config["data"]["lookback_days_for_live"],
        cache_min_refresh_seconds=config["data"]["cache_min_refresh_seconds"],
        interval=config["provider"]["yahoo"]["interval"],
    )
    if df.empty:
        return {"data": {}, "meta": {"error": meta.error}}
    df = add_features(df)
    if config.get("multimodal", {}).get("enabled"):
        df = add_multimodal_features(
            df,
            stock_id,
            fundamentals_path=config["multimodal"]["fundamentals_csv"],
            news_sentiment_path=config["multimodal"]["news_sentiment_csv"],
            prefix=config["multimodal"]["feature_prefix"],
        )

    shared_id = forecast_id or config["forecast"].get("shared_id", "shared_v1")
    shared_dir = os.path.join(output_dir, "forecast", "shared", shared_id)
    scaler = joblib.load(os.path.join(shared_dir, "scaler.joblib"))

    feature_cols = get_feature_columns(df, config.get("multimodal", {}).get("feature_prefix", "mm_"))
    scaled = df.copy()
    scaled[feature_cols] = scaler.transform(df[feature_cols])
    values = scaled[feature_cols].values

    if len(df) < lookback + max(horizons):
        return {"data": {}, "meta": {"error": "Insufficient data for shared forecast"}}

    X = values[-lookback:]
    X_tensor = torch.tensor(X[None, ...], dtype=torch.float32)

    stock_ids = config["universe"]["stock_ids"]
    onehot = np.zeros(len(stock_ids), dtype=np.float32)
    if stock_id in stock_ids:
        onehot[stock_ids.index(stock_id)] = 1.0
    sid_tensor = torch.tensor(onehot[None, ...], dtype=torch.float32)

    shared_model = config["forecast"].get("shared_model", "lstm")
    if shared_model == "transformer":
        model = SharedTransformerForecast(
            input_dim=len(FEATURE_COLUMNS),
            d_model=config["forecast"]["transformer_d_model"],
            nhead=config["forecast"]["transformer_nhead"],
            num_layers=config["forecast"]["transformer_layers"],
            dropout=config["forecast"]["transformer_dropout"],
            stock_id_dim=len(stock_ids),
            out_dim=len(horizons),
        )
    else:
        model = SharedForecastModel(
            input_dim=len(FEATURE_COLUMNS),
            hidden_dim=config["forecast"]["lstm_hidden"],
            num_layers=config["forecast"]["lstm_layers"],
            stock_id_dim=len(stock_ids),
            out_dim=len(horizons),
        )
    state = torch.load(os.path.join(shared_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        mu = model(X_tensor, sid_tensor).numpy().squeeze(0)

    train_log_path = os.path.join(shared_dir, "train_log.json")
    residual_std = {}
    if os.path.exists(train_log_path):
        with open(train_log_path, "r", encoding="utf-8") as f:
            residual_std = json.load(f).get("residual_std", {})

    close = float(df["Close"].iloc[-1])
    outputs = {}
    for idx, h in enumerate(horizons):
        sigma = float(residual_std.get(str(h), np.std(df["returns"].values)))
        mu_h = float(mu[idx])
        p_up = 1 / (1 + np.exp(-np.clip(mu_h / (sigma + 1e-6), -20, 20)))
        pred_close = close * float(np.exp(mu_h))
        z = 1.96
        pi_low = close * float(np.exp(mu_h - z * sigma))
        pi_high = close * float(np.exp(mu_h + z * sigma))
        outputs[str(h)] = {
            "mu": mu_h,
            "sigma": sigma,
            "p_up": float(p_up),
            "pred_close": pred_close,
            "pi_low": pi_low,
            "pi_high": pi_high,
        }

    return {
        "data": outputs,
        "meta": {
            "data_source_used": meta.data_source_used,
            "last_data_date": meta.last_data_date,
            "cache_updated": meta.cache_updated,
            "forecast_model": shared_id,
            "stock_id_in_universe": stock_id in stock_ids,
        },
    }
