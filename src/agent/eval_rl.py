from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.backtest.metrics import compute_metrics
from src.data.dataset import FEATURE_COLUMNS, apply_scaler, train_val_test_split
from src.data.features import add_features
from src.data.ingest import load_price_data
from src.env.trading_env import TradingEnv
import joblib
import os


def _build_eval_data(config: Dict[str, Dict], stock_id: str, source: str) -> Tuple[TradingEnv, pd.Series]:
    prices_dir = config["data"]["prices_dir"]
    output_dir = config["project"]["output_dir"]
    df, meta = load_price_data(
        stock_id,
        prices_dir,
        source=source,
        lookback_days_for_live=config["data"]["lookback_days_for_live"],
        cache_min_refresh_seconds=config["data"]["cache_min_refresh_seconds"],
        interval=config["provider"]["yahoo"]["interval"],
    )
    if df.empty:
        raise ValueError(f"No data for {stock_id}. Error: {meta.error}")

    df = add_features(df)
    train_ratio = config["forecast"]["train_ratio"]
    val_ratio = config["forecast"]["val_ratio"]
    _, _, test_df = train_val_test_split(df, train_ratio, val_ratio)

    forecast_dir = os.path.join(output_dir, "forecast", stock_id)
    scaler = joblib.load(os.path.join(forecast_dir, "scaler.joblib"))
    scaled = apply_scaler(test_df, scaler, FEATURE_COLUMNS)

    forecast_df = pd.read_csv(os.path.join(forecast_dir, "forecast_features.csv"))
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    scaled["Date"] = pd.to_datetime(scaled["Date"])
    merged = scaled.merge(forecast_df, on="Date", how="left")
    merged[["mu", "sigma", "p_up"]] = merged[["mu", "sigma", "p_up"]].fillna(0.0)

    market_features = merged[FEATURE_COLUMNS].values.astype(np.float32)
    forecast_features = merged[["mu", "sigma", "p_up"]].values.astype(np.float32)
    prices = merged["Close"].values.astype(np.float32)

    env = TradingEnv(
        market_features,
        forecast_features,
        prices,
        transaction_cost=config["rl"]["transaction_cost"],
        drawdown_limit=config["rl"]["drawdown_limit"],
        drawdown_penalty=config["rl"]["drawdown_penalty"],
    )
    return env, merged["Date"]


def evaluate_rl(config: Dict[str, Dict], stock_id: str, source: str = "local") -> Tuple[pd.DataFrame, Dict[str, float]]:
    output_dir = config["project"]["output_dir"]
    env, dates = _build_eval_data(config, stock_id, source)
    model = PPO.load(os.path.join(output_dir, "rl", stock_id, "rl_model.zip"))

    obs, _ = env.reset()
    done = False
    values = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        values.append(info["portfolio_value"])

    equity = pd.DataFrame({"Date": dates.values[: len(values)], "equity": values})
    metrics = compute_metrics(equity["equity"].values)
    return equity, metrics


def evaluate_rl_shared(config: Dict[str, Dict], stock_id: str, source: str = "local") -> Tuple[pd.DataFrame, Dict[str, float]]:
    output_dir = config["project"]["output_dir"]
    env, dates = _build_eval_data(config, stock_id, source)

    include_stock_id = config["rl"]["state"]["include_stock_id"]
    stock_ids = config["universe"]["stock_ids"]
    if include_stock_id:
        if stock_id not in stock_ids:
            raise ValueError(f"stock_id {stock_id} not in universe")
        onehot = np.zeros(len(stock_ids), dtype=np.float32)
        onehot[stock_ids.index(stock_id)] = 1.0

    model_path = os.path.join(output_dir, "rl", "shared", config["rl"]["shared_policy_id"], "rl_model.zip")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    if include_stock_id:
        obs = np.concatenate([obs, onehot]).astype(np.float32)

    done = False
    values = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        if include_stock_id:
            obs = np.concatenate([obs, onehot]).astype(np.float32)
        values.append(info["portfolio_value"])

    equity = pd.DataFrame({"Date": dates.values[: len(values)], "equity": values})
    metrics = compute_metrics(equity["equity"].values)
    return equity, metrics
