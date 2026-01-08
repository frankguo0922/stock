from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data.dataset import FEATURE_COLUMNS, apply_scaler, train_val_test_split
from src.data.features import add_features
from src.data.ingest import load_price_data
from src.env.trading_env import TradingEnv
from src.env.multi_stock_env import EnvBundle, MultiStockTradingEnv
from src.utils.logger import setup_logger
import joblib
import os


def _load_env_bundle(config: Dict[str, Dict], stock_id: str, source: str) -> EnvBundle:
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
    train_df, _, _ = train_val_test_split(df, train_ratio, val_ratio)

    forecast_dir = os.path.join(output_dir, "forecast", stock_id)
    scaler = joblib.load(os.path.join(forecast_dir, "scaler.joblib"))
    train_scaled = apply_scaler(train_df, scaler, FEATURE_COLUMNS)

    forecast_df = pd.read_csv(os.path.join(forecast_dir, "forecast_features.csv"))
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    train_scaled["Date"] = pd.to_datetime(train_scaled["Date"])
    merged = train_scaled.merge(forecast_df, on="Date", how="left")
    merged[["mu", "sigma", "p_up"]] = merged[["mu", "sigma", "p_up"]].fillna(0.0)

    market_features = merged[FEATURE_COLUMNS].values.astype(np.float32)
    forecast_features = merged[["mu", "sigma", "p_up"]].values.astype(np.float32)
    prices = merged["Close"].values.astype(np.float32)

    return EnvBundle(market_features=market_features, forecast_features=forecast_features, prices=prices)


def train_rl(config: Dict[str, Dict], stock_id: str, source: str = "local") -> str:
    logger = setup_logger("train_rl")
    output_dir = config["project"]["output_dir"]
    tx_cost = config["rl"]["transaction_cost"]

    bundle = _load_env_bundle(config, stock_id, source)
    env = TradingEnv(
        bundle.market_features,
        bundle.forecast_features,
        bundle.prices,
        transaction_cost=tx_cost,
        drawdown_limit=config["rl"]["drawdown_limit"],
        drawdown_penalty=config["rl"]["drawdown_penalty"],
    )
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=config["rl"]["total_timesteps"])

    rl_dir = os.path.join(output_dir, "rl", stock_id)
    os.makedirs(rl_dir, exist_ok=True)
    model_path = os.path.join(rl_dir, "rl_model.zip")
    model.save(model_path)
    logger.info("RL training complete for %s. Saved to %s", stock_id, model_path)
    return model_path


def train_rl_shared(config: Dict[str, Dict], stock_ids: List[str], source: str = "local") -> str:
    logger = setup_logger("train_rl_shared")
    output_dir = config["project"]["output_dir"]
    tx_cost = config["rl"]["transaction_cost"]
    episode_steps = config["rl"]["episode_steps"]
    include_stock_id = config["rl"]["state"]["include_stock_id"]

    lookback = config["features"]["lookback"]
    min_len = lookback + episode_steps
    bundles: Dict[str, EnvBundle] = {}
    for sid in stock_ids:
        bundle = _load_env_bundle(config, sid, source)
        if len(bundle.prices) < min_len:
            raise ValueError(f"Insufficient data for {sid}: {len(bundle.prices)} < {min_len}")
        bundles[sid] = bundle

    env = MultiStockTradingEnv(
        stock_ids=stock_ids,
        env_map=bundles,
        transaction_cost=tx_cost,
        episode_steps=episode_steps,
        include_stock_id=include_stock_id,
        reset_on_stock_change=config["rl"]["reset_on_stock_change"],
        drawdown_limit=config["rl"]["drawdown_limit"],
        drawdown_penalty=config["rl"]["drawdown_penalty"],
    )
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=config["rl"]["total_timesteps"])

    shared_id = config["rl"]["shared_policy_id"]
    rl_dir = os.path.join(output_dir, "rl", "shared", shared_id)
    os.makedirs(rl_dir, exist_ok=True)
    model_path = os.path.join(rl_dir, "rl_model.zip")
    model.save(model_path)
    logger.info("Shared RL training complete. Saved to %s", model_path)
    return model_path
