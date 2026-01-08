from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import numpy as np

from src.data.features import add_features
from src.data.ingest import load_price_data
from src.data.dataset import FEATURE_COLUMNS, apply_scaler, get_feature_columns
from src.forecast.predict import predict_forecast, predict_shared_forecast
from src.utils.config import Config
import joblib


class PredictRequest(BaseModel):
    stock_id: str
    mode: str = "latest"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    source: Optional[str] = None
    horizons: Optional[list[int]] = None
    forecast_id: Optional[str] = None


class RecommendRequest(BaseModel):
    stock_id: str
    source: Optional[str] = None
    policy: str = "shared"
    shared_policy_id: Optional[str] = None
    forecast_id: Optional[str] = None
    horizons: Optional[list[int]] = None


class RecommendResponse(BaseModel):
    action: str
    target_weight: float
    reason: str
    meta: dict


def _latest_obs_for_stock(cfg: Config, stock_id: str, source: str, horizon_id: int | None = None) -> tuple[np.ndarray, dict]:
    prices_dir = cfg.get("data", "prices_dir")
    df, meta = load_price_data(
        stock_id,
        prices_dir,
        source=source,
        lookback_days_for_live=cfg.get("data", "lookback_days_for_live"),
        cache_min_refresh_seconds=cfg.get("data", "cache_min_refresh_seconds"),
        interval=cfg.get("provider", "yahoo", "interval", default="1d"),
    )
    if df.empty:
        raise ValueError(meta.error or "No data")
    df = add_features(df)

    forecast_dir = os.path.join(cfg.get("project", "output_dir"), "forecast", stock_id)
    scaler = joblib.load(os.path.join(forecast_dir, "scaler.joblib"))
    feature_cols = get_feature_columns(df, cfg.get("multimodal", "feature_prefix", default="mm_"))
    scaled = apply_scaler(df, scaler, feature_cols)

    forecast_df = predict_forecast(cfg.raw, stock_id, source=source)
    if forecast_df.empty:
        raise ValueError("No forecast data")
    merged = scaled.merge(forecast_df, on="Date", how="left")
    merged[["mu", "sigma", "p_up"]] = merged[["mu", "sigma", "p_up"]].fillna(0.0)
    latest = merged.iloc[-1]

    cash_ratio = 1.0
    position = 0.0
    last_action = 0.0
    obs = np.concatenate(
        [
            latest[feature_cols].values.astype(np.float32),
            np.array([latest["mu"], latest["sigma"], latest["p_up"]], dtype=np.float32),
            np.array([cash_ratio, position, last_action], dtype=np.float32),
        ]
    )
    if cfg.get("rl", "state", "include_horizon_id", default=False) and horizon_id is not None:
        horizon_ids = cfg.get("rl", "state", "horizon_ids", default=[])
        if horizon_id in horizon_ids:
            onehot = np.zeros(len(horizon_ids), dtype=np.float32)
            onehot[horizon_ids.index(horizon_id)] = 1.0
            obs = np.concatenate([obs, onehot]).astype(np.float32)
    return obs, {
        "data_source_used": meta.data_source_used,
        "last_data_date": meta.last_data_date,
        "cache_updated": meta.cache_updated,
        "error": meta.error,
    }


def create_app(config_path: str) -> FastAPI:
    cfg = Config.load(config_path)
    app = FastAPI()

    @app.get("/")
    def root():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(req: PredictRequest):
        source = req.source or "yahoo"
        if req.mode not in {"latest", "range"}:
            raise HTTPException(status_code=400, detail="mode must be latest or range")
        start = req.start_date if req.mode == "range" else None
        end = req.end_date if req.mode == "range" else None
        shared_id = req.forecast_id or cfg.get("forecast", "shared_id", default="shared_v1")
        shared_dir = os.path.join(cfg.get("project", "output_dir"), "forecast", "shared", shared_id)
        if os.path.exists(shared_dir):
            out = predict_shared_forecast(cfg.raw, req.stock_id, source=source, forecast_id=shared_id)
            if req.horizons:
                out["data"] = {str(h): out["data"].get(str(h)) for h in req.horizons if str(h) in out["data"]}
            return out

        _, meta = load_price_data(
            req.stock_id,
            cfg.get("data", "prices_dir"),
            source=source,
            start=start,
            end=end,
            lookback_days_for_live=cfg.get("data", "lookback_days_for_live"),
            cache_min_refresh_seconds=cfg.get("data", "cache_min_refresh_seconds"),
            interval=cfg.get("provider", "yahoo", "interval", default="1d"),
        )
        data = predict_forecast(cfg.raw, req.stock_id, start, end, source=source)
        if req.mode == "latest" and len(data) > 0:
            data = data.tail(1)
        return {
            "data": data.to_dict(orient="records"),
            "meta": {
                "data_source_used": meta.data_source_used,
                "last_data_date": meta.last_data_date,
                "cache_updated": meta.cache_updated,
                "error": meta.error,
                "forecast_model": "per_stock",
            },
        }

    @app.post("/recommend")
    def recommend(req: RecommendRequest):
        source = req.source or "yahoo"
        policy = req.policy
        shared_id = req.shared_policy_id or cfg.get("rl", "shared_policy_id")

        forecast_id = req.forecast_id or cfg.get("forecast", "shared_id", default="shared_v1")
        shared_dir = os.path.join(cfg.get("project", "output_dir"), "forecast", "shared", forecast_id)
        horizons = req.horizons or cfg.get("forecast", "horizons", default=[21, 126, 252])
        horizon = int(horizons[0])

        if os.path.exists(shared_dir):
            out = predict_shared_forecast(cfg.raw, req.stock_id, source=source, forecast_id=forecast_id)
            horizon_data = out["data"].get(str(horizon))
            if not horizon_data:
                raise HTTPException(status_code=400, detail="No shared forecast data")
            mu = float(horizon_data["mu"])
            sigma = float(horizon_data["sigma"])
            p_up = float(horizon_data["p_up"])
            meta = out["meta"]
            meta["forecast_model"] = forecast_id
        else:
            _, meta = load_price_data(
                req.stock_id,
                cfg.get("data", "prices_dir"),
                source=source,
                lookback_days_for_live=cfg.get("data", "lookback_days_for_live"),
                cache_min_refresh_seconds=cfg.get("data", "cache_min_refresh_seconds"),
                interval=cfg.get("provider", "yahoo", "interval", default="1d"),
            )
            data = predict_forecast(cfg.raw, req.stock_id, None, None, source=source)
            if len(data) == 0:
                raise HTTPException(status_code=400, detail="No data")
            latest = data.iloc[-1]
            mu = float(latest["mu"])
            sigma = float(latest["sigma"])
            p_up = float(latest["p_up"])
            meta = {
                "data_source_used": meta.data_source_used,
                "last_data_date": meta.last_data_date,
                "cache_updated": meta.cache_updated,
                "error": meta.error,
                "forecast_model": "per_stock",
            }

        buy_th = float(cfg.get("recommend", "buy_threshold", default=0.0))
        sell_th = float(cfg.get("recommend", "sell_threshold", default=0.0))
        p_up_th = float(cfg.get("recommend", "p_up_threshold", default=0.5))
        risk_limit = float(cfg.get("recommend", "risk_limit", default=0.05))

        target_weight = 0.5
        used_policy = "rule"
        if policy == "shared":
            model_path = os.path.join(cfg.get("project", "output_dir"), "rl", "shared", shared_id, "rl_model.zip")
            if os.path.exists(model_path):
                obs, data_meta = _latest_obs_for_stock(cfg, req.stock_id, source, horizon_id=horizon)
                if cfg.get("rl", "state").get("include_stock_id"):
                    stock_ids = cfg.get("universe", "stock_ids", default=[])
                    if req.stock_id not in stock_ids:
                        raise HTTPException(status_code=400, detail="stock_id not in universe")
                    onehot = np.zeros(len(stock_ids), dtype=np.float32)
                    onehot[stock_ids.index(req.stock_id)] = 1.0
                    obs = np.concatenate([obs, onehot]).astype(np.float32)
                from stable_baselines3 import PPO
                model = PPO.load(model_path)
                action, _ = model.predict(obs, deterministic=True)
                target_weight = float(np.clip(action[0], 0.0, 1.0))
                used_policy = "shared"
            else:
                policy = "per_stock"

        if policy == "per_stock":
            model_path = os.path.join(cfg.get("project", "output_dir"), "rl", req.stock_id, "rl_model.zip")
            if os.path.exists(model_path):
                obs, data_meta = _latest_obs_for_stock(cfg, req.stock_id, source, horizon_id=horizon)
                from stable_baselines3 import PPO
                model = PPO.load(model_path)
                action, _ = model.predict(obs, deterministic=True)
                target_weight = float(np.clip(action[0], 0.0, 1.0))
                used_policy = "per_stock"

        if mu > buy_th and p_up > p_up_th and sigma < risk_limit:
            action = "BUY"
            reason = "Expected return is positive, upside probability is high, and risk is within limit."
        elif mu < sell_th or sigma > risk_limit:
            action = "SELL"
            reason = "Expected return is negative or risk exceeds limit."
        else:
            action = "HOLD"
            reason = "Signals are mixed; maintain neutral exposure."

        meta = {
            "policy_used": used_policy,
            "mu": mu,
            "sigma": sigma,
            "p_up": p_up,
            "horizon": horizon,
            **meta,
        }
        if "data_meta" in locals():
            meta.update(data_meta)

        return RecommendResponse(
            action=action,
            target_weight=target_weight,
            reason=reason,
            meta=meta,
        )

    @app.get("/backtest_report")
    def backtest_report(stock_id: str, policy: str = "per_stock", shared_policy_id: Optional[str] = None):
        shared_id = shared_policy_id or cfg.get("rl", "shared_policy_id")
        if policy == "shared":
            report_path = os.path.join(cfg.get("project", "output_dir"), "backtest", stock_id, "report.json")
        else:
            report_path = os.path.join(cfg.get("project", "output_dir"), "backtest", stock_id, "report.json")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")

    return app
