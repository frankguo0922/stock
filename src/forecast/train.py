from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

from src.data.dataset import (
    FEATURE_COLUMNS,
    apply_scaler,
    build_multi_stock_dataset,
    build_supervised_arrays,
    fit_scaler,
    get_feature_columns,
    train_val_test_split,
)
from src.data.features import add_features, add_multimodal_features
from src.data.ingest import load_price_data
from src.forecast.models import LSTMForecast, SharedForecastModel, SharedTransformerForecast
from src.utils.logger import setup_logger
import json


@dataclass
class ForecastArtifacts:
    scaler: StandardScaler
    model_type: str
    linear_model: LinearRegression | None
    lstm_model: LSTMForecast | None


def train_linear(train_df: pd.DataFrame) -> LinearRegression:
    X = train_df[FEATURE_COLUMNS].values
    y = train_df["returns"].shift(-1).dropna().values
    X = X[:-1]
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
) -> LSTMForecast:
    model = LSTMForecast(input_dim, hidden_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()
    return model


def infer_mu(
    df: pd.DataFrame,
    scaler: StandardScaler,
    model_type: str,
    linear_model: LinearRegression | None,
    lstm_model: LSTMForecast | None,
    lookback: int,
    feature_cols: list[str],
) -> np.ndarray:
    scaled = apply_scaler(df, scaler, feature_cols)
    if model_type == "linear":
        X = scaled[feature_cols].values
        mu = linear_model.predict(X)
        return mu
    X_seq, _ = build_supervised_arrays(scaled, lookback, feature_cols)
    lstm_model.eval()
    with torch.no_grad():
        mu_seq = lstm_model(torch.tensor(X_seq, dtype=torch.float32)).numpy()
    pad = np.full(lookback, np.nan, dtype=np.float32)
    mu = np.concatenate([pad, mu_seq])
    return mu


def train_forecast(config: Dict[str, Dict], stock_id: str, source: str = "local") -> pd.DataFrame:
    logger = setup_logger("forecast")
    prices_dir = config["data"]["prices_dir"]
    output_dir = config["project"]["output_dir"]
    lookback = config["features"]["lookback"]
    train_ratio = config["forecast"]["train_ratio"]
    val_ratio = config["forecast"]["val_ratio"]
    model_type = config["forecast"]["model"]

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
    if config.get("multimodal", {}).get("enabled"):
        df = add_multimodal_features(
            df,
            stock_id,
            fundamentals_path=config["multimodal"]["fundamentals_csv"],
            news_sentiment_path=config["multimodal"]["news_sentiment_csv"],
            prefix=config["multimodal"]["feature_prefix"],
        )
    train_df, _, _ = train_val_test_split(df, train_ratio, val_ratio)

    feature_cols = get_feature_columns(train_df, config.get("multimodal", {}).get("feature_prefix", "mm_"))
    scaler = fit_scaler(train_df, feature_cols)
    train_scaled = apply_scaler(train_df, scaler, feature_cols)

    linear_model = None
    lstm_model = None
    if model_type == "linear":
        linear_model = train_linear(train_scaled)
    else:
        X_train, y_train = build_supervised_arrays(train_scaled, lookback, feature_cols)
        if X_train.ndim != 3 or X_train.shape[0] == 0:
            raise ValueError(
                f"Insufficient data for LSTM: got shape {X_train.shape}. "
                f"Increase data length or reduce lookback (current {lookback})."
            )
        lstm_model = train_lstm(
            X_train,
            y_train,
            input_dim=len(FEATURE_COLUMNS),
            hidden_dim=config["forecast"]["lstm_hidden"],
            num_layers=config["forecast"]["lstm_layers"],
            epochs=config["forecast"]["lstm_epochs"],
            lr=config["forecast"]["lstm_lr"],
        )

    mu = infer_mu(df, scaler, model_type, linear_model, lstm_model, lookback, feature_cols)
    sigma = df["returns"].rolling(20).std().bfill().values
    if len(mu) < len(sigma):
        mu = np.concatenate([mu, np.full(len(sigma) - len(mu), np.nan, dtype=np.float32)])
    p_up = 1 / (1 + np.exp(-mu / (sigma + 1e-6)))

    forecast_df = pd.DataFrame({
        "Date": df["Date"].values,
        "mu": mu,
        "sigma": sigma,
        "p_up": p_up,
    })

    forecast_dir = os.path.join(output_dir, "forecast", stock_id)
    os.makedirs(forecast_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(forecast_dir, "scaler.joblib"))
    if linear_model is not None:
        joblib.dump(linear_model, os.path.join(forecast_dir, "linear_model.joblib"))
    if lstm_model is not None:
        torch.save(lstm_model.state_dict(), os.path.join(forecast_dir, "lstm_model.pt"))
    forecast_df.to_csv(os.path.join(forecast_dir, "forecast_features.csv"), index=False)

    logger.info("Forecast training complete for %s. Saved to %s", stock_id, forecast_dir)
    return forecast_df


def train_shared_forecast(
    config: Dict[str, Dict],
    stock_ids: list[str],
    forecast_id: str,
    source: str = "local",
) -> dict:
    logger = setup_logger("forecast_shared")
    prices_dir = config["data"]["prices_dir"]
    output_dir = config["project"]["output_dir"]
    lookback = config["features"]["lookback"]
    horizons = config["forecast"]["horizons"]
    loss_weights = config["forecast"]["loss_weights"]

    stock_dfs: dict[str, pd.DataFrame] = {}
    val_dfs: dict[str, pd.DataFrame] = {}
    train_parts: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    do_wf = bool(config["forecast"]["walk_forward"]["enabled"])
    wf_folds = int(config["forecast"]["walk_forward"]["folds"])
    wf_train_ratio = float(config["forecast"]["walk_forward"]["train_ratio"])
    wf_val_ratio = float(config["forecast"]["walk_forward"]["val_ratio"])
    for stock_id in stock_ids:
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
        if config.get("multimodal", {}).get("enabled"):
            df = add_multimodal_features(
                df,
                stock_id,
                fundamentals_path=config["multimodal"]["fundamentals_csv"],
                news_sentiment_path=config["multimodal"]["news_sentiment_csv"],
                prefix=config["multimodal"]["feature_prefix"],
            )
        if do_wf:
            n = len(df)
            fold_size = max(int(n * wf_val_ratio), 1)
            fold_idx = max(wf_folds - 1, 0)
            start = min(fold_idx * fold_size, n - fold_size)
            end = min(start + fold_size, n)
            train_end = max(int(n * wf_train_ratio), lookback + max(horizons) + 1)
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[start:end].copy()
            stock_dfs[stock_id] = train_df
            val_dfs[stock_id] = val_df
            train_parts.append(train_df)
        else:
            train_df, _, _ = train_val_test_split(df, config["forecast"]["train_ratio"], config["forecast"]["val_ratio"])
            stock_dfs[stock_id] = train_df
            train_parts.append(train_df)

    all_train = pd.concat(train_parts, ignore_index=True)
    feature_cols = get_feature_columns(all_train, config.get("multimodal", {}).get("feature_prefix", "mm_"))
    scaler = fit_scaler(all_train, feature_cols)

    X_train, y_train, sid = build_multi_stock_dataset(
        stock_dfs=stock_dfs,
        lookback=lookback,
        horizons=horizons,
        stock_ids=stock_ids,
        scaler=scaler,
        feature_cols=feature_cols,
    )
    if X_train.shape[0] == 0:
        raise ValueError("Insufficient data for shared forecast training.")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=config["forecast"]["lstm_lr"])
    if config["forecast"].get("loss", "mse") == "huber":
        loss_fn = nn.HuberLoss(delta=float(config["forecast"]["huber_delta"]))
    else:
        loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    sid_tensor = torch.tensor(sid, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor, sid_tensor)
    batch_size = int(config["forecast"].get("batch_size", 256))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X_val = None
    y_val = None
    sid_val = None
    if do_wf and val_dfs:
        X_val, y_val, sid_val = build_multi_stock_dataset(
            stock_dfs=val_dfs,
            lookback=lookback,
            horizons=horizons,
            stock_ids=stock_ids,
            scaler=scaler,
            feature_cols=feature_cols,
        )

    model.train()
    epoch_losses = []
    val_losses = []
    epochs = int(config["forecast"]["lstm_epochs"])
    best_loss = float("inf")
    patience = int(config["forecast"]["early_stopping_patience"])
    no_improve = 0
    if config["forecast"].get("lr_scheduler") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(config["forecast"]["lr_factor"]),
            patience=int(config["forecast"]["lr_patience"]),
        )
    else:
        scheduler = None

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb, sb in loader:
            optimizer.zero_grad()
            pred = model(xb, sb)
            loss = 0.0
            for idx, h in enumerate(horizons):
                weight = float(loss_weights[str(h)])
                loss = loss + weight * loss_fn(pred[:, idx], yb[:, idx])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        epoch_losses.append(total_loss / max(len(loader), 1))
        if X_val is not None and X_val.shape[0] > 0:
            model.eval()
            with torch.no_grad():
                pred_val = model(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(sid_val, dtype=torch.float32),
                )
                val_loss = 0.0
                for idx, h in enumerate(horizons):
                    weight = float(loss_weights[str(h)])
                    val_loss = val_loss + weight * loss_fn(
                        pred_val[:, idx],
                        torch.tensor(y_val[:, idx], dtype=torch.float32),
                    )
                val_loss = float(val_loss.item())
            val_losses.append(val_loss)
            model.train()
            if scheduler is not None:
                scheduler.step(val_loss)
            metric = val_loss
        else:
            if scheduler is not None:
                scheduler.step(epoch_losses[-1])
            metric = epoch_losses[-1]

        if metric < best_loss - 1e-6:
            best_loss = metric
            no_improve = 0
        else:
            no_improve += 1

        if val_losses:
            logger.info(
                "Shared forecast epoch %d/%d train_loss=%.6f val_loss=%.6f",
                epoch + 1,
                epochs,
                epoch_losses[-1],
                val_losses[-1],
            )
        else:
            logger.info("Shared forecast epoch %d/%d loss=%.6f", epoch + 1, epochs, epoch_losses[-1])
        if no_improve >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break
        logger.info("Shared forecast epoch %d/%d loss=%.6f", epoch + 1, epochs, epoch_losses[-1])

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor, sid_tensor).numpy()
    residual_std = {}
    for idx, h in enumerate(horizons):
        resid = y_train[:, idx] - preds[:, idx]
        residual_std[str(h)] = float(np.std(resid))

    shared_dir = os.path.join(output_dir, "forecast", "shared", forecast_id)
    os.makedirs(shared_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(shared_dir, "model.pt"))
    joblib.dump(scaler, os.path.join(shared_dir, "scaler.joblib"))

    train_log = {
        "forecast_id": forecast_id,
        "stock_ids": stock_ids,
        "lookback": lookback,
        "horizons": horizons,
        "loss_weights": loss_weights,
        "residual_std": residual_std,
        "train_size": int(X_train.shape[0]),
        "batch_size": batch_size,
        "epoch_losses": epoch_losses,
        "shared_model": shared_model,
        "early_stopping_patience": patience,
        "best_loss": best_loss,
        "val_losses": val_losses,
    }
    with open(os.path.join(shared_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    logger.info("Shared forecast training complete. Saved to %s", shared_dir)
    return train_log
