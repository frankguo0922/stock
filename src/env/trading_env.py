from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        market_features: np.ndarray,
        forecast_features: np.ndarray,
        prices: np.ndarray,
        transaction_cost: float = 0.001,
        drawdown_limit: float = 0.2,
        drawdown_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.market_features = market_features
        self.forecast_features = forecast_features
        self.prices = prices
        self.transaction_cost = transaction_cost
        self.drawdown_limit = drawdown_limit
        self.drawdown_penalty = drawdown_penalty

        obs_dim = market_features.shape[1] + forecast_features.shape[1] + 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0
        self.position = 0.0
        self.cash = 1.0
        self.last_action = 0.0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        market = self.market_features[self.t]
        forecast = self.forecast_features[self.t]
        pv = self.portfolio_value
        if not np.isfinite(pv) or pv <= 0:
            pv = 1e-9
        cash_ratio = self.cash / pv
        obs = np.concatenate([market, forecast, np.array([cash_ratio, self.position, self.last_action], dtype=np.float32)])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        target_weight = float(np.clip(action[0], 0.0, 1.0))
        prev_price = self.prices[self.t]
        self.t += 1
        done = self.t >= len(self.prices) - 1
        cur_price = self.prices[self.t]

        port_before = self.portfolio_value
        trade_cost = abs(target_weight - self.position) * self.transaction_cost
        self.position = target_weight
        self.last_action = target_weight

        denom = max(prev_price, 1e-9)
        price_return = (cur_price - prev_price) / denom
        if not np.isfinite(price_return):
            price_return = 0.0
        step_return = 1 + self.position * price_return - trade_cost
        step_return = np.clip(step_return, 1e-6, 1e6)
        self.portfolio_value = port_before * step_return
        if not np.isfinite(self.portfolio_value) or self.portfolio_value <= 0:
            self.portfolio_value = 1e-9
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.cash = self.portfolio_value * (1 - self.position)

        reward = np.log(self.portfolio_value / max(port_before, 1e-9))
        drawdown = 1 - self.portfolio_value / max(self.peak_value, 1e-9)
        if drawdown > self.drawdown_limit:
            reward -= self.drawdown_penalty * (drawdown - self.drawdown_limit)
        if not np.isfinite(reward):
            reward = 0.0
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "cost": trade_cost,
            "drawdown": drawdown,
        }
        return self._get_obs(), float(reward), done, False, info
