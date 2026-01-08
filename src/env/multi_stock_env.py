from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np

from src.env.trading_env import TradingEnv


@dataclass
class EnvBundle:
    market_features: np.ndarray
    forecast_features: np.ndarray
    prices: np.ndarray


class MultiStockTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        stock_ids: List[str],
        env_map: Dict[str, EnvBundle],
        transaction_cost: float = 0.001,
        episode_steps: int = 252,
        include_stock_id: bool = True,
        reset_on_stock_change: bool = True,
        drawdown_limit: float = 0.2,
        drawdown_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.stock_ids = stock_ids
        self.env_map = env_map
        self.transaction_cost = transaction_cost
        self.episode_steps = episode_steps
        self.include_stock_id = include_stock_id
        self.reset_on_stock_change = reset_on_stock_change
        self.drawdown_limit = drawdown_limit
        self.drawdown_penalty = drawdown_penalty

        sample = self._make_env(self.stock_ids[0])
        obs_dim = sample.observation_space.shape[0]
        if self.include_stock_id:
            obs_dim += len(self.stock_ids)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = sample.action_space

        self.current_stock_id = self.stock_ids[0]
        self.current_env = sample
        self.steps_in_episode = 0

    def _make_env(self, stock_id: str) -> TradingEnv:
        bundle = self.env_map[stock_id]
        return TradingEnv(
            bundle.market_features,
            bundle.forecast_features,
            bundle.prices,
            self.transaction_cost,
            drawdown_limit=self.drawdown_limit,
            drawdown_penalty=self.drawdown_penalty,
        )

    def _stock_onehot(self, stock_id: str) -> np.ndarray:
        vec = np.zeros(len(self.stock_ids), dtype=np.float32)
        idx = self.stock_ids.index(stock_id)
        vec[idx] = 1.0
        return vec

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.include_stock_id:
            return obs
        return np.concatenate([obs, self._stock_onehot(self.current_stock_id)]).astype(np.float32)

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if self.reset_on_stock_change or self.current_stock_id is None:
            self.current_stock_id = np.random.choice(self.stock_ids)
        self.current_env = self._make_env(self.current_stock_id)
        self.steps_in_episode = 0
        obs, info = self.current_env.reset(seed=seed)
        info["current_stock_id"] = self.current_stock_id
        return self._augment_obs(obs), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, truncated, info = self.current_env.step(action)
        self.steps_in_episode += 1
        if self.steps_in_episode >= self.episode_steps:
            done = True
        info["current_stock_id"] = self.current_stock_id
        return self._augment_obs(obs), reward, done, truncated, info
