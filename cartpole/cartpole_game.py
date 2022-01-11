from collections import deque
from typing import Dict, Tuple

import gym
import numpy as np

from game import Action, Game


class CartPoleGame(Game):
    def __init__(self, env: gym.Env, discount: float, game_unroll_steps: int):
        self.game_unroll_steps = game_unroll_steps
        self.frame_stack = deque([], maxlen = game_unroll_steps)
        self.env = env
        self.game_unroll_steps = game_unroll_steps
        super(CartPoleGame, self).__init__(env=env, discount=discount)
        
    def unroll_obs(self, idx: int) -> np.ndarray:
        frames = self.obs_history[idx : idx + self.game_unroll_steps]
        return np.asarray(frames).flatten()

    def step(self, action: 'Action') -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)

        self.rewards.append(reward)
        self.history.append(action)
        self.obs_history.append(obs)

        return self.unroll_obs(len(self.rewards)), reward, done, info
    
    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)
        
        self.rewards = []
        self.history = []
        self.obs_history = []

        for _ in range(self.game_unroll_steps):
            self.obs_history.append(obs)
        
        return self.unroll_obs(0)
