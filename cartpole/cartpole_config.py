from typing import List

import gym
from gym.wrappers import Monitor
import numpy as np
import tensorflow as tf

from muzero_config import MuZeroConfig
from .cartpole_network import CartPoleNetwork
from .cartpole_game import CartPoleGame


class CartPoleConfig(MuZeroConfig):
    def __init__(self):
        super(CartPoleConfig, self).__init__(
            num_actors=32,
            max_episode_moves=1000,
            num_simulations=50,
            discount=0.997,
            dirichlet_alpha=0.25,
            num_training_steps=20000,
            checkpoint_interval=20,
            td_steps=5,
            lr_init=0.05,
            lr_decay_rate=0.01,
            lr_decay_steps=10000,
            test_interval=100,
            test_num_episodes=5,
            window_size=1000,
            batch_size=128,
            value_support=20,
            reward_support=5
        )
    
    def get_init_network_params(self) -> List[np.ndarray]:
        return self.get_init_network_obj(training=True).get_params()
    
    def get_init_network_obj(self, training: bool) -> "CartPoleNetwork":
        reward_dim = 2*self.reward_support + 1
        value_dim = 2*self.value_support + 1
        return CartPoleNetwork(
            self.obs_shape, reward_dim, value_dim, self.action_space_size, training
        )

    def set_game(self, env_name = str):
        self.env_name = env_name
        game = self.new_game()
        self.obs_shape = game.reset().shape[0]
        self.action_space_size = game.action_space_size
    
    def new_game(
        self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None
    ) -> "Game":
        env = gym.make(self.env_name)
        if seed is not None:
            env.seed(seed)
        
        if save_video:
            env = Monitor(
                env, directory=save_path, force=True, video_callable=video_callable, uid=uid
            )

        return CartPoleGame(env, discount=self.discount, game_unroll_steps=4)
    
    def visit_softmax_temperature_fn(self, num_moves: int, num_trained_steps: int) -> float:
        if num_trained_steps < 0.5 * self.num_training_steps:
            return 1
        elif num_trained_steps < 0.75 * self.num_training_steps:
            return 0.5
        else:
            return 0.25
    