from typing import List, Tuple
from dataclasses import dataclass

import ray
import numpy as np
import tensorflow as tf

from ray_constants import FRAC_CPUS_PER_WORKER, FRAC_GPUS_PER_WORKER


@dataclass
class BatchData:
    obs: tf.Tensor
    action: tf.Tensor
    reward: tf.Tensor
    value: tf.Tensor
    policy: tf.Tensor
    indices: tf.Tensor
    weights: tf.Tensor


@ray.remote(num_gpus=FRAC_GPUS_PER_WORKER, num_cpus=FRAC_CPUS_PER_WORKER)
class ReplayMemory:
    def __init__(
        self,
        batch_size: int,
        capacity: int,
        priorities_alpha: float = 1,
        weights_beta: float = 1,
    ):
        self.num_games_collected = 0
        self.priorities_alpha = priorities_alpha
        self.weights_beta = weights_beta
        self.memory = []
        self.priorities = []
        self.game_and_step_by_idx = []
        self.total_game_idx = 0
        self.batch_size = batch_size
        self.capacity = capacity

    def save_game(self, game: 'Game', priorities: List[float]):
        if priorities is None:
            max_priority = self.priorities.max() if self.memory else 1
            self.priorities = np.concatenate(
                (self.priorities, [max_priority for _ in range(len(game))])
            )
        else:
            self.priorities = np.concatenate((self.priorities, priorities))
        
        self.memory.append(game)
        for step_idx in range(len(game)):
            self.game_and_step_by_idx += [(self.total_game_idx + len(self.memory) - 1, step_idx)]
        self.num_games_collected += 1

    def get_batch(
        self,
        num_unroll_steps: int,
        td_steps: int,
        network_params: List[np.ndarray],
        config: 'MuZeroConfig'
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        obs_output, action_output, reward_output, value_output, policy_output = [], [], [], [], []
    
        probs = np.asarray(self.priorities) ** self.priorities_alpha
        probs = probs / sum(probs)
        indices = np.random.choice(len(self.priorities), self.batch_size, p=probs)
        weights = (len(self.priorities) * probs[indices]) ** (-self.weights_beta)
        weights = weights / weights.max()

        indices = tf.convert_to_tensor(indices)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        network = None
        if config.use_target_model:
            network = config.get_init_network_obj(training=False)
            network.set_params(network_params)

        for idx in indices:
            game_idx, game_step_idx = self.game_and_step_by_idx[idx]
            game_idx -= self.total_game_idx
            game = self.memory[game_idx]
            actions = game.history[game_step_idx : game_step_idx + num_unroll_steps]
            if len(actions) < num_unroll_steps:
                diff = num_unroll_steps - len(actions)
                actions += [
                    np.random.randint(0, game.action_space_size) for _ in range(diff)
                ]
            
            policy, value, reward = game.make_target(
                game_step_idx, num_unroll_steps, td_steps, network, config
            )
            obs_output.append(game.unroll_obs(game_step_idx))
            action_output.append(actions)
            reward_output.append(reward)
            value_output.append(value)
            policy_output.append(policy)

        return BatchData(
            obs=tf.convert_to_tensor(obs_output, dtype=tf.float32),
            action=tf.convert_to_tensor(action_output, dtype=tf.int32),
            reward=tf.convert_to_tensor(reward_output, dtype=tf.float32),
            value=tf.convert_to_tensor(value_output, dtype=tf.float32),
            policy=tf.convert_to_tensor(policy_output, dtype=tf.float32),
            indices=indices,
            weights=weights
        )

    def get_priorities(self) -> List[float]:
        return self.priorities

    def set_priorities(self, indices: List[int], priorities: List[float]):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def remove_excess_games(self):
        if len(self.memory) > self.capacity:
            num_excess_games = len(self.memory) - self.capacity
            num_excess_steps = sum(
                [len(game) for game in self.memory[:num_excess_games]]
            )
            del self.memory[:num_excess_games]
            del self.game_and_step_by_idx[:num_excess_steps]
            self.priorities = self.priorities[num_excess_steps:]
            self.total_game_idx += num_excess_games

    def size(self) -> int:
        return len(self.memory)

    def num_games_collected(self) -> int:
        return self.num_games_collected
