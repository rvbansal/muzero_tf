from dataclasses import dataclass
from typing import Callable, Dict, List

from tensorflow.keras import Model
import numpy as np
import tensorflow as tf

import model_utils as mu


@dataclass
class NetworkOutput:
    value: tf.Tensor
    reward: tf.Tensor
    policy_logits: tf.Tensor
    hidden_state: tf.Tensor


class MuZeroNetwork(Model):
    def __init__(
        self,
        scalar_transform: Callable = mu.atari_scalar_transform,
        inverse_scalar_transform: Callable = mu.inverse_atari_scalar_transform,
        training: bool = True
    ):
        super(MuZeroNetwork, self).__init__()
        self.scalar_transform = scalar_transform
        self.inverse_scalar_transform = inverse_scalar_transform
        self.training = training

    def get_params(self) -> List[np.ndarray]:
        return self.get_weights()

    def set_params(self, params: List[np.ndarray]):
        self.set_weights(params)

    def dynamics(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def representation(self, obs_history: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def prediction(self, state: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def initial_inference(self, obs: tf.Tensor, value_support: int) -> NetworkOutput:
        state = self.representation(obs)
        policy_logits, value = self.prediction(state)

        if not self.training:
            value = self.support_to_scalar(value, value_support)

        return NetworkOutput(value, 0, policy_logits, state)

    def recurrent_inference(
        self, state: tf.Tensor, action: tf.Tensor, value_support: int, reward_support: int
    ) -> NetworkOutput:
        state, reward = self.dynamics(state, action)
        policy_logits, value = self.prediction(state)

        if not self.training:
            value = self.support_to_scalar(value, value_support)
            reward = self.support_to_scalar(reward, reward_support)

        return NetworkOutput(value, reward, policy_logits, state)
    
    def scalar_to_support(self, x: tf.Tensor, support: int):
        return mu.scalar_to_support_calc(x, self.scalar_transform, support)

    def support_to_scalar(self, x: tf.Tensor, reward_support: int):
        return mu.support_to_scalar_calc(x, self.inverse_scalar_transform, reward_support)
    
    def scalar_loss_func(self, prediction: tf.Tensor, target: tf.Tensor) -> float:
        raise NotImplementedError
    