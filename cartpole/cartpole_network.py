from typing import Tuple

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from muzero_network import MuZeroNetwork


class CartPoleNetwork(MuZeroNetwork):
    def __init__(
        self,
        input_dim: int,
        reward_dim: int,
        value_dim: int,
        action_space_size: int,
        training: bool,
        hidden_state_dim: int = 32,
        dynamics_dim: int = 64,
    ):
        super(CartPoleNetwork, self).__init__(training=training)
        self.input_dim = input_dim
        self.reward_dim = reward_dim
        self.value_dim = value_dim
        self.hidden_state_dim = hidden_state_dim
        self.dynamics_dim = dynamics_dim
        self.action_space_size = action_space_size
    
        self._representation = Sequential([
            Dense(self.hidden_state_dim), 
            Activation(tf.nn.leaky_relu),
            Dense(self.hidden_state_dim),
            Activation(tf.nn.tanh)
        ])
        self._dynamics_state = Sequential([
            Dense(self.dynamics_dim), 
            Activation(tf.nn.tanh), 
            Dense(self.hidden_state_dim),
            Activation(tf.nn.tanh)
        ])
        self._dynamics_reward = Sequential([
            Dense(self.dynamics_dim),
            Activation(tf.nn.leaky_relu),
            Dense(self.reward_dim, kernel_initializer=initializers.Zeros())
        ])
        self._prediction_actor = Sequential([
            Dense(self.dynamics_dim),
            Activation(tf.nn.leaky_relu),
            Dense(self.action_space_size)
        ])
        self._prediction_value = Sequential([
            Dense(self.dynamics_dim),
            Activation(tf.nn.leaky_relu),
            Dense(self.value_dim, kernel_initializer=initializers.Zeros())
        ])
        self._representation.build((1, self.input_dim))
        self._dynamics_state.build((1, self.hidden_state_dim + self.action_space_size))
        self._dynamics_reward.build((1, self.hidden_state_dim + self.action_space_size))
        self._prediction_actor.build((1, self.hidden_state_dim))
        self._prediction_value.build((1, self.hidden_state_dim))

    def prediction(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self._prediction_actor(state), self._prediction_value(state)
    
    def representation(self, obs_history: tf.Tensor) -> tf.Tensor:
        return self._representation(obs_history)
    
    def dynamics(self, state: tf.Tensor, action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        action_encode = tf.one_hot(action, depth=self.action_space_size)
        x = tf.concat([state, action_encode], axis=1)
        return self._dynamics_state(x), self._dynamics_reward(x)
    
    def scalar_loss_func(self, prediction: tf.Tensor, target: tf.Tensor) -> float:
        return -tf.reduce_sum((tf.nn.log_softmax(prediction, axis=1) * target), axis=1)
    