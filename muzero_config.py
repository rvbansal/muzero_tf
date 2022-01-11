import argparse
import os
from typing import Callable, Dict, List

import numpy as np


class MuZeroConfig:
    def __init__(
        self,
        num_actors: int,
        max_episode_moves: int,
        num_simulations: int,
        discount: float,
        dirichlet_alpha: float,
        value_support: float,
        reward_support: float,
        num_training_steps: int,
        checkpoint_interval: int,
        batch_size: int,
        td_steps: int,
        lr_init: float,
        lr_decay_rate: float,
        lr_decay_steps: float,
        test_interval: int,
        test_num_episodes: int,
        exploration_frac: float = 0.25,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
        window_size: int = int(1e6),
        num_unroll_steps: int = 5,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        lr_floor: float = 0.001,
        value_loss_coef: float = 1,
        soft_update_tau: float = 0.2,
        max_grad_norm: float = 5,
        use_target_model: bool = False,
        revisit_policy_search_rate: float = 0,
        priorities_alpha: float = 1,
        weights_beta: float = 1,
        use_max_priority: bool = False,
        replay_memory_checkpoint: int = 50,
    ):
        # Self-play
        self.num_actors = num_actors
        self.max_episode_moves = max_episode_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_frac = exploration_frac

        # UCB formula
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init

        # Value and reward supports
        self.value_support = value_support
        self.reward_support = reward_support

        # Training
        self.num_training_steps = num_training_steps
        self.checkpoint_interval = checkpoint_interval
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.value_loss_coef = value_loss_coef
        self.soft_update_tau = soft_update_tau

        # Optimization params
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.lr_floor = lr_floor
        self.max_grad_norm = max_grad_norm

        # Testing params
        self.test_interval = test_interval
        self.test_num_episodes = test_num_episodes

        # Replay memory
        self.use_target_model = use_target_model
        self.revisit_policy_search_rate = revisit_policy_search_rate
        self.priorities_alpha = priorities_alpha
        self.weights_beta = weights_beta
        self.use_max_priority = use_max_priority
        self.replay_memory_checkpoint = replay_memory_checkpoint

        # To be set with command line args
        self.seed = None
        self.exp_path = None
        self.network_path = None
        self.debug = None

    def new_game(
        self,
        seed: int = None,
        save_video: bool = False,
        save_path: bool = False,
        video_callable: Callable = None,
        uid: int = None,
    ) -> 'Game':
        raise NotImplementedError

    def set_game(self, env_name=str):
        raise NotImplementedError

    def get_init_network_params(self) -> List[np.ndarray]:
        raise NotImplementedError

    def get_init_network_obj(self, training: bool) -> 'MuZeroNetwork':
        raise NotImplementedError

    def visit_softmax_temperature_fn(self, num_moves: int, trained_steps: int) -> float:
        raise NotImplementedError

    def get_hyperparams(self) -> Dict:
        hyperparams = {}
        for k, v in self.__dict__.items():
            if "path" not in k and (v is not None):
                hyperparams[k] = v
        return hyperparams

    def set_config(self, args: argparse.Namespace) -> str:
        self.set_game(args.env)
        self.seed = args.seed

        self.debug = args.debug

        self.priorities_alpha = 1 if args.use_priority else 0
        self.use_target_model = args.use_target_model
        self.use_max_priority = args.use_max_priority and args.use_priority

        if args.value_loss_coef is not None:
            self.value_loss_coef = args.value_loss_coef

        if args.revisit_policy_search_rate is not None:
            self.revisit_policy_search_rate = args.revisit_policy_search_rate

        self.exp_path = os.path.join(args.result_dir, args.env, 'seed_{}'.format(self.seed))
        self.network_path = os.path.join(self.exp_path, 'network.h5')
        return self.exp_path
