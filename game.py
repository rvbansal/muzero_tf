import random
from typing import List, Tuple

from gym import Env
import tensorflow as tf
from mcts import MonteCarloTreeSearch, Node


class Action:
    def __init__(self, index: int):
        self.index = index

    def __hash__(self) -> int:
        return self.index

    def __eq__(self, other) -> bool:
        if not isinstance(other, Action):
            raise NotImplementedError
        return self.index == other.index

    def __gt__(self, other) -> bool:
        if not isinstance(other, Action):
            raise NotImplementedError
        return self.index > other.index


class Player:
    def __init__(self, index: int = 1):
        self.index = index

    def __hash__(self) -> int:
        return self.index

    def __eq__(self, other) -> bool:
        if not isinstance(other, Player):
            raise NotImplementedError
        return self.index == other.index


class ActionHistory:
    """
    Simple history container used inside the search. Only used to keep track of the actions executed.
    """

    def __init__(self, history: 'List[Action]', action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size
    
    def action_space(self) -> 'List[Action]':
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> 'Player':
        return Player()

    def clone(self) -> 'ActionHistory':
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: 'Action'):
        self.history.append(action)

    def last_action(self) -> 'Action':
        return self.history[-1]


class Game:
    """
    A single episode of interaction with the environment.
    """

    def __init__(self, env: Env, discount: float):
        self.env = env
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.obs_history = []
        self.action_space_size = self.env.action_space.n
        self.discount = discount
    
    def action_history(self, index: int = None) -> 'ActionHistory':
        if index is None:
            return ActionHistory(self.history, self.action_space_size)
        else:
            return ActionHistory(self.history[:index], self.action_space_size)
    
    def legal_actions(self) -> 'List[Action]':
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> 'Player':
        return Player()

    def make_target(
        self,
        state_index: int,
        num_unroll_steps: int,
        td_steps: int,
        network: 'MuZeroNetwork' = None,
        config: 'MuZeroConfig' = None,
    ) -> Tuple[tf.Tensor, float, float]:

        target_policies, target_values, target_rewards = [], [], []

        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index >= len(self.root_values):
                value = 0
            else:
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
                if network is not None:
                    obs = tf.expand_dims(
                        tf.convert_to_tensor(self.unroll_obs(bootstrap_index)),
                        axis=0,
                    )
                    output = network.initial_inference(obs, config.value_support)
                    value = float(output.value) * self.discount ** td_steps

            for i, reward in enumerate(self.rewards[current_index : bootstrap_index]):
                value += reward * self.discount ** i

            if current_index < len(self.root_values):
                target_values.append(float(value))
                target_rewards.append(self.rewards[current_index])
                if (
                    network is not None
                    and random.random() < config.revisit_policy_search_rate
                ):
                    root = Node(0)
                    obs = tf.expand_dims(
                        tf.convert_to_tensor(self.unroll_obs(current_index)), axis=0
                    )
                    network_output = network.initial_inference(obs, config.value_support)
                    root.expand(self.to_play(), self.legal_actions(), network_output)
                    MonteCarloTreeSearch(config).run(
                        root, network, self.action_history(current_index)
                    )
                    self.store_search_statistics(root, current_index)

                target_policies.append(self.child_visits[current_index])
            else:
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append([0 for _ in range(len(self.child_visits[0]))])
        
        return target_policies, target_values, target_rewards

    def store_search_statistics(self, root: 'Node', index: int = None):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = [Action(index) for index in range(self.action_space_size)]
        search_child_visits = [
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ]
        if index is None:
            self.root_values.append(root.value())
            self.child_visits.append(search_child_visits)
        else:
            self.child_visits[index] = search_child_visits
            self.root_values[index] = root.value()
    
    def step(self, action: 'Action'):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    def __len__(self):
        return len(self.rewards)
