from typing import Dict, List, Tuple
import math

import numpy as np
import tensorflow as tf
from scipy.stats import entropy


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(
        self, to_play: 'Player', actions: 'List[Action]', output: 'NetworkOutput'
    ):
        self.to_play = to_play
        self.hidden_state = output.hidden_state
        self.reward = output.reward

        policy = {a: math.exp(output.policy_logits[0][a.index]) for a in actions}
        policy_sum = sum(policy.values())
        for action, policy_val in policy.items():
            self.children[action] = Node(prior=policy_val / policy_sum)

    def add_noise(self, alpha: float, explore_frac: float):
        actions = list(self.children.keys())
        noises = np.random.dirichlet([alpha for _ in range(len(actions))])
        for i in range(len(actions)):
            action = actions[i]
            noise = noises[i]
            self.children[action].prior = (
                self.children[action].prior * (1 - explore_frac) + noise * explore_frac
            )
    
    def select_action(node, temperature: float = 1, random: bool = False) -> Tuple[int, float]:
        child_info = [(child.visit_count, action) for action, child in node.children.items()]
        probs = [visit_count ** (1 / temperature) for visit_count, _ in child_info]
        probs_total = sum(probs)
        probs = [p / probs_total for p in probs]

        if random == False:
            index = np.argmax([v for v, _ in child_info])
        else:
            index = np.random.choice(len(child_info), p=probs)

        return child_info[index][1], entropy(probs, base=2)


class MonteCarloTreeSearch:
    def __init__(self, config: 'MuZeroConfig'):
        self.config = config

    def run(self, root: Node, network: 'MuZeroNetwork', action_history: 'List[Action]'):
        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            history = action_history.clone()
            node = root
            search_path = [root]

            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                history.add_action(action)
                search_path.append(node)

            parent_state = search_path[-2]
            network_output = network.recurrent_inference(
                parent_state.hidden_state,
                tf.convert_to_tensor([history.last_action().index]),
                self.config.value_support,
                self.config.reward_support
            )
            node.expand(history.to_play(), history.action_space(), network_output)

            self.backpropagate(
                search_path,
                float(network_output.value),
                history.to_play(),
                min_max_stats,
            )

    def select_child(
        self, node: 'Node', min_max_stats: 'MinMaxStats'
    ) -> 'Tuple[Action, Node]':
        max_score, selected_child, selected_action = float('-inf'), None, None

        for action, child in node.children.items():
            score = self.compute_ucb_score(node, child, min_max_stats)
            if score > max_score:
                max_score = score
                selected_child = child
                selected_action = action

        return selected_action, selected_child

    def compute_ucb_score(
        self, parent: 'Node', child: 'Node', min_max_stats: 'MinMaxStats'
    ) -> float:
        pb_c_log_term = np.log(
            (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
        )
        pb_c = pb_c_log_term + self.config.pb_c_init
        prior_score = (
            pb_c * child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
        )
        value_score = min_max_stats.normalize(child.value())

        return value_score + prior_score

    def backpropagate(
        self,
        search_path: "List[Node]",
        value: float,
        to_play: "Player",
        min_max_stats: "MinMaxStats",
    ):
        for node in search_path[::-1]:
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + self.config.discount * value
