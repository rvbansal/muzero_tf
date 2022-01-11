from typing import List, Tuple

import ray
import numpy as np
from ray_constants import FRAC_GPUS_PER_WORKER, FRAC_CPUS_PER_WORKER


@ray.remote(num_gpus=FRAC_GPUS_PER_WORKER, num_cpus=FRAC_CPUS_PER_WORKER)
class CentralActorStorage:
    def __init__(self, network_params: List[np.ndarray]):
        self.network_params = network_params
        self.total_training_steps = 0
        self.stored_rewards = []
        self.stored_episode_lens = []
        self.stored_temperatures = []
        self.stored_test_scores = []
        self.stored_visit_entropies = []

    def get_params(self) -> List[np.ndarray]:
        return self.network_params

    def set_params(self, params: List[np.ndarray]) -> List[np.ndarray]:
        self.network_params = params
        return params

    def get_total_training_steps(self) -> int:
        return self.total_training_steps

    def store_experience(
        self, episode_len: int, reward: float, temperature: float, visit_entropy: float
    ):
        self.stored_episode_lens.append(episode_len)
        self.stored_rewards.append(reward)
        self.stored_temperatures.append(temperature)
        self.stored_visit_entropies.append(visit_entropy)

    def get_actors_log(self) -> Tuple[float, float, float, float]:
        reward_output = self.get_stored_attribute(self.stored_rewards)
        episode_len_output = self.get_stored_attribute(self.stored_episode_lens)
        temperature_output = self.get_stored_attribute(self.stored_temperatures)
        visit_entropy_output = self.get_stored_attribute(self.stored_visit_entropies)

        self.stored_rewards = []
        self.stored_episode_lens = []
        self.stored_temperatures = []
        self.stored_visit_entropies = []

        return (
            reward_output, episode_len_output, temperature_output, visit_entropy_output
        )
    
    def get_test_scores_log(self) -> List[float]:
        if len(self.stored_test_scores) > 0:
            test_score_output = self.get_stored_attribute(self.stored_test_scores)
            self.stored_test_scores = []
            return test_score_output 

    @staticmethod
    def get_stored_attribute(attribute) -> float:
        return np.asarray(attribute).mean() if len(attribute) > 0 else None

    def add_test_score(self, score: float):
        self.stored_test_scores.append(score)

    def increment_total_training_steps(self):
        self.total_training_steps += 1

    def total_training_steps(self) -> int:
        return self.total_training_steps
