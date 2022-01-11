import ray
import tensorflow as tf

from mcts import MonteCarloTreeSearch, Node
from ray_constants import FRAC_CPUS_PER_WORKER, FRAC_GPUS_PER_WORKER
from replay_memory import ReplayMemory


@ray.remote(num_gpus=FRAC_GPUS_PER_WORKER, num_cpus=FRAC_CPUS_PER_WORKER)
class ExperienceActor:
    def __init__(
        self,
        index: int,
        config: 'MuZeroConfig',
        central_storage: 'CentralActorStorage',
        replay_memory: 'ReplayMemory'
    ):
        self.index = index
        self.config = config
        self.central_storage = central_storage
        self.replay_memory = replay_memory
        self.eps = 1e-5

    def total_training_steps(self) -> int:
        return ray.get(self.central_storage.total_training_steps.remote())

    def run(self):
        network = self.config.get_init_network_obj(training=False)
        while self.total_training_steps() < self.config.num_training_steps:
            stored_params = ray.get(self.central_storage.get_params.remote())
            network.set_params(stored_params)

            env = self.config.new_game(self.config.seed + self.index)
            obs = env.reset()

            temperature = self.config.visit_softmax_temperature_fn(
                len(env.history), self.total_training_steps()
            )
            episode_moves, episode_reward, episode_entropies = 0, 0, 0
            priorities = []
            done = False

            while episode_moves <= self.config.max_episode_moves and not done:
                root = Node(0)
                obs = tf.expand_dims(
                    tf.convert_to_tensor(obs, dtype=tf.float32), axis=0
                )
                network_output = network.initial_inference(obs, self.config.value_support)
                root.expand(env.to_play(), env.legal_actions(), network_output)
                root.add_noise(
                    self.config.root_dirichlet_alpha, self.config.root_exploration_frac
                )
                MonteCarloTreeSearch(self.config).run(root, network, env.action_history())
                action, prob_entropy = root.select_action(temperature, random=True)
                obs, reward, done, _ = env.step(action.index)
                env.store_search_statistics(root)

                episode_reward += reward
                episode_moves += 1
                episode_entropies += prob_entropy

                if not self.config.use_max_priority:
                    error = tf.abs(network_output.value - root.value())
                    priorities.append(float(error + self.eps))
            
            env.close()

            self.replay_memory.save_game.remote(
                env, priorities = None if self.config.use_max_priority else priorities
            )
            self.central_storage.store_experience.remote(
                episode_moves, episode_reward, temperature, episode_entropies / episode_moves
            )
