import multiprocessing
import os
from typing import Dict, List

import ray
import tensorflow as tf

from mcts import MonteCarloTreeSearch, Node


def _test(
    config: "MuZeroConfig",
    render: bool,
    network: "MuZeroNetwork",
    storage: "CentralActorStorage",
    save_path: bool,
    ep_idx: int,
    save_video: bool = False
):
    env = config.new_game(
        save_video=save_video,
        save_path=save_path,
        uid=ep_idx,
        video_callable=lambda episode_id: True,
    )

    done = False
    ep_reward = 0
    obs = env.reset()

    if network is None:
        network = config.get_init_network_obj(training=False)
        network.set_params(ray.get(storage.get_params.remote()))

    while not done:
        if render:
            env.render()
        root = Node(0)
        obs = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), axis=0)
        output = network.initial_inference(obs, config.value_support)
        root.expand(env.to_play(), env.legal_actions(), output)
        MonteCarloTreeSearch(config).run(root, network, env.action_history())
        action, _ = root.select_action(temperature=1, random=False)
        obs, reward, done, _ = env.step(action.index)
        ep_reward += reward
    env.close()

    return ep_reward


def test(
    config: "MuZeroConfig",    
    num_episodes: int,
    render: bool,
    network: "MuZeroNetwork" = None,
    storage: "CentralActorStorage" = None,
    save_video: bool = False,
) -> float:
    save_path = os.path.join(config.exp_path, "recordings")

    ep_reward = 0
    for ep_idx in range(num_episodes):
        ep_reward += _test(
            config, render, network, storage, save_path, ep_idx, save_video
        )

    return ep_reward / num_episodes
