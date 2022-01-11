import logging
from typing import List, Tuple
from dataclasses import astuple, dataclass

import ray
from keras.optimizers import Optimizer
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from central_actor_storage import CentralActorStorage
from experience_actor import ExperienceActor
from muzero_config import MuZeroConfig
from muzero_network import MuZeroNetwork, NetworkOutput
from model_utils import scale_gradient, soft_network_params_update
from replay_memory import ReplayMemory
from test_runner import test
from ray_constants import FRAC_CPUS_PER_WORKER, FRAC_GPUS_PER_WORKER


train_logger = logging.getLogger("train")
train_test_logger = logging.getLogger("train_test")


def train(config: 'MuZeroConfig', summary_writer=None) -> 'MuZeroNetwork':
    replay_memory = ReplayMemory.remote(
        config.batch_size, config.window_size, config.priorities_alpha, config.weights_beta,
    )
    storage = CentralActorStorage.remote(config.get_init_network_params())

    actors = []
    actors += [run_test_evals.remote(config, storage)]
    for index in range(config.num_actors):
        actor = ExperienceActor.remote(index, config, storage, replay_memory)
        actors.append(actor.run.remote())

    run_training_steps(config, storage, replay_memory, summary_writer)
    ray.wait(actors, num_returns=len(actors))

    trained_network = config.get_init_network_obj(training=False)
    trained_network.set_weights(ray.get(storage.get_params.remote()))
    return trained_network


def run_training_steps(
    config: 'MuZeroConfig',
    storage: 'CentralActorStorage',
    replay_memory: 'ReplayMemory',
    summary_writer: tf.summary.SummaryWriter,
):
    network = config.get_init_network_obj(training=True)
    target_network = config.get_init_network_obj(training=False)
    optimizer = tfa.optimizers.SGDW(config.weight_decay, config.lr_init, config.momentum)

    while ray.get(replay_memory.size.remote()) == 0:
        pass

    for step in range(config.num_training_steps):
        storage.increment_total_training_steps.remote()
        step_log_data = StepLogData()
        step_log_data.step = step

        if step % config.checkpoint_interval:
            storage.set_params.remote(network.get_params())
        
        step_log_data.lr = adjust_lr(config, optimizer, step)
        step_log_data = update_network_params(
            network, target_network, optimizer, replay_memory, config, step_log_data
        )
        if config.use_target_model:
            soft_network_params_update(target_network, network, config.soft_update_tau)
        
        step_log_data.actors_log = ray.get(storage.get_actors_log.remote())
        step_log_data.test_score = ray.get(storage.get_test_scores_log.remote())
        step_log_data.num_games_collected = ray.get(replay_memory.num_games_collected.remote())
        step_log_data.replay_memory_size = ray.get(replay_memory.size.remote())
        step_log_data.replay_memory_priorities = ray.get(replay_memory.get_priorities.remote())

        log_data(step_log_data, summary_writer)

        if step % config.replay_memory_checkpoint == 0:
            replay_memory.remove_excess_games.remote()

    storage.set_params.remote(network.get_params())


def adjust_lr(config: 'MuZeroConfig', optimizer: Optimizer, step: int) -> float:
    lr = config.lr_init * config.lr_decay_rate ** (step / config.lr_decay_steps)
    lr = max(lr, config.lr_floor)
    optimizer.lr.assign(lr)
    return lr


def update_network_params(
    network: 'MuZeroNetwork',
    target_network: 'MuZeroNetwork',
    optimizer: Optimizer,
    replay_memory: 'ReplayMemory',
    config: 'MuZeroConfig',
    step_log_data: 'StepLogData'
):
    # Pull a batch
    batch_data = get_batch_from_replay_memory(network, target_network, replay_memory, config)
    target_value = network.scalar_to_support(batch_data.value, config.value_support)
    target_reward = network.scalar_to_support(batch_data.reward, config.reward_support)
    target_policy = batch_data.policy

    # Save targets and info on batch samples
    step_log_data.targets = [target_value, target_reward, target_policy]
    step_log_data.scalar_targets = [batch_data.value, batch_data.reward, batch_data.policy]
    step_log_data.batch_samples = [batch_data.weights, batch_data.indices]

    # Forward pass
    with tf.GradientTape() as tape:
        network_output = network.initial_inference(batch_data.obs, config.value_support)
        value, reward, policy_logits, hidden_state = astuple(network_output)
        
        unscaled_value = tf.squeeze(network.support_to_scalar(value, config.value_support), axis=-1)

        # Compute losses and store predictions
        value_loss = network.scalar_loss_func(value, target_value[:, 0])
        reward_loss = tf.zeros(config.batch_size)
        policy_loss = network.scalar_loss_func(value, target_value[:, 0])
        step_log_data = store_predictions(step_log_data, network, network_output, config, 0)

        for i in range(config.num_unroll_steps):
            network_output = network.recurrent_inference(
                hidden_state, batch_data.action[:, i], config.value_support, config.reward_support
            )
            value, reward, policy_logits, hidden_state = astuple(network_output)

            value_loss += network.scalar_loss_func(value, target_value[:, i + 1])
            reward_loss += network.scalar_loss_func(reward, target_reward[:, i])
            policy_loss += network.scalar_loss_func(policy_logits, target_policy[:, i + 1])
            hidden_state = scale_gradient(hidden_state, 0.5)
            step_log_data = store_predictions(step_log_data, network, network_output, config, i + 1)
    
        # Compute and save losses 
        total_loss = value_loss * config.value_loss_coef + policy_loss + reward_loss
        weighted_loss = tf.reduce_mean(batch_data.weights * total_loss)
        weighted_loss = scale_gradient(weighted_loss, float(1 / config.num_unroll_steps))

    step_log_data.losses = [
        float(tf.reduce_mean(total_loss)), 
        float(weighted_loss),
        float(tf.reduce_mean(value_loss)), 
        float(tf.reduce_mean(reward_loss)), 
        float(tf.reduce_mean(policy_loss))
    ]

    # Get gradients and update params
    grads = tape.gradient(weighted_loss, network.trainable_variables)
    grads = [tf.clip_by_norm(grad, config.max_grad_norm) for grad in grads]
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    step_log_data.grad_norms = [tf.norm(grad) for grad in grads]
    step_log_data.param_norms = [tf.norm(param) for param in network.get_params()]

    # Update priorities in replay memory per Appendix G of Schrittwieser et al 2020
    new_priorities = tf.abs(unscaled_value - batch_data.value[:, 0])
    replay_memory.set_priorities.remote(batch_data.indices, new_priorities.numpy().tolist())

    return step_log_data


def get_batch_from_replay_memory(
    network: 'MuZeroNetwork',
    target_network: 'MuZeroNetwork',
    replay_memory: 'ReplayMemory',
    config: 'MuZeroConfig'
) -> Tuple[tf.Tensor]:
    batch_network = target_network if config.use_target_model else network
    params = batch_network.get_params()
    batch = ray.get(
        replay_memory.get_batch.remote(config.num_unroll_steps, config.td_steps, params, config)
    )
    return batch


def store_predictions(
    step_log_data: 'StepLogData',
    network: 'MuZeroNetwork',
    network_output: 'NetworkOutput',
    config: 'MuZeroConfig',
    step: int
):
    value, reward, policy_logits, _ = astuple(network_output)

    predictions_value = network.support_to_scalar(value, config.value_support)
    predictions_policy = tf.nn.softmax(policy_logits, axis=1)

    if step == 0:
        step_log_data.predictions = [predictions_value, None, predictions_policy]
    else:
        stored_value, stored_reward, stored_policy = step_log_data.predictions
        predictions_reward = network.support_to_scalar(reward, config.reward_support)
        if stored_reward is None:
            add_reward = predictions_reward
        else:
            add_reward = tf.concat([stored_reward, predictions_reward], axis=0)
        step_log_data.predictions = [
            tf.concat([stored_value, predictions_value], axis=0),
            add_reward,
            tf.concat([stored_policy, predictions_policy], axis=0),
        ]
    return step_log_data


@ray.remote(num_gpus=FRAC_GPUS_PER_WORKER, num_cpus=FRAC_CPUS_PER_WORKER)
def run_test_evals(config: 'MuZeroConfig', storage: 'CentralActorStorage'):
    best_test_score = float('-inf')

    while ray.get(storage.total_training_steps.remote()) < config.num_training_steps:
        test_score = test(config, config.test_num_episodes, False, storage=storage)
        if test_score >= best_test_score:
            best_test_score = test_score
            test_network = config.get_init_network_obj(training=False)
            test_network.set_params(ray.get(storage.get_params.remote()))
            test_network.save_weights(config.network_path)

        storage.add_test_score.remote(test_score)


@dataclass
class StepLogData:
    step: int = None
    lr: float = None
    targets: List[tf.Tensor] = None
    scalar_targets: List[tf.Tensor] = None
    batch_samples: List[tf.Tensor] = None
    predictions: List[tf.Tensor] = None
    losses: List[float] = None
    grad_norms: List[float] = None
    param_norms: List[float] = None
    actors_log: List[float] = None
    test_score: float = None
    num_games_collected: int = None
    replay_memory_size: int = None
    replay_memory_priorities: List[float] = None


def log_data(step_log_data: 'StepLogData', summary_writer: tf.summary.SummaryWriter):
    key_metrics = [
        "Step: {:<10}",
        "Loss: {:<8.3f}",
        "Weighted Loss: {:<8.3f}",
        "Value Loss: {:<8.3f}",
        "Reward Loss: {:<8.3f}",
        "Policy Loss: {:<8.3f}",
        "# of Games Collected: {:<10d}",
        "Replay Memory Size: {:<10d}",
        "Lr: {:<8.3f}",
    ]
    msg_format = "".join(key_metrics)
    msg = msg_format.format(
        step_log_data.step,
        *step_log_data.losses,
        step_log_data.num_games_collected,
        step_log_data.replay_memory_size,
        step_log_data.lr
    )
    train_logger.info(msg)

    if step_log_data.test_score is not None:
        msg = "#{:<10} Test Score: {:<10}".format(step_log_data.step, step_log_data.test_score)
        train_test_logger.info(msg)

    if summary_writer is not None:
        with summary_writer.as_default(step=step_log_data.step):
            batch_weights, batch_indices = step_log_data.batch_samples
            tf.summary.histogram("replay_memory/batch_weights", batch_weights)
            tf.summary.histogram("replay_memory/batch_indices", batch_indices)
            tf.summary.histogram(
                "replay_memory/priorities", np.asarray(step_log_data.replay_memory_priorities)
            )

            scalar_value, scalar_reward, scalar_policy = step_log_data.scalar_targets
            tf.summary.histogram("train_data/scalar_value", tf.reshape(scalar_value, [-1]))
            tf.summary.histogram("train_data/scalar_reward", tf.reshape(scalar_reward, [-1]))
            tf.summary.histogram("train_data/scalar_policy", tf.reshape(scalar_policy, [-1]))

            target_value, target_reward, target_policy = step_log_data.targets
            tf.summary.histogram(
                "train_data/target_value", tf.unique(tf.reshape(target_value, [-1]))[0]
            )
            tf.summary.histogram(
                "train_data/target_reward", tf.unique(tf.reshape(target_reward, [-1]))[0]
            )
            tf.summary.histogram("train_data/target_policy", tf.reshape(target_policy, [-1]))

            pred_value, pred_reward, pred_policy = step_log_data.predictions
            tf.summary.histogram("train_data/pred_value", tf.reshape(pred_value, [-1]))
            tf.summary.histogram("train_data/pred_reward", tf.reshape(pred_reward, [-1]))
            tf.summary.histogram("train_data/pred_policy", tf.reshape(pred_policy, [-1]))

            loss, weighted_loss, value_loss, reward_loss, policy_loss = step_log_data.losses
            tf.summary.scalar("train_opt/loss", loss)
            tf.summary.scalar("train_opt/weighted_loss", weighted_loss)
            tf.summary.scalar("train_opt/policy_loss", policy_loss)
            tf.summary.scalar("train_opt/reward_loss", reward_loss)
            tf.summary.scalar("train_opt/value_loss", value_loss)

            tf.summary.scalar("train_opt/num_games_collected", step_log_data.num_games_collected)
            tf.summary.scalar("train_opt/replay_memory_size", step_log_data.replay_memory_size)
            tf.summary.scalar("train_opt/lr", step_log_data.lr)
            tf.summary.histogram("train_opt/grad_norms", np.asarray(step_log_data.grad_norms))
            tf.summary.histogram("train_opt/param_norms", np.asarray(step_log_data.param_norms))

            if step_log_data.test_score is not None:
                tf.summary.scalar("train_opt/test_score", step_log_data.test_score)
            
            reward, episode_lens, temperature, entropy = step_log_data.actors_log
            if reward is not None:
                tf.summary.scalar("actors/reward", reward)
                tf.summary.scalar("actors/episode_lens", episode_lens)
                tf.summary.scalar("actors/temperature", temperature)
                tf.summary.scalar("actors/entropy", entropy)

        tf.summary.flush(summary_writer)

