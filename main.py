import argparse
import logging
import os

import ray
import numpy as np
import tensorflow as tf

from env_config_mapping import ENV_CONFIG_MAPPING
from logging_utils import init_logger, make_results_dir
from ray_constants import  NUM_GPUS, NUM_CPUS

from test_runner import test
from train_runner import train


ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPUS)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuZero TensorFlow Implementation")
    parser.add_argument(
        "--env", required=True, help="Name of environment"
    )
    parser.add_argument(
        "--result_dir", 
        default=os.path.join(os.getcwd(), "results"),
        help="Directory to store results (default: %(default)s)",
    )
    parser.add_argument(
        "--mode", required=True, choices=["train", "test"]
    )
    parser.add_argument(
        "--no_gpu", 
        action="store_true", 
        default=False, 
        help="No GPU usage (default: %(default)s)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Renders the environment (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overrides past results (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed (default: %(default)s)"
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=None,
        help="Scale for value loss (default: %(default)s)",
    )
    parser.add_argument(
        "--revisit_policy_search_rate",
        type=float,
        default=None,
        help="Rate at which target policy is re-estimated (default: %(default)s)",
    )
    parser.add_argument(
        "--use_target_model",
        action="store_true",
        default=False,
        help="Use target model for value estimation (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode (default: %(default)s)",
    )
    parser.add_argument(
        "--use_max_priority",
        action="store_true",
        default=False,
        help="Forces max priority assignment for new incoming data in replay buffer(default: %(default)s)",
    )
    parser.add_argument(
        "--use_priority",
        action="store_true",
        default=False,
        help="Uses priority for data sampling in replay buffer",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=10,
        help="Evaluation episode count (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.no_gpu:
        tf.config.set_visible_devices([], "GPU")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    muzero_config = ENV_CONFIG_MAPPING[args.env]
    exp_path = muzero_config.set_config(args)

    exp_path, log_path = make_results_dir(exp_path, args)
    init_logger(log_path)

    try:
        if args.mode == "train":
            summary_writer = tf.summary.create_file_writer(exp_path)
            train(muzero_config, summary_writer)
        elif args.mode == "test":
            path = muzero_config.network_path
            assert os.path.exists(path), "Network not found in {}".format(path)
            network = muzero_config.get_init_network_obj(training=False)
            network.built = True
            network.load_weights(path)
            test_score = test(
                muzero_config, args.test_episodes, args.render, network=network
            )
            logging.getLogger("test").info("Test Score: {}".format(test_score))
        ray.shutdown()
    except Exception as e:
        logging.getLogger("root").error(e, exc_info=True)
