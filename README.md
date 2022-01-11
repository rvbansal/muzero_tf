## MuZero

A TensorFlow implementation of DeepMind's MuZero algorithm for self-learning games without any knowledge of the rules. The algorithm is implemented as described in the [original paper](https://arxiv.org/abs/1911.08265) and [pseudocode](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py). It supports prioritized replay and is parallelized with the help of [Ray](https://github.com/ray-project/ray). The repo structure is based on a [muzero-pytorch](https://github.com/koulanurag/muzero-pytorch).

**Train**: ```python main.py --mode train --env CartPole-v1 --force```

**Test**: ```python main.py --mode test --env CartPole-v1 --force```

**TensorBoard**: ```tensorboard --logdir=result_dir```

At the moment, the code has only been tested for simple OpenAI gym environments like CartPole. Results are fairly sensitive to choices of hyperparameters.