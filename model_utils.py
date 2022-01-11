from typing import Callable

import tensorflow as tf


def atari_scalar_transform(x: tf.Tensor, eps: float = 0.001) -> tf.Tensor:
    """
    Transform for values and rewards per Appendix F of Schrittwieser et al 2020.
    """
    return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1 + eps * x)


def inverse_atari_scalar_transform(x: tf.Tensor, eps: float = 0.001) -> tf.Tensor:
    """
    Corresponding inverse transform.
    """
    numerator = tf.math.sqrt(1 + 4 * eps * (tf.abs(x) + 1 + eps)) - 1
    denominator = 2 * eps
    return tf.math.sign(x) * ((numerator / denominator) ** 2 - 1)


def scalar_to_support_calc(
    x: tf.Tensor, scalar_transform: Callable, support: int, **kwargs
) -> tf.Tensor:
    """
    Transforms and then discretizes the scalars into bins.
    """
    min_int, max_int = -support, support
    x = tf.clip_by_value(scalar_transform(x, **kwargs), min_int, max_int)

    x_floor = tf.cast(tf.math.floor(x), dtype=tf.int32)
    x_ceil = tf.cast(tf.math.ceil(x), dtype=tf.int32)
    x_ceil_idx = tf.one_hot(x_ceil - min_int, 2 * support + 1)
    x_floor_idx = tf.one_hot(x_floor - min_int, 2 * support + 1)
    prop_ceil = tf.stack(
        [x - tf.cast(x_floor, dtype=tf.float32) for _ in range(2 * support + 1)], axis=2
    )
    return prop_ceil * x_ceil_idx + (1 - prop_ceil) * x_floor_idx


def support_to_scalar_calc(
    logits: tf.Tensor, inverse_scalar_transform: Callable, support: int, **kwargs
) -> tf.Tensor:
    """
    Corresponding inverse transform from bins to scalars.
    """
    min_int, max_int = -support, support
    probs = tf.nn.softmax(logits, axis=1)
    bins = tf.ones(probs.shape, dtype=tf.float32)
    bins = bins * tf.convert_to_tensor(
        [i for i in range(min_int, max_int + 1)], dtype=tf.float32
    )
    values = tf.math.reduce_sum(bins * probs, axis=1, keepdims=True)
    return inverse_scalar_transform(values, **kwargs)


def scale_gradient(x: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Scales a gradient for reverse autodiff, but will leave it unchanged for forward pass.
    """
    return x * scale + tf.stop_gradient(x) * (1 - scale)


def soft_network_params_update(
    target_network: 'MuZeroNetwork', source_network: 'MuZeroNetwork', tau: float
):
    param_zip = zip(
        target_network.trainable_variables, source_network.trainable_variables
    )
    for target_param, source_param in param_zip:
        target_param.assign(target_param * (1 - tau) + source_param * tau)
