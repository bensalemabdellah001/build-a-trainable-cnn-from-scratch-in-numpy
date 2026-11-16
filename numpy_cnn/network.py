"""LeNet-style network: init, forward, backward, predict."""

import numpy as np

from .layers import (
    conv2d_forward, conv2d_backward,
    maxpool2d_forward, maxpool2d_backward,
    relu_forward, relu_backward,
    flatten_forward, flatten_backward,
    linear_forward, linear_backward,
)
from .loss import argmax_rows


# ── weight initialisation ─────────────────────────────────────────────────────

def he_std(fan_in):
    return np.sqrt(2.0 / fan_in)


def he_init(shape, fan_in, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) * he_std(fan_in)


def init_zero_bias(length):
    return np.zeros(length)


def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    fan_in = in_channels * kernel_size ** 2
    W = he_init((out_channels, in_channels, kernel_size, kernel_size), fan_in, seed)
    return {"W": W, "b": init_zero_bias(out_channels)}


def init_linear_layer(in_features, out_features, seed=0):
    W = he_init((in_features, out_features), in_features, seed)
    return {"W": W, "b": init_zero_bias(out_features)}


def init_lenet(in_channels, num_classes, seed=0):
    """LeNet-5 style: conv1(6) → conv2(16) → fc1(120) → fc2(num_classes).

    Input assumed 28×28; after two conv(5,pad=0,stride=1)+pool(2) passes the
    spatial size is ((28-4)//2 - 4)//2 = 4, giving 16*4*4 = 256 flat features.
    """
    return {
        "conv1": init_conv_layer(6, in_channels, 5, seed),
        "conv2": init_conv_layer(16, 6, 5, seed + 1),
        "fc1": init_linear_layer(256, 120, seed + 2),
        "fc2": init_linear_layer(120, num_classes, seed + 3),
    }


# ── forward blocks ────────────────────────────────────────────────────────────

def forward_conv_block(x, W, b, pool_size, stride, pad):
    y, conv_cache = conv2d_forward(x, W, b, stride, pad)
    z, relu_cache = relu_forward(y)
    out, pool_cache = maxpool2d_forward(z, pool_size, pool_size)
    return out, {"conv_cache": conv_cache, "relu_cache": relu_cache, "pool_cache": pool_cache}


def forward_classifier_block(x, fc1, fc2):
    fx, flatten_cache = flatten_forward(x)
    f1, fc1_cache = linear_forward(fx, fc1["W"], fc1["b"])
    r1, relu_cache = relu_forward(f1)
    out, fc2_cache = linear_forward(r1, fc2["W"], fc2["b"])
    return out, {
        "flatten_cache": flatten_cache,
        "fc1_cache": fc1_cache,
        "relu_cache": relu_cache,
        "fc2_cache": fc2_cache,
    }


def lenet_forward(x, params):
    x1, c1 = forward_conv_block(x, params["conv1"]["W"], params["conv1"]["b"], 2, 1, 0)
    x2, c2 = forward_conv_block(x1, params["conv2"]["W"], params["conv2"]["b"], 2, 1, 0)
    out, cls = forward_classifier_block(x2, params["fc1"], params["fc2"])
    return out, {"block1": c1, "block2": c2, "classifier": cls}


# ── backward blocks ───────────────────────────────────────────────────────────

def backward_conv_block(dout, cache):
    dp = maxpool2d_backward(dout, cache["pool_cache"])
    dr = relu_backward(dp, cache["relu_cache"])
    return conv2d_backward(dr, cache["conv_cache"])  # (dx, dW, db)


def backward_classifier_block(dlogits, cache):
    fc2_dx, dW2, db2 = linear_backward(dlogits, cache["fc2_cache"])
    dr = relu_backward(fc2_dx, cache["relu_cache"])
    fc1_dx, dW1, db1 = linear_backward(dr, cache["fc1_cache"])
    dx = flatten_backward(fc1_dx, cache["flatten_cache"])
    return {"dx": dx, "fc1": {"dW": dW1, "db": db1}, "fc2": {"dW": dW2, "db": db2}}


def lenet_backward(dlogits, caches):
    cls = backward_classifier_block(dlogits, caches["classifier"])
    dx2, dW2, db2 = backward_conv_block(cls["dx"], caches["block2"])
    _,   dW1, db1 = backward_conv_block(dx2,       caches["block1"])
    return {
        "conv1": {"dW": dW1, "db": db1},
        "conv2": {"dW": dW2, "db": db2},
        "fc1": cls["fc1"],
        "fc2": cls["fc2"],
    }


def lenet_predict(x, params):
    logits, _ = lenet_forward(x, params)
    return argmax_rows(logits)
