"""Synthetic dataset generation and data-loading utilities."""

import numpy as np


def shuffle_indices(n, seed=0):
    return np.random.RandomState(seed).permutation(n)


def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    gen = np.random.default_rng(seed)
    y = gen.integers(0, num_classes, size=num_samples)
    x = gen.standard_normal((num_samples, in_channels, image_size, image_size))
    shift = y - (num_classes - 1) / 2
    x = x + shift.reshape(-1, 1, 1, 1)
    return x, y


def train_test_split(x, y, test_fraction=0.2, seed=0):
    N = len(y)
    idx = shuffle_indices(N, seed)
    x, y = x[idx], y[idx]
    cut = int(N * test_fraction)
    return x[cut:], y[cut:], x[:cut], y[:cut]


def iterate_minibatches(x, y, batch_size, seed=0):
    N = len(y)
    idx = shuffle_indices(N, seed)
    x, y = x[idx], y[idx]
    for i in range(N // batch_size):
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
