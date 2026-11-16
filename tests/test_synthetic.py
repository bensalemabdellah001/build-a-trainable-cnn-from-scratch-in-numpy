"""Train on the linearly-separable synthetic dataset and verify ~100% accuracy."""

import numpy as np
import pytest

from numpy_cnn import (
    init_lenet, train_loop, evaluate, build_synthetic_image_dataset, train_test_split,
)


def test_synthetic_convergence():
    x, y = build_synthetic_image_dataset(
        num_samples=200, num_classes=3, image_size=28, in_channels=1, seed=0
    )
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_fraction=0.2, seed=0)

    params = init_lenet(in_channels=1, num_classes=3, seed=0)
    params, _ = train_loop(
        params, x_train, y_train,
        num_epochs=15, batch_size=16, lr=1e-3, seed=0,
    )

    train_acc = evaluate(params, x_train, y_train)
    test_acc = evaluate(params, x_test, y_test)
    print(f"Synthetic train={train_acc:.3f}  test={test_acc:.3f}")
    assert train_acc >= 0.95, f"train_acc={train_acc:.3f}"
    assert test_acc >= 0.90, f"test_acc={test_acc:.3f}"
