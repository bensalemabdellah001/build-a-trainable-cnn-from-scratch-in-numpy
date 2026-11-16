"""PyTorch parity tests: same weights, same input → same outputs and gradients."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from numpy_cnn import (
    lenet_forward, lenet_backward, softmax_cross_entropy_backward,
    softmax_cross_entropy_forward, init_lenet,
    conv2d_forward, conv2d_backward,
    linear_forward, linear_backward,
    relu_forward, relu_backward,
    maxpool2d_forward, maxpool2d_backward,
    flatten_forward, flatten_backward,
)

ATOL = 1e-5


def _t(arr):
    return torch.tensor(arr, dtype=torch.float64, requires_grad=False)


def _tc(arr):
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)


# ── individual layers ─────────────────────────────────────────────────────────

def test_conv2d_parity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 1, 10, 10))
    W = rng.standard_normal((4, 1, 3, 3)) * 0.1
    b = rng.standard_normal(4) * 0.1

    # NumPy forward
    y_np, cache = conv2d_forward(x, W, b, stride=1, padding=1)

    # PyTorch forward
    xt = _tc(x)
    Wt = _t(W)
    bt = _t(b)
    y_pt = nn.functional.conv2d(xt, Wt, bt, stride=1, padding=1)

    assert np.allclose(y_np, y_pt.detach().numpy(), atol=ATOL), \
        f"conv2d forward max_diff={np.max(np.abs(y_np - y_pt.detach().numpy())):.2e}"

    # backward
    d_out = rng.standard_normal(y_np.shape)
    dx_np, dW_np, db_np = conv2d_backward(d_out, cache)

    y_pt.backward(_t(d_out))
    dx_pt = xt.grad.numpy()

    assert np.allclose(dx_np, dx_pt, atol=ATOL), \
        f"conv2d dx max_diff={np.max(np.abs(dx_np - dx_pt)):.2e}"


def test_linear_parity():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 8))
    W = rng.standard_normal((8, 5)) * 0.1
    b = rng.standard_normal(5) * 0.1

    y_np, cache = linear_forward(x, W, b)

    xt = _tc(x)
    Wt = _t(W.T)   # torch linear uses (out, in)
    bt = _t(b)
    y_pt = nn.functional.linear(xt, Wt, bt)

    assert np.allclose(y_np, y_pt.detach().numpy(), atol=ATOL)

    d_out = rng.standard_normal(y_np.shape)
    dx_np, dW_np, db_np = linear_backward(d_out, cache)

    y_pt.backward(_t(d_out))
    assert np.allclose(dx_np, xt.grad.numpy(), atol=ATOL)


def test_relu_parity():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 8)) * 2

    y_np, cache = relu_forward(x)

    xt = _tc(x)
    y_pt = torch.relu(xt)

    assert np.allclose(y_np, y_pt.detach().numpy(), atol=ATOL)

    d_out = rng.standard_normal(y_np.shape)
    dx_np = relu_backward(d_out, cache)
    y_pt.backward(_t(d_out))
    assert np.allclose(dx_np, xt.grad.numpy(), atol=ATOL)


def test_maxpool_parity():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 2, 8, 8))

    y_np, cache = maxpool2d_forward(x, kernel=2, stride=2)

    xt = _tc(x)
    y_pt = nn.functional.max_pool2d(xt, kernel_size=2, stride=2)

    assert np.allclose(y_np, y_pt.detach().numpy(), atol=ATOL)

    d_out = rng.standard_normal(y_np.shape)
    dx_np = maxpool2d_backward(d_out, cache)
    y_pt.backward(_t(d_out))
    assert np.allclose(dx_np, xt.grad.numpy(), atol=ATOL)


# ── full LeNet ────────────────────────────────────────────────────────────────

class TorchLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(256, 120, bias=True)
        self.fc2 = nn.Linear(120, 10, bias=True)

    def forward(self, x):
        x = nn.functional.max_pool2d(torch.relu(self.conv1(x)), 2, 2)
        x = nn.functional.max_pool2d(torch.relu(self.conv2(x)), 2, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _copy_weights_to_torch(np_params, model):
    """Copy NumPy LeNet weights into a TorchLeNet."""
    with torch.no_grad():
        model.conv1.weight.copy_(torch.tensor(np_params["conv1"]["W"], dtype=torch.float64))
        model.conv1.bias.copy_(torch.tensor(np_params["conv1"]["b"], dtype=torch.float64))
        model.conv2.weight.copy_(torch.tensor(np_params["conv2"]["W"], dtype=torch.float64))
        model.conv2.bias.copy_(torch.tensor(np_params["conv2"]["b"], dtype=torch.float64))
        model.fc1.weight.copy_(torch.tensor(np_params["fc1"]["W"].T, dtype=torch.float64))
        model.fc1.bias.copy_(torch.tensor(np_params["fc1"]["b"], dtype=torch.float64))
        model.fc2.weight.copy_(torch.tensor(np_params["fc2"]["W"].T, dtype=torch.float64))
        model.fc2.bias.copy_(torch.tensor(np_params["fc2"]["b"], dtype=torch.float64))


def test_lenet_forward_parity():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 1, 28, 28)).astype(np.float64)
    params = init_lenet(1, 10, seed=0)

    logits_np, _ = lenet_forward(x, params)

    model = TorchLeNet().double()
    _copy_weights_to_torch(params, model)
    with torch.no_grad():
        logits_pt = model(_t(x)).numpy()

    max_diff = np.max(np.abs(logits_np - logits_pt))
    print(f"lenet forward max_diff={max_diff:.2e}")
    assert max_diff < ATOL, f"forward max_diff={max_diff:.2e}"


def test_lenet_grad_parity():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 1, 28, 28)).astype(np.float64)
    labels = np.array([0, 3, 7, 2])
    params = init_lenet(1, 10, seed=0)

    # NumPy grads
    logits_np, caches = lenet_forward(x, params)
    dlogits = softmax_cross_entropy_backward(logits_np, labels)
    grads_np = lenet_backward(dlogits, caches)

    # PyTorch grads
    model = TorchLeNet().double()
    _copy_weights_to_torch(params, model)
    xt = _tc(x)
    logits_pt = model(xt)
    loss_pt = nn.functional.cross_entropy(logits_pt, torch.tensor(labels))
    loss_pt.backward()

    pairs = [
        ("conv1.W", grads_np["conv1"]["dW"], model.conv1.weight.grad.numpy()),
        ("conv1.b", grads_np["conv1"]["db"], model.conv1.bias.grad.numpy()),
        ("conv2.W", grads_np["conv2"]["dW"], model.conv2.weight.grad.numpy()),
        ("conv2.b", grads_np["conv2"]["db"], model.conv2.bias.grad.numpy()),
        ("fc1.W",   grads_np["fc1"]["dW"].T, model.fc1.weight.grad.numpy()),
        ("fc1.b",   grads_np["fc1"]["db"],   model.fc1.bias.grad.numpy()),
        ("fc2.W",   grads_np["fc2"]["dW"].T, model.fc2.weight.grad.numpy()),
        ("fc2.b",   grads_np["fc2"]["db"],   model.fc2.bias.grad.numpy()),
    ]
    for name, gnp, gpt in pairs:
        d = np.max(np.abs(gnp - gpt))
        print(f"  grad {name:12s} max_diff={d:.2e}")
        assert d < ATOL, f"{name} grad max_diff={d:.2e}"
