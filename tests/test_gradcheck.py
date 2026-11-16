"""Numerical gradient checks (central differences) for every layer and the full network."""

import numpy as np
import pytest
from numpy_cnn import (
    conv2d_forward, conv2d_backward,
    maxpool2d_forward, maxpool2d_backward,
    relu_forward, relu_backward,
    linear_forward, linear_backward,
    softmax_cross_entropy_forward, softmax_cross_entropy_backward,
    lenet_forward, lenet_backward, init_lenet,
)


EPS = 1e-5
TOL = 1e-4   # max relative error we accept


def numerical_grad(f, x, eps=EPS):
    """Central-difference gradient of scalar f with respect to flat array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))


# ── conv2d ───────────────────────────────────────────────────────────────────

def test_conv2d_grad_input():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 1, 8, 8))
    W = rng.standard_normal((3, 1, 3, 3)) * 0.1
    b = np.zeros(3)
    _, cache = conv2d_forward(x, W, b, stride=1, padding=1)
    # upstream grad (all ones for simplicity)
    y, _ = conv2d_forward(x, W, b, stride=1, padding=1)
    d_out = rng.standard_normal(y.shape)

    dx_analytic, _, _ = conv2d_backward(d_out, cache)

    def f(xf):
        out, _ = conv2d_forward(xf, W, b, stride=1, padding=1)
        return (out * d_out).sum()

    dx_num = numerical_grad(f, x.copy())
    err = rel_error(dx_analytic, dx_num)
    print(f"conv2d dx  rel_err={err:.2e}")
    assert err < TOL


def test_conv2d_grad_weights():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 1, 8, 8))
    W = rng.standard_normal((3, 1, 3, 3)) * 0.1
    b = np.zeros(3)
    y, cache = conv2d_forward(x, W, b, stride=1, padding=1)
    d_out = rng.standard_normal(y.shape)

    _, dW_analytic, _ = conv2d_backward(d_out, cache)

    def f(wf):
        out, _ = conv2d_forward(x, wf, b, stride=1, padding=1)
        return (out * d_out).sum()

    dW_num = numerical_grad(f, W.copy())
    err = rel_error(dW_analytic, dW_num)
    print(f"conv2d dW  rel_err={err:.2e}")
    assert err < TOL


def test_conv2d_grad_bias():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 1, 8, 8))
    W = rng.standard_normal((3, 1, 3, 3)) * 0.1
    b = rng.standard_normal(3) * 0.1
    y, cache = conv2d_forward(x, W, b, stride=1, padding=1)
    d_out = rng.standard_normal(y.shape)

    _, _, db_analytic = conv2d_backward(d_out, cache)

    def f(bf):
        out, _ = conv2d_forward(x, W, bf, stride=1, padding=1)
        return (out * d_out).sum()

    db_num = numerical_grad(f, b.copy())
    err = rel_error(db_analytic, db_num)
    print(f"conv2d db  rel_err={err:.2e}")
    assert err < TOL


# ── maxpool2d ────────────────────────────────────────────────────────────────

def test_maxpool2d_backward():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 2, 8, 8))
    y, cache = maxpool2d_forward(x, kernel=2, stride=2)
    d_out = rng.standard_normal(y.shape)

    dx_analytic = maxpool2d_backward(d_out, cache)

    def f(xf):
        out, _ = maxpool2d_forward(xf, kernel=2, stride=2)
        return (out * d_out).sum()

    dx_num = numerical_grad(f, x.copy())
    err = rel_error(dx_analytic, dx_num)
    print(f"maxpool dx rel_err={err:.2e}")
    assert err < TOL


# ── relu ─────────────────────────────────────────────────────────────────────

def test_relu_backward():
    rng = np.random.default_rng(4)
    # avoid values near zero to prevent flat gradient at the kink
    x = rng.standard_normal((4, 8)) * 2 + 0.5
    y, cache = relu_forward(x)
    d_out = rng.standard_normal(y.shape)

    dx_analytic = relu_backward(d_out, cache)

    def f(xf):
        out, _ = relu_forward(xf)
        return (out * d_out).sum()

    dx_num = numerical_grad(f, x.copy())
    err = rel_error(dx_analytic, dx_num)
    print(f"relu   dx  rel_err={err:.2e}")
    assert err < TOL


# ── linear ───────────────────────────────────────────────────────────────────

def test_linear_grad_input():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 4))
    W = rng.standard_normal((4, 5)) * 0.1
    b = np.zeros(5)
    y, cache = linear_forward(x, W, b)
    d_out = rng.standard_normal(y.shape)

    dx_analytic, _, _ = linear_backward(d_out, cache)

    def f(xf):
        out, _ = linear_forward(xf, W, b)
        return (out * d_out).sum()

    dx_num = numerical_grad(f, x.copy())
    err = rel_error(dx_analytic, dx_num)
    print(f"linear dx  rel_err={err:.2e}")
    assert err < TOL


def test_linear_grad_weights():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((3, 4))
    W = rng.standard_normal((4, 5)) * 0.1
    b = np.zeros(5)
    y, cache = linear_forward(x, W, b)
    d_out = rng.standard_normal(y.shape)

    _, dW_analytic, _ = linear_backward(d_out, cache)

    def f(wf):
        out, _ = linear_forward(x, wf, b)
        return (out * d_out).sum()

    dW_num = numerical_grad(f, W.copy())
    err = rel_error(dW_analytic, dW_num)
    print(f"linear dW  rel_err={err:.2e}")
    assert err < TOL


# ── softmax-cross-entropy ────────────────────────────────────────────────────

def test_softmax_xent_backward():
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((4, 5))
    labels = np.array([0, 2, 4, 1])

    dlogits_analytic = softmax_cross_entropy_backward(logits, labels)

    def f(lf):
        return softmax_cross_entropy_forward(lf, labels)

    dlogits_num = numerical_grad(f, logits.copy())
    err = rel_error(dlogits_analytic, dlogits_num)
    print(f"xent   dL  rel_err={err:.2e}")
    assert err < TOL


# ── full network ─────────────────────────────────────────────────────────────

def test_lenet_full_gradcheck():
    """Central-difference check on the loss w.r.t. the input image batch."""
    rng = np.random.default_rng(42)
    # tiny spatial size to keep the check fast: 16×16 input
    x = rng.standard_normal((2, 1, 16, 16)) * 0.3
    # 16×16 → conv(5,s1,p0)→12 → pool(2)→6 → conv(5,s1,p0)→2 → pool(2)→1
    # 16×4*4 = 16 flat — but we use the standard 28×28 init; let's use a custom init
    # Actually let's just check gradients wrt conv1 weights (small)
    params = init_lenet(1, 3, seed=0)
    # Adjust fc1 for smaller spatial size: after two (conv5+pool2) blocks on 16×16:
    # 16 → (16-5+1)=12 → 12//2=6 → (6-5+1)=2 → 2//2=1; so 16*1*1=16 features
    # We'll use 28×28 (standard) so fc1 input is 256 as expected
    x = rng.standard_normal((2, 1, 28, 28)) * 0.3
    labels = np.array([0, 2])

    logits, caches = lenet_forward(x, params)
    dlogits = softmax_cross_entropy_backward(logits, labels)
    grads = lenet_backward(dlogits, caches)

    # Check dW for conv1 (small tensor: 6×1×5×5 = 150 elements)
    def f(wf):
        p2 = {k: dict(v) for k, v in params.items()}
        p2["conv1"] = {"W": wf, "b": params["conv1"]["b"]}
        log, _ = lenet_forward(x, p2)
        return softmax_cross_entropy_forward(log, labels)

    dW1_num = numerical_grad(f, params["conv1"]["W"].copy())
    err = rel_error(grads["conv1"]["dW"], dW1_num)
    print(f"lenet  dW1 rel_err={err:.2e}")
    assert err < TOL
