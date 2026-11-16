"""Verify that our Adam matches torch.optim.Adam over several steps."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from numpy_cnn import adam_step


def test_adam_matches_torch():
    rng = np.random.default_rng(0)
    # Use a simple 2-D parameter matrix; simulate 10 gradient steps
    param_np = rng.standard_normal((4, 4))
    param_pt = torch.tensor(param_np.copy(), dtype=torch.float64, requires_grad=True)

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    opt = torch.optim.Adam([param_pt], lr=lr, betas=(beta1, beta2), eps=eps)

    m = np.zeros_like(param_np)
    v = np.zeros_like(param_np)

    for t in range(1, 11):
        grad = rng.standard_normal(param_np.shape)

        # NumPy Adam
        param_np, m, v = adam_step(param_np, grad, m, v, t, lr, beta1, beta2, eps)

        # PyTorch Adam (inject the same gradient)
        opt.zero_grad()
        param_pt.grad = torch.tensor(grad, dtype=torch.float64)
        opt.step()

        diff = np.max(np.abs(param_np - param_pt.detach().numpy()))
        assert diff < 1e-10, f"step {t}: Adam diff={diff:.2e}"

    print("Adam matches PyTorch over 10 steps.")
