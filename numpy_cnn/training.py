"""train_step, train_one_epoch, train_loop, evaluate."""

import numpy as np

from .network import lenet_forward, lenet_backward, lenet_predict
from .loss import softmax_cross_entropy_forward, softmax_cross_entropy_backward
from .optimizer import adam_step
from .data import iterate_minibatches


def train_step(params, opt_state, xb, yb, lr, beta1, beta2, eps, step):
    logits, cache = lenet_forward(xb, params)
    loss = softmax_cross_entropy_forward(logits, yb)
    dout = softmax_cross_entropy_backward(logits, yb)
    grads = lenet_backward(dout, cache)

    new_params, new_state = {}, {}
    for layer in params:
        new_params[layer], new_state[layer] = {}, {}
        for pname in params[layer]:
            grad = grads[layer]["d" + pname]
            state = opt_state[layer][pname]
            p, m, v = adam_step(
                params[layer][pname], grad,
                state["m"], state["v"],
                step, lr, beta1, beta2, eps,
            )
            new_params[layer][pname] = p
            new_state[layer][pname] = {"m": m, "v": v}
    return new_params, new_state, loss


def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta1, beta2, eps, step_counter, seed=0):
    losses = []
    for xb, yb in iterate_minibatches(x, y, batch_size, seed):
        step_counter += 1
        params, opt_state, loss = train_step(params, opt_state, xb, yb, lr, beta1, beta2, eps, step_counter)
        losses.append(loss)
    return params, opt_state, step_counter, losses


def train_loop(params, x_train, y_train, num_epochs, batch_size,
               lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, seed=0):
    opt_state = {
        layer: {pname: {"m": np.zeros_like(p), "v": np.zeros_like(p)}
                for pname, p in params[layer].items()}
        for layer in params
    }
    step_counter = 0
    loss_history = []
    for epoch in range(num_epochs):
        params, opt_state, step_counter, losses = train_one_epoch(
            params, opt_state, x_train, y_train,
            batch_size, lr, beta1, beta2, eps,
            step_counter, seed=seed + epoch,
        )
        loss_history.extend(losses)
    return params, loss_history


def evaluate(params, x, y):
    return float(np.mean(lenet_predict(x, params) == y))
