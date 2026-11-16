"""SGD and Adam optimizers."""

import numpy as np


def sgd_step(param, grad, lr):
    return param - lr * grad


def adam_update_m(m, grad, beta1):
    return beta1 * m + (1 - beta1) * grad


def adam_update_v(v, grad, beta2):
    return beta2 * v + (1 - beta2) * grad ** 2


def adam_bias_correct(moment, beta, t):
    return moment / (1 - beta ** t)


def adam_param_step(param, m_hat, v_hat, lr, eps):
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)


def adam_step(param, grad, m, v, t, lr, beta1, beta2, eps):
    m = adam_update_m(m, grad, beta1)
    v = adam_update_v(v, grad, beta2)
    m_hat = adam_bias_correct(m, beta1, t)
    v_hat = adam_bias_correct(v, beta2, t)
    return adam_param_step(param, m_hat, v_hat, lr, eps), m, v
