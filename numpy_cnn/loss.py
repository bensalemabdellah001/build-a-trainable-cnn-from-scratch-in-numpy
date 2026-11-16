"""Numerically stable softmax utilities, cross-entropy loss, and accuracy."""

import numpy as np


def argmax_rows(matrix):
    return np.argmax(matrix, axis=1)


def row_max(matrix):
    return matrix.max(axis=1, keepdims=True)


def row_sum(matrix):
    return matrix.sum(axis=1, keepdims=True)


def exp_shifted(logits):
    return np.exp(logits - row_max(logits))


def stable_softmax(logits):
    e = exp_shifted(logits)
    return e / row_sum(e)


def one_hot(labels, num_classes):
    N = len(labels)
    matrix = np.zeros((N, num_classes))
    matrix[np.arange(N), labels] = 1.0
    return matrix


def gather_true_class_probs(probs, labels):
    return probs[np.arange(len(labels)), labels]


def cross_entropy_loss(probs, labels, eps=1e-12):
    true_probs = gather_true_class_probs(probs, labels)
    return -np.log(np.clip(true_probs, eps, 1.0)).mean()


def accuracy(logits_or_probs, labels):
    return float(np.mean(argmax_rows(logits_or_probs) == labels))


def softmax_cross_entropy_forward(logits, y):
    return cross_entropy_loss(stable_softmax(logits), y)


def softmax_cross_entropy_backward(logits, y):
    N, C = logits.shape
    return (stable_softmax(logits) - one_hot(y, C)) / N
