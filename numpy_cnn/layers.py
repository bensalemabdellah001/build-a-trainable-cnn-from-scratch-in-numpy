"""Primitive layer operations: pad, im2col/col2im, conv2d, maxpool, relu, flatten, linear."""

import numpy as np


# ── spatial helpers ──────────────────────────────────────────────────────────

def output_spatial_size(input_size, kernel, stride, padding):
    return (input_size + 2 * padding - kernel) // stride + 1


def pad_2d(images, pad):
    N, C, H, W = images.shape
    out = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=images.dtype)
    out[:, :, pad:H + pad, pad:W + pad] = images
    return out


# ── im2col / col2im ───────────────────────────────────────────────────────────

def im2col(images, kernel_h, kernel_w, stride, padding):
    N, C, H, W = images.shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    x = pad_2d(images, padding)
    cols = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=x.dtype)
    for a in range(kernel_h):
        for b in range(kernel_w):
            cols[:, :, a, b] = x[:, :, a:a + out_h * stride:stride, b:b + out_w * stride:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = input_shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    x = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)
    c6 = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    for a in range(kernel_h):
        for b in range(kernel_w):
            x[:, :, a:a + stride * out_h:stride, b:b + stride * out_w:stride] += c6[:, :, a, b]
    return x[:, :, padding:padding + H, padding:padding + W]


# ── conv2d ───────────────────────────────────────────────────────────────────

def conv2d_forward(x, weights, bias, stride, padding):
    F, _, kh, kw = weights.shape
    N, C, H, W = x.shape
    out_h = output_spatial_size(H, kh, stride, padding)
    out_w = output_spatial_size(W, kw, stride, padding)
    cols = im2col(x, kh, kw, stride, padding)
    w_mat = weights.reshape(F, -1)
    y = cols @ w_mat.T + bias
    y = y.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
    cache = {
        "x_shape": x.shape,
        "weights": weights,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": kh,
        "kernel_w": kw,
    }
    return y, cache


def conv2d_grad_input(d_out, cache):
    weights = cache["weights"]
    x_shape = cache["x_shape"]
    padding = cache["padding"]
    stride = cache["stride"]
    kh = cache["kernel_h"]
    kw = cache["kernel_w"]
    F = weights.shape[0]
    d_mat = d_out.transpose(0, 2, 3, 1).reshape(-1, F)
    dcols = d_mat @ weights.reshape(F, -1)
    return col2im(dcols, x_shape, kh, kw, stride, padding)


def conv2d_grad_weights(d_out, cache):
    weights = cache["weights"]
    cols = cache["cols"]
    F = weights.shape[0]
    d_mat = d_out.transpose(0, 2, 3, 1).reshape(-1, F)
    return (d_mat.T @ cols).reshape(weights.shape)


def conv2d_grad_bias(d_out):
    return d_out.sum(axis=(0, 2, 3))


def conv2d_backward(d_out, cache):
    return conv2d_grad_input(d_out, cache), conv2d_grad_weights(d_out, cache), conv2d_grad_bias(d_out)


# ── maxpool2d ────────────────────────────────────────────────────────────────

def maxpool2d_forward(x, kernel, stride):
    N, C, H, W = x.shape
    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=int)
    for i in range(out_h):
        for j in range(out_w):
            win = x[:, :, i * stride:i * stride + kernel, j * stride:j * stride + kernel].reshape(N, C, -1)
            out[:, :, i, j] = win.max(axis=2)
            argmax[:, :, i, j] = win.argmax(axis=2)
    return out, {"x_shape": x.shape, "argmax": argmax, "kernel": kernel, "stride": stride}


def scatter_grad_window(grad_value, argmax_index, kernel):
    block = np.zeros((kernel, kernel))
    block[argmax_index // kernel, argmax_index % kernel] = grad_value
    return block


def maxpool2d_backward(d_out, cache):
    N, C, H, W = cache["x_shape"]
    stride = cache["stride"]
    kernel = cache["kernel"]
    argmax = cache["argmax"]
    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)
    grad_res = np.zeros((N, C, H, W))
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    grad_res[n, c, i * stride:i * stride + kernel, j * stride:j * stride + kernel] += \
                        scatter_grad_window(d_out[n, c, i, j], argmax[n, c, i, j], kernel)
    return grad_res


# ── relu ─────────────────────────────────────────────────────────────────────

def relu_forward(x):
    return np.maximum(0, x), {"x": x}


def relu_backward(d_out, cache):
    return d_out * (cache["x"] > 0)


# ── flatten ──────────────────────────────────────────────────────────────────

def flatten_forward(x):
    N, C, H, W = x.shape
    return x.reshape(N, -1), {"x_shape": x.shape}


def flatten_backward(d_out, cache):
    return d_out.reshape(cache["x_shape"])


# ── linear ───────────────────────────────────────────────────────────────────

def linear_forward(x, weights, bias):
    return x @ weights + bias, {"x": x, "weights": weights}


def linear_grad_input(d_out, cache):
    return d_out @ cache["weights"].T


def linear_grad_weights(x, dout):
    return x.T @ dout


def linear_grad_bias(dout):
    return dout.sum(axis=0)


def linear_backward(dout, cache):
    x = cache["x"]
    return linear_grad_input(dout, cache), linear_grad_weights(x, dout), linear_grad_bias(dout)
