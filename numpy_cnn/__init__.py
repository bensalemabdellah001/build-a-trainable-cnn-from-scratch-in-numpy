"""numpy_cnn – pure-NumPy LeNet-style CNN."""

from .layers import (
    output_spatial_size, pad_2d,
    im2col, col2im,
    conv2d_forward, conv2d_grad_input, conv2d_grad_weights, conv2d_grad_bias, conv2d_backward,
    maxpool2d_forward, scatter_grad_window, maxpool2d_backward,
    relu_forward, relu_backward,
    flatten_forward, flatten_backward,
    linear_forward, linear_grad_input, linear_grad_weights, linear_grad_bias, linear_backward,
)
from .loss import (
    argmax_rows, row_max, row_sum, exp_shifted, stable_softmax,
    one_hot, gather_true_class_probs, cross_entropy_loss, accuracy,
    softmax_cross_entropy_forward, softmax_cross_entropy_backward,
)
from .optimizer import (
    sgd_step,
    adam_update_m, adam_update_v, adam_bias_correct, adam_param_step, adam_step,
)
from .network import (
    he_std, he_init, init_zero_bias,
    init_conv_layer, init_linear_layer, init_lenet,
    forward_conv_block, forward_classifier_block,
    lenet_forward, lenet_backward, lenet_predict,
    backward_conv_block, backward_classifier_block,
)
from .data import (
    shuffle_indices, build_synthetic_image_dataset,
    train_test_split, iterate_minibatches,
)
from .training import train_step, train_one_epoch, train_loop, evaluate

__all__ = [
    "output_spatial_size", "pad_2d", "im2col", "col2im",
    "conv2d_forward", "conv2d_grad_input", "conv2d_grad_weights",
    "conv2d_grad_bias", "conv2d_backward",
    "maxpool2d_forward", "scatter_grad_window", "maxpool2d_backward",
    "relu_forward", "relu_backward",
    "flatten_forward", "flatten_backward",
    "linear_forward", "linear_grad_input", "linear_grad_weights",
    "linear_grad_bias", "linear_backward",
    "argmax_rows", "row_max", "row_sum", "exp_shifted", "stable_softmax",
    "one_hot", "gather_true_class_probs", "cross_entropy_loss", "accuracy",
    "softmax_cross_entropy_forward", "softmax_cross_entropy_backward",
    "sgd_step", "adam_update_m", "adam_update_v", "adam_bias_correct",
    "adam_param_step", "adam_step",
    "he_std", "he_init", "init_zero_bias",
    "init_conv_layer", "init_linear_layer", "init_lenet",
    "forward_conv_block", "forward_classifier_block",
    "lenet_forward", "lenet_backward", "lenet_predict",
    "backward_conv_block", "backward_classifier_block",
    "shuffle_indices", "build_synthetic_image_dataset",
    "train_test_split", "iterate_minibatches",
    "train_step", "train_one_epoch", "train_loop", "evaluate",
]
