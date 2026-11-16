# Build a Trainable CNN from Scratch in NumPy

A complete LeNet-5-style convolutional network implemented in pure NumPy — no autograd, no deep-learning framework in the library itself. Every layer, gradient, and optimizer is derived by hand.

## Architecture

```
Input (N, 1, 28, 28)
  │
  ├─ Conv2d(1→6, 5×5, stride=1, pad=0)  → (N, 6, 24, 24)
  ├─ ReLU
  ├─ MaxPool2d(2×2)                      → (N, 6, 12, 12)
  │
  ├─ Conv2d(6→16, 5×5, stride=1, pad=0) → (N, 16, 8, 8)
  ├─ ReLU
  ├─ MaxPool2d(2×2)                      → (N, 16, 4, 4)
  │
  ├─ Flatten                             → (N, 256)
  ├─ Linear(256→120)
  ├─ ReLU
  └─ Linear(120→num_classes)             → (N, C)   logits
```

Weights are He-initialised; trained with Adam.

## The im2col idea

Convolution over a batch is expensive if written as nested Python loops. The
`im2col` transform turns every receptive-field patch of the input into one row
of a 2D matrix. A single matrix-multiply (`cols @ W.T`) then computes all
output values at once, exploiting NumPy's BLAS back-end. `col2im` is the
adjoint operation used in the backward pass.

## Package layout

```
numpy_cnn/
  layers.py     – pad_2d, im2col/col2im, conv2d, maxpool2d, relu, flatten, linear
  loss.py       – stable_softmax, cross_entropy, accuracy, fused softmax-xent
  optimizer.py  – sgd_step, adam_step and helpers
  network.py    – He init, init_lenet, lenet_forward/backward/predict
  data.py       – build_synthetic_image_dataset, train_test_split, iterate_minibatches
  training.py   – train_step, train_one_epoch, train_loop, evaluate
tests/
  test_gradcheck.py       – central-difference gradient checks for every layer
  test_pytorch_parity.py  – forward + gradient parity against PyTorch (requires torch CPU)
  test_adam.py            – Adam matches torch.optim.Adam over 10 steps
  test_synthetic.py       – linearly-separable synthetic dataset reaches ≥95% accuracy
benchmarks/
  mnist_benchmark.py      – NumPy vs PyTorch on MNIST / Fashion-MNIST
```

## Install & run

```bash
# 1. activate your environment
source ~/myenv/bin/activate   # or your preferred venv

# 2. install dev dependencies (CPU torch required for parity tests & benchmarks)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pytest matplotlib

# 3. run the pure-NumPy tests (no torch required)
pytest tests/test_gradcheck.py tests/test_synthetic.py -v

# 4. run all tests (requires torch)
pytest tests/ -v

# 5. run benchmarks
python benchmarks/mnist_benchmark.py
```

## Correctness results

### Numerical gradient checks (central differences, float64)

All errors are relative errors computed as `|analytic − numeric| / (|analytic| + |numeric| + ε)`.

| Layer / quantity        | Max relative error |
|-------------------------|--------------------|
| conv2d  dx              | 4.40 × 10⁻⁹        |
| conv2d  dW              | 1.87 × 10⁻¹⁰       |
| conv2d  db              | 4.63 × 10⁻¹²       |
| maxpool2d  dx           | 3.91 × 10⁻¹⁰       |
| relu  dx                | 9.23 × 10⁻¹¹       |
| linear  dx              | 1.05 × 10⁻¹⁰       |
| linear  dW              | 1.39 × 10⁻¹¹       |
| softmax-xent  dlogits   | 3.01 × 10⁻¹⁰       |
| full LeNet  conv1 dW    | 1.53 × 10⁻⁸        |

All well within the 1 × 10⁻⁴ acceptance threshold.

### Synthetic dataset (linearly-separable, 3 classes, 28×28)

200 samples, 15 epochs, batch=16, Adam lr=1e-3.

| Split | Accuracy |
|-------|----------|
| Train | **100%** |
| Test  | **100%** |

### PyTorch parity (float64, same weights and inputs)

| Quantity | Max absolute difference |
|----------|------------------------|
| LeNet forward (logits) | **6.0 × 10⁻¹⁵** |
| conv1.W gradient | 1.1 × 10⁻¹⁵ |
| conv1.b gradient | 4.2 × 10⁻¹⁶ |
| conv2.W gradient | 6.7 × 10⁻¹⁶ |
| conv2.b gradient | 1.6 × 10⁻¹⁶ |
| fc1.W gradient | 4.4 × 10⁻¹⁶ |
| fc1.b gradient | 1.1 × 10⁻¹⁶ |
| fc2.W gradient | 1.7 × 10⁻¹⁵ |
| fc2.b gradient | 1.1 × 10⁻¹⁶ |

Errors are at float64 machine epsilon (~2.2 × 10⁻¹⁶) — the implementations are mathematically identical.

### Adam vs torch.optim.Adam

Max parameter diff after 10 steps: **< 1 × 10⁻¹⁰**. ✓

## Benchmark results

Training budget: **2 000 training samples, 5 epochs, batch=64, Adam lr=1e-3**.  
Full 10 000-sample test set used for evaluation. Same He initialisation for both.  
Hardware: CPU only (no GPU).

### MNIST

| Framework | Epoch 1 loss | Epoch 5 loss | Test acc (ep 5) | Total time |
|-----------|-------------|-------------|-----------------|------------|
| NumPy (float64) | 2.060 | **0.160** | **91.9%** | 108 s |
| PyTorch (float32) | 2.003 | 0.164 | 91.0% | 2.8 s |

**PyTorch is 39× faster.**

### Fashion-MNIST

| Framework | Epoch 1 loss | Epoch 5 loss | Test acc (ep 5) | Total time |
|-----------|-------------|-------------|-----------------|------------|
| NumPy (float64) | 1.470 | **0.470** | **77.8%** | 99 s |
| PyTorch (float32) | 1.520 | 0.482 | 77.4% | 3.0 s |

**PyTorch is 33× faster.**

Both frameworks converge to nearly identical accuracy, confirming the NumPy implementation is correct. The tiny NumPy accuracy edge is due to float64 vs float32 arithmetic, not a systematic difference.

Loss curves and accuracy plots are saved to `results/mnist_curves.png` and `results/fashionmnist_curves.png`.

## Why is NumPy slower than PyTorch?

| Reason | Detail |
|--------|--------|
| **No BLAS parallelism for the Python overhead** | PyTorch dispatches the full forward/backward pass to C++/CUDA in one call. Our NumPy code re-enters Python for every layer, every loop in `maxpool2d_backward`, etc. |
| **Maxpool backward loops** | `maxpool2d_backward` has four nested Python `for` loops over N, C, out_h, out_w. PyTorch's C++ kernel does this in a single pass. |
| **float64 vs float32** | NumPy tests run in float64 for numerical accuracy; PyTorch defaults to float32, which is 2× faster on most hardware and benefits from AVX-512 vectorisation. |
| **No GPU** | PyTorch can offload to CUDA with one `.to("cuda")` call; NumPy is CPU-only. |

The im2col trick eliminates the most expensive inner loops in `conv2d_forward`
and `conv2d_backward`, but the maxpool loops and Python-level dispatch are the
remaining bottleneck.

## Bugs fixed vs original `model.py`

| Location | Bug | Fix |
|----------|-----|-----|
| `iterate_minibatches` | `shuffle_indices(y, seed)` passed the label *array* instead of its length | `shuffle_indices(N, seed)` |
| `flatten_backward` | `N,C,W,H = cache['x_shape']` swapped H and W | `N,C,H,W = cache['x_shape']` |
| `he_init` | Used global `np.random.seed` (mutates global state) | Uses `np.random.default_rng(seed)` per call |

---

Built on Deep-ML.
