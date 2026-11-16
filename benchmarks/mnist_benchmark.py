"""
Benchmark: NumPy LeNet vs PyTorch LeNet on MNIST and Fashion-MNIST.

Training budget: 2000 training images, 5 epochs, batch_size=64.
Full test set used for evaluation.

Run with:
    /home/ensimag/myenv/bin/python benchmarks/mnist_benchmark.py
"""

import time
import json
import pathlib
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numpy_cnn import init_lenet, train_loop, evaluate


# ── hyper-parameters ───────────────────────────────────────────────────────
TRAIN_SUBSET = 2000   # number of training samples used (keeps NumPy runtime sane)
NUM_EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
SEED = 0


# ── download helpers ───────────────────────────────────────────────────────

def load_dataset(name):
    """Return (x_train, y_train, x_test, y_test) as float64 numpy arrays, NCHW."""
    cls = torchvision.datasets.MNIST if name == "MNIST" else torchvision.datasets.FashionMNIST
    transform = T.ToTensor()
    train_ds = cls(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_ds = cls(root=str(DATA_DIR), train=False, download=True, transform=transform)

    def to_numpy(ds, n=None):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
        x, y = next(iter(loader))
        x = x.numpy().astype(np.float64)  # (N,1,28,28)
        y = y.numpy().astype(np.int64)
        if n is not None:
            x, y = x[:n], y[:n]
        return x, y

    x_train, y_train = to_numpy(train_ds, TRAIN_SUBSET)
    x_test, y_test = to_numpy(test_ds)
    # normalise to zero mean unit variance (per training set)
    mu = x_train.mean()
    sigma = x_train.std() + 1e-8
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    return x_train, y_train, x_test, y_test


# ── PyTorch LeNet ──────────────────────────────────────────────────────────

class TorchLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = nn.functional.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = nn.functional.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc1(x)))


def copy_np_weights_to_torch(np_params, model):
    """Initialise TorchLeNet with the same He weights as the NumPy model."""
    with torch.no_grad():
        model.conv1.weight.copy_(torch.tensor(np_params["conv1"]["W"], dtype=torch.float32))
        model.conv1.bias.copy_(torch.tensor(np_params["conv1"]["b"], dtype=torch.float32))
        model.conv2.weight.copy_(torch.tensor(np_params["conv2"]["W"], dtype=torch.float32))
        model.conv2.bias.copy_(torch.tensor(np_params["conv2"]["b"], dtype=torch.float32))
        model.fc1.weight.copy_(torch.tensor(np_params["fc1"]["W"].T, dtype=torch.float32))
        model.fc1.bias.copy_(torch.tensor(np_params["fc1"]["b"], dtype=torch.float32))
        model.fc2.weight.copy_(torch.tensor(np_params["fc2"]["W"].T, dtype=torch.float32))
        model.fc2.bias.copy_(torch.tensor(np_params["fc2"]["b"], dtype=torch.float32))


def train_torch(model, x_train, y_train, x_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    xte = torch.tensor(x_test, dtype=torch.float32).to(device)
    yte = torch.tensor(y_test, dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(xt, yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         generator=torch.Generator().manual_seed(SEED))

    epoch_records = []
    t0 = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ep_loss = []
        ep_t0 = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        model.eval()
        with torch.no_grad():
            preds = model(xte).argmax(1)
            acc = (preds == yte).float().mean().item()
        ep_wall = time.perf_counter() - ep_t0
        rec = {"epoch": epoch, "train_loss": float(np.mean(ep_loss)),
               "test_acc": acc, "wall_sec": ep_wall}
        epoch_records.append(rec)
        print(f"  [PyTorch] epoch {epoch}/{NUM_EPOCHS}  loss={rec['train_loss']:.4f}"
              f"  acc={acc:.4f}  ({ep_wall:.1f}s)")
    total = time.perf_counter() - t0
    return epoch_records, total


def train_numpy(params, x_train, y_train, x_test, y_test):
    opt_state = {
        layer: {pname: {"m": np.zeros_like(p), "v": np.zeros_like(p)}
                for pname, p in params[layer].items()}
        for layer in params
    }
    from numpy_cnn import train_one_epoch

    epoch_records = []
    step = 0
    t0 = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        ep_t0 = time.perf_counter()
        params, opt_state, step, losses = train_one_epoch(
            params, opt_state, x_train, y_train,
            BATCH_SIZE, LR, 0.9, 0.999, 1e-8, step, seed=SEED + epoch,
        )
        acc = evaluate(params, x_test, y_test)
        ep_wall = time.perf_counter() - ep_t0
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "test_acc": acc, "wall_sec": ep_wall}
        epoch_records.append(rec)
        print(f"  [NumPy]   epoch {epoch}/{NUM_EPOCHS}  loss={rec['train_loss']:.4f}"
              f"  acc={acc:.4f}  ({ep_wall:.1f}s)")
    total = time.perf_counter() - t0
    return params, epoch_records, total


# ── plotting ───────────────────────────────────────────────────────────────

def plot_results(np_recs, pt_recs, dataset_name, out_path):
    epochs = [r["epoch"] for r in np_recs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{dataset_name}  (train subset={TRAIN_SUBSET}, epochs={NUM_EPOCHS})")

    ax1.plot(epochs, [r["train_loss"] for r in np_recs], label="NumPy")
    ax1.plot(epochs, [r["train_loss"] for r in pt_recs], label="PyTorch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.legend()
    ax1.set_title("Training loss")

    ax2.plot(epochs, [r["test_acc"] for r in np_recs], label="NumPy")
    ax2.plot(epochs, [r["test_acc"] for r in pt_recs], label="PyTorch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test accuracy")
    ax2.legend()
    ax2.set_title("Test accuracy")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved plot → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────

def run_dataset(name):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  |  train_subset={TRAIN_SUBSET}  epochs={NUM_EPOCHS}")
    print(f"{'='*60}")

    x_train, y_train, x_test, y_test = load_dataset(name)
    print(f"  Loaded: train={x_train.shape}  test={x_test.shape}")

    # shared initial weights
    params = init_lenet(in_channels=1, num_classes=10, seed=SEED)

    # PyTorch (float32, same init)
    torch_model = TorchLeNet(10)
    copy_np_weights_to_torch(params, torch_model)
    print("\n  --- PyTorch ---")
    pt_recs, pt_total = train_torch(torch_model, x_train, y_train, x_test, y_test)
    print(f"  PyTorch total: {pt_total:.1f}s")

    # NumPy (float64)
    print("\n  --- NumPy ---")
    params, np_recs, np_total = train_numpy(params, x_train, y_train, x_test, y_test)
    print(f"  NumPy total: {np_total:.1f}s")
    print(f"  Speedup PyTorch/NumPy: {np_total/pt_total:.1f}×")

    result = {
        "dataset": name,
        "train_subset": TRAIN_SUBSET,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "numpy": {"epochs": np_recs, "total_sec": np_total},
        "pytorch": {"epochs": pt_recs, "total_sec": pt_total},
        "speedup_pytorch_over_numpy": np_total / pt_total,
    }

    out_json = RESULTS_DIR / f"{name.lower()}_results.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved results → {out_json}")

    plot_results(np_recs, pt_recs, name, RESULTS_DIR / f"{name.lower()}_curves.png")
    return result


if __name__ == "__main__":
    all_results = {}
    for ds in ["MNIST", "FashionMNIST"]:
        all_results[ds] = run_dataset(ds)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ds, r in all_results.items():
        np_final = r["numpy"]["epochs"][-1]
        pt_final = r["pytorch"]["epochs"][-1]
        print(f"\n{ds}:")
        print(f"  NumPy   — loss={np_final['train_loss']:.4f}  acc={np_final['test_acc']:.4f}"
              f"  total={r['numpy']['total_sec']:.1f}s")
        print(f"  PyTorch — loss={pt_final['train_loss']:.4f}  acc={pt_final['test_acc']:.4f}"
              f"  total={r['pytorch']['total_sec']:.1f}s")
        print(f"  PyTorch is {r['speedup_pytorch_over_numpy']:.1f}× faster")
