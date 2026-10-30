import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

NUM_CLASSES = 10
VAL_SIZE = 5000
SPLIT_SEED = 2026
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(root="data", fake=False):
    """Returns x_train, y_train, x_test, y_test with x as uint8 NCHW."""
    if fake:  # random data, only for testing the pipeline without a download
        rng = np.random.default_rng(0)
        y_tr = np.repeat(np.arange(NUM_CLASSES), 5000)
        y_te = np.repeat(np.arange(NUM_CLASSES), 1000)
        base = rng.integers(0, 256, size=(NUM_CLASSES, 3, 1, 1))

        def make(y):
            noise = rng.integers(-60, 60, size=(len(y), 3, 32, 32))
            return np.clip(base[y] + noise, 0, 255).astype(np.uint8)
        return make(y_tr), y_tr, make(y_te), y_te

    from torchvision.datasets import CIFAR10
    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    x_tr = np.ascontiguousarray(tr.data.transpose(0, 3, 1, 2))
    x_te = np.ascontiguousarray(te.data.transpose(0, 3, 1, 2))
    return x_tr, np.array(tr.targets), x_te, np.array(te.targets)


def stratified_train_val(y, val_size=VAL_SIZE, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    per_class = val_size // NUM_CLASSES
    train_idx, val_idx = [], []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val_idx.append(idx[:per_class])
        train_idx.append(idx[per_class:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def nested_subsets(y, pool_idx, fractions, seed):
    # one permutation per class -> the 10% subset is always inside the 30% one, etc.
    rng = np.random.default_rng(10_000 + seed)
    perms = {}
    for c in range(NUM_CLASSES):
        idx = pool_idx[y[pool_idx] == c].copy()
        rng.shuffle(idx)
        perms[c] = idx
    out = {}
    for f in fractions:
        parts = [perms[c][: int(round(f * len(perms[c])))] for c in range(NUM_CLASSES)]
        out[f] = np.sort(np.concatenate(parts))
    return out


def save_split(path, train_idx, val_idx):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"split_seed": SPLIT_SEED, "val_size": VAL_SIZE,
                   "train_pool_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, fh)


def channel_stats(x_uint8):
    x = x_uint8.astype(np.float64) / 255.0
    return x.mean(axis=(0, 2, 3)).tolist(), x.std(axis=(0, 2, 3)).tolist()


def train_transform(aug, mean, std):
    ops = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
    if aug == "strong":
        ops.append(v2.RandAugment(num_ops=2, magnitude=9))
    elif aug != "basic":
        raise ValueError(f"unknown aug: {aug}")
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)


def batch_mixer():
    # strong aug only: Mixup or CutMix per batch, gives soft labels
    return v2.RandomChoice([v2.MixUp(alpha=0.8, num_classes=NUM_CLASSES),
                            v2.CutMix(alpha=1.0, num_classes=NUM_CLASSES)])


class TensorImageDataset(Dataset):
    def __init__(self, x_uint8, y, transform):
        self.x = torch.from_numpy(np.ascontiguousarray(x_uint8))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]


def to_eval_tensor(x_uint8, mean, std, device):
    x = torch.from_numpy(x_uint8).to(device=device, dtype=torch.float32) / 255.0
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


def default_num_workers():
    return max(0, min(8, (os.cpu_count() or 2) - 1))
