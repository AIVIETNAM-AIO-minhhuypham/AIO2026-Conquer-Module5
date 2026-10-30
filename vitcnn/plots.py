"""Plotting helpers. Nothing here runs during training, call them once results exist."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import data as D
from .analysis import load_runs

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLOR = {"vit_p4": C1, "resnet18_w05": C2, "vit_p8": C3}
MODEL_LABEL = {"vit_p4": "ViT (patch 4)", "vit_p8": "ViT (patch 8)", "resnet18_w05": "ResNet-18 (w=0.5)"}
AUG_STYLE = {"strong": "-", "basic": "--"}

STYLE = {
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
}


def _save(fig, out_dir, name, formats=("png", "pdf")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    return out_dir / f"{name}.{formats[0]}"


def plot_lr_sweep(result_dir="result", out_dir=None):
    runs = load_runs(result_dir, plan="lr_sweep")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
        for ax, model in zip(axes, ("vit_p4", "resnet18_w05")):
            rs = sorted([v for v in runs.values() if v["config"]["model"] == model],
                        key=lambda v: v["config"]["lr"])
            for color, r in zip((C1, C2, C3, C4), rs):
                lg = r["log"]
                ax.plot(lg["step"] / 1000, lg["val_acc"], color=color, label=f"lr {r['config']['lr']:g}")
            ax.set_title(MODEL_LABEL[model], fontsize=9.5, color=INK)
            ax.set_xlabel("update step (k)")
            ax.set_xlim(0, 10.3)
            ax.legend(loc="lower right", fontsize=8)
        axes[0].set_ylabel("validation accuracy")
        fig.tight_layout()
        return _save(fig, out_dir or Path(result_dir) / "figures", "lr_sweep")


def plot_curves(result_dir="result", plan="pilot10", out_dir=None):
    runs = load_runs(result_dir, plan=plan)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        for r in sorted(runs.values(), key=lambda v: (v["config"]["model"], v["config"]["aug"])):
            c, lg = r["config"], r["log"]
            kw = dict(color=MODEL_COLOR[c["model"]], linestyle=AUG_STYLE[c["aug"]],
                      label=f"{MODEL_LABEL[c['model']]}, {c['aug']} aug")
            axes[0].plot(lg["step"] / 1000, lg["val_acc"], **kw)
            axes[1].plot(lg["step"] / 1000, lg["val_loss"], **kw)
        axes[0].set_ylabel("validation accuracy")
        axes[1].set_ylabel("validation loss")
        for ax in axes:
            ax.set_xlabel("update step (k)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout()
        return _save(fig, out_dir or Path(result_dir) / "figures", f"curves_{plan}")


def plot_timing(result_dir="result", plan="pilot10", out_dir=None):
    """Actual time per eval interval divided by the time predicted from bench.json."""
    bench = json.loads((Path(result_dir) / "bench" / "bench.json").read_text(encoding="utf-8"))
    sps = {(r["model"], r["aug"]): r["steps_per_s"] for r in bench["results"]}
    runs = load_runs(result_dir, plan=plan)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        for r in sorted(runs.values(), key=lambda v: (v["config"]["model"], v["config"]["aug"])):
            c, lg = r["config"], r["log"]
            dt = lg["train_time_s"].diff().fillna(lg["train_time_s"].iloc[0])
            dsteps = lg["step"].diff().fillna(lg["step"].iloc[0])
            ratio = dt / dsteps * sps[(c["model"], c["aug"])]
            ax.plot(lg["step"] / 1000, ratio, color=MODEL_COLOR[c["model"]], linestyle=AUG_STYLE[c["aug"]],
                    marker="o", markersize=3, label=f"{MODEL_LABEL[c['model']]}, {c['aug']} aug")
        ax.axhline(1.0, color=MUTED, linewidth=1)
        ax.set_ylabel("actual / expected time")
        ax.set_xlabel("update step (k)")
        ax.set_ylim(0, None)
        ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
        fig.tight_layout()
        return _save(fig, out_dir or Path(result_dir) / "figures", f"timing_{plan}")


def _pick_examples(data_root, fake=False):
    x_tr, y_tr, _, _ = D.load_cifar10(data_root, fake=fake)
    pool, _ = D.stratified_train_val(y_tr)
    mean, std = D.channel_stats(x_tr[pool])
    rng = np.random.default_rng(0)
    pick = [int(rng.choice(pool[y_tr[pool] == c])) for c in range(10)]
    return x_tr, y_tr, pick, mean, std


def plot_aug_examples(data_root="data", result_dir="result", out_dir=None, fake=False):
    x_tr, y_tr, pick, mean, std = _pick_examples(data_root, fake)
    m, s = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    show = lambda t: (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()

    torch.manual_seed(0)
    basic = D.train_transform("basic", mean, std)
    strong = D.train_transform("strong", mean, std)
    raw = [torch.from_numpy(x_tr[i]) for i in pick]
    rows = {"original": [r.permute(1, 2, 0).numpy() for r in raw],
            "basic": [show(basic(r)) for r in raw],
            "RandAugment": [show(strong(r)) for r in raw]}
    xb = torch.stack([strong(r) for r in raw])
    mixed, _ = D.batch_mixer()(xb, torch.tensor(y_tr[pick]))
    rows["Mixup/CutMix"] = [show(t) for t in mixed]

    fig, axes = plt.subplots(len(rows), 10, figsize=(13, 5.6))
    for r, (title, imgs) in enumerate(rows.items()):
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(imgs[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(D.CLASSES[y_tr[pick[c]]], fontsize=9)
            if c == 0:
                ax.set_ylabel(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir or Path(result_dir) / "figures", "aug_examples", formats=("png",))


def plot_patch_grid(data_root="data", result_dir="result", out_dir=None, fake=False):
    x_tr, _, pick, _, _ = _pick_examples(data_root, fake)
    img = x_tr[pick[0]].transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))
    for ax, p in zip(axes, (4, 8)):
        ax.imshow(img, interpolation="nearest")
        for k in range(0, 33, p):
            ax.axhline(k - 0.5, color="white", lw=0.8)
            ax.axvline(k - 0.5, color="white", lw=0.8)
        ax.set_title(f"patch {p}x{p}: {(32 // p) ** 2} patches + 1 class token", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _save(fig, out_dir or Path(result_dir) / "figures", "patch_grid", formats=("png",))


# TODO: attention distance per layer (needs the main runs' checkpoints)
# TODO: confusion matrix grid for the main runs
