import csv
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from . import data as D
from .models import build_model, count_params, param_groups


@dataclass
class RunConfig:
    model: str
    aug: str              # basic | strong
    data_frac: float      # fraction of the 45k train pool
    seed: int             # controls the subset and the init
    lr: float
    total_steps: int      # fixed number of updates, independent of data size
    plan: str = "adhoc"
    weight_decay: float = 0.05
    batch_size: int = 128
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    n_evals: int = 20
    eval_test: bool = True
    num_workers: int = -1
    fake_data: bool = False
    compile: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def run_id(self):
        return (f"{self.plan}__{self.model}__{self.aug}__f{int(round(self.data_frac * 100)):03d}"
                f"__s{self.seed}__lr{self.lr:g}__st{self.total_steps}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info(device):
    import torchvision
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "platform": platform.platform(),
            "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2)
    return info


def amp_setup(device):
    # bf16 on Ampere/Ada, fp16 + scaler otherwise (e.g. T4)
    if device.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


class Context:
    """Data shared by all runs in one process."""

    def __init__(self, data_root, out_root, fake, device):
        self.device = device
        self.out_root = Path(out_root)
        self.fake = fake
        self.x_tr, self.y_tr, self.x_te, self.y_te = D.load_cifar10(data_root, fake=fake)
        self.pool_idx, self.val_idx = D.stratified_train_val(self.y_tr)
        D.save_split(self.out_root / "splits" / "train_val_split.json", self.pool_idx, self.val_idx)
        # normalisation stats from the train pool only
        self.mean, self.std = D.channel_stats(self.x_tr[self.pool_idx])
        self.x_val = D.to_eval_tensor(self.x_tr[self.val_idx], self.mean, self.std, device)
        self.y_val = torch.from_numpy(self.y_tr[self.val_idx]).to(device)
        self.x_test = D.to_eval_tensor(self.x_te, self.mean, self.std, device)
        self.y_test = torch.from_numpy(self.y_te).to(device)

    def subset(self, frac, seed):
        subs = D.nested_subsets(self.y_tr, self.pool_idx, (0.1, 0.3, 1.0), seed)
        if frac not in subs:
            subs = D.nested_subsets(self.y_tr, self.pool_idx, (frac,), seed)
        return subs[frac]


@torch.no_grad()
def evaluate(model, x, y, amp_dtype, bs=1000):
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(y), bs):
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x[i:i + bs])
        total_loss += F.cross_entropy(logits.float(), y[i:i + bs], reduction="sum").item()
        preds.append(logits.argmax(1))
    preds = torch.cat(preds)
    return total_loss / len(y), (preds == y).float().mean().item(), preds


@torch.no_grad()
def inference_speed(model, x, amp_dtype, bs=500, warmup=3, reps=3):
    model.eval()
    dev = x.device
    ctx = lambda: torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    for _ in range(warmup):
        with ctx():
            model(x[:bs])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0, n = time.perf_counter(), 0
    for _ in range(reps):
        for i in range(0, len(x), bs):
            with ctx():
                model(x[i:i + bs])
            n += min(bs, len(x) - i)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def lr_lambda(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


def train_run(cfg, ctx, verbose=True):
    run_dir = ctx.out_root / "runs" / cfg.run_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        if verbose:
            print(f"[skip] {cfg.run_id} already done")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    device = ctx.device
    amp_dtype, scaler = amp_setup(device)

    sub_idx = ctx.subset(cfg.data_frac, cfg.seed)
    np.save(run_dir / "train_subset_idx.npy", sub_idx)
    ds = D.TensorImageDataset(ctx.x_tr[sub_idx], ctx.y_tr[sub_idx],
                              D.train_transform(cfg.aug, ctx.mean, ctx.std))
    mixer = D.batch_mixer() if cfg.aug == "strong" else None
    workers = D.default_num_workers() if cfg.num_workers < 0 else cfg.num_workers

    set_seed(cfg.seed)
    model = build_model(cfg.model).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    warmup = max(1, int(cfg.warmup_frac * cfg.total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg.total_steps, warmup))
    fwd = torch.compile(model) if cfg.compile else model

    eval_every = max(1, cfg.total_steps // cfg.n_evals)
    state = {"step": 0, "best_val_acc": -1.0, "best_step": -1, "train_time_s": 0.0,
             "log": [], "resumed_at": []}
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists():  # resume an interrupted run
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        state["resumed_at"].append(state["step"])
        if verbose:
            print(f"[resume] {cfg.run_id} from step {state['step']}")

    remaining = cfg.total_steps - state["step"]
    loader = []
    if remaining > 0:
        g = torch.Generator().manual_seed(cfg.seed * 1_000_003 + state["step"])
        # sample with repetition across epochs so that one loader covers all remaining steps
        sampler = RandomSampler(ds, num_samples=remaining * cfg.batch_size, generator=g)
        loader = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True,
                            num_workers=workers, pin_memory=device.type == "cuda",
                            persistent_workers=workers > 0,
                            prefetch_factor=4 if workers > 0 else None)

    if verbose:
        print(f"[run] {cfg.run_id} | {len(sub_idx)} train imgs | {n_params:,} params | "
              f"{cfg.total_steps} steps | amp={amp_dtype} | workers={workers}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_loss, run_n, run_correct, run_seen = 0.0, 0, 0, 0
    model.train()
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        target = yb
        if mixer is not None:
            xb, target = mixer(xb, yb)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = fwd(xb)
            loss = F.cross_entropy(logits.float(), target)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
        state["step"] += 1

        # .item() syncs the GPU, so only sample the train loss every 50 steps
        if state["step"] % 50 == 0 or state["step"] % eval_every == 0:
            run_loss += loss.item()
            run_n += 1
            if mixer is None:
                run_correct += (logits.argmax(1) == yb).sum().item()
                run_seen += len(yb)

        if state["step"] % eval_every == 0 or state["step"] == cfg.total_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            state["train_time_s"] += time.perf_counter() - t0
            val_loss, val_acc, _ = evaluate(model, ctx.x_val, ctx.y_val, amp_dtype)
            row = {"step": state["step"],
                   "epoch": round(state["step"] * cfg.batch_size / len(sub_idx), 2),
                   "lr": sched.get_last_lr()[0],
                   "train_loss": run_loss / max(1, run_n),
                   "train_acc": (run_correct / run_seen) if run_seen else float("nan"),
                   "val_loss": val_loss, "val_acc": val_acc,
                   "train_time_s": round(state["train_time_s"], 2)}
            state["log"].append(row)
            if val_acc > state["best_val_acc"]:  # ties keep the earlier step
                state["best_val_acc"], state["best_step"] = val_acc, state["step"]
                torch.save(model.state_dict(), best_path)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "state": state}, last_path)
            if verbose:
                print(f"  step {row['step']:>6} | ep {row['epoch']:>7} | train_loss {row['train_loss']:.4f} "
                      f"| val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | {row['train_time_s']:.0f}s")
            run_loss, run_n, run_correct, run_seen = 0.0, 0, 0, 0
            model.train()
            t0 = time.perf_counter()

    with open(run_dir / "log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(state["log"][0].keys()))
        w.writeheader()
        w.writerows(state["log"])

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, best_val_acc, _ = evaluate(model, ctx.x_val, ctx.y_val, amp_dtype)
    result = {"run_id": cfg.run_id, "config": asdict(cfg), "n_params": n_params,
              "n_train": int(len(sub_idx)), "best_step": state["best_step"],
              "val_acc": best_val_acc, "final_val_acc": state["log"][-1]["val_acc"],
              "train_time_s": state["train_time_s"],
              "steps_per_s": cfg.total_steps / max(1e-9, state["train_time_s"]),
              "resumed_at": state["resumed_at"], "env": env_info(device),
              "fake_data": ctx.fake, "mean": ctx.mean, "std": ctx.std}
    if device.type == "cuda":
        result["peak_mem_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)

    if cfg.eval_test:
        from sklearn.metrics import confusion_matrix, f1_score
        test_loss, test_acc, preds = evaluate(model, ctx.x_test, ctx.y_test, amp_dtype)
        p, t = preds.cpu().numpy(), ctx.y_test.cpu().numpy()
        result.update({"test_loss": test_loss, "test_acc": test_acc,
                       "test_macro_f1": float(f1_score(t, p, average="macro")),
                       "confusion": confusion_matrix(t, p, labels=list(range(10))).tolist(),
                       "infer_img_per_s": inference_speed(model, ctx.x_test, amp_dtype)})
        np.save(run_dir / "test_preds.npy", p.astype(np.int8))  # kept for McNemar

    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    last_path.unlink(missing_ok=True)
    if verbose:
        msg = f"[done] {cfg.run_id} | best val_acc {best_val_acc:.4f} @ step {state['best_step']}"
        if cfg.eval_test:
            msg += f" | test_acc {result['test_acc']:.4f} | macro_f1 {result['test_macro_f1']:.4f}"
        print(msg)
    return result
