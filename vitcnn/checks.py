import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .engine import amp_setup, set_seed
from .models import MODEL_NAMES, build_model, count_params


def check_splits(ctx):
    y, pool, val = ctx.y_tr, ctx.pool_idx, ctx.val_idx
    res = {"n_train_pool": int(len(pool)), "n_val": int(len(val)), "n_test": int(len(ctx.y_te)),
           "pool_val_disjoint": bool(np.intersect1d(pool, val).size == 0),
           "pool_val_cover_all_50000": bool(len(np.union1d(pool, val)) == len(y)),
           "val_per_class": np.bincount(y[val], minlength=10).tolist(),
           "mean": ctx.mean, "std": ctx.std, "subsets": {}}
    for seed in (0, 1, 2, 3):
        subs = D.nested_subsets(y, pool, (0.1, 0.3, 1.0), seed)
        s10, s30, s100 = subs[0.1], subs[0.3], subs[1.0]
        res["subsets"][f"seed{seed}"] = {
            "sizes": {"10%": int(len(s10)), "30%": int(len(s30)), "100%": int(len(s100))},
            "per_class_10%": np.bincount(y[s10], minlength=10).tolist(),
            "nested_10_in_30": bool(np.isin(s10, s30).all()),
            "nested_30_in_100": bool(np.isin(s30, s100).all()),
            "no_overlap_with_val": bool(np.intersect1d(s100, val).size == 0),
        }
    # subsets should differ between seeds (~450 shared images if independent)
    a = D.nested_subsets(y, pool, (0.1,), 1)[0.1]
    b = D.nested_subsets(y, pool, (0.1,), 2)[0.1]
    res["subset_10%_overlap_seed1_vs_seed2"] = int(np.intersect1d(a, b).size)
    return res


def check_models(device):
    out = {}
    x = torch.randn(4, 3, 32, 32, device=device)
    for name in MODEL_NAMES:
        m = build_model(name).to(device)
        info = {"n_params": count_params(m), "logits_shape": list(m(x).shape)}
        if name.startswith("vit"):
            info["tokens_shape"] = list(m.tokens(x).shape)
            _, attns = m(x, return_attn=True)
            info["attn_shape_per_block"] = list(attns[0].shape)
        out[name] = info
    return out


def overfit_tiny(ctx, n=256, steps=300, lr=1e-3):
    """Each model should be able to memorise a small fixed batch (no aug, no wd)."""
    device = ctx.device
    amp_dtype, scaler = amp_setup(device)
    idx = ctx.pool_idx[:n]
    x = D.to_eval_tensor(ctx.x_tr[idx], ctx.mean, ctx.std, device)
    y = torch.from_numpy(ctx.y_tr[idx]).to(device)
    out = {}
    for name in MODEL_NAMES:
        set_seed(0)
        m = build_model(name).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
        losses, t0 = [], time.perf_counter()
        for _ in range(steps):
            m.train()
            perm = torch.randperm(n, device=device)[:64]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = F.cross_entropy(m(x[perm]).float(), y[perm])
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            losses.append(loss.item())
        m.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                             enabled=amp_dtype is not None):
            acc = (m(x).argmax(1) == y).float().mean().item()
        out[name] = {"loss_first": losses[0], "loss_last": float(np.mean(losses[-10:])),
                     "train_acc_end": acc, "passed": acc >= 0.98, "seconds": time.perf_counter() - t0}
    return out


def run_all(ctx, overfit_steps=300):
    out_dir = ctx.out_root / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"splits": check_splits(ctx), "models": check_models(ctx.device),
           "overfit_tiny": overfit_tiny(ctx, steps=overfit_steps), "fake_data": ctx.fake}
    (out_dir / "sanity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
