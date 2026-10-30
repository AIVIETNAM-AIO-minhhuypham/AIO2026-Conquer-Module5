import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from . import data as D
from .engine import amp_setup, env_info, inference_speed, set_seed
from .models import build_model, count_params, param_groups
from .plans import DEFAULT_STEPS


def bench_train(ctx, model_name, aug, steps, warmup, batch_size=128, num_workers=-1):
    device = ctx.device
    amp_dtype, scaler = amp_setup(device)
    set_seed(0)
    model = build_model(model_name).to(device)
    opt = torch.optim.AdamW(param_groups(model, 0.05), lr=1e-3)
    ds = D.TensorImageDataset(ctx.x_tr[ctx.pool_idx], ctx.y_tr[ctx.pool_idx],
                              D.train_transform(aug, ctx.mean, ctx.std))
    mixer = D.batch_mixer() if aug == "strong" else None
    workers = D.default_num_workers() if num_workers < 0 else num_workers
    sampler = RandomSampler(ds, num_samples=(steps + warmup) * batch_size,
                            generator=torch.Generator().manual_seed(0))
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True,
                        num_workers=workers, pin_memory=device.type == "cuda",
                        prefetch_factor=4 if workers > 0 else None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    t0 = None
    for i, (xb, yb) in enumerate(loader):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        target = yb
        if mixer is not None:
            xb, target = mixer(xb, yb)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = F.cross_entropy(model(xb).float(), target)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    res = {"model": model_name, "aug": aug, "n_params": count_params(model),
           "steps_measured": steps, "steps_per_s": steps / elapsed,
           "train_img_per_s": steps * batch_size / elapsed,
           "infer_img_per_s": inference_speed(model, ctx.x_test, amp_dtype),
           "num_workers": workers}
    if device.type == "cuda":
        res["peak_mem_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    return res


def estimate_hours(results, steps_main, steps_sweep):
    sps = {(r["model"], r["aug"]): r["steps_per_s"] for r in results}

    def h(model, aug, steps, n):
        return n * steps / sps[(model, aug)] / 3600 * 1.05  # +5% for validation passes

    mains = ("vit_p4", "resnet18_w05")
    est = {
        "lr_sweep": sum(h(m, "strong", steps_sweep, 3) for m in mains),
        "pilot10": sum(h(m, a, steps_main, 1) for m in mains for a in ("basic", "strong")),
        "tier1": sum(h(m, a, steps_main, 6) for m in mains for a in ("basic", "strong")),
        "tier2": h("vit_p8", "strong", steps_main, 6),
        "tier3": sum(h(m, a, steps_main, 3) for m in mains for a in ("basic", "strong")),
    }
    est["total"] = sum(est.values())
    return {k: round(v, 2) for k, v in est.items()}


def run_bench(ctx, steps=200, num_workers=-1, steps_main=None, steps_sweep=None):
    warmup = max(2, min(30, steps // 5))
    results = []
    for model in ("vit_p4", "vit_p8", "resnet18_w05"):
        for aug in ("basic", "strong"):
            r = bench_train(ctx, model, aug, steps, warmup, num_workers=num_workers)
            print(f"  {model:13s} {aug:6s} | {r['steps_per_s']:6.1f} steps/s | "
                  f"{r['train_img_per_s']:7.0f} img/s train | {r['infer_img_per_s']:8.0f} img/s infer"
                  + (f" | {r['peak_mem_mb']} MB" if "peak_mem_mb" in r else ""))
            results.append(r)
    sm = steps_main or DEFAULT_STEPS["main"]
    ss = steps_sweep or DEFAULT_STEPS["lr_sweep"]
    out = {"env": env_info(ctx.device), "batch_size": 128, "results": results,
           "steps_main": sm, "steps_sweep": ss, "estimated_gpu_hours": estimate_hours(results, sm, ss)}
    (ctx.out_root / "bench").mkdir(parents=True, exist_ok=True)
    (ctx.out_root / "bench" / "bench.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("estimated GPU hours:", out["estimated_gpu_hours"])
    return out
