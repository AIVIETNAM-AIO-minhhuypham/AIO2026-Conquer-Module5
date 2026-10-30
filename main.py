import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import torch


class Tee:
    """Mirror stdout into result/console.log."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.out.flush()
        self.fh.flush()


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return dev


def make_ctx(args):
    from vitcnn.engine import Context
    t0 = time.perf_counter()
    ctx = Context(args.data_root, args.out, args.fake, get_device(args.device))
    print(f"[data] train pool {len(ctx.pool_idx)} | val {len(ctx.val_idx)} | test {len(ctx.y_te)} "
          f"| device {ctx.device} | {time.perf_counter() - t0:.1f}s")
    return ctx


def cmd_sanity(args):
    from vitcnn import checks
    ctx = make_ctx(args)
    res = checks.run_all(ctx, overfit_steps=args.overfit_steps)
    s = res["splits"]
    print(f"[sanity] pool/val disjoint: {s['pool_val_disjoint']} | val per class: {s['val_per_class']}")
    for k, v in s["subsets"].items():
        print(f"  {k}: {v['sizes']} | nested: {v['nested_10_in_30'] and v['nested_30_in_100']} "
              f"| no val overlap: {v['no_overlap_with_val']}")
    for k, v in res["models"].items():
        print(f"  {k}: {v['n_params']:,} params | logits {v['logits_shape']}"
              + (f" | tokens {v['tokens_shape']}" if "tokens_shape" in v else ""))
    print_overfit(res["overfit_tiny"], args.overfit_steps)


def cmd_overfit(args):
    from vitcnn.checks import overfit_tiny
    ctx = make_ctx(args)
    res = overfit_tiny(ctx, steps=args.overfit_steps)
    out = ctx.out_root / "sanity" / f"overfit_{args.overfit_steps}steps.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print_overfit(res, args.overfit_steps)


def print_overfit(res, steps):
    for k, v in res.items():
        print(f"  overfit 256 imgs, {steps} steps - {k}: loss {v['loss_first']:.3f} -> {v['loss_last']:.4f}, "
              f"acc {v['train_acc_end']:.3f} -> {'PASS' if v['passed'] else 'FAIL'}")


def cmd_bench(args):
    from vitcnn.bench import run_bench
    run_bench(make_ctx(args), steps=args.bench_steps, num_workers=args.workers, steps_main=args.steps)


def cmd_queue(args):
    from vitcnn.engine import train_run
    from vitcnn.plans import build_plan
    ctx = make_ctx(args)
    steps = args.sweep_steps if args.plan in ("lr_sweep", "lr_sweep_ext") else args.steps
    runs = build_plan(args.plan, args.out, steps=steps, num_workers=args.workers,
                      fake_data=args.fake, compile=args.compile, n_evals=args.n_evals)
    if args.limit:
        runs = runs[: args.limit]
    print(f"[queue] {args.plan}: {len(runs)} runs")
    t0 = time.perf_counter()
    for i, cfg in enumerate(runs, 1):
        print(f"--- {i}/{len(runs)} | elapsed {(time.perf_counter() - t0) / 60:.1f} min ---")
        train_run(cfg, ctx)


def cmd_select_lr(args):
    from vitcnn.plans import select_lr
    print(json.dumps(select_lr(args.out), indent=2))


def cmd_summarize(args):
    from vitcnn.analysis import write_summary
    write_summary(args.out)


COMMANDS = {"sanity": cmd_sanity, "overfit": cmd_overfit, "bench": cmd_bench, "queue": cmd_queue,
            "select-lr": cmd_select_lr, "summarize": cmd_summarize}


def main():
    p = argparse.ArgumentParser(description="ViT vs ResNet on CIFAR-10 under limited data")
    p.add_argument("command", choices=list(COMMANDS))
    p.add_argument("--plan", default="tier1",
                   choices=["lr_sweep", "lr_sweep_ext", "pilot10", "tier1", "tier2", "tier3", "budget_check"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="result")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--sweep-steps", type=int, default=None)
    p.add_argument("--bench-steps", type=int, default=200)
    p.add_argument("--n-evals", type=int, default=20)
    p.add_argument("--overfit-steps", type=int, default=300)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fake", action="store_true", help="random data, for testing only")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.stdout = Tee(Path(args.out) / "console.log")
    print(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} | python main.py {' '.join(sys.argv[1:])} =====")
    COMMANDS[args.command](args)


if __name__ == "__main__":  # needed on Windows for DataLoader workers
    main()
