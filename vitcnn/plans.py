import json
from itertools import product
from pathlib import Path

from .engine import RunConfig

MAIN_MODELS = ("vit_p4", "resnet18_w05")
LR_GRID = (5e-4, 1e-3, 2e-3)
MAIN_SEEDS = (1, 2, 3)   # seed 0 is used for lr search and pilot runs
DEFAULT_STEPS = {"lr_sweep": 10_000, "pilot10": 30_000, "main": 30_000}


def load_chosen_lr(out_root):
    p = Path(out_root) / "chosen_lr.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found, run the lr_sweep plan and select-lr first")
    lr = json.loads(p.read_text(encoding="utf-8"))["chosen_lr"]
    lr.setdefault("vit_p8", lr["vit_p4"])
    return lr


def build_plan(plan, out_root, steps=None, **kw):
    if plan == "lr_sweep":
        st = steps or DEFAULT_STEPS["lr_sweep"]
        return [RunConfig(model=m, aug="strong", data_frac=1.0, seed=0, lr=lr, total_steps=st,
                          plan=plan, eval_test=False, **kw)
                for m, lr in product(MAIN_MODELS, LR_GRID)]
    if plan == "lr_sweep_ext":
        # best ResNet lr kept landing on the upper edge of the grid (2e-3, then 4e-3)
        st = steps or DEFAULT_STEPS["lr_sweep"]
        return [RunConfig(model="resnet18_w05", aug="strong", data_frac=1.0, seed=0, lr=lr,
                          total_steps=st, plan="lr_sweep", eval_test=False, **kw)
                for lr in (4e-3, 8e-3)]

    lr = load_chosen_lr(out_root)
    st = steps or DEFAULT_STEPS["pilot10" if plan == "pilot10" else "main"]
    if plan == "pilot10":
        return [RunConfig(model=m, aug=a, data_frac=0.1, seed=0, lr=lr[m], total_steps=st,
                          plan=plan, eval_test=False, **kw)
                for m, a in product(MAIN_MODELS, ("basic", "strong"))]
    if plan == "budget_check":
        # same as the 100% strong runs but with 2x steps
        return [RunConfig(model=m, aug="strong", data_frac=1.0, seed=1, lr=lr[m], total_steps=2 * st,
                          plan="budget", eval_test=True, **kw) for m in MAIN_MODELS]
    if plan == "tier1":
        combos = product(MAIN_MODELS, (0.1, 1.0), ("basic", "strong"), MAIN_SEEDS)
    elif plan == "tier2":
        combos = product(("vit_p8",), (0.1, 1.0), ("strong",), MAIN_SEEDS)
    elif plan == "tier3":
        combos = product(MAIN_MODELS, (0.3,), ("basic", "strong"), MAIN_SEEDS)
    else:
        raise ValueError(f"unknown plan {plan}")
    # seed-major order: if a queue stops early every cell still has some seeds
    combos = sorted(combos, key=lambda c: (c[3], c[1], c[0], c[2]))
    return [RunConfig(model=m, aug=a, data_frac=f, seed=s, lr=lr[m], total_steps=st,
                      plan="main", eval_test=True, **kw) for m, f, a, s in combos]


def select_lr(out_root):
    """Best val acc per model over all lr_sweep runs; ties go to the smaller lr."""
    out_root = Path(out_root)
    rows = []
    for p in sorted((out_root / "runs").glob("lr_sweep__*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        rows.append((m["config"]["model"], m["config"]["lr"], m["val_acc"]))
    if not rows:
        raise FileNotFoundError("no lr_sweep results")
    chosen, table = {}, {}
    for model in MAIN_MODELS:
        cand = [(lr, acc) for mm, lr, acc in rows if mm == model]
        if not cand:
            continue
        best = sorted(cand, key=lambda t: (-t[1], t[0]))[0]
        chosen[model] = best[0]
        table[model] = {f"{lr:g}": acc for lr, acc in sorted(cand)}
        lrs = sorted(lr for lr, _ in cand)
        if best[0] in (lrs[0], lrs[-1]):
            print(f"[warn] {model}: best lr {best[0]:g} is on the edge of the grid {lrs}")
    out = {"chosen_lr": chosen, "val_acc_by_lr": table}
    (out_root / "chosen_lr.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
