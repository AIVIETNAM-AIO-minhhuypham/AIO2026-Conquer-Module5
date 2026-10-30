import csv
import json
from pathlib import Path

import pandas as pd

SUMMARY_FIELDS = ["run_id", "plan", "model", "aug", "data_frac", "seed", "lr", "total_steps",
                  "n_params", "n_train", "best_step", "val_acc", "final_val_acc", "test_acc",
                  "test_macro_f1", "train_time_s", "steps_per_s", "infer_img_per_s", "peak_mem_mb", "gpu"]


def load_runs(result_dir="result", plan=None):
    """run_id -> metrics dict (+ the log as a DataFrame under 'log')."""
    runs = {}
    for d in sorted((Path(result_dir) / "runs").iterdir()):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if plan and m["config"]["plan"] != plan:
            continue
        m["log"] = pd.read_csv(d / "log.csv")
        runs[d.name] = m
    return runs


def write_summary(result_dir="result"):
    rows = []
    for m in load_runs(result_dir).values():
        c = m["config"]
        rows.append({"run_id": m["run_id"], "plan": c["plan"], "model": c["model"], "aug": c["aug"],
                     "data_frac": c["data_frac"], "seed": c["seed"], "lr": c["lr"],
                     "total_steps": c["total_steps"], "n_params": m["n_params"], "n_train": m["n_train"],
                     "best_step": m["best_step"], "val_acc": m["val_acc"], "final_val_acc": m["final_val_acc"],
                     "test_acc": m.get("test_acc"), "test_macro_f1": m.get("test_macro_f1"),
                     "train_time_s": m["train_time_s"], "steps_per_s": m["steps_per_s"],
                     "infer_img_per_s": m.get("infer_img_per_s"), "peak_mem_mb": m.get("peak_mem_mb"),
                     "gpu": m["env"].get("gpu", "cpu")})
    path = Path(result_dir) / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {path}")
    return path


def run_table(runs):
    rows = []
    for k, v in runs.items():
        c, lg = v["config"], v["log"]
        rows.append({
            "model": c["model"], "aug": c["aug"], "frac": c["data_frac"], "seed": c["seed"],
            "lr": c["lr"], "steps": c["total_steps"],
            "epochs": round(c["total_steps"] * c["batch_size"] / v["n_train"], 1),
            "best_step": v["best_step"], "val_acc_best": v["val_acc"], "val_acc_final": v["final_val_acc"],
            "val_loss_min": lg["val_loss"].min(), "val_loss_min_step": int(lg.loc[lg["val_loss"].idxmin(), "step"]),
            "val_loss_final": lg["val_loss"].iloc[-1], "train_loss_final": lg["train_loss"].iloc[-1],
            "train_time_s": v["train_time_s"], "test_acc": v.get("test_acc"),
        })
    return pd.DataFrame(rows).sort_values(["model", "aug", "frac", "seed", "lr"]).reset_index(drop=True)


def aug_gaps(runs, metric="val_acc"):
    """Gap (ResNet - ViT) per aug level and aug gain G per model, in accuracy points."""
    acc = {(v["config"]["model"], v["config"]["aug"]): v[metric] for v in runs.values()}
    gap = {a: acc[("resnet18_w05", a)] - acc[("vit_p4", a)] for a in ("basic", "strong")}
    gain = {m: acc[(m, "strong")] - acc[(m, "basic")] for m in ("vit_p4", "resnet18_w05")}
    return {"gap_basic": gap["basic"], "gap_strong": gap["strong"],
            "G_vit": gain["vit_p4"], "G_resnet": gain["resnet18_w05"],
            "G_vit_minus_G_resnet": gain["vit_p4"] - gain["resnet18_w05"]}


def timing_vs_bench(runs, bench):
    sps = {(r["model"], r["aug"]): r["steps_per_s"] for r in bench["results"]}
    rows = []
    for v in runs.values():
        c = v["config"]
        expected = c["total_steps"] / sps[(c["model"], c["aug"])]
        rows.append({"model": c["model"], "aug": c["aug"], "actual_s": v["train_time_s"],
                     "expected_s": expected, "ratio": v["train_time_s"] / expected})
    return pd.DataFrame(rows)
