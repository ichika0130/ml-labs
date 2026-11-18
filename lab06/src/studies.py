"""Tasks 1, 4-7: run the training studies through train.py, then build tables and plots.

    python src/studies.py run baseline|batch|schedules|accum|resume|all
    python src/studies.py report

Every run is a separate train.py process with config overrides, so peak memory is measured
per run and every run is logged to MLflow.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline import LAB_DIR, RESULTS_DIR, load_config  # noqa: E402

PY = sys.executable
TRAIN = str(LAB_DIR / "src" / "train.py")
RUNS_DIR = RESULTS_DIR / "runs"

BATCH_SIZES = [32, 64, 128, 256, 512]
SCHEDULE_RUNS = {   # equal 10-epoch budget
    "adam_step": ["optimizer.name=adam", "optimizer.lr=0.001", "scheduler.name=step"],
    "adam_cosine": ["optimizer.name=adam", "optimizer.lr=0.001", "scheduler.name=cosine"],
    "adam_plateau": ["optimizer.name=adam", "optimizer.lr=0.001", "scheduler.name=plateau"],
    "sgd_cosine": ["optimizer.name=sgd", "optimizer.lr=0.05", "optimizer.momentum=0.9", "scheduler.name=cosine"],
}

GROUPS = {
    "baseline": [("baseline", [])],
    "batch": [(f"bs{b}", [f"data.batch_size={b}"]) for b in BATCH_SIZES if b != 128],  # bs128 = baseline
    "schedules": [(name, ov + ["train.epochs=10"]) for name, ov in SCHEDULE_RUNS.items()],
    "accum": [("accum64x4", ["data.batch_size=64", "train.grad_accum_steps=4"])],
    "resume": [("resume_test", ["train.stop_after_epoch=3"]), ("resume_test", ["train.resume=true"])],
}


def train(name, overrides):
    cmd = [PY, TRAIN, "--set", f"run_name={name}"] + [a for o in overrides for a in ("--set", o)]
    print(">>", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)


def load_log(name):
    with open(RUNS_DIR / f"{name}_log.csv") as f:
        return list(csv.DictReader(f))


def load_summary(name):
    return json.loads((RUNS_DIR / f"{name}_summary.json").read_text())


def write_csv(filename, rows):
    with open(RESULTS_DIR / filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


def col(log, key):
    return [float(r[key]) for r in log]


def report():
    # ---- task 4: batch size ----------------------------------------------------------------
    with open(RESULTS_DIR / "throughput_bench.csv") as f:
        bench = {int(r["batch_size"]): float(r["median_samples_per_s"]) for r in csv.DictReader(f)}
    rows = []
    for b in BATCH_SIZES:
        name = "baseline" if b == 128 else f"bs{b}"
        s, log = load_summary(name), load_log(name)
        rows.append({"batch_size": b, "run": name, "samples_per_s": bench[b],
                     "samples_per_s_during_5_epoch_run": s["mean_samples_per_s"],
                     "peak_rss_mb": s["peak_rss_mb"], "optimizer_steps_per_epoch": log[0]["optimizer_steps"],
                     "final_val_acc": s["final_val_acc"], "test_acc_after_5_epochs": s["test_acc"],
                     "train_seconds_5_epochs": s["train_seconds"]})
    write_csv("batch_size_study.csv", rows)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    bs = [r["batch_size"] for r in rows]
    for a, key, title in zip(ax, ["samples_per_s", "peak_rss_mb", "test_acc_after_5_epochs"],
                             ["Training throughput (samples/s, benchmark)", "Peak RSS (MB)", "Test accuracy after 5 epochs"]):
        a.plot(bs, [r[key] for r in rows], marker="o")
        a.set_xscale("log", base=2)
        a.set_xticks(bs, [str(b) for b in bs])
        a.set_xlabel("batch size")
        a.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "batch_size_study.png", dpi=150)

    # ---- task 5: schedulers and optimizers -----------------------------------------------------
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in SCHEDULE_RUNS:
        s, log = load_summary(name), load_log(name)
        rows.append({"run": name, "optimizer": name.split("_")[0], "scheduler": name.split("_")[1],
                     "best_val_acc": max(col(log, "val_acc")), "final_val_acc": s["final_val_acc"],
                     "final_train_loss": log[-1]["train_loss"], "test_acc_after_10_epochs": s["test_acc"],
                     "final_lr": log[-1]["lr"]})
        epochs = col(log, "epoch")
        ax[0].plot(epochs, col(log, "val_acc"), marker="o", label=name)
        ax[1].plot(epochs, col(log, "lr"), marker="o", label=name)
    write_csv("schedule_optimizer_study.csv", rows)
    ax[0].set_title("Validation accuracy")
    ax[1].set_title("Learning rate at start of epoch")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "schedule_optimizer_study.png", dpi=150)

    # ---- task 6: gradient accumulation ---------------------------------------------------------
    real, acc = load_log("bs256"), load_log("accum64x4")
    rows = [{"epoch": int(a["epoch"]), "bs256_train_loss": a["train_loss"], "accum64x4_train_loss": b["train_loss"],
             "bs256_val_acc": a["val_acc"], "accum64x4_val_acc": b["val_acc"],
             "bs256_optimizer_steps": a["optimizer_steps"], "accum64x4_optimizer_steps": b["optimizer_steps"]}
            for a, b in zip(real, acc)]
    s1, s2 = load_summary("bs256"), load_summary("accum64x4")
    rows.append({"epoch": "test_acc", "bs256_train_loss": "", "accum64x4_train_loss": "",
                 "bs256_val_acc": s1["test_acc"], "accum64x4_val_acc": s2["test_acc"],
                 "bs256_optimizer_steps": "", "accum64x4_optimizer_steps": ""})
    write_csv("grad_accum_comparison.csv", rows)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for log, label in ((real, "batch 256"), (acc, "batch 64 x accumulation 4")):
        ax[0].plot(col(log, "epoch"), col(log, "train_loss"), marker="o", label=label)
        ax[1].plot(col(log, "epoch"), col(log, "val_acc"), marker="o", label=label)
    ax[0].set_title("Training loss")
    ax[1].set_title("Validation accuracy")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "grad_accum_comparison.png", dpi=150)

    # ---- task 7: resume ------------------------------------------------------------------------
    full, resumed = load_log("baseline"), load_log("resume_test")
    rows = [{"epoch": int(a["epoch"]), "segment": "before interruption" if int(a["epoch"]) <= 3 else "after resume",
             "uninterrupted_val_loss": a["val_loss"], "resumed_val_loss": b["val_loss"],
             "uninterrupted_val_acc": a["val_acc"], "resumed_val_acc": b["val_acc"],
             "identical": a["val_loss"] == b["val_loss"] and a["train_loss"] == b["train_loss"]}
            for a, b in zip(full, resumed)]
    write_csv("resume_check.csv", rows)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(col(full, "epoch"), col(full, "val_loss"), marker="o", lw=3, alpha=0.4, label="uninterrupted")
    ax.plot(col(resumed, "epoch"), col(resumed, "val_loss"), marker="x", ls="--", label="stopped at 3, resumed")
    ax.axvline(3.5, color="grey", ls=":")
    ax.set_xlabel("epoch")
    ax.set_title("Validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "resume_check.png", dpi=150)

    # ---- num_workers plot (task 3) and profile plot (task 2) --------------------------------------
    with open(RESULTS_DIR / "num_workers_study.csv") as f:
        w = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    nw = [int(r["num_workers"]) for r in w]
    ax.plot(nw, [float(r["train_epoch_s"]) for r in w], marker="o", label="training epoch")
    ax.plot(nw, [float(r["data_only_s"]) for r in w], marker="o", label="data loading only")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("seconds per epoch")
    ax.set_xticks(nw)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "num_workers_study.png", dpi=150)

    with open(RESULTS_DIR / "profile_breakdown.csv") as f:
        p = [r for r in csv.DictReader(f) if r["phase"] != "total"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh([r["phase"] for r in p], [float(r["profiler_share"]) * 100 for r in p])
    ax.set_xlabel("% of profiled CPU time in one epoch")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "profile_breakdown.png", dpi=150)

    # ---- MLflow runs (graded evidence) -----------------------------------------------------------
    import mlflow
    cfg = load_config()
    mlflow.set_tracking_uri((LAB_DIR / cfg["mlflow"]["tracking_dir"]).resolve().as_uri())
    df = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    keep = [c for c in df.columns if c in ("run_id", "tags.mlflow.runName", "status", "start_time")
            or c.startswith("metrics.") or c in ("params.data.batch_size", "params.optimizer.name",
                                                 "params.scheduler.name", "params.train.epochs",
                                                 "params.train.grad_accum_steps")]
    df[keep].to_csv(RESULTS_DIR / "mlflow_runs.csv", index=False)
    print(f"{len(df)} MLflow runs exported")


def main():
    if sys.argv[1] == "run":
        groups = list(GROUPS) if sys.argv[2] == "all" else sys.argv[2:]
        for g in groups:
            for name, overrides in GROUPS[g]:
                train(name, overrides)
    elif sys.argv[1] == "report":
        report()


if __name__ == "__main__":
    main()
