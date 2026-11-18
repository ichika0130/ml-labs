"""Task 1 (+6, 7): the training pipeline, driven by a single YAML config object.

    python src/train.py                                   # baseline from config.yaml
    python src/train.py --set data.batch_size=256 --set run_name=bs256

Every run: fixed seed, a checkpoint after each epoch (model, optimizer, scheduler, epoch,
all RNG states and the shuffle generator), and MLflow logging of params, per-epoch metrics
and artifacts. train.grad_accum_steps > 1 accumulates gradients over several mini-batches;
train.stop_after_epoch simulates an interruption; train.resume continues from the checkpoint.
"""
import argparse
import csv
import json
import time

import mlflow
import torch
import yaml
from torch import nn

from pipeline import (LAB_DIR, RESULTS_DIR, build_model, build_optimizer, build_scheduler, flatten,
                      load_config, make_loaders, peak_rss_mb, rng_state, set_rng_state, set_seeds)

DEVICE = torch.device("cpu")


def evaluate(model, loader, loss_fn):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x)
            total += loss_fn(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total / n, correct / n


def train_epoch(model, loader, optimizer, loss_fn, accum):
    """One epoch. With accum > 1 the loss is divided by accum and the optimizer steps once
    every accum mini-batches, so the update matches one batch of size batch_size * accum."""
    model.train()
    total, correct, n, steps = 0.0, 0, 0, 0
    optimizer.zero_grad()
    start = time.perf_counter()
    for i, (x, y) in enumerate(loader, start=1):
        logits = model(x)
        loss = loss_fn(logits, y)
        (loss / accum).backward()
        if i % accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
        total += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    if i % accum != 0:          # leftover mini-batches at the end of the epoch
        optimizer.step()
        optimizer.zero_grad()
        steps += 1
    seconds = time.perf_counter() - start
    return total / n, correct / n, seconds, n, steps


def run(cfg):
    name = cfg["run_name"]
    ckpt_dir = LAB_DIR / cfg["checkpoint"]["dir"]
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    runs_dir = RESULTS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(cfg["seed"])
    train_loader, val_loader, test_loader, gen = make_loaders(cfg)
    model = build_model(cfg).to(DEVICE)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = nn.CrossEntropyLoss()
    accum = cfg["train"]["grad_accum_steps"]

    mlflow.set_tracking_uri((LAB_DIR / cfg["mlflow"]["tracking_dir"]).resolve().as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    start_epoch, log, run_id = 1, [], None
    if cfg["train"]["resume"]:
        state = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        set_rng_state(state["rng"])
        gen.set_state(state["loader_generator"])
        start_epoch, log, run_id = state["epoch"] + 1, state["log"], state["mlflow_run_id"]
        print(f"[{name}] resumed from {ckpt_path.name} after epoch {state['epoch']}")

    with mlflow.start_run(run_id=run_id, run_name=None if run_id else name) as mlrun:
        if run_id is None:
            mlflow.log_params(flatten(cfg))
            mlflow.log_dict(cfg, "config.yaml")
        for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc, secs, n, steps = train_epoch(model, train_loader, optimizer, loss_fn, accum)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()
            row = {"run": name, "epoch": epoch, "lr": lr, "train_loss": round(tr_loss, 4),
                   "train_acc": round(tr_acc, 4), "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
                   "epoch_seconds": round(secs, 2), "samples_per_s": round(n / secs, 1),
                   "optimizer_steps": steps}
            log.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k not in ("run", "epoch")}, step=epoch)
            print(f"[{name}] epoch {epoch}: lr={lr:.2e} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.4f} ({secs:.1f}s, {n / secs:.0f} samples/s)", flush=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "rng": rng_state(), "loader_generator": gen.get_state(), "log": log,
                        "mlflow_run_id": mlrun.info.run_id, "config": cfg}, ckpt_path)
            if cfg["train"]["stop_after_epoch"] == epoch:
                print(f"[{name}] simulated interruption after epoch {epoch}")
                mlflow.set_tag("interrupted_after_epoch", epoch)
                return None

        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        train_logs = log
        summary = {"run": name, "test_loss": round(test_loss, 4), "test_acc": round(test_acc, 4),
                   "final_val_acc": train_logs[-1]["val_acc"],
                   "mean_samples_per_s": round(sum(r["samples_per_s"] for r in train_logs) / len(train_logs), 1),
                   "peak_rss_mb": round(peak_rss_mb(), 1),
                   "train_seconds": round(sum(r["epoch_seconds"] for r in train_logs), 1)}
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc, "peak_rss_mb": summary["peak_rss_mb"]})
        log_path = runs_dir / f"{name}_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(train_logs[0]))
            writer.writeheader()
            writer.writerows(train_logs)
        (runs_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
        (runs_dir / f"{name}_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        mlflow.log_artifact(str(log_path))
        mlflow.log_artifact(str(ckpt_path))
        print(f"[{name}] TEST acc={test_acc:.4f}  peak RSS={summary['peak_rss_mb']} MB", flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(LAB_DIR / "src" / "config.yaml"))
    parser.add_argument("--set", action="append", default=[], help="override, e.g. data.batch_size=256")
    args = parser.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
