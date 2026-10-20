"""Task 4: train each model for 5 epochs with Adam (lr 1e-3, batch 128) and record accuracy.

Each model starts from torch.manual_seed(42) and sees the same shuffle order. The model after
the final epoch is evaluated on the test set (validation is logged per epoch, not used for
model selection). An ablation trains the attention model without the positional embedding.
"""
import csv
import time

import torch
from torch import nn

from common import RESULTS_DIR, make_loaders, set_seeds
from models import MODEL_FACTORIES, PatchTransformer

EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cpu")

RUNS = dict(MODEL_FACTORIES)
RUNS["Attention (no positional embedding)"] = lambda: PatchTransformer(positional=False)


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def train_one(name, factory):
    set_seeds()
    train_loader, val_loader, test_loader = make_loaders()
    model = factory().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    log = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        epoch_s = time.perf_counter() - start
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        log.append({"model": name, "epoch": epoch, "train_loss": round(total_loss / n, 4),
                    "train_acc": round(correct / n, 4), "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4), "epoch_seconds": round(epoch_s, 2)})
        print(f"{name:38s} epoch {epoch}  train_loss={total_loss / n:.4f}  train_acc={correct / n:.4f}  "
              f"val_acc={val_acc:.4f}  ({epoch_s:.1f}s)")
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"{name:38s} TEST acc={test_acc:.4f}")
    return log, {"model": name, "params": sum(p.numel() for p in model.parameters()),
                 "test_loss": round(test_loss, 4), "test_accuracy": round(test_acc, 4),
                 "final_val_accuracy": log[-1]["val_acc"],
                 "train_seconds_total": round(sum(r["epoch_seconds"] for r in log), 1)}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    logs, finals = [], []
    for name, factory in RUNS.items():
        log, final = train_one(name, factory)
        logs += log
        finals.append(final)
    for filename, rows in (("train_log.csv", logs), ("test_accuracy.csv", finals)):
        with open(RESULTS_DIR / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
