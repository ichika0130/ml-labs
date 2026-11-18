"""Task 2: profile one training epoch of the baseline config with torch.profiler.

Each phase of every training step is wrapped in a named record_function region, so the
profiler reports how much CPU time goes to data loading, the forward pass (incl. loss),
the backward pass and the optimizer step. Wall-clock timers around the same phases are
recorded as a cross-check.
"""
import sys
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from pipeline import RESULTS_DIR, build_model, build_optimizer, load_config, make_loaders, set_seeds

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]


def main():
    cfg = load_config(overrides=sys.argv[1:])
    set_seeds(cfg["seed"])
    train_loader, *_ = make_loaders(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    wall = dict.fromkeys(PHASES, 0.0)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        it = iter(train_loader)
        epoch_start = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            with record_function("data_loading"):
                try:
                    x, y = next(it)
                except StopIteration:
                    break
            t1 = time.perf_counter()
            with record_function("forward"):
                loss = loss_fn(model(x), y)
            t2 = time.perf_counter()
            with record_function("backward"):
                optimizer.zero_grad()
                loss.backward()
            t3 = time.perf_counter()
            with record_function("optimizer_step"):
                optimizer.step()
            t4 = time.perf_counter()
            for k, a, b in zip(PHASES, (t0, t1, t2, t3), (t1, t2, t3, t4)):
                wall[k] += b - a
        epoch_s = time.perf_counter() - epoch_start

    averages = {e.key: e for e in prof.key_averages()}
    prof_ms = {k: averages[k].cpu_time_total / 1e3 for k in PHASES}
    total_prof, total_wall = sum(prof_ms.values()), sum(wall.values())
    rows = ["phase,profiler_cpu_ms,profiler_share,wall_ms,wall_share"]
    for k in PHASES:
        rows.append(f"{k},{prof_ms[k]:.0f},{prof_ms[k] / total_prof:.3f},"
                    f"{wall[k] * 1e3:.0f},{wall[k] / total_wall:.3f}")
    rows.append(f"total,{total_prof:.0f},1.000,{total_wall * 1e3:.0f},1.000")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "profile_breakdown.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"epoch wall time {epoch_s:.1f}s")
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
    (RESULTS_DIR / "profile_top_ops.txt").write_text(
        f"One epoch, batch {cfg['data']['batch_size']}, num_workers {cfg['data']['num_workers']}, "
        f"epoch wall time {epoch_s:.1f}s\n\n{table}\n")


if __name__ == "__main__":
    main()
