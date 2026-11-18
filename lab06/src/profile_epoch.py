"""Task 2: profile one training epoch of the baseline config with torch.profiler.

Each phase of every training step is wrapped in a named record_function region, so the
profiler reports how much CPU time goes to data loading, the forward pass (incl. loss),
the backward pass and the optimizer step. Wall-clock timers around the same phases are
recorded as a cross-check.

With num_workers=0 every image transform is recorded as profiler events (~6,700 events per
step), and keeping a whole epoch of events in memory exceeded 3 GB of RAM. The profiler
therefore runs with a schedule that covers every step of the epoch in consecutive windows
of CHUNK steps; each window is aggregated in on_trace_ready and then discarded.
"""
import sys
import time
from collections import defaultdict

from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from pipeline import RESULTS_DIR, build_model, build_optimizer, load_config, make_loaders, set_seeds

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]
CHUNK = 40

phase_us = defaultdict(float)                         # inclusive CPU time of our four regions
op_self_us, op_calls = defaultdict(float), defaultdict(int)   # per-operator self CPU time


def collect(prof):
    for e in prof.key_averages():
        if e.key in PHASES:
            phase_us[e.key] += e.cpu_time_total
        else:
            op_self_us[e.key] += e.self_cpu_time_total
            op_calls[e.key] += e.count


def main():
    cfg = load_config(overrides=sys.argv[1:])
    set_seeds(cfg["seed"])
    train_loader, *_ = make_loaders(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    wall = dict.fromkeys(PHASES, 0.0)
    steps = 0

    with profile(activities=[ProfilerActivity.CPU], on_trace_ready=collect,
                 schedule=schedule(wait=0, warmup=0, active=CHUNK, repeat=0)) as prof:
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
            steps += 1
            prof.step()
        epoch_s = time.perf_counter() - epoch_start

    prof_ms = {k: phase_us[k] / 1e3 for k in PHASES}
    total_prof, total_wall = sum(prof_ms.values()), sum(wall.values())
    rows = ["phase,profiler_cpu_ms,profiler_share,wall_ms,wall_share"]
    for k in PHASES:
        rows.append(f"{k},{prof_ms[k]:.0f},{prof_ms[k] / total_prof:.3f},"
                    f"{wall[k] * 1e3:.0f},{wall[k] / total_wall:.3f}")
    rows.append(f"total,{total_prof:.0f},1.000,{total_wall * 1e3:.0f},1.000")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "profile_breakdown.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"{steps} steps, epoch wall time {epoch_s:.1f}s (with profiler overhead)")

    top = sorted(op_self_us.items(), key=lambda kv: -kv[1])[:20]
    total_self = sum(op_self_us.values())
    lines = [f"One epoch ({steps} steps), batch {cfg['data']['batch_size']}, "
             f"num_workers {cfg['data']['num_workers']}, wall time {epoch_s:.1f}s with profiler on",
             "", f"{'operator':45s} {'self CPU ms':>12s} {'share':>7s} {'calls':>9s}"]
    for name, us in top:
        lines.append(f"{name[:45]:45s} {us / 1e3:12.0f} {us / total_self:7.1%} {op_calls[name]:9d}")
    (RESULTS_DIR / "profile_top_ops.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
