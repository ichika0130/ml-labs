"""Task 3: parameters (torchinfo), hand-calculated vs. thop FLOPs, and inference latency.

FLOP convention (Ch. 4 formulas): 2 FLOPs per multiply-accumulate, matrix products only;
biases, activations, pooling, softmax and normalisation are not counted. thop reports MACs,
so its numbers are doubled before comparing.

thop and the attention model: thop always profiles under eval() + no_grad(), where PyTorch
runs TransformerEncoderLayer through a fused fast-path kernel that bypasses every submodule
hook, so the whole encoder counts as 0. Even with the fast path off, thop has no rule for
nn.MultiheadAttention (its QKV projection and attention matmuls are functional calls). Both
fixes are applied step by step below to explain the gap.

Latency protocol: each (device, model, batch) is measured in 5 independent repetitions; each
repetition warms up, then takes the median of 100 runs (batch 1) or 20 runs (batch 128). The
median of the 5 repetition medians is reported with its range. Single-threaded CPU is the
headline setting: with PyTorch's default 10 threads on this 4 performance + 6 efficiency core
chip, small-batch timings varied by more than 2x between repetitions.
"""
import copy
import csv
import statistics
import time

import thop
import torch
from torch import nn
from torchinfo import summary

from common import RESULTS_DIR, load_datasets, set_seeds
from models import D_MODEL, N_PATCHES, PATCH, build_models

LATENCY_RUNS = {1: 100, 128: 20}
WARMUP = 10
REPETITIONS = 5
CPU_SETTINGS = (1, 4)  # threads: headline single-thread, and the 4 performance cores
SEQ, FF = N_PATCHES, 128


# ---- hand-calculated FLOPs (per image) ----------------------------------------------------
def linear_flops(n_in, n_out, rows=1):
    return 2 * n_in * n_out * rows


def conv_flops(k, c_in, c_out, h_out, w_out):
    return 2 * k * k * c_in * c_out * h_out * w_out


def attention_flops(seq=SEQ, d=D_MODEL):
    return {
        "QKV projections": 2 * seq * d * 3 * d,
        "attention scores QK^T": 2 * seq * seq * d,
        "attention-weighted values AV": 2 * seq * seq * d,
        "output projection": 2 * seq * d * d,
    }


def hand_flops_by_layer():
    rows = {
        "MLP": [
            ("Linear 784->256", "Linear", linear_flops(784, 256)),
            ("Linear 256->128", "Linear", linear_flops(256, 128)),
            ("Linear 128->10", "Linear", linear_flops(128, 10)),
        ],
        "CNN": [
            ("Conv2d 1->16, 3x3, 28x28 out", "Conv2d", conv_flops(3, 1, 16, 28, 28)),
            ("Conv2d 16->32, 3x3, 14x14 out", "Conv2d", conv_flops(3, 16, 32, 14, 14)),
            ("Linear 1568->10", "Linear", linear_flops(32 * 7 * 7, 10)),
        ],
        "Attention": [("Patch projection 49->64 (x16 patches)", "Linear",
                       linear_flops(PATCH * PATCH, D_MODEL, rows=SEQ))],
    }
    for layer in (1, 2):
        for part, flops in attention_flops().items():
            rows["Attention"].append((f"Encoder {layer}: {part}", "Attention", flops))
        rows["Attention"].append((f"Encoder {layer}: FFN 64->128", "Linear", linear_flops(D_MODEL, FF, rows=SEQ)))
        rows["Attention"].append((f"Encoder {layer}: FFN 128->64", "Linear", linear_flops(FF, D_MODEL, rows=SEQ)))
    rows["Attention"].append(("Head 64->10", "Linear", linear_flops(D_MODEL, 10)))
    return rows


# ---- thop ----------------------------------------------------------------------------------
def count_mha(m, x, y):
    """thop rule for nn.MultiheadAttention: QKV + QK^T + AV + output projection (MACs)."""
    q, k = x[0], x[1]
    b, l, e = q.shape
    s = k.shape[1]
    m.total_ops += torch.DoubleTensor([b * (l * e * e + 2 * s * e * e + 2 * l * s * e + l * e * e)])


def thop_flops(model, x, fastpath=True, mha_rule=False):
    torch.backends.mha.set_fastpath_enabled(fastpath)
    try:
        macs, _ = thop.profile(copy.deepcopy(model), inputs=(x,), verbose=False,
                               custom_ops={nn.MultiheadAttention: count_mha} if mha_rule else None)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    return 2 * int(macs)


def representative_layers():
    """One layer of each type, profiled on its own, with its hand-calculated FLOPs."""
    enc = nn.TransformerEncoderLayer(D_MODEL, 4, FF, batch_first=True).self_attn
    tokens = torch.randn(1, SEQ, D_MODEL)
    cases = [
        ("Linear", "MLP Linear 784->256", nn.Linear(784, 256), (torch.randn(1, 784),),
         linear_flops(784, 256), "2 x 784 x 256"),
        ("Conv2d", "CNN Conv2d 16->32 3x3 (14x14 out)", nn.Conv2d(16, 32, 3, padding=1),
         (torch.randn(1, 16, 14, 14),), conv_flops(3, 16, 32, 14, 14), "2 x 3 x 3 x 16 x 32 x 14 x 14"),
        ("Attention", "Encoder self-attention (seq 16, d 64)", enc, (tokens, tokens, tokens),
         sum(attention_flops().values()),
         "2*16*64*192 + 2*16^2*64 + 2*16^2*64 + 2*16*64*64"),
    ]
    rows = []
    for kind, name, layer, inputs, hand, formula in cases:
        default = 2 * int(thop.profile(copy.deepcopy(layer), inputs=inputs, verbose=False)[0])
        ruled = 2 * int(thop.profile(copy.deepcopy(layer), inputs=inputs, verbose=False,
                                     custom_ops={nn.MultiheadAttention: count_mha})[0])
        rows.append({"type": kind, "layer": name, "formula": formula, "hand_flops": hand,
                     "thop_flops_default": default, "thop_flops_with_mha_rule": ruled,
                     "thop_over_hand": round(default / hand, 4)})
    return rows


# ---- latency -------------------------------------------------------------------------------
def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def median_latency_ms(model, x, runs, device):
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
        sync(device)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            sync(device)
            times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def main() -> None:
    set_seeds()
    RESULTS_DIR.mkdir(exist_ok=True)
    models = build_models()
    x1 = torch.randn(1, 1, 28, 28)
    hand = hand_flops_by_layer()

    # parameters
    param_rows = []
    for name, model in models.items():
        stats = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                        col_names=("input_size", "output_size", "num_params"))
        (RESULTS_DIR / f"torchinfo_{name}.txt").write_text(str(stats) + "\n")
        param_rows.append({"model": name, "torchinfo_params": stats.total_params,
                           "trainable_params": stats.trainable_params,
                           "in_target_100k_150k": 100_000 <= stats.total_params <= 150_000})
    write_csv("params.csv", param_rows)

    # FLOPs: per layer (hand), representative layers, and whole models
    write_csv("flops_hand_by_layer.csv", [
        {"model": m, "layer": layer, "type": kind, "hand_flops": f}
        for m, rows in hand.items() for layer, kind, f in rows])
    write_csv("flops_representative_layers.csv", representative_layers())
    totals = []
    for name, model in models.items():
        model.eval()
        h = sum(f for _, _, f in hand[name])
        t_default = thop_flops(model, x1)
        t_nofast = thop_flops(model, x1, fastpath=False)
        t_rule = thop_flops(model, x1, fastpath=False, mha_rule=True)
        totals.append({"model": name, "hand_flops": h, "thop_default": t_default,
                       "thop_fastpath_off": t_nofast, "thop_fastpath_off_mha_rule": t_rule,
                       "default_over_hand": round(t_default / h, 4),
                       "fixed_over_hand": round(t_rule / h, 4)})
    write_csv("flops_totals.csv", totals)

    # latency on real test images: CPU 1 thread (headline), CPU 4 threads and MPS (supplementary)
    _, _, test = load_datasets()
    images = torch.stack([test[i][0] for i in range(128)])
    settings = [(torch.device("cpu"), t) for t in CPU_SETTINGS]
    if torch.backends.mps.is_available():
        settings.append((torch.device("mps"), 1))
    default_threads = torch.get_num_threads()
    lat_rows = []
    for device, threads in settings:
        torch.set_num_threads(threads)
        label = f"cpu-{threads}t" if device.type == "cpu" else "mps"
        for name, model in models.items():
            m = copy.deepcopy(model).to(device).eval()
            for batch, runs in LATENCY_RUNS.items():
                x = images[:batch].to(device)
                reps = [median_latency_ms(m, x, runs, device) for _ in range(REPETITIONS)]
                ms = statistics.median(reps)
                lat_rows.append({"setting": label, "model": name, "batch": batch,
                                 "runs_per_repetition": runs, "repetitions": REPETITIONS,
                                 "median_ms": round(ms, 4), "min_rep_ms": round(min(reps), 4),
                                 "max_rep_ms": round(max(reps), 4),
                                 "per_image_ms": round(ms / batch, 5),
                                 "images_per_s": round(batch / ms * 1e3, 1)})
    torch.set_num_threads(default_threads)
    write_csv("latency.csv", lat_rows)

    for row in param_rows + totals + lat_rows:
        print(row)


def write_csv(filename, rows):
    with open(RESULTS_DIR / filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
