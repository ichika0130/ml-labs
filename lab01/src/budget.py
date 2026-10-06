"""Task 4: check each model's measured cost against the Cloud/Edge/Mobile/TinyML budgets.

Memory requirement = peak RSS of a Python process serving 100 single-sample predictions.
A memory range "lo-hi" means devices in that class have between lo and hi MB available:
the model fits every device if it needs <= lo, only the larger devices if it needs <= hi.
"""
import csv

from common import RESULTS_DIR

KB, MB = 1 / 1024, 1.0  # express everything in MB

BUDGETS = {
    #           memory (lo, hi) MB    latency ms  size MB
    "Cloud":  {"mem": (1024, None), "lat": 100, "size": 500},
    "Edge":   {"mem": (256, 1024),  "lat": 50,  "size": 50},
    "Mobile": {"mem": (64, 256),    "lat": 20,  "size": 10},
    "TinyML": {"mem": (None, 256 * KB), "lat": 10, "size": 100 * KB},
}


def memory_verdict(need_mb, lo, hi):
    if lo is not None and need_mb <= lo:
        return "yes"
    if hi is None or need_mb <= hi:
        return "partial"
    return "no"


def main() -> None:
    with open(RESULTS_DIR / "system_cost.csv") as f:
        models = list(csv.DictReader(f))

    rows = []
    for m in models:
        mem = float(m["infer_peak_rss_mb"])
        lat = float(m["infer_latency_median_ms"])
        size = int(m["model_size_bytes"]) / 2**20
        for target, b in BUDGETS.items():
            mem_ok = memory_verdict(mem, *b["mem"])
            lat_ok = lat <= b["lat"]
            size_ok = size <= b["size"]
            if mem_ok == "no" or not lat_ok or not size_ok:
                verdict = "no"
            else:
                verdict = mem_ok
            failed = [k for k, ok in (("memory", mem_ok != "no"), ("latency", lat_ok), ("size", size_ok)) if not ok]
            rows.append({
                "model": m["model"], "target": target,
                "memory": mem_ok, "latency": "yes" if lat_ok else "no", "size": "yes" if size_ok else "no",
                "verdict": verdict, "failed_constraints": " ".join(failed) or "-",
            })

    with open(RESULTS_DIR / "deployment_fit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
