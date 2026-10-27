"""Task 2: read the raw UCI Adult data without modifying it, and log its shape."""
from sklearn.datasets import fetch_openml

from common import DATA_DIR, sha256_file, write_csv


def ingest(cfg: dict, log):
    ds = cfg["dataset"]
    raw = fetch_openml(name=ds["name"], version=ds["version"], as_frame=True).frame
    rows, cols = raw.shape
    log(f"[ingest] fetch_openml(name='{ds['name']}', version={ds['version']}): "
        f"{rows} rows, {cols} columns")
    log(f"[ingest] expected {ds['expected_rows']} rows and {ds['expected_columns']} columns: "
        f"{'OK' if (rows, cols) == (ds['expected_rows'], ds['expected_columns']) else 'MISMATCH'}")

    # Untouched snapshot of exactly what was read, so later steps can be checked against it.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = DATA_DIR / "adult_raw.csv"
    write_csv(raw, snapshot)
    log(f"[ingest] raw snapshot {snapshot.name} sha256={sha256_file(snapshot)}")
    return raw
