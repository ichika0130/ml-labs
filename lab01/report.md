# Lab 1: Environment and first system measurements

## 1. Goal

Set up a reproducible Python 3.11.8 environment with pinned dependencies, train two baseline classifiers on Breast Cancer Wisconsin (Diagnostic), measure what each model costs as a system (training time, single-sample latency, model size, peak memory), and decide which deployment targets (Cloud, Edge, Mobile, TinyML) each model can actually run on.

## 2. Method

**Environment.** Python 3.11.8 in a repo-local `.venv`, packages from the pinned `requirements.txt`. The versions captured by `src/print_versions.py` are in `results/versions.txt`. Measured on an Apple Silicon Mac (macOS 26.6, arm64), CPU only.

> Note: the manual's pins conflict. `mlflow==2.14.1` requires `pyarrow<16`, while `pyarrow==16.1.0` is also pinned, so a plain `pip install -r requirements.txt` cannot resolve. I installed with `pyarrow==16.1.0` forced as an override. Nothing in this lab uses mlflow.

**Data and models.** `load_breast_cancer(return_X_y=True)` gives 569 samples and 30 features. A 70/30 stratified split with `random_state=42` leaves 398 for training and 171 for testing. The two models are exactly as specified: `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`. As the spec requires, the features are not scaled. Seeds are set for `random`, NumPy and PyTorch.

**Measurement protocol** (all timings use `time.perf_counter`):

| Quantity | How it was measured |
|---|---|
| Training time | 1 warm-up fit, then 5 timed fits on fresh estimators; the median is reported (all 5 runs are kept in the CSV) |
| Inference latency | 1 sample (`X_test[:1]`), 1 warm-up, then 100 timed `predict` calls; the median is reported |
| Model size | `joblib.dump` of each model to `results/model_<name>.joblib`; both models are also stored together in `results/model.joblib` |
| Peak memory | Process RSS sampled every 1 ms with `memory_profiler` during one fit, and during 100 single-sample predictions. Each (model, phase) runs in a fresh subprocess, so earlier runs cannot inflate the high-water mark. I report both the peak RSS and the increase over the RSS just before the call |

The memory subprocess does not import PyTorch. These models never use it, and importing it alone raised RSS from 116 MB to 227 MB, which would have been wrongly counted as model memory.

**Budget check.** The memory requirement is the peak RSS while serving predictions. For a range such as Mobile 64–256 MB, a model "fits" if it needs no more than the lower bound, so it runs on every device in the class. It is "partial" if it needs no more than the upper bound, so it runs only on the larger devices. Latency and size are compared directly against their limits. All results are reproduced with `src/run_all.sh`.

## 3. Results

**Test accuracy** (`results/baseline_accuracy.csv`, 171 test samples)

| Model | Test accuracy |
|---|---|
| LogisticRegression | 0.9415 |
| RandomForest | 0.9357 |

**System cost** (`results/system_cost.csv`)

| Model | Train time, median (ms) | Single-sample latency, median (ms) | Size (bytes) | Size (KB) | Peak RSS, train (MB) | Peak RSS, inference (MB) | Δ RSS, train / inference (MB) |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 57.77 | 0.014 | 1,103 | 1.08 | 117.5 | 116.6 | 0.69 / 0.00 |
| RandomForest | 46.30 | 0.643 | 290,905 | 284.09 | 115.8 | 117.7 | 0.02 / 0.02 |

Repeated runs were consistent. RandomForest training stayed between 45 and 48 ms. LogisticRegression training was noisier, between 43 and 100 ms per run, with medians of 43–58 ms. LogisticRegression latency medians ranged from 0.014 to 0.035 ms, and peak RSS stayed within ±1 MB.

**Deployment fit** (`results/deployment_fit.csv`)

| Target (memory / latency / size budget) | LogisticRegression | RandomForest |
|---|---|---|
| Cloud (≥ 1 GB / ≤ 100 ms / ≤ 500 MB) | **Fits** | **Fits** |
| Edge (256–1024 MB / ≤ 50 ms / ≤ 50 MB) | **Fits** | **Fits** |
| Mobile (64–256 MB / ≤ 20 ms / ≤ 10 MB) | **Partial:** needs ~117 MB, so only devices with ≥ 117 MB available | **Partial:** same, ~118 MB |
| TinyML (≤ 256 KB / ≤ 10 ms / ≤ 100 KB) | **No:** memory 117 MB ≫ 256 KB (size 1.08 KB and latency pass) | **No:** memory 118 MB ≫ 256 KB, and size 284 KB > 100 KB |

**Deployment argument.**
- **Latency is never the deciding factor.** The slower model, RandomForest, takes 0.64 ms per prediction. That is 15× below even the TinyML limit of 10 ms.
- **Memory decides Mobile and TinyML, and it is the runtime's memory, not the model's.** The Python, NumPy, SciPy and scikit-learn process already uses about 116 MB before a model is loaded. Fitting or running a model adds less than 1 MB.
- **Cloud and Edge.** Both models fit easily on Cloud and Edge: 117 MB is below even the smallest Edge device (256 MB).
- **Mobile.** On Mobile, both models fit only on devices with roughly 120 MB or more available. A 64 MB device cannot even start the Python stack.
- **TinyML.** In their current form, both models fail on TinyML. The two differ, though:
  - LogisticRegression is 30 weights plus one bias, about 124 bytes as float32. Its prediction is one dot product. Exported to C, it would fit every TinyML budget.
  - RandomForest has 100 trees with 3,138 nodes in total. Even its serialized file (284 KB) is larger than the TinyML size limit, so it would need far fewer or shallower trees first.

## 4. Conclusions

1. On this dataset the simple linear model does slightly better than the 100-tree forest (0.9415 vs 0.9357, one more correct test sample). It is also 264× smaller and about 46× faster per prediction. Extra model complexity did not buy accuracy here.
2. For small classical models, the deployment bottleneck is the software stack rather than the model. About 116 MB of the 117 MB peak is the Python/scikit-learn runtime. To move down to Mobile or TinyML, the first thing to change is the runtime, for example exporting the model to C or ONNX, not the model itself.
3. Training time can mislead. LogisticRegression trains no faster than the forest, because on unscaled features L-BFGS reaches the 1000-iteration limit without converging (scikit-learn raises a ConvergenceWarning). Its cost reflects optimizer behaviour, not model size. Medians over repeated runs were necessary, because individual LogisticRegression training runs ranged from 43 to 100 ms.
