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

**Supplementary checks (beyond the spec).** These were run to test two claims in the conclusions. The graded baselines above are unchanged.
1. 10-fold stratified cross-validation of both models on all 569 samples, to check whether the single-split accuracy gap is real.
2. The same LogisticRegression trained on `StandardScaler` features (scaler fitted on the training split only), to explain its training time.

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

**Supplementary: is the accuracy gap real?** (`results/cv_accuracy.csv`)

| Model | Single split (171 test samples) | 10-fold CV, mean ± std |
|---|---|---|
| LogisticRegression | 0.9415 (161 correct) | 0.9526 ± 0.0249 |
| RandomForest | 0.9357 (160 correct) | 0.9561 ± 0.0239 |

On the single split the two models disagree on 9 test samples, but they make almost the same number of errors. Under cross-validation the ranking flips. In both cases the gap is far smaller than one standard deviation.

**Supplementary: why LogisticRegression trains slowly** (`results/logreg_scaling.csv`, a separate run)

| Features | L-BFGS iterations | Converged | Train time, median (ms) | Test accuracy |
|---|---|---|---|---|
| Unscaled (as specified) | 1000 (limit) | No | 69.39 | 0.9415 |
| Standardized | 19 | Yes | 1.90 | 0.9883 |

The unscaled training time here (69 ms) differs from the main table (58 ms). It comes from a separate run, which matches the run-to-run spread noted above.

## 4. Conclusions

1. **The two baselines are equally accurate, but their system costs are very different.** The 0.006 gap on the test split is one sample out of 171. Cross-validation reverses it, and in both cases it is well inside one standard deviation. At the same accuracy, LogisticRegression is 264× smaller (1.08 KB vs 284 KB) and about 46× faster per prediction (0.014 ms vs 0.64 ms). Its size is what keeps a TinyML port possible. When accuracy is tied, cost should decide the model.
2. **For small classical models, the software stack is the deployment bottleneck, not the model.** About 116 MB of the ~117 MB peak is the Python/NumPy/scikit-learn runtime. The models themselves add less than 1 MB. This is why both models only partially fit Mobile and fail TinyML even though latency passes everywhere. Moving down a tier means replacing the runtime, for example by exporting the model to C or ONNX, not shrinking the model.
3. **LogisticRegression's training cost comes from the optimizer, not the model.** On unscaled features, L-BFGS uses all 1000 iterations without converging, so training takes ~45–90 ms, no faster than a 100-tree forest. After standardizing, it converges in 19 iterations and trains in ~2 ms, about 30× faster, with higher test accuracy (0.9883). A timing number therefore reflects the whole pipeline, including preprocessing. It says little about model size unless the training setup is described with it.
