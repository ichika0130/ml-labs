# Lab 6: Training pipelines and project proposal

## 1. Goal

Turn the training of my best Lab 3 model into a reproducible pipeline driven by one YAML config, with checkpoints and MLflow logging. Then use it to study where training time goes (profiling, `num_workers`), how batch size, learning-rate schedules and optimizers change speed, memory and accuracy, whether gradient accumulation can replace a large batch, and whether interrupted training can be resumed. Finally, write the individual project proposal (`proposal.md`).

## 2. Method

**Environment.** Python 3.10.12, PyTorch 2.3.1, torchvision 0.18.1, MLflow 2.14.1 (versions in `results/versions.txt`). These results were measured in a Linux ARM virtual machine running on the MacBook (4 vCPUs, 3.9 GB RAM, CPU only, PyTorch uses 4 threads), not directly in macOS as in Labs 1–4. All results are reproduced with `bash lab06/src/run_all.sh` (it uses the repo `.venv`; set `PY=...` to use another Python).

**Model and data.** The CNN from Lab 3 (20,490 parameters, test accuracy 0.892 in Lab 3), imported directly from `lab03/src/models.py`. Fashion-MNIST with the same normalisation and the same 55,000 / 5,000 / 10,000 train/validation/test split (seed 42). Unless a study changes it: Adam, lr 1e-3, batch 128, `num_workers` 0, 5 epochs, cross-entropy loss.

**Pipeline (task 1).** `src/config.yaml` is the single config object. `src/train.py` reads it, and any key can be overridden from the command line (`--set data.batch_size=256`). Every run:
- sets the seed and builds data, model, optimizer and scheduler from the config (`src/pipeline.py`);
- saves a checkpoint after every epoch with the model, optimizer and scheduler states, the epoch, all random-number-generator states and the state of the shuffle generator (`lab06/checkpoints/`, git-ignored);
- logs all config values as MLflow params, train/validation loss and accuracy, learning rate, epoch time and samples/s as per-epoch metrics, test accuracy and peak memory as final metrics, and the config, the epoch log and the final checkpoint as artifacts (`lab06/mlruns/`, git-ignored; view with `mlflow ui --backend-store-uri lab06/mlruns`). A table of all runs is exported to `results/mlflow_runs.csv`.

Each study run is a separate `train.py` process (`src/studies.py`), so peak memory (max resident set size) is measured per run.

**Studies.**

| Task | Script | Setup |
|---|---|---|
| 2 Profiling | `src/profile_epoch.py` | One epoch of the baseline under `torch.profiler`. Each step is split into four `record_function` regions: data loading, forward (incl. loss), backward, optimizer step. Wall-clock timers around the same regions as a cross-check |
| 3 num_workers | `src/workers_study.py` | `num_workers` ∈ {0, 1, 2, 4, 8}: time one pass over the training loader with no model ("data only") and one full training epoch |
| 4 Batch size | `src/studies.py`, `src/throughput_bench.py` | Batch ∈ {32, 64, 128, 256, 512}, 5 epochs each, same lr. Throughput is measured separately (see below) |
| 5 Schedules / optimizers | `src/studies.py` | 10 epochs each: Adam + StepLR (×0.5 every 3 epochs), Adam + CosineAnnealingLR, Adam + ReduceLROnPlateau (×0.5, patience 1, on validation loss), SGD (momentum 0.9, lr 0.05) + cosine |
| 6 Gradient accumulation | `src/studies.py` | Batch 64 × 4 accumulation steps (loss divided by 4, optimizer step every 4 mini-batches) vs a real batch of 256 |
| 7 Resume | `src/studies.py` | Stop the baseline config after epoch 3, start again with `train.resume=true`, compare with the uninterrupted baseline |

**Two measurement notes.**
- Profiling with `num_workers=0` records every image transform as profiler events. Keeping a whole epoch of events in memory crashed the 3.9 GB VM, so the profiler records the epoch in consecutive windows of 40 steps that are summarised and then discarded; all 430 steps are covered.
- The speed of the VM drifted during the session: the same baseline config ran at about 4,400 samples/s early and about 5,600 samples/s later. The 5-epoch batch-size runs were made one after another, so their speeds are not comparable. Throughput is therefore measured in a separate benchmark that runs all batch sizes back to back in 3 interleaved rounds (5 warm-up steps, then ≥12,800 samples), and the median is reported.

## 3. Results

### 3.1 Pipeline and MLflow (`results/mlflow_runs.csv`, `results/runs/`)

11 MLflow runs were logged (baseline, 4 batch sizes, 4 schedule/optimizer runs, accumulation, resume test). The baseline reproduces Lab 3: test accuracy **0.8919** (Lab 3: 0.892).

### 3.2 Profile of one epoch (`results/profile_breakdown.csv`, `results/profile_top_ops.txt`)

| Phase | CPU time (ms) | Share |
|---|---|---|
| Data loading | 5,545 | 35.6% |
| Forward (incl. loss) | 3,885 | 24.9% |
| Backward | 5,913 | 38.0% |
| Optimizer step | 233 | 1.5% |

The wall-clock timers give the same shares (within 0.1 points). The most expensive single operators are `convolution_backward` (32.6% of self CPU time), the DataLoader loop itself (27.3%) and the forward convolution (12.5%). Note: the profiler adds overhead to the many small transform operations, so data loading looks bigger here (5.5 s) than without the profiler (3.2 s in §3.3).

### 3.3 num_workers (`results/num_workers_study.csv`, `results/num_workers_study.png`)

| num_workers | Data only (s) | Training epoch (s) | Training samples/s |
|---|---|---|---|
| 0 | 3.23 | **12.21** | 4,504 |
| 1 | 3.73 | 12.52 | 4,393 |
| 2 | 1.77 | 15.93 | 3,452 |
| 4 | 1.21 | 18.58 | 2,961 |
| 8 | 2.30 | 20.33 | 2,706 |

More workers make data loading alone faster (up to 4 workers), but make training slower.

### 3.4 Batch size (`results/batch_size_study.csv`, `results/throughput_bench.csv`, `results/batch_size_study.png`)

| Batch | Throughput (samples/s) | Peak memory (MB) | Optimizer steps per epoch | Test accuracy after 5 epochs |
|---|---|---|---|---|
| 32 | 5,271 | 564 | 1,719 | **0.9024** |
| 64 | 5,614 | 564 | 860 | 0.9010 |
| 128 | **5,795** | 580 | 430 | 0.8919 |
| 256 | 5,398 | 605 | 215 | 0.8891 |
| 512 | 4,364 | 729 | 108 | 0.8782 |

Peak memory includes the whole process (Python, PyTorch and the dataset in memory), so the part caused by the batch itself is the growth from about 564 MB to 729 MB.

### 3.5 Schedules and optimizers, 10 epochs each (`results/schedule_optimizer_study.csv`, `results/schedule_optimizer_study.png`)

| Optimizer | Schedule | Final lr | Best val acc | Final train loss | Test accuracy |
|---|---|---|---|---|---|
| Adam 1e-3 | StepLR | 1.25e-4 | 0.9070 | 0.2196 | 0.9014 |
| Adam 1e-3 | Cosine | 2.4e-5 | 0.9090 | 0.2140 | 0.9014 |
| Adam 1e-3 | ReduceLROnPlateau | 1e-3 (never reduced) | 0.9064 | 0.2082 | 0.8985 |
| SGD + momentum 0.05 | Cosine | 1.2e-3 | **0.9154** | **0.1570** | **0.9145** |

ReduceLROnPlateau never lowered the learning rate: validation loss improved in 8 of 9 epochs, and patience 1 needs two epochs in a row without improvement. Each configuration was run once (one seed).

### 3.6 Gradient accumulation (`results/grad_accum_comparison.csv`, `results/grad_accum_comparison.png`)

| Epoch | Train loss, batch 256 | Train loss, 64 × 4 | Val acc, batch 256 | Val acc, 64 × 4 |
|---|---|---|---|---|
| 1 | 0.6175 | 0.6175 | 0.8552 | 0.8544 |
| 3 | 0.3384 | 0.3385 | 0.8696 | 0.8682 |
| 5 | 0.2936 | 0.2936 | 0.8920 | 0.8920 |
| Test accuracy | | | 0.8891 | 0.8885 |

Both make 215 optimizer steps per epoch. Training loss differs by at most 0.0002 and validation accuracy by at most 0.0014. Peak memory: 605 MB (batch 256) vs 563 MB (64 × 4).

### 3.7 Resume from checkpoint (`results/resume_check.csv`, `results/resume_check.png`)

Training was stopped after epoch 3 and resumed in a new process. Epochs 4 and 5 after resuming have exactly the same training loss, validation loss and validation accuracy as the uninterrupted run (e.g. epoch 5 validation loss 0.2864 in both), and the final test accuracy is the same (0.8919). The curve continues smoothly because the checkpoint also restores the optimizer state, the random-number generators and the shuffle order.

### 3.8 Project proposal

See `proposal.md`.

## 4. Conclusions

1. On this 4-core machine, more DataLoader workers made training slower, not faster. The profile shows that with `num_workers=0`, data loading takes about 36% of the epoch, so I expected workers to help. Data loading alone did get faster (3.23 s → 1.21 s with 4 workers), but a full training epoch got slower, from 12.2 s with 0 workers to 20.3 s with 8. The reason is that the workers are extra processes that run on the same 4 CPU cores that PyTorch already uses for the convolutions, so they take CPU time away from the training itself. So the real bottleneck here is the CPU, not data loading. Workers help when there are free cores or when the model trains on a GPU, and the right number has to be measured on the machine we use.
2. A bigger batch is not automatically faster, and with the same number of epochs it learns less. Throughput was highest at batch 128 (5,795 samples/s) and dropped at 512 (4,364 samples/s), while peak memory grew from 564 MB to 729 MB. After 5 epochs, test accuracy went down from 0.902 at batch 32 to 0.878 at batch 512. With the same number of epochs and the same learning rate, a bigger batch means fewer updates: 1,719 steps per epoch at batch 32 but only 108 at batch 512. Gradient accumulation (64 × 4) gave almost the same training curve as a real batch of 256 with less memory, so it is a good way to use a big batch when memory is not enough. But a big batch only makes sense if we also change the learning rate or train longer.
3. In a short training budget, the choice of optimizer mattered more than the choice of schedule. With 10 epochs each, SGD with momentum and a cosine schedule reached the best test accuracy (0.9145) and the lowest training loss (0.157). The three Adam runs were all close together (0.8985–0.9014). ReduceLROnPlateau never lowered the learning rate in 10 epochs, because the validation loss almost always improved, so in such a short run it behaves like a constant learning rate. This is only one run per setting with one learning rate each, so the gap of about 1.3 points is not final proof. But it shows that we should compare optimizers with the same epoch budget, instead of just using the usual default.
