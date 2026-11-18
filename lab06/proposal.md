# Individual Project Proposal

**Student:** Xiang Deng  **Student ID:** 73799
**Course:** Machine Learning (ML Systems labs), Vistula University

## Problem

Recognise **isolated signs** of American Sign Language (one sign per clip) from body landmarks, with a model that is small and fast enough to run in real time on an ordinary laptop CPU. This is the isolated-sign counterpart of our team project (continuous sign language recognition on Isharah): it lets me study the deployment side (size, latency, compression) on a smaller, well-defined task.

## Dataset

**Kaggle "Google – Isolated Sign Language Recognition" (asl-signs, 2023).**
- 250 ASL signs, recorded by 21 participants with smartphone selfie cameras; about 94.5k sequences (exact count to be checked in EDA).
- Every frame already contains 543 MediaPipe Holistic landmarks (face, pose, both hands) with x, y, z; undetected landmarks are missing values.
- `train.csv` gives `participant_id` for every sequence, so I can make a **participant-disjoint** split: test participants never appear in training.
- Input: a sequence of selected landmarks (hands, upper body, part of the face). Target: one of 250 sign labels.

## Success metric

- **Main:** top-1 accuracy on held-out participants (participant-disjoint test split).
- **Also:** macro-F1 (some signs are rarer than others) and top-5 accuracy.
- **Target:** the optimised model must meet both deployment constraints below while losing at most 2 points of top-1 accuracy compared with my unoptimised model. A first baseline (a small BiLSTM) sets the reference accuracy; I will report the actual numbers, not a promised value.

## Deployment constraints

| Constraint | Limit | How it is measured |
|---|---|---|
| Latency | **≤ 50 ms** per sign | Median over 200 runs, batch 1, laptop CPU, ONNX Runtime, after warm-up. MediaPipe landmark extraction is timed separately so the end-to-end cost is visible |
| Model size | **≤ 5 MB** | Size of the exported model file on disk |

## Planned model and optimisation

- **Model:** a small landmark Transformer (2–3 encoder layers) over the frame sequence, trained with the YAML/MLflow pipeline from Lab 6 (checkpoints, seeds, logged runs).
- **Planned optimisation: quantization.** Export to ONNX and apply INT8 post-training quantization with ONNX Runtime. A float32 model with about 1–1.5 M parameters is 4–6 MB; INT8 storage is about 4× smaller, which leaves room under 5 MB and should also reduce CPU latency.
- **Fallback:** if quantization costs more than 2 points of accuracy, try **knowledge distillation** from the larger float32 model into a smaller student, then quantize the student.

## Evaluation plan

Compare float32 vs INT8 (and the distilled student, if used) on accuracy, macro-F1, model size and latency, all on the same participant-disjoint test set, and report which versions meet both constraints.

## Risks

- About 94.5k sequences of 543 landmarks is a lot of data for free GPUs; I may train on a landmark subset first.
- MediaPipe extraction time may be larger than the model latency, so the 50 ms limit is applied to the model and the end-to-end time is reported separately.
- Dynamic INT8 quantization may help less for small models; static quantization with a calibration set is the alternative.
