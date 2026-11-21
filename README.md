# Webcam-Based Dual-Mode Hand Gesture Interface for Computer Control & Indian Sign Language (ISL)

[![tests](https://github.com/Nittaany/dual-mode-gesture-isl/actions/workflows/tests.yml/badge.svg)](https://github.com/Nittaany/dual-mode-gesture-isl/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **TL;DR** — A CPU-only, single-webcam assistive interface unifying a touchless **virtual mouse (HCI)** and **Indian Sign Language (ISL) recognition** with temporal sentence buffering and hybrid LLM translation. Built on macOS Apple Silicon (ARM64) with zero-copy shared-memory multiprocessing (**30 FPS**, up from 14 FPS - single-threaded), evaluated at **72.7% ± 4.1% accuracy (0.73 macro-F1)** via 5-fold cross-validation (**85.2%** on the shipped model's held-out split), and backed by **52 hardware-free unit tests - `pytest`**.

## Demo Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=6OAXb4nhgCM">
    <img src="https://img.youtube.com/vi/6OAXb4nhgCM/maxresdefault.jpg" width="480" alt="Dual-Mode Gesture AI: 30 FPS Virtual Mouse & ISL Translator Demo" />
  </a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=6OAXb4nhgCM"><strong>Watch full walkthrough (4:52)</strong></a> ·
  <a href="https://www.youtube.com/watch?v=6OAXb4nhgCM&t=149s"><strong>Jump straight to live Virtual Mouse & ISL demo (2:29)</strong></a>
</p>

## At a Glance

| Specification            | Details                                                                                                              |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Author**               | Satyam Chaudhary ([ORCID: 0009-0004-0677-1495](https://orcid.org/0009-0004-0677-1495))                               |
| **Research Context**     | B.Tech Major Research Project (15 credits), Dept. of Computing Technologies, SRMIST                                  |
| **Publication**          | Presented at **AdComSys 2026** (UEM Kolkata, June 26–27, 2026; Paper ID 307); to appear in Springer proceedings      |
| **Core Stack**           | Python 3.10 · MediaPipe (Hands + Holistic) · TensorFlow Lite (Bi-LSTM, ~159K params) · OpenCV · PyAutoGUI · Eel      |
| **Post-Paper Extension** | Temporal Intelligence Engine + Hybrid LLM (**Apple MLX** `Qwen1.5-1.8B-Chat-4bit` + **Groq** `llama-3.1-8b-instant`) |
| **Throughput**           | **30 FPS** locked via zero-copy shared memory (up from 14 FPS GIL-blocked baseline)                                  |
| **Verification**         | 52 passing `pytest` unit tests · reproducible evaluation & 5-fold CV scripts                                         |

---

## Research Motivation & Challenges Overcome

Most assistive gesture tools force an either-or tradeoff—offering - either OS cursor navigation or isolated sign recognition—and often rely on depth cameras, wearable gloves, GPUs, or voice wake words that exclude non-verbal users. This research project was engineered to overcome five core bottlenecks using only a standard 640×480 laptop webcam:

1. **Functional Dichotomy (Unified Dual-Mode):** Combines deterministic virtual mouse control (Mode 1) and probabilistic ISL translation (Mode 2) inside one lightweight background application, with hands-free **Thumbs-Up (2.0 s)** wake and **Open-Palm (3.5 s)** mode switching.
2. **Python GIL Concurrency Bottleneck:** Running MediaPipe tracking alongside Bi-LSTM inference in one loop dropped throughput to **14 FPS** and froze the cursor. Moving inference to isolated OS processes connected via `multiprocessing.shared_memory` eliminated ~27.6 MB/s of frame pickling and restored a locked **30 FPS**.
3. **Studio-to-Webcam Domain Shift:** Models trained on straight-on studio clips (INCLUDE-50 at ~2 m) break on tilted, close-range laptop webcams (~0.5 m) due to scale and translation variance. **Shoulder-Invariant Normalization** maps all 201 landmarks relative to the user's chest center and shoulder width.
4. **Softmax Hallucinations & UI Flickering:** Without a null state, softmax classifiers emit random signs whenever a user types or rests their hands. Injecting a 101-clip **`Idle`** negative-control class plus a temporal debouncing state machine stabilizes real-world output.
5. **Ergonomic Fatigue ("Gorilla Arm") & Micro-Tremor:** Absolute screen mapping causes rapid arm strain, while raw hand landmarks jitter. Relative knuckle-delta mapping (trackpad-style) paired with a velocity-adaptive **One Euro Filter** delivers smooth cursor control with minimal wrist movement.

---

## System Architecture

<p align="center">
  <img src="https://q4rcgfyyqou1ialf.public.blob.vercel-storage.com/architecture.jpeg" width="680" alt="Zero-Copy Multiprocessing System Architecture" />
</p>

```text
Webcam (640×480 @ 30 FPS) ─► src/vision_backend.py (producer: capture, wake/mode gestures, LLM worker thread)
                                 │  numpy.copyto → SharedMemory "jarvis_shm" (921,600 B/frame, O(1) zero-copy read)
                                 ├─► HCI process  (MediaPipe Hands, 21 landmarks)     active when mode = 0
                                 └─► ISL process  (MediaPipe Holistic, 201 features)  active when mode = 1
                                          └─► temporal buffer ─► rule parser / Hybrid LLM ─► Eel Web GUI + TTS
```

| Module                              | Responsibility                                                                                         |
| :---------------------------------- | :----------------------------------------------------------------------------------------------------- |
| `src/Jarvis.py`                     | Main orchestrator: process lifecycle, voice/text fallback, UI bridge watcher                           |
| `src/app.py`                        | Eel web GUI and live LLM backend toggle (`MLX` / `GROQ` / `AUTO`)                                      |
| `src/vision_backend.py`             | 30 FPS webcam producer, zero-copy shared-memory writer, wake/toggle gestures, async LLM thread         |
| `src/controllers/HCI_Controller.py` | 21-point hand heuristics, One Euro Filter, relative-motion cursor, macOS actions                       |
| `src/controllers/ISL_Controller.py` | 201-point Holistic extraction, normalization, TFLite Bi-LSTM, temporal state machine, `parse_intent()` |
| `src/controllers/llm_engine.py`     | 3-tier `HybridLLM` (local Apple MLX 4-bit + Groq cloud fallback, 5 s timeout, retry, cache)            |
| `src/utils/normalization.py`        | Shoulder-invariant 201-feature spatial normalization                                                   |
| `scripts/` & `tests/`               | Reproducible TFLite evaluation, 5-fold CV, confusion matrix plot, and 52 `pytest` unit tests           |

---

## Benchmarks & Evaluation

### 1. Pipeline Throughput (14 FPS vs. 30 FPS)

Offloading Bi-LSTM inference (~50–80 ms) to a GIL-isolated process keeps the Vision + HCI loop within **17–24 ms/frame** (well under the 33.3 ms budget for 30 FPS).

<p align="center">
  <img src="https://q4rcgfyyqou1ialf.public.blob.vercel-storage.com/fps_benchmark.jpg" width="460" alt="Vision Pipeline Throughput: 14 FPS vs 30 FPS" />
</p>

### 2. ISL Model Evaluation (532 Curated Clips, 11 Classes)

The dataset merges studio-recorded **INCLUDE-50** subsets with custom real-world webcam captures across 11 classes (`hello`, `thank you`, `good morning`, `good`, `happy`, `time`, `father`, `teacher`, `i`, `you (plural)`, and 101 `idle` clips).

Because real-world laptop captures introduce varied lighting, camera tilt, and background noise compared to clean studio data—and total data is constrained to 532 clips (34–55 per sign except idle - 101)—the model fits the training set tightly (~96%) while unseen generalization sits in the 73–85% range. All metrics below are directly reproducible from the repository scripts and training notebook (`notebooks/train_isl_bilstm.ipynb`):

| Evaluation Protocol                   |      Accuracy      |     Macro-F1      | Scope & Interpretation                                                                                               |
| :------------------------------------ | :----------------: | :---------------: | :------------------------------------------------------------------------------------------------------------------- |
| **5-Fold Stratified CV** _(headline)_ | **72.74% ± 4.06%** | **0.726 ± 0.041** | Every clip tested once across 5 folds (70.1%, 76.6%, 74.5%, 67.0%, 75.5%) via `scripts/cross_validate.py`.           |
| **Shipped `.tflite` Held-Out Split**  | **85.19%** (46/54) |     **0.859**     | Reproducible 10% stratified split (`random_state=42`) via `scripts/evaluate_tflite.py` (95% CI ≈ 73–92%).            |
| Training Set Convergence (Epoch 150)  |       96.02%       |         —         | Final training `sparse_categorical_accuracy` (peak 97.81%); reflects training fit rather than unseen generalization. |

_Note: 5-fold CV across 80, 100, and 150 epochs produced 72.4%, 71.8%, and 72.7% respectively—confirming accuracy is bounded by dataset size and lighting/noise/environment diversity rather than epoch count. This repository documents the full codebase plus post-paper Phase 3 extensions (temporal engine and Hybrid LLM) under strict reproducible cross-validation._

<p align="center">
  <img src="docs/assets/confusion_matrix.png" width="580" alt="Held-out confusion matrix of shipped ISL_Model.tflite (85.2%, n=54)" />
</p>

_Held-out confusion matrix of the shipped `models/ISL_Model.tflite` (single 10% split, 46/54 correct; cells show row recall and raw clip counts)._

<details>
<summary><strong>View Pooled 5-Fold Per-Class Report, Dataset Distribution & Training Curves</strong></summary>

| Class        | Precision | Recall |    F1     | Support |
| :----------- | :-------: | :----: | :-------: | :-----: |
| father       |   0.930   | 0.870  | **0.899** |   46    |
| happy        |   0.851   | 0.909  | **0.879** |   44    |
| thank you    |   0.741   | 0.851  |   0.792   |   47    |
| teacher      |   0.757   | 0.824  |   0.789   |   34    |
| time         |   0.737   | 0.718  |   0.727   |   39    |
| i            |   0.765   | 0.667  |   0.712   |   39    |
| good morning |   0.630   | 0.773  |   0.694   |   44    |
| idle         |   0.705   | 0.663  |   0.684   |   101   |
| hello        |   0.661   | 0.673  |   0.667   |   55    |
| you (plural) |   0.649   | 0.615  |   0.632   |   39    |
| good         |   0.622   | 0.523  | **0.568** |   44    |

<p align="center">
  <img src="https://q4rcgfyyqou1ialf.public.blob.vercel-storage.com/dataset.jpg" width="600" alt="ISL Dataset Class Distribution (532 Clips)" />
</p>

<p align="center">────────────────────</p>

<p align="center">
  <img src="https://q4rcgfyyqou1ialf.public.blob.vercel-storage.com/accuracy.png" width="46%" alt="Training vs Validation Accuracy over 150 Epochs" />
  &nbsp;&nbsp;
  <img src="https://q4rcgfyyqou1ialf.public.blob.vercel-storage.com/Loss.png" width="46%" alt="Training vs Validation Loss over 150 Epochs" />
</p>

</details>

---

## Core Pipelines & Technical Details

### ISL Normalization, Bi-LSTM & Hybrid LLM Extension (Phase 3)

- **Shoulder-Invariant Normalization (`src/utils/normalization.py`):** Extracts 201 features (25 upper-body pose + 21 left-hand + 21 right-hand 3D points), centers on the thoracic midpoint `c = (S_L + S_R) / 2`, scales by inter-shoulder distance `D = ||S_L - S_R||_2`, and sanitizes with `nan_to_num` + `clip(-5, 5)` identically at training and inference.
- **Bi-LSTM (~159K params, Float32 TFLite):** `Bidirectional(LSTM 64, return_sequences) → BatchNorm → Dropout(0.3) → LSTM(32) → BatchNorm → Dropout(0.3) → Dense(32, ReLU) → Dense(11, Softmax)` over 30 linearly interpolated frames.
- **Temporal Intelligence Engine:** Converts noisy sliding-window predictions into stable sentences using a **0.80 / 0.70 hysteresis confidence gate**, leaky 5-frame majority-vote `deque`, **0.5 s word-hold confirmation**, **2.0 s cooldown**, **0.8 s post-word stability**, **3.0–6.0 s context-aware timeouts**, and a **10-frame Idle commit**.
- **3-Tier Sentence Generation (`src/controllers/llm_engine.py`):** Fast deterministic templates (`parse_intent()`, 0 ms) → local offline **Apple MLX** (`Qwen1.5-1.8B-Chat-4bit`, ~1.2 GB RAM) → **Groq** (`llama-3.1-8b-instant`) cloud fallback, protected by a **5.0 s `ThreadPoolExecutor` timeout**, 2-attempt retry, result cache, and raw-gloss fallback.

<p align="center">
  <img src="https://q4rcgfyyqou1ialf.public.blob.vercel-storage.com/temporal_engine.png" width="45%" alt="Temporal State Machine Output" />
  &nbsp;&nbsp;
  <img src="https://q4rcgfyyqou1ialf.public.blob.vercel-storage.com/mlx_llm_test.png.png" width="45%" alt="Apple MLX and Groq LLM Bridge Test" />
</p>

### Virtual Mouse (Mode 1 — HCI)

| Action                 | Gesture              | Geometric Rule (MediaPipe Hands, 21 Landmarks)                                      |
| :--------------------- | :------------------- | :---------------------------------------------------------------------------------- |
| **Move**               | Index up             | Relative knuckle (`lm 5`) delta × `1.3`, One Euro Filter (`min_cutoff=0.01, β=1.0`) |
| **Left / Right Click** | Index / Middle pinch | Thumb distance `< 0.04` with `> 0.06` mutual-exclusion margin; 0.3 s cooldown       |
| **Drag & Drop**        | Closed fist          | All four fingers curled (`mouseDown`, release on open)                              |
| **App Switch**         | Three fingers up     | Index, middle, ring extended (`⌘-Tab`, 1.0 s cooldown)                              |
| **Screenshot**         | Pinky up             | Pinky extended only (`⌘⇧3`, 1.0 s cooldown)                                         |
| **Clutch (Neutral)**   | Open hand            | Pauses cursor to reposition hand without moving pointer                             |

_ISO 9241-9 Fitts' Law pilot (n = 1, CMU HCII instrument):_ **418.7 ms** movement time with **0% error** on standard 1.0 cm GUI targets at 4.0 cm distance; 0.25 cm targets hit the 640×480 webcam sub-pixel precision floor (60–90% error).

---

## Limitations & Future Research Roadmap

- **Dataset Scale & Environmental Robustness:** The current model covers 11 classes across 532 clips with clip-level splits. Next: collect multi-signer, multi-lighting datasets for signer-disjoint evaluation, apply temporal/lighting augmentation, and scale vocabulary via **few-shot learning** (adding new signs from 3–5 user examples without full retraining).
- **Frame Synchronization & Continuous Signing:** Consumers currently free-run on shared memory, and multi-word signing relies on isolated-sign buffering via the temporal state machine. Next: add frame-index synchronization and explore CTC / sequence-level architectures for unconstrained continuous ISL.
- **Cross-Platform & Edge Deployment:** OS bindings (`osascript` / ⌘ hotkeys) and Apple MLX currently target macOS Apple Silicon (ARM64). Next: containerize with Docker, add Linux/Windows bindings (`xdotool` / `pywin32`), and deploy quantized headless builds on low-cost edge hardware (e.g., Raspberry Pi 5).
- **Multi-User Usability Trials:** Expand the single-participant (n = 1) Fitts' Law pilot to multi-user trials with motor- and speech-impaired participants.

---

## Quick Start & Reproducibility (macOS Apple Silicon)

```bash
git clone https://github.com/Nittaany/dual-mode-gesture-isl.git
cd dual-mode-gesture-isl
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add GROQ_API_KEY for cloud LLM fallback
python3 src/Jarvis.py
```

Grant camera, microphone, and macOS accessibility permissions on first run (falls back to Web UI text input if audio dependencies are unavailable).

### Reproduce Every Result

| Verification                    | Command                                                      | Requirements                            |
| :------------------------------ | :----------------------------------------------------------- | :-------------------------------------- |
| **52 unit tests pass**          | `pip install -r requirements-dev.txt && python -m pytest -v` | No webcam / TF needed (~0.2 s)          |
| **85.19% held-out TFLite eval** | `python scripts/evaluate_tflite.py`                          | Local `data/processed/{X,y,labels}.npy` |
| **72.7% ± 4.1% 5-fold CV**      | `python scripts/cross_validate.py`                           | Same + TensorFlow                       |
| **Regenerate confusion matrix** | `python scripts/plot_confusion_matrix.py`                    | Same + Matplotlib                       |
| **Full training logs & curves** | Open `notebooks/train_isl_bilstm.ipynb`                      | Included in repo                        |

_Note: Raw INCLUDE-50 videos and `data/processed/X.npy` are excluded from git due to size and dataset licensing; regenerate via `src/ml_tools/data_extractor.py`._

---

## Citation, datasets & license

- **Paper:** S. Chaudhary, A. Goyal, C. Venkatesh, _"Webcam-Based Dual-Mode Hand Gesture Interface for Computer Control and Indian Sign Language,"_ Third International Conference on Advanced Computing and Systems (AdComSys 2026), UEM Kolkata, June 26–27, 2026 (Paper ID 307).
- **Dataset:** A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra, _"INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition,"_ ACM Multimedia (MM '20), 2020.
- **License:** [MIT](LICENSE).

---

---

---

## For AI assistants & indexers (machine-readable summary)

```yaml
project: dual-mode-gesture-isl # formerly Jarvis-EcoSign
purpose: CPU-only webcam virtual mouse + isolated Indian Sign Language recognition with LLM sentence generation
platform: macOS Apple Silicon (ARM64), Python 3.10
entry_point: src/Jarvis.py
isl_model:
  type: Bi-LSTM, TFLite float32, ~159K params
  input: [1, 30, 201]
  classes: 11 # 10 signs + idle
  dataset_clips: 532
metrics:
  cv_5fold_accuracy:
    { mean: 0.7274, std: 0.0406, macro_f1: 0.726, generalization: true }
  shipped_model_single_split:
    {
      accuracy: 0.8519,
      correct: 46,
      total: 54,
      generalization: true,
      note: small test set,
    }
  keras_val_sparse_categorical_accuracy_final:
    { value: 0.8519, generalization: partial, note: same 54-clip split }
  keras_sparse_categorical_accuracy_final:
    { value: 0.9602, generalization: false, note: training-set fit }
citation_guidance:
  - Cite 72.7% +/- 4.1% (5-fold CV) as the model's accuracy; 85.2% only as a single-split figure.
  - Do not present training-set sparse_categorical_accuracy, or a best-epoch value, as accuracy.
  - Throughput 14->30 FPS and the n=1 Fitts' law result are author-measured.
reproduce:
  [scripts/evaluate_tflite.py, scripts/cross_validate.py, "python -m pytest"]
```

---
