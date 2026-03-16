# 10th ABAW VA Estimation Pipeline

This repository contains the **audio** and **multimodal fusion** pipeline for the ABAW Valence-Arousal estimation challenge.

The pipeline is organized as follows:

1. **Install dependencies**
2. **Prepare the data**
3. **Build segment-level metadata for audio training**
4. **Train an audio model**
5. **Run audio evaluation / feature extraction**
6. **Train a multimodal fusion model**
7. **Run multimodal evaluation / submission generation**
8. **Optionally perform late fusion of several submissions**

---

## 1. Installation

The project uses **Poetry** and depends on a local editable copy of [Chimera ML](https://github.com/markitantov/chimera_ml).

### Requirements

- Python `>=3.12,<3.13`
- Poetry
- `ffmpeg` and `ffprobe` available in `PATH`
- CUDA-enabled PyTorch environment for training/evaluation on GPU
- A local checkout of `chimera_ml` located one level above this repository:

```text
../chimera_ml
```

### Install with Poetry

From the repository root, run:

```bash
poetry install
```

```bash
poetry shell
```

or run commands through Poetry directly:

```bash
poetry run chimera_ml train --config-path configs/mm_cfg_va_s2s.yaml
```

### Project entry points

This repository registers a Chimera plugin:

```toml
[tool.poetry.plugins."chimera_ml.plugins"]
10th_ABAW = "chimera_plugin:register"
```

That means `chimera_ml` will discover the custom datamodules, models, losses, metrics, callbacks, and optimizers defined in this codebase.

---

## 2. Repository structure

```text
configs/
  cfg_va_s2s.yaml                # audio training config
  cfg_test_w_audio.yaml        # audio evaluation / pickle export config
  mm_cfg_va_s2s.yaml             # multimodal training config
  test_mm_cfg_va_s2s.yaml        # multimodal evaluation / submission export config

scripts/
  extract_audio.py               # extract wav files from videos
  scan_fps.py                    # compute FPS per original video
  make_s2s_windows.py            # build segment-level windows and labels
  make_audio_windows.py          # filter windows using open-mouth information
  predictions_fusion.py          # late fusion of multiple prediction folders

src/
  audio/                         # audio datasets and models
  fusion/                        # multimodal datasets and models
  callbacks/                     
  losses/
  metrics/
  optimizers/
  schedulers/
  chimera_plugin.py
```

---

## 3. Expected data layout

The code assumes a `data/` directory similar to the following:

```text
data/
  features/
    ...
  labels/
    VA_Estimation_Challenge/
      Train_Set/
      Validation_Set/
      Test_Set/
  labels_segmented_s2s_wav/
    train_audio_segment.csv
    val_audio_segment.csv
    test_audio_segment.csv
  openmouth/
    *.csv
  ABAW_VA_test_set_sample.txt
  videos_fps.csv
```

---

## 4. Full pipeline

## Step 1. Extract WAV audio from source videos

Use `scripts/extract_audio.py` to convert videos into mono 16 kHz WAV files.

Example:

```bash
python scripts/extract_audio.py scripts/extract_audio.yaml
```

Adjust config if your videos are stored elsewhere.
This step creates the WAV tree used later by the audio datamodules.

---

## Step 2. Scan FPS for all videos

The audio-window filtering step uses FPS values from the original videos.

Run:

```bash
python scripts/scan_fps.py scripts/scan_fps.yaml
```

This produces:

```text
data/videos_fps.csv
```

---

## Step 3. Create sequence-to-sequence windows

Use `make_s2s_windows.py` to create temporal windows and aggregate frame-level VA labels into segment-level targets.

Run:

```bash
python scripts/make_s2s_windows.py scripts/make_s2s_windows.yaml
```

Main parameters in `scripts/make_s2s_windows.yaml`:

- `win_max_length: 4`
- `win_shift: 2`
- `win_min_length: 1`
- `s2s_frames: 4`

This script creates window-level CSVs with segment metadata for train/val/test.

Typical output is a directory such as:

```text
data/labels_segmented_s2s_wav/
```

---

## Step 4. Filter audio windows using mouth openness

Audio training uses only windows with enough visible speaking activity.

Run:

```bash
python scripts/make_audio_windows.py scripts/make_audio_windows.yaml
```

This script combines:

- segment metadata from Step 3
- FPS from `data/videos_fps.csv`
- open-mouth CSV files from `data/openmouth/`

It adds the following fields to each window:

- `open_sec`
- `coverage_ratio`
- `use_for_audio`
- `fps`

and writes audio-specific CSV files such as:

```text
data/labels_segmented_s2s_wav/train_audio_segment.csv
data/labels_segmented_s2s_wav/val_audio_segment.csv
data/labels_segmented_s2s_wav/test_audio_segment.csv
```

These CSVs are consumed by the audio datamodules.

---

## Step 5. Train an audio model

Audio training is done through [Chimera ML](https://github.com/markitantov/chimera_ml).

Example:

```bash
chimera_ml train --config-path configs/cfg_va_s2s.yaml
```

The default audio training config uses:

- dataset root: `data/Chunk_wav`
- train CSV: `data/labels_segmented_s2s_wav/train_audio_segment.csv`
- val CSV: `data/labels_segmented_s2s_wav/val_audio_segment.csv`
- model: `emotion2vec_s2s_model_v1`
- sequence length: `s2s_steps: 4`

Training outputs are stored under `logs/` together with:

- checkpoints
- MLflow logs
- code snapshot
- config snapshot

### Notes

- `cfg_va_s2s.yaml` uses the `abaw_va_datamodule`.
- The validation callback computes frame-wise metrics from the window-level predictions.

---

## Step 6. Run audio evaluation and export features

After training the audio model, run evaluation with the test config.
This step reconstructs window-level predictions and dumps them to pickle files for train/val/test.

Example:

```bash
chimera_ml eval \
  --config-path configs/cfg_test_w_audio.yaml \
  --checkpoint-path <path_to_audio_checkpoint>
```

The callback configuration in `configs/cfg_test_w_audio.yaml` writes pickle files such as:

```text
data/features/audio_features/train.pkl
data/features/audio_features/val.pkl
data/features/audio_features/test.pkl
```

Each pickle is later used as the audio input for multimodal fusion.

### Important note

The multimodal configs currently point to:

```text
data/features/audio_features_w_fcl_new
```

---

## Step 7. Prepare frame-level modality features for multimodal fusion

The multimodal datamodule expects one pickle per split for each non-audio modality.

According to `configs/mm_cfg_va_s2s.yaml` and `configs/test_mm_cfg_va_s2s.yaml`, the expected inputs are:

```text
data/features/face_features/train.pkl
data/features/face_features/val.pkl
data/features/face_features/test.pkl

data/features/qwen_mm/train.pkl
data/features/qwen_mm/val.pkl
data/features/qwen_mm/test.pkl
```

And the audio modality is expected at:

```text
data/features/audio_features_w_fcl_new/train.pkl
data/features/audio_features_w_fcl_new/val.pkl
data/features/audio_features_w_fcl_new/test.pkl
```

These pickles are not created by this folder except for the audio ones. Face and other modality features must be prepared beforehand (see others folders).

---

## Step 8. Train a multimodal model

Once all modality pickles are ready, train the multimodal fusion model:

```bash
chimera_ml train --config-path configs/mm_cfg_va_s2s.yaml
```

Example from your workflow:

```bash
chimera_ml train --config-path configs/mm_cfg_va_s2s.yaml
```

The default multimodal config uses:

- datamodule: `abaw_mm_datamodule`
- model: `multimodal_fusion_model_v4`
- frame modalities: `face`, `qwen_mm`
- audio modality from the exported audio pickles
- `use_predictions: true`

Validation is done with `multimodal_framewise_v2_callback`, which reconstructs frame-wise predictions and computes CCC.

---

## Step 9. Evaluate the multimodal model and create a submission

Run evaluation with the multimodal test config:

```bash
chimera_ml eval \
  --config-path configs/test_mm_cfg_va_s2s.yaml \
  --checkpoint-path <path_to_multimodal_checkpoint>
```

This config generates a test submission file using:

- `submission_template_path: data/ABAW_VA_test_set_example.txt`
- `submission_out_path: data/Submit_4.txt`

So the multimodal evaluation stage serves two purposes:

1. evaluate the checkpoint on train/val/test
2. export predictions for leaderboard submission or later late fusion

---

## Step 10. Perform late fusion of several submissions

If you have several prediction folders under `data/features/submit/`, you can fuse them with `scripts/predictions_fusion.py`.

Example:

```bash
python scripts/predictions_fusion.py \
  --test_template_path data/ABAW_VA_test_set_example.txt \
  --root data/features/submit/ \
  --include submit_1_134f submit_2_14f submit_3_1 submit_4_134w \
  --max-comb-size 3 \
  --pair-grid-step 0.05 \
  --dirichlet-samples 8000 \
  --dirichlet-alpha 1.0 \
  --channelwise \
  --output-dir ./fusion_out_with_seeds \
  --split train
```

What this script does:

- loads prediction pickles from several submission folders
- searches for the best weighted combination
- supports channel-wise fusion for valence and arousal separately
- evaluates combinations with CCC
- exports fused outputs and fusion metadata

Typical usage pattern:

- fit fusion weights on `train` or `val`
- apply the selected combination to `test`
- create the final fused submission

---

## 5. Minimal end-to-end command sequence

Below is the full order of operations.

### A. Audio pipeline

```bash
poetry install

python scripts/extract_audio.py scripts/extract_audio.yaml
python scripts/scan_fps.py scripts/scan_fps.yaml
python scripts/make_s2s_windows.py scripts/make_s2s_windows.yaml
python scripts/make_audio_windows.py scripts/make_audio_windows.yaml

chimera_ml train --config-path configs/cfg_va_s2s.yaml

chimera_ml eval \
  --config-path configs/cfg_test_w_audio.yaml \
  --checkpoint-path <path_to_audio_checkpoint>
```

### B. Multimodal pipeline

```bash
chimera_ml train --config-path configs/mm_cfg_va_s2s.yaml

chimera_ml eval \
  --config-path configs/test_mm_cfg_va_s2s.yaml \
  --checkpoint-path <path_to_multimodal_checkpoint>
```

### C. Late fusion

```bash
python scripts/predictions_fusion.py \
  --test_template_path data/ABAW_VA_test_set_example.txt \
  --root data/features/submit/ \
  --include submit_1_134f submit_2_14f submit_3_1 submit_4_134w \
  --max-comb-size 3 \
  --pair-grid-step 0.05 \
  --dirichlet-samples 8000 \
  --dirichlet-alpha 1.0 \
  --channelwise \
  --output-dir ./fusion_out_with_seeds \
  --split train
```
