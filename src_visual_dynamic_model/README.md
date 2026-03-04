# src_visual_dynamic_model

Minimal project for dynamic visual VA modeling with cached frame features.

## What is included

- AffWild2 index building from `Train_Set` and `Validation_Set` TXT annotations.
- Cached feature loading from `.npz` (`frames`, `features`).
- Temporal model `VisualDynamicModel`.
- CCC-based training and validation metrics (`ccc_v`, `ccc_a`, `va_score`).
- Best-checkpoint selection by validation `va_score`.
- Optional GRADA feature extraction from cropped faces.
- Export of PKL records and test TXT predictions.

## Project files

- `dataset.py`: index building, windowing, and dataset.
- `model.py`: `VisualDynamicModel`.
- `losses.py`: CCC loss.
- `metrics.py`: CCC metric.
- `train.py`: training + validation.
- `extract_grada_features.py`: feature extraction to `.npz`.
- `export_pkls_from_best.py`: export `train/val/test` PKL and test annotation-style TXT.
- `generate_test_submission.py`: flat test file (`image_location,valence,arousal`).
- `config_best.toml`: example config.

## Setup

1. Open `config_best.toml` and set your paths:
- path to annotations root
- path to frames root
- path to test sample file
- path to features cache root
- path to output directory
- path to GRADA repo and GRADA weights

2. Install dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

## Run pipeline

1. Extract features (if cache is missing):

```powershell
python extract_grada_features.py --config <path_to_config.toml> --split all
```

Quick debug run:

```powershell
python extract_grada_features.py --config <path_to_config.toml> --split val --max-videos 2 --max-frames-per-video 200
```

2. Train:

```powershell
python train.py --config <path_to_config.toml>
```

Main outputs are under `<path_to_output_dir>/<run_id>/`:
- `checkpoints/best_<run_id>_va*.pt`
- `logs/metrics_<run_id>.csv`
- `logs/config_<run_id>.json`

3. Export PKL records and test TXT from the best checkpoint:

```powershell
python export_pkls_from_best.py `
  --config <path_to_config.toml> `
  --checkpoint <path_to_best_checkpoint.pt> `
  --output-dir <path_to_export_dir>
```

Exported files:
- `train.pkl`
- `val.pkl`
- `test.pkl`
- `test_predictions_txt/Test_Set/*.txt`
- `summary.json`

4. Generate flat test submission file:

```powershell
python generate_test_submission.py `
  --config <path_to_config.toml> `
  --checkpoint <path_to_best_checkpoint.pt> `
  --sample-file <path_to_test_sample.txt> `
  --output-path <path_to_output_txt>
```

Output format:
- header: `image_location,valence,arousal`
- same row count and order as sample file

## Grad-CAM visualization

Single target frame:

```powershell
python visualize_gradcam.py `
  --config <path_to_config.toml> `
  --checkpoint <path_to_best_checkpoint.pt> `
  --split val `
  --video <video_name> `
  --target-frame 300 `
  --target-output valence
```

Uniform targets example:

```powershell
python visualize_gradcam.py `
  --config <path_to_config.toml> `
  --checkpoint <path_to_best_checkpoint.pt> `
  --split val `
  --video <video_name> `
  --uniform-targets 10 `
  --target-output both
```

Saved artifacts under `<path_to_output_dir>/gradcam/<split>_<video>_<timestamp>/`:
- `frames/*.png`
- `frames_temporal_weighted/*.png`
- `frame_scores.csv`
- `runs_summary.csv`
- `summary.json`
