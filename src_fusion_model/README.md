# src_fusion_model

Fusion training project for AffWild2 VA from precomputed multimodal PKL features.

## Expected feature format

For each modality subfolder inside `features_root`:

- `train.pkl`
- `val.pkl`
- `test.pkl`

Each PKL must be a dict:

- key: `"video_name/00001.jpg"`
- value: `{"embedding": np.ndarray, "prediction": np.ndarray, "label": np.ndarray}`

## Configure

Edit `config_best.toml`:

- paths (`annotations_root`, `test_list_path`, `features_root`, `output_dir`)
- modality list (`modalities`)
- Q/K/V modality selection (`q_modality`, `k_modality`, `v_modality`)
- training hyperparameters

## Train

```powershell
python train.py --config <path_to_config.toml>
```

Outputs:

- best checkpoint: `output_dir/<run_id>/checkpoints/best_<run_id>_va*.pt`
- logs: `output_dir/<run_id>/logs/metrics_<run_id>.csv`
- validation per-video txt: `output_dir/<run_id>/best_val_predictions_txt/Validation_Set/*.txt`
- test common txt: `output_dir/<run_id>/test_predictions/va_test_predictions_<timestamp>.txt`

## Grid search

Configure grid in `search_params_fusion.toml`, then run:

```powershell
python search_train.py `
  --config <path_to_config.toml> `
  --search-config <path_to_search_params.toml>
```

Outputs:
- `output_dir/search_<timestamp>/search_results.csv`
- `output_dir/search_<timestamp>/best_trial.json`
- `output_dir/search_<timestamp>/trials/<run_id>/...`

## Test submission from checkpoint

```powershell
python generate_test_submission.py `
  --config <path_to_config.toml> `
  --checkpoint <path_to_best_checkpoint.pt> `
  --sample-file <path_to_test_sample.txt> `
  --output-path <path_to_output_txt>
```
