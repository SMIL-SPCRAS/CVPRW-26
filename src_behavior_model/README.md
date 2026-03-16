# src_behavior_model

Training pipeline for valence/arousal prediction from precomputed behavior embeddings.
By default, the system loads segment-level Qwen embeddings from cached pickle/PT files, trains a regression model on top of them, evaluates on validation data, and saves the best checkpoint and training history.

## Entry Point

Run training from the repository root:

```bash
python src_behavior_model/main.py
```

Main config:

- `src_behavior_model/configs/text_va.toml`

## Config

Sections in `configs/text_va.toml`:

- `[pipeline]` - training mode: `segment` or `model`
- `[search]` - single run or hyperparameter search
- `[embeddings]` - embedding source and cache settings
- `[frame_eval]` - frame-level validation settings
- `[train]` - data paths and main training parameters
- `[model]` - sequence-model settings used when `pipeline.mode="model"`
- `[predict]` - kept only for config compatibility, not used by this training-only entrypoint

## Notes

- The default setup uses `embeddings.source = "qwen"`.
- The current default pipeline is `pipeline.mode = "model"` with `model_arch = "mamba"`.
- Relative dataset paths in the config should point to your local `dataset/` and `features/` folders.
