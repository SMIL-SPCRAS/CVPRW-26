from typing import Any, Dict, Optional
from pathlib import Path
from typing import Optional, Dict

from audio.data.abaw_va_dataset import AbawAudioVADataset
from chimera_ml.data.datamodule import DataModule
from chimera_ml.data.masking_collate import masking_collate
from chimera_ml.core.registry import DATAMODULES


@DATAMODULES.register("abaw_va_test_datamodule")
def abaw_va_datamodule(
    *,
    dataset_path: str,
    train_csv: str,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    filter_non_speech: bool = True,
    augment: bool = True,
    augment_params: Dict[str, float] = {},
    s2s: bool = False,
    s2s_steps: int = 4,
    **_,
) -> DataModule:
    """ABAW audio VA DataModule.

    Provides:
      - train loader from `train_csv` (optionally filtered by `use_for_audio`)
      - validation loaders (if `val_csv` is provided):
          * "val"      : filtered (use_for_audio=True) — original audio-focused validation
          * "val_full" : unfiltered (all windows)       — for frame-wise / alternative validation
      - optional test loader (if `test_csv` is provided), unfiltered by default

    Note: chimera_ml's base DataModule supports multi-loader validation/test
    when val_dataset/test_dataset are passed as a dict[str, Dataset].
    """

    dataset_path = Path(dataset_path)

    train_ds = AbawAudioVADataset(
        csv_path=train_csv, 
        wav_root=dataset_path / "train",
        filter_non_speech=False,
        labeled=True,
        split="train_full",
        augment=False,
        s2s=s2s,
        s2s_steps=s2s_steps
    )

    val_ds = None
    if val_csv:
        val_ds = AbawAudioVADataset(
                csv_path=val_csv,
                wav_root=dataset_path / "val",
                filter_non_speech=False,
                labeled=True,
                split="val_full",
                augment=False,
                s2s=s2s,
                s2s_steps=s2s_steps
            )

    test_ds = None
    if test_csv:
        test_ds = AbawAudioVADataset(
            csv_path=Path(test_csv),
            wav_root=dataset_path / "test",
            filter_non_speech=False,
            labeled=False,
            split="test_full",
            augment=False,
            s2s=s2s,
            s2s_steps=s2s_steps
        )

    return DataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=masking_collate(),
    )
