from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

from audio.data.augmentations import *


class AbawAudioVADataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        wav_root: Path,
        sample_rate: int = 16000,
        window_size: float = 4.0,
        filter_non_speech: bool = True,
        labeled: Optional[bool] = True,
        split: str = "train",
        augment: bool = False,
        augment_params: Dict[str, float] = {},
        s2s: bool = False,
        s2s_steps: int = 4,
    ) -> None:
        """ABAW audio dataset for Valence/Arousal.

        Args:
            csv_path: Path to a CSV with at least `video_name` (and optionally labels).
            wav_root: Root directory containing wav files.
        """
        df = pd.read_csv(csv_path)

        if filter_non_speech:
            if "use_for_audio" not in df.columns:
                raise ValueError("CSV must contain 'use_for_audio' to filter.")
            df = df[df["use_for_audio"].astype(bool)]

        self.s2s = s2s
        self.s2s_steps = s2s_steps

        self._seq_cols = []
        for i in range(self.s2s_steps):
            self._seq_cols += [f"valence_s{i}", f"arousal_s{i}"]

        has_seq_cols = all(c in df.columns for c in self._seq_cols)
        has_one_cols = ("valence" in df.columns) and ("arousal" in df.columns)

        if labeled is None:
            labeled = has_seq_cols if self.s2s else has_one_cols
        
        self.labeled = bool(labeled)

        if self.labeled:
            if self.s2s and not has_seq_cols:
                raise ValueError("s2s=true, but valence_s*/arousal_s* columns are missing")
            if (not self.s2s) and not has_one_cols:
                raise ValueError("s2s=false, but valence/arousal columns are missing")

        if self.labeled:
            if self.s2s:
                t = df[self._seq_cols].astype(float).to_numpy().reshape(-1, self.s2s_steps, 2)
                valid_sec = (t[:, :, 0] != -5.0) & (t[:, :, 1] != -5.0) & np.isfinite(t).all(axis=2)
                # df = df[valid_sec.any(axis=1)] # 1 / 4
                # df = df[valid_sec.all(axis=1)] # all valid
                
                valid_cnt = valid_sec.sum(axis=1)
                df = df[valid_cnt >= math.ceil(self.s2s_steps / 2)] # 2 / 4 or 4 / 8
            else:
                df = df[
                    (df["valence"].astype(float) != -5.0)
                    & (df["arousal"].astype(float) != -5.0)
                ]

        self.df = df.reset_index(drop=True)
        self.wav_root = wav_root
        self.sample_rate = int(sample_rate)
        self.window_size = float(window_size)

        self.target_len = int(round(self.sample_rate * self.window_size))

        self.split = split
        self.augment = augment
        self.augment_params = augment_params

    def __len__(self) -> int:
        return len(self.df)

    def _get_wav_path(self, segment_video_name: str) -> Path:
        """Convert a segment video filename (mp4/avi/mov/...) to corresponding wav filename."""
        segment_p = Path(segment_video_name)
        wav_name = f"{segment_p.stem}.wav"
        return self.wav_root / str(segment_p).split("___")[0] / wav_name

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load wav -> mono -> resample -> pad/crop to fixed length. Returns [T]."""
        wav, sr = torchaudio.load(str(path))  # [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        x = wav.squeeze(0)  # [T]
        t = x.numel()
        if t < self.target_len:
            t_eff = t
            x = torch.nn.functional.pad(x, (0, self.target_len - t))
        else:
            # if longer, crop, and valid length becomes target_len
            t_eff = self.target_len
            x = x[: self.target_len]

        return x, int(t_eff)
    
    def _apply_augmentations(self, audio: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.augment_params["aug_gain_prob"]:
            audio = apply_gain(audio, 
                               self.augment_params["aug_gain_db_min"], 
                               self.augment_params["aug_gain_db_max"])

        if torch.rand(1).item() < self.augment_params["aug_noise_prob"]:
            audio = add_noise_snr(audio, 
                                  self.augment_params["aug_snr_db_min"], 
                                  self.augment_params["aug_snr_db_max"])

        if torch.rand(1).item() < self.augment_params["aug_band_prob"]:
            audio = simple_bandlimit(audio)

        if torch.rand(1).item() < self.augment_params["aug_reverb_prob"]:
            audio = simple_reverb(audio, sr=self.sample_rate)

        # safety clip
        audio = audio.clamp(-1.0, 1.0)
        return audio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        segment_name = str(row["video_name"])
        wav_path = self._get_wav_path(segment_name)
        audio, audio_len = self._load_audio(wav_path)

        if self.augment and self.split == "train":
            audio = self._apply_augmentations(audio)

        inputs = {
            "audio": audio,
        }

        meta: Dict[str, Any] = {
            "segment_name": segment_name,
            "full_video_name": str(row.get("full_video_name")),
            "wav_path": str(wav_path),
            "start_frame": int(row.get("start_frame", 0)) if not pd.isna(row.get("start_frame", 0)) else 0,
            "end_frame": int(row.get("end_frame", 0)) if not pd.isna(row.get("end_frame", 0)) else 0,
            "open_sec": float(row.get("open_sec", 0.0)) if not pd.isna(row.get("open_sec", 0.0)) else 0.0,
            "coverage_ratio": float(row.get("coverage_ratio", 0.0)) if not pd.isna(row.get("coverage_ratio", 0.0)) else 0.0,
            "fps": float(row.get("fps")) if ("fps" in self.df.columns and not pd.isna(row.get("fps"))) else None,
            "audio_len": torch.tensor(audio_len, dtype=torch.long),
        }

        target = None
        if self.s2s:
            arr = np.array([float(row[c]) for c in self._seq_cols], dtype=np.float32).reshape(self.s2s_steps, 2)
            target = torch.from_numpy(arr)  # [S,2]
        else:
            target = torch.tensor([float(row["valence"]), float(row["arousal"])], dtype=torch.float32)  # [2]

        return {"inputs": inputs, "target": target, "meta": meta}
    