from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_EXT_RE = re.compile(r"\.(mp4|avi)$", flags=re.IGNORECASE)
_SEGMENT_SUFFIX_RE = re.compile(r"___\d+_\d+_\d+$")


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Failed to read CSV {path}. Last error: {last_error}")


def ensure_columns(df: pd.DataFrame, required: List[str], csv_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_name}: {missing}")


def to_stream_name(value: str) -> str:
    text = str(value).strip()
    text = _EXT_RE.sub("", text)
    text = _SEGMENT_SUFFIX_RE.sub("", text)
    return text


def add_stream_name_column(df: pd.DataFrame, source_col: str = "video_name", target_col: str = "stream_name") -> pd.DataFrame:
    if source_col not in df.columns:
        return df
    out = df.copy()
    out[target_col] = out[source_col].astype(str).map(to_stream_name)
    return out


class TextRegressionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        has_targets: bool,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_targets = has_targets

        self.df["text"] = self.df["text"].fillna("").astype(str)
        self.video_names = self.df["video_name"].astype(str).tolist()
        self.texts = self.df["text"].tolist()

        self.targets = None
        if has_targets:
            self.targets = self.df[["valence", "arousal"]].astype(float).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "row_idx": idx,
            "video_name": self.video_names[idx],
        }
        if self.has_targets:
            item["labels"] = self.targets[idx]
        return item


def make_collate_fn(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict:
        text_inputs = [
            {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
            }
            for sample in batch
        ]
        padded = tokenizer.pad(
            text_inputs,
            padding=True,
            return_tensors="pt",
        )

        out = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "row_idx": torch.tensor([sample["row_idx"] for sample in batch], dtype=torch.long),
            "video_name": [sample["video_name"] for sample in batch],
        }
        if "labels" in batch[0]:
            labels_np = np.asarray([sample["labels"] for sample in batch], dtype=np.float32)
            out["labels"] = torch.from_numpy(labels_np)
        return out

    return collate_fn
