from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import pickle
import torch
from torch.utils.data import Dataset


class AbawMMDataset(Dataset):
    def __init__(
        self,
        *,
        audio_windows_pkl: Path,
        audio_emb_size: int,
        frame_modalities: Mapping[str, Path],
        modalities_emb_sizes: Mapping[str, int],
        use_predictions: bool = True,
        target_source_modality: str = "face",
        target_source_field: str = "label",
        frame_index_offset: int = 1,
        labeled: bool = True,

        # audio masking using open-mouth meta
        mask_audio: bool = False,
        audio_min_open_sec: float = 1.0,
        audio_min_coverage_ratio: float = 0.8,
    ):
        self.audio_windows = self._load_pickle(Path(audio_windows_pkl))
        self.audio_emb_size = int(audio_emb_size)
        self.window_keys = sorted(self.audio_windows.keys())

        self.use_predictions = bool(use_predictions)
        self.frame_index_offset = int(frame_index_offset)
        self.labeled = bool(labeled)

        self.target_source_modality = str(target_source_modality)
        self.target_source_field = str(target_source_field)

        self.mask_audio = bool(mask_audio)
        self.audio_min_open_sec = float(audio_min_open_sec)
        self.audio_min_coverage_ratio = float(audio_min_coverage_ratio)

        self.frame_modalities = dict(frame_modalities)
        self.modalities_emb_sizes = {k: int(v) for k, v in modalities_emb_sizes.items()}
        self.mod_data = {m: self._load_pickle(Path(p)) for m, p in self.frame_modalities.items()}

        if self.target_source_modality not in self.mod_data:
            raise ValueError(
                f"target_source_modality='{self.target_source_modality}' must exist in frame_modalities. "
                f"Have: {list(self.mod_data.keys())}"
            )

    def __len__(self) -> int:
        return len(self.window_keys)

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        with path.open("rb") as f:
            return pickle.load(f)

    def _frame_key(self, vid: str, fr: int) -> str:
        return f"{vid}/{int(fr + self.frame_index_offset):05d}.jpg"

    @staticmethod
    def _make_sequence_mask(length: int) -> torch.Tensor:
        return torch.ones((max(int(length), 1),), dtype=torch.bool)

    def _audio_om_filter_pass(self, rec: dict) -> bool:
        if not self.mask_audio:
            return True
        
        meta = rec.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}

        open_sec = float(meta.get("open_sec", 0.0) or 0.0)
        coverage_ratio = float(meta.get("coverage_ratio", 0.0) or 0.0)

        return (
            open_sec >= self.audio_min_open_sec
            and coverage_ratio >= self.audio_min_coverage_ratio
        )
    
    def _extract_audio_inputs(
        self,
        rec: dict,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns:
            inputs_part:
                audio_embedding:  [T_a, D_a]
                audio_prediction: [T_a, 2]        if use_predictions=True

            masks_part:
                sequence_mask:  [T_a]
                valid_mask:     [T_a]
                om_filter_mask: [T_a]
        """
        emb = rec.get("embedding", None)
        if emb is None:
            raise ValueError("Audio record does not contain 'embedding'")

        audio_emb = torch.as_tensor(emb, dtype=torch.float32)
        if audio_emb.ndim != 2:
            raise ValueError(
                f"Audio embedding must have shape [T_a, D_a], got {tuple(audio_emb.shape)}"
            )
        
        if audio_emb.shape[1] != self.audio_emb_size:
            raise ValueError(
                f"Audio embedding dim mismatch: expected {self.audio_emb_size}, "
                f"got {audio_emb.shape[1]}"
            )

        T_a = int(audio_emb.shape[0])
        sequence_mask = self._make_sequence_mask(T_a)

        om_pass = self._audio_om_filter_pass(rec)
        om_filter_mask = torch.full((T_a,), fill_value=om_pass, dtype=torch.bool)

        valid_mask = sequence_mask & om_filter_mask

        inputs_part: Dict[str, torch.Tensor] = {
            "audio_embedding": audio_emb,
        }

        if self.use_predictions:
            pr = rec.get("prediction", None)
            if pr is None:
                raise ValueError(
                    "use_predictions=True, but audio record does not contain 'prediction'"
                )

            audio_pr = torch.as_tensor(pr, dtype=torch.float32)
            if audio_pr.ndim != 2 or audio_pr.shape[0] != T_a or audio_pr.shape[1] != 2:
                raise ValueError(
                    f"Audio prediction must have shape [T_a, 2], got {tuple(audio_pr.shape)}"
                )

            inputs_part["audio_prediction"] = audio_pr

        masks_part = {
            "sequence_mask": sequence_mask,
            "valid_mask": valid_mask,
            "om_filter_mask": om_filter_mask,
        }

        return inputs_part, masks_part
    
    def _extract_targets(
        self,
        *,
        vid: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tgt_data = self.mod_data[self.target_source_modality]

        y_list = []
        valid_list = []

        for fr in range(start_frame, end_frame):
            fk = self._frame_key(vid, fr)
            item = tgt_data.get(fk, None)

            if (not self.labeled) or item is None or (self.target_source_field not in item):
                y_list.append(torch.tensor([-5.0, -5.0], dtype=torch.float32))
                valid_list.append(False)
                continue

            lab = torch.as_tensor(item[self.target_source_field], dtype=torch.float32).view(-1)
            if lab.numel() != 2:
                raise ValueError(f"Bad label shape at {fk}: got {tuple(lab.shape)}")

            is_valid = bool((lab[0].item() != -5.0) and (lab[1].item() != -5.0))
            y_list.append(lab)
            valid_list.append(is_valid)

        if len(y_list) == 0: # no frames
            y = torch.full((1, 2), -5.0, dtype=torch.float32)
            valid = torch.zeros((1,), dtype=torch.bool)
        else:
            y = torch.stack(y_list, dim=0)                  # [L, 2]
            valid = torch.tensor(valid_list, dtype=torch.bool)  # [L]

        return y, valid

    def _extract_frame_modality_inputs(
        self,
        *,
        mod: str,
        vid: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns:
            inputs_part:
                f"{mod}_embedding":  [L, D]
                f"{mod}_prediction": [L, 2]      if use_predictions=True

            masks_part:
                sequence_mask: [L]
                valid_mask:    [L]
        """
        data = self.mod_data[mod]
        D_emb = self.modalities_emb_sizes[mod]

        L = max(int(end_frame - start_frame), 1)

        emb_seq = torch.zeros((L, D_emb), dtype=torch.float32)
        valid_mask = torch.zeros((L,), dtype=torch.bool)

        if self.use_predictions:
            pr_seq = torch.zeros((L, 2), dtype=torch.float32)

        for j, fr in enumerate(range(start_frame, end_frame)):
            fk = self._frame_key(vid, fr)
            item = data.get(fk, None)
            if item is None:
                continue

            emb = item.get("embedding", None)
            if emb is None:
                continue

            v = torch.as_tensor(emb, dtype=torch.float32).view(-1)
            if v.numel() != D_emb:
                raise ValueError(
                    f"{mod} embedding dim mismatch at {fk}: expected {D_emb}, got {v.numel()}"
                )

            emb_seq[j] = v
            valid_mask[j] = True

            if self.use_predictions:
                pr = item.get("prediction", None)
                if pr is None:
                    raise ValueError(
                        f"use_predictions=True, but {mod} record at {fk} has no 'prediction'"
                    )

                p = torch.as_tensor(pr, dtype=torch.float32).view(-1)
                if p.numel() != 2:
                    raise ValueError(
                        f"{mod} prediction dim mismatch at {fk}: expected 2, got {p.numel()}"
                    )
                pr_seq[j] = p

        sequence_mask = self._make_sequence_mask(L)

        inputs_part: Dict[str, torch.Tensor] = {
            f"{mod}_embedding": emb_seq,
        }
        if self.use_predictions:
            inputs_part[f"{mod}_prediction"] = pr_seq

        masks_part = {
            "sequence_mask": sequence_mask,
            "valid_mask": valid_mask,
        }

        return inputs_part, masks_part

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win_key = self.window_keys[idx]
        rec = self.audio_windows[win_key]

        vid = str(rec.get("vid", "") or str(win_key).split("___", 1)[0])
        start_frame = int(rec.get("use_start_frame", rec.get("start_frame", 0)) or 0)
        end_frame = int(rec.get("use_end_frame", rec.get("end_frame", 0)) or 0)

        inputs: Dict[str, torch.Tensor] = {}
        audio_meta_raw = rec.get("meta", {})
        audio_meta = dict(audio_meta_raw) if isinstance(audio_meta_raw, dict) else {}
        audio_meta.pop("audio_len", None)

        meta: Dict[str, Any] = {
            "window_key": win_key,
            "vid": vid,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "audio_meta": audio_meta,
            "masks": {
                "targets_sequence_mask": None,
                "targets_valid_mask": None,
                "input_sequence_mask": {},
                "input_valid_mask": {},
                "audio_om_filter_mask": None,
            },
        }
        # audio
        audio_inputs, audio_masks = self._extract_audio_inputs(rec)
        inputs.update(audio_inputs)

        meta["masks"]["input_sequence_mask"]["audio"] = audio_masks["sequence_mask"]
        meta["masks"]["input_valid_mask"]["audio"] = audio_masks["valid_mask"]
        meta["masks"]["audio_om_filter_mask"] = audio_masks["om_filter_mask"]

        # targets
        target, targets_valid_mask = self._extract_targets(
            vid=vid,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        targets_sequence_mask = self._make_sequence_mask(target.shape[0])

        meta["masks"]["targets_sequence_mask"] = targets_sequence_mask
        meta["masks"]["targets_valid_mask"] = targets_valid_mask

        # frame-modalities input
        for mod in self.mod_data.keys():
            mod_inputs, mod_masks = self._extract_frame_modality_inputs(
                mod=mod,
                vid=vid,
                start_frame=start_frame,
                end_frame=end_frame,
            )

            inputs.update(mod_inputs)

            meta["masks"]["input_sequence_mask"][mod] = mod_masks["sequence_mask"]
            meta["masks"]["input_valid_mask"][mod] = mod_masks["valid_mask"]

        return {"inputs": inputs, "target": target, "meta": meta}