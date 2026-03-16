from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import COLLATES


def _pad_tensor_first_dim(
    x: torch.Tensor,
    target_len: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pads tensor x along dim=0 up to target_len.
    Works for [T], [T, D], [T, ...].
    """
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        raise ValueError(f"Cannot pad tensor: current len={x.shape[0]} > target_len={target_len}")

    pad_shape = (target_len - x.shape[0],) + tuple(x.shape[1:])
    pad = torch.full(
        pad_shape,
        fill_value=pad_value,
        dtype=x.dtype
    )

    return torch.cat([x, pad], dim=0)


def _stack_padded_tensors(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pads a list of tensors along dim=0 to max length and stacks to [B, T_max, ...]
    or [B, T_max] for 1D tensors.
    """
    if len(tensors) == 0:
        raise ValueError("Empty tensor list")

    max_len = max(int(t.shape[0]) for t in tensors)
    padded = [_pad_tensor_first_dim(t, max_len, pad_value=pad_value) for t in tensors]
    return torch.stack(padded, dim=0)


def _batch_pad_optional_sequence_map(
    seqs: List[Optional[torch.Tensor]],
    *,
    pad_value: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    seqs: list of tensors [T] or None
    returns: stacked [B, T_max]
    """
    example = next((x for x in seqs if x is not None), None)
    if example is None:
        raise ValueError("All sequences are None; cannot infer length/device")

    filled = []
    for x in seqs:
        if x is None:
            x = torch.empty((0,), dtype=dtype)
        filled.append(x)

    return _stack_padded_tensors(filled, pad_value=pad_value)


def _batch_pad_optional_input_tensors(
    tensors: List[Optional[torch.Tensor]],
) -> torch.Tensor:
    """
    tensors: list of [T, D] or [T] tensors or None
    returns: padded+stacked [B, T_max, ...]
    """
    example = next((x for x in tensors if x is not None), None)
    if example is None:
        raise ValueError("All tensors are None; cannot infer shape")

    shape_tail = tuple(example.shape[1:])
    dtype = example.dtype

    prepared = []
    for x in tensors:
        if x is None:
            x = torch.empty((0,) + shape_tail, dtype=dtype)
        prepared.append(x)

    return _stack_padded_tensors(prepared, pad_value=0.0)


@dataclass
class VariableLengthMaskingCollate:
    """
    Collate for variable-length multimodal sequences with masks.

    Expected sample format:
    {
        "inputs": {
            "audio_embedding":  [T_a, D_a],
            "audio_prediction": [T_a, 2],   # optional
            "face_embedding":   [L, D_f],
            "face_prediction":  [L, 2],     # optional
            ...
        },
        "target": [L_tgt, 2],
        "meta": {
            "masks": {
                "targets_sequence_mask": [L_tgt],
                "targets_valid_mask":    [L_tgt],
                "input_sequence_mask": {
                    "audio": [T_a],
                    "face":  [L],
                    ...
                },
                "input_valid_mask": {
                    "audio": [T_a],
                    "face":  [L],
                    ...
                },
                "audio_om_filter_mask": [T_a],
            }
        }
    }

    Output meta:
    {
        "present_modalities": [...],          # union of input keys
        "mask": {input_key: FloatTensor[B]},  # batch-level presence, legacy-compatible
        "sample_meta": [...],                 # original per-sample meta
        "masks": {
            "targets_sequence_mask": BoolTensor[B, T_tgt_max],
            "targets_valid_mask":    BoolTensor[B, T_tgt_max],
            "input_sequence_mask": {
                "audio": BoolTensor[B, T_audio_max],
                "face":  BoolTensor[B, T_face_max],
                ...
            },
            "input_valid_mask": {
                "audio": BoolTensor[B, T_audio_max],
                "face":  BoolTensor[B, T_face_max],
                ...
            },
            "audio_om_filter_mask": BoolTensor[B, T_audio_max],
        }
    }
    """

    require_at_least_one_modality: bool = True

    def __call__(self, batch: List[Dict[str, Any]]):
        if len(batch) == 0:
            raise ValueError("Empty batch")

        inputs_list = [b["inputs"] for b in batch]
        targets_list = [b["target"] for b in batch]
        sample_metas = [b.get("meta", {}) for b in batch]

        all_input_keys = sorted(set().union(*[d.keys() for d in inputs_list]))
        if self.require_at_least_one_modality and not all_input_keys:
            raise ValueError("No modalities found in any sample of the batch.")

        # inputs
        inputs: Dict[str, torch.Tensor] = {}
        batch_presence_mask: Dict[str, torch.Tensor] = {}

        for key in all_input_keys:
            example = next((d[key] for d in inputs_list if key in d), None)
            if example is None:
                continue

            tensors_for_key: List[Optional[torch.Tensor]] = []
            present_flags: List[float] = []

            shape_tail = tuple(example.shape[1:])
            dtype = example.dtype

            for d in inputs_list:
                if key in d:
                    tensors_for_key.append(d[key])
                    present_flags.append(1.0)
                else:
                    tensors_for_key.append(
                        torch.empty((0,) + shape_tail, dtype=dtype)
                    )
                    present_flags.append(0.0)

            inputs[key] = _stack_padded_tensors(tensors_for_key, pad_value=0.0)
            batch_presence_mask[key] = torch.tensor(present_flags, dtype=torch.float32)

        # targets
        targets = _stack_padded_tensors(targets_list, pad_value=0.0)

        # masks from meta
        collated_masks: Dict[str, Any] = {
            "targets_sequence_mask": None,
            "targets_valid_mask": None,
            "input_sequence_mask": {},
            "input_valid_mask": {},
            "audio_om_filter_mask": None,
        }

        # target masks
        tgt_seq_masks = []
        tgt_valid_masks = []
        for m in sample_metas:
            mm = m.get("masks", {})
            tgt_seq_masks.append(mm.get("targets_sequence_mask", None))
            tgt_valid_masks.append(mm.get("targets_valid_mask", None))

        collated_masks["targets_sequence_mask"] = _batch_pad_optional_sequence_map(
            tgt_seq_masks,
            pad_value=0,
            dtype=torch.bool,
        )
        collated_masks["targets_valid_mask"] = _batch_pad_optional_sequence_map(
            tgt_valid_masks,
            pad_value=0,
            dtype=torch.bool,
        )

        # infer modality names from per-sample masks
        input_mask_modalities = sorted(
            set().union(
                *[
                    set(m.get("masks", {}).get("input_sequence_mask", {}).keys())
                    for m in sample_metas
                ]
            )
        )

        for mod in input_mask_modalities:
            seqs = []
            vals = []

            for m in sample_metas:
                mm = m.get("masks", {})
                seqs.append(mm.get("input_sequence_mask", {}).get(mod, None))
                vals.append(mm.get("input_valid_mask", {}).get(mod, None))

            collated_masks["input_sequence_mask"][mod] = _batch_pad_optional_sequence_map(
                seqs,
                pad_value=0,
                dtype=torch.bool,
            )
            collated_masks["input_valid_mask"][mod] = _batch_pad_optional_sequence_map(
                vals,
                pad_value=0,
                dtype=torch.bool,
            )

        # audio om mask
        audio_om_masks = []
        for m in sample_metas:
            mm = m.get("masks", {})
            audio_om_masks.append(mm.get("audio_om_filter_mask", None))

        collated_masks["audio_om_filter_mask"] = _batch_pad_optional_sequence_map(
            audio_om_masks,
            pad_value=0,
            dtype=torch.bool,
        )

        meta = {
            "present_modalities": all_input_keys,
            "mask": batch_presence_mask,   # legacy-compatible field
            "sample_meta": sample_metas,
            "masks": collated_masks,
        }

        return Batch(inputs=inputs, targets=targets, meta=meta)


@COLLATES.register("variable_length_masking_collate")
def variable_length_masking_collate(**params):
    return VariableLengthMaskingCollate(**params)