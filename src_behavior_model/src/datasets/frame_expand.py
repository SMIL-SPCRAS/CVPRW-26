from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def expand_segment_predictions_to_frames(
    df: pd.DataFrame,
    group_col: str,
    start_col: str = "start_frame",
    end_col: str = "end_frame",
    val_col: str = "valence",
    aro_col: str = "arousal",
    output_video_col: str = "video_name",
) -> pd.DataFrame:
    required = [group_col, start_col, end_col, val_col, aro_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for frame expansion: {missing}")

    rows: List[pd.DataFrame] = []

    for group_value, grp in df.groupby(group_col, sort=False):
        grp = grp.copy()
        grp[start_col] = grp[start_col].astype(int)
        grp[end_col] = grp[end_col].astype(int)

        max_end = int(grp[end_col].max())
        if max_end <= 0:
            continue

        # Frame GT tables in this project are 1-based. A segment [start_frame, end_frame]
        # covers frames start_frame + 1 ... end_frame.
        val_sum = np.zeros(max_end + 1, dtype=np.float64)
        aro_sum = np.zeros(max_end + 1, dtype=np.float64)
        count = np.zeros(max_end + 1, dtype=np.int32)

        for _, r in grp.iterrows():
            start = int(r[start_col])
            end = int(r[end_col])
            if end <= start:
                continue
            left = max(1, start + 1)
            right = min(end + 1, max_end + 1)
            if right <= left:
                continue

            v = float(r[val_col])
            a = float(r[aro_col])
            val_sum[left:right] += v
            aro_sum[left:right] += a
            count[left:right] += 1

        covered = np.nonzero(count > 0)[0]
        if covered.size == 0:
            continue

        frame_df = pd.DataFrame(
            {
                output_video_col: [group_value] * int(covered.size),
                "frame_idx": covered,
                val_col: val_sum[covered] / count[covered],
                aro_col: aro_sum[covered] / count[covered],
            }
        )
        rows.append(frame_df)

    if not rows:
        return pd.DataFrame(columns=[output_video_col, "frame_idx", val_col, aro_col])
    return pd.concat(rows, ignore_index=True)
