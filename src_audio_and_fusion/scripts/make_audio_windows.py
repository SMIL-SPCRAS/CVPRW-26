from pathlib import Path
from typing import Dict, List, Tuple, Any, Sequence
import sys

import pandas as pd
import yaml


def load_cfg(path: Path) -> Dict[str, Any]:
    """Load YAML config into a plain dictionary."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return data


def _rle_runs(x: List[bool]) -> List[Tuple[bool, int, int]]:
    """Run-length encoding: (value, start, end_exclusive)."""
    if not x:
        return []
    out: List[Tuple[bool, int, int]] = []
    cur, s = x[0], 0
    for i in range(1, len(x)):
        if x[i] != cur:
            out.append((cur, s, i))
            cur, s = x[i], i
    out.append((cur, s, len(x)))
    return out


def smooth_open_flags(flags: List[bool], fps: float, min_open_sec: float, max_gap_sec: float) -> List[bool]:
    """Time-aware smoothing of mouth_open flags."""
    if not flags:
        return flags

    max_gap_frames = max(1, int(round(max_gap_sec * fps)))
    min_open_frames = max(1, int(round(min_open_sec * fps)))

    y = flags[:]

    # Fill short False gaps between True segments
    for val, s, e in _rle_runs(y):
        if (val is False) and (e - s) <= max_gap_frames:
            left_open = (s > 0 and y[s - 1] is True)
            right_open = (e < len(y) and y[e] is True) if e < len(y) else False
            if left_open and right_open:
                for i in range(s, e):
                    y[i] = True

    # Remove short True blips
    for val, s, e in _rle_runs(y):
        if (val is True) and (e - s) <= min_open_frames:
            for i in range(s, e):
                y[i] = False

    return y


def load_fps_map(fps_csv: Path) -> Dict[str, float]:
    """Map full_video_name -> fps (expects columns: video_name,fps)."""
    df = pd.read_csv(fps_csv)
    return {str(r["video_name"]): float(r["fps"]) for _, r in df.iterrows()}


def load_openmouth_dense(openmouth_csv: Path) -> Tuple[List[bool], List[bool]]:
    """
    Build dense arrays indexed by absolute frame number:
      flags[f]   = mouth_open at frame f (missing frames -> False)
      present[f] = whether frame f exists in the CSV
    Handles frames like '00304' (starts late).
    """
    df = pd.read_csv(openmouth_csv)
    frames = df["frame"].astype(str).astype(int).to_numpy()
    mouth_open = df["mouth_open"].astype(int).to_numpy() != 0

    max_f = int(frames.max()) if frames.size else -1
    flags = [False] * (max_f + 1)
    present = [False] * (max_f + 1)

    for f, mo in zip(frames, mouth_open):
        if f >= 0:
            flags[int(f)] = bool(mo)
            present[int(f)] = True

    return flags, present


def openmouth_path_for_row(openmouth_dir: Path, video_name: str) -> Path:
    """
    video_name example:
      10-60-1280x720_right___240_360_0004.mp4

    We use prefix before '___' as the openmouth CSV stem:
      openmouth/10-60-1280x720_right.csv
    """
    prefix = str(video_name).split("___", 1)[0]
    path = openmouth_dir / f"{prefix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Openmouth CSV not found: {path}")
    return path


def main(argv: Sequence[str]) -> int:
    """Usage: python make_audio_windows.py config.yaml"""
    if len(argv) != 2:
        print("Usage: python make_audio_windows.py config.yaml", file=sys.stderr)
        return 2
    
    cfg = load_cfg(Path(argv[1]))

    labels_dir = Path(cfg["paths"]["labels_dir"])
    openmouth_dir = Path(cfg["paths"]["openmouth_dir"])
    fps_csv = Path(cfg["paths"]["fps_csv"])
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    min_open_sec = float(cfg["smoothing"]["min_open_sec"])
    max_gap_sec = float(cfg["smoothing"]["max_gap_sec"])
    min_open_ratio = float(cfg["window_filter"]["min_open_ratio"])
    min_coverage_ratio = float(cfg["window_filter"]["min_coverage_ratio"])

    fps_map = load_fps_map(fps_csv)

    # Cache per openmouth CSV path (since multiple windows refer to the same track)
    flags_cache: Dict[str, List[bool]] = {}
    present_cache: Dict[str, List[bool]] = {}

    for split_csv in sorted(labels_dir.glob("*_segment.csv")):
        df = pd.read_csv(split_csv)

        required = {"full_video_name", "video_name", "start_frame", "end_frame"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{split_csv} missing columns: {sorted(missing)}")

        open_secs: List[float] = []
        coverage_ratios: List[float] = []
        use_flags: List[bool] = []
        openmouth_files: List[str] = []
        seg_fps: List[float] = []

        for _, row in df.iterrows():
            full_video = str(row["full_video_name"])
            seg_video = str(row["video_name"])
            start_f = int(row["start_frame"])
            end_f = int(row["end_frame"])

            fps = fps_map.get(full_video)
            if fps is None:
                # If FPS is keyed by a different name, you can extend mapping logic here.
                open_secs.append(0.0)
                coverage_ratios.append(0.0)
                use_flags.append(False)
                openmouth_files.append("missing_fps")
                seg_fps.append(0.0)
                continue

            om_path = openmouth_path_for_row(openmouth_dir, seg_video)
            om_key = str(om_path)

            if om_key not in flags_cache:
                flags, present = load_openmouth_dense(om_path)
                flags = smooth_open_flags(flags, fps, min_open_sec, max_gap_sec)
                flags_cache[om_key] = flags
                present_cache[om_key] = present

            flags = flags_cache[om_key]
            present = present_cache[om_key]

            # Window bounds; if window exceeds available openmouth length -> missing part treated as not present/closed
            s = max(0, start_f)
            e = max(0, end_f)
            if e <= s:
                open_secs.append(0.0)
                coverage_ratios.append(0.0)
                use_flags.append(False)
                openmouth_files.append(om_path.name)
                seg_fps.append(0.0)
                continue

            max_len = len(flags)
            inter_s = min(s, max_len)
            inter_e = min(e, max_len)

            if inter_e <= inter_s:
                cov = 0.0
                open_sec = 0.0
            else:
                seg_present = present[inter_s:inter_e]
                seg_flags = flags[inter_s:inter_e]
                cov = sum(seg_present) / float(e - s)          # coverage over full window length
                open_sec = sum(seg_flags) / float(fps)         # open time (seconds)

            has_time = ("start_time" in df.columns) and ("end_time" in df.columns)

            if has_time:
                window_len_sec = float(row["end_time"]) - float(row["start_time"])
            else:
                window_len_sec = (e - s) / float(fps)

            need_open_sec = min_open_ratio * max(0.0, window_len_sec)
            use_for_audio = (open_sec >= need_open_sec) and (cov >= min_coverage_ratio)

            open_secs.append(float(open_sec))
            coverage_ratios.append(float(cov))
            use_flags.append(bool(use_for_audio))
            openmouth_files.append(om_path.name)
            seg_fps.append(fps)

        df["open_sec"] = open_secs
        df["coverage_ratio"] = coverage_ratios
        df["use_for_audio"] = use_flags
        df["fps"] = seg_fps

        out_path = out_dir / split_csv.name.replace("_segment", "_audio_segment")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} | kept {sum(use_flags)}/{len(use_flags)}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
