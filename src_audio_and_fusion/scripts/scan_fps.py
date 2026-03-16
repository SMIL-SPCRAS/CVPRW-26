from pathlib import Path
from typing import Optional, Dict, Any, Sequence
import csv
import subprocess
import sys

import yaml
from tqdm import tqdm


def load_cfg(path: Path) -> Dict[str, Any]:
    """Load YAML config into a plain dictionary."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return data


def fps(path: Path, ffprobe: str) -> Optional[float]:
    """Return FPS from ffprobe (avg_frame_rate)."""
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None
    if not out or out == "0/0":
        return None
    if "/" in out:
        n, d = out.split("/", 1)
        return None if float(d) == 0 else float(n) / float(d)
    return float(out)


def main(argv: Sequence[str]) -> int:
    """Usage: python scan_fps.py config.yaml"""
    if len(sys.argv) != 2:
        print("Usage: python scan_fps.py config.yaml", file=sys.stderr)
        return 2

    cfg = load_cfg(Path(argv[1]))

    root = Path(cfg["video_dir"])
    out_csv = Path(cfg["output_csv"])
    ffprobe = cfg.get("ffprobe_bin", "ffprobe")
    exts = {"." + e.lower().lstrip(".") for e in cfg.get("video_exts", [])}

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "fps"])
        for p in tqdm(files, desc="Scanning", unit="file"):
            v = fps(p, ffprobe)
            w.writerow([p.name, "" if v is None else round(v, 3)])

    print(f"Saved: {out_csv} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
