from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional
import multiprocessing as mp
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


def iter_videos(root: Path, exts: Sequence[str]) -> Iterable[Path]:
    """Yield video files under root matching the given extensions."""
    norm_exts = {("." + e.lower().lstrip(".")) for e in exts}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in norm_exts:
            yield p


def to_wav_path(input_root: Path, output_root: Path, video_path: Path) -> Path:
    """Map input_root/.../file.ext -> output_root/.../file.wav."""
    rel = video_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".wav")


def extract_wav(
    ffmpeg_bin: str,
    src: Path,
    dst: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
) -> None:
    """Run ffmpeg to extract audio track to WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(dst),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _worker(job: Tuple[str, Path, Path, int, int, bool]) -> Tuple[str, bool, Optional[str]]:
    """
    Worker: process exactly one file.

    Returns:
      (status, did_work, error_message)
      status: "ok" | "skip" | "fail"
      did_work: True if ffmpeg was run, False if skipped
    """
    ffmpeg_bin, src, dst, sample_rate, channels, overwrite = job

    if dst.exists() and not overwrite:
        return ("skip", False, None)

    try:
        extract_wav(ffmpeg_bin, src, dst, sample_rate, channels, overwrite)
        return ("ok", True, None)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")
        return ("fail", True, err)


def main(argv: Sequence[str]) -> int:
    """Usage: python extract_audio.py config.yaml"""
    if len(argv) != 2:
        print("Usage: python extract_audio.py config.yaml", file=sys.stderr)
        return 2

    cfg = load_cfg(Path(argv[1]))

    input_root = Path(str(cfg.get("input_root", "data")))
    output_root = Path(str(cfg.get("output_root", "wav_data")))

    ffmpeg_bin = str(cfg.get("ffmpeg_bin", "ffmpeg"))
    sample_rate = int(cfg.get("sample_rate", 16000))
    channels = int(cfg.get("channels", 1))

    video_exts = cfg.get("video_exts", ["mp4"])
    if not isinstance(video_exts, list) or not all(isinstance(x, str) for x in video_exts):
        raise ValueError("video_exts must be a list of strings, e.g. [mp4, mkv].")

    overwrite = bool(cfg.get("overwrite", False))
    workers = int(cfg.get("workers", 4))

    if not input_root.exists():
        print(f"Input root not found: {input_root}", file=sys.stderr)
        return 2

    videos = list(iter_videos(input_root, video_exts))
    jobs: List[Tuple[str, Path, Path, int, int, bool]] = [
        (ffmpeg_bin, v, to_wav_path(input_root, output_root, v), sample_rate, channels, overwrite)
        for v in videos
    ]

    n_ok = 0
    n_skip = 0
    n_fail = 0

    if workers <= 1:
        iterator = map(_worker, jobs)
    else:
        # spawn is safer across platforms (esp. macOS/Windows)
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=workers)
        iterator = pool.imap_unordered(_worker, jobs, chunksize=8)

    try:
        for status, _did_work, err in tqdm(iterator, total=len(jobs), desc="Extracting audio", unit="file"):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                if err:
                    tqdm.write(err.strip())
    finally:
        # Close pool if it was created
        if workers > 1:
            pool.close()
            pool.join()

    print(f"Done. ok={n_ok} skip={n_skip} fail={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
