import json
import math
import os
import subprocess
import multiprocessing as mp
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_CONFIG = Path('scripts/make_s2s_windows.yaml')
ACTIVE_POOL = None


def load_yaml_config(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    current_list_key: Optional[str] = None
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '#' in line:
                line = line.split('#', 1)[0].strip()
            if not line:
                continue

            if line.startswith('- '):
                if current_list_key is None:
                    continue
                item = line[2:].strip()
                if item.startswith(("'", '"')) and item.endswith(("'", '"')):
                    item = item[1:-1]
                if item:
                    data[current_list_key].append(item)
                continue

            current_list_key = None
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if value == '':
                data[key] = []
                current_list_key = key
                continue
            if value.startswith('[') and value.endswith(']'):
                inner = value[1:-1].strip()
                if not inner:
                    data[key] = []
                else:
                    items = []
                    for part in inner.split(','):
                        item = part.strip()
                        if item.startswith(("'", '"')) and item.endswith(("'", '"')):
                            item = item[1:-1]
                        if item:
                            items.append(item)
                    data[key] = items
                continue
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                data[key] = value[1:-1]
                continue
            if value.lower() in ('true', 'false'):
                data[key] = value.lower() == 'true'
                continue
            try:
                if '.' in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
                continue
            except ValueError:
                data[key] = value
    return data


def slice_audio(
    start_time: float,
    end_time: float,
    win_max_length: float,
    win_shift: float,
    win_min_length: float,
) -> List[Dict[str, float]]:
    if end_time < start_time:
        return []
    if (end_time - start_time) > win_max_length:
        timings = []
        while start_time < end_time:
            end_time_chunk = start_time + win_max_length
            if end_time_chunk < end_time:
                timings.append({'start': start_time, 'end': end_time_chunk})
            elif end_time_chunk == end_time:
                timings.append({'start': start_time, 'end': end_time_chunk})
                break
            else:
                if end_time - start_time < win_min_length:
                    break
                timings.append({'start': start_time, 'end': end_time})
                break
            start_time += win_shift
        return timings
    return [{'start': start_time, 'end': end_time}]


def ffprobe_duration(path: Path) -> Optional[float]:
    cmd = [
        'ffprobe',
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=nw=1:nk=1',
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if res.returncode != 0:
        return None
    out = res.stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def get_video_info(path: Path) -> Tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if frame_count > 0:
        duration = frame_count / fps
    else:
        duration = ffprobe_duration(path)
        if duration is None:
            raise RuntimeError(f'Cannot determine duration for {path}')
    return float(fps), float(duration), int(frame_count)


def mean_with_missing(values: np.ndarray, missing: float) -> float:
    valid = values != missing
    if not np.any(valid):
        return float(missing)
    return float(np.mean(values[valid]))


def segment_mean(values: np.ndarray, start: int, end: int, missing: float) -> float:
    start = max(0, int(start))
    end = max(start, int(end))
    if start >= len(values):
        return float(missing)
    end = min(end, len(values))
    if end <= start:
        return float(missing)
    segment = values[start:end]
    return mean_with_missing(segment, missing)


def segment_mean_per_bins(
    values: np.ndarray,
    start_time: float,
    end_time: float,
    fps: float,
    missing: float,
    n_bins: int,
    win_seconds: float,
) -> List[float]:
    out: List[float] = []
    bin_len = win_seconds / n_bins

    for i in range(n_bins):
        t0 = start_time + i * bin_len
        t1 = start_time + (i + 1) * bin_len

        if t0 >= end_time:
            out.append(float(missing))
            continue

        t1 = min(t1, end_time)

        a = int(np.floor(t0 * fps))
        b = int(np.ceil(t1 * fps))

        if b <= a:
            b = a + 1

        out.append(segment_mean(values, a, b, missing))

    return out


def segment_video(
    input_path: Path,
    output_dir: Path,
    name_file: str,
    fps: float,
    duration: float,
    valence: np.ndarray,
    arousal: np.ndarray,
    missing_value: float,
    win_max_length: float,
    win_shift: float,
    win_min_length: float,
    s2s_frames: int,
    skip_existing: bool,
    show_progress: bool,
) -> Tuple[List[List[object]], List[object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = input_path.suffix
    metadata_segments: List[List[object]] = []

    timings = slice_audio(
        start_time=0,
        end_time=duration,
        win_max_length=win_max_length,
        win_shift=win_shift,
        win_min_length=win_min_length,
    )

    timing_iter = timings
    if show_progress:
        timing_iter = tqdm(timings, desc=f'segments {name_file}', leave=False)

    for segment_index, timing in enumerate(timing_iter):
        start_time = timing['start']
        end_time = timing['end']
        start_frame = int(np.round(start_time * fps))
        end_frame = int(np.round(end_time * fps))

        seg_vals = segment_mean_per_bins(
            valence, start_time, end_time, fps, missing_value,
            n_bins=s2s_frames, win_seconds=win_max_length
        )

        seg_ars = segment_mean_per_bins(
            arousal, start_time, end_time, fps, missing_value,
            n_bins=s2s_frames, win_seconds=win_max_length
        )
        
        seg_val = segment_mean(valence, start_frame, end_frame, missing_value)
        seg_ar = segment_mean(arousal, start_frame, end_frame, missing_value)

        new_name = f"{name_file}___{start_frame}_{end_frame}_{segment_index:04d}{ext}"
        output_path = output_dir / new_name

        row = [
            input_path.name,
            new_name,
            '',
            start_frame,
            end_frame,
            float(start_time),
            float(end_time),
        ]

        for i in range(s2s_frames):
            row.extend([seg_vals[i], seg_ars[i]])

        row.extend([seg_val, seg_ar])
        metadata_segments.append(row)

    full_val = mean_with_missing(valence, missing_value)
    full_ar = mean_with_missing(arousal, missing_value)
    metadata_full = [input_path.name, name_file, '', full_val, full_ar]
    return metadata_segments, metadata_full


def segment_video_test(
    input_path: Path,
    output_dir: Path,
    name_file: str,
    fps: float,
    duration: float,
    win_max_length: float,
    win_shift: float,
    win_min_length: float,
    s2s_frames: int,
    skip_existing: bool,
    show_progress: bool,
    test_fill_mode: str,
    missing_value: float,
) -> Tuple[List[List[object]], List[object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = input_path.suffix
    metadata_segments: List[List[object]] = []

    timings = slice_audio(
        start_time=0,
        end_time=duration,
        win_max_length=win_max_length,
        win_shift=win_shift,
        win_min_length=win_min_length,
    )

    timing_iter = timings
    if show_progress:
        timing_iter = tqdm(timings, desc=f'segments {name_file}', leave=False)

    if test_fill_mode == 'minus5':
        fill_val = float(missing_value)
        fill_ar = float(missing_value)
    else:
        fill_val = None
        fill_ar = None
        seg_vals = (None,) * s2s_frames
        seg_ars  = (None,) * s2s_frames

    for segment_index, timing in enumerate(timing_iter):
        start_time = timing['start']
        end_time = timing['end']
        start_frame = int(np.round(start_time * fps))
        end_frame = int(np.round(end_time * fps))

        new_name = f"{name_file}___{start_frame}_{end_frame}_{segment_index:04d}{ext}"
        output_path = output_dir / new_name

        row = [
           input_path.name,
            new_name,
            '',
            start_frame,
            end_frame,
            float(start_time),
            float(end_time),
        ]

        for i in range(s2s_frames):
            row.extend([seg_vals[i], seg_ars[i]])

        row.extend([fill_val, fill_ar])
        metadata_segments.append(row)

    metadata_full = [input_path.name, name_file, '', fill_val, fill_ar]
    return metadata_segments, metadata_full


def build_video_index(videos_dir: Path) -> Dict[str, Path]:
    return {p.stem: p for p in videos_dir.iterdir() if p.is_file()}


def resolve_video_path(name_file: str, index: Dict[str, Path]) -> Optional[Path]:
    if name_file in index:
        return index[name_file]
    if name_file.endswith("_left") or name_file.endswith("_right"):
        base = name_file.rsplit("_", 1)[0]
        return index.get(base)
    return None


def process_single_video(
    ann_path_str: str,
    video_path_str: str,
    name_file: str,
    subset_name: str,
    output_root_str: str,
    win_max_length: float,
    win_shift: float,
    win_min_length: float,
    s2s_frames: int,
    missing_value: float,
    skip_existing: bool,
    show_progress: bool,
) -> Tuple[str, str, Optional[List[List[object]]], Optional[List[object]]]:
    ann_path = Path(ann_path_str)
    video_path = Path(video_path_str)
    output_root = Path(output_root_str)

    df = pd.read_csv(ann_path)
    if 'valence' not in df.columns or 'arousal' not in df.columns:
        return 'bad_ann', name_file, None, None

    valence = df['valence'].to_numpy(dtype=float)
    arousal = df['arousal'].to_numpy(dtype=float)
    fps, duration, _ = get_video_info(video_path)
    output_dir = output_root / subset_name / name_file
    seg_meta, full_meta = segment_video(
        video_path,
        output_dir,
        name_file,
        fps,
        duration,
        valence,
        arousal,
        missing_value,
        win_max_length,
        win_shift,
        win_min_length,
        s2s_frames,
        skip_existing,
        show_progress,
    )

    return 'ok', name_file, seg_meta, full_meta


def process_task(task: Tuple[object, ...]) -> Tuple[str, str, Optional[List[List[object]]], Optional[List[object]]]:
    return process_single_video(*task)


def _init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _handle_sigint(sig, frame) -> None:
    global ACTIVE_POOL
    if ACTIVE_POOL is not None:
        try:
            ACTIVE_POOL.terminate()
        except Exception:
            pass
    sys.stderr.write("Hard exit on Ctrl+C\n")
    sys.stderr.flush()
    os._exit(1)


def process_subset(
    subset_name: str,
    ann_dir: Path,
    videos_dir: Path,
    output_root: Path,
    win_max_length: float,
    win_shift: float,
    win_min_length: float,
    s2s_frames: int,
    missing_value: float,
    max_videos: int,
    skip_existing: bool,
    workers: int,
    show_segment_progress: bool,
) -> None:
    if not ann_dir.exists():
        raise FileNotFoundError(f'Missing annotations dir: {ann_dir}')

    index = build_video_index(videos_dir)
    ann_files = sorted([p for p in ann_dir.iterdir() if p.suffix == '.txt'])

    segment_metadata: List[List[object]] = []
    video_metadata: List[List[object]] = []

    chunks_dir = output_root / subset_name
    chunks_dir.mkdir(parents=True, exist_ok=True)

    show_progress = show_segment_progress and workers <= 1
    tasks: List[Tuple[object, ...]] = []
    for ann_path in ann_files:
        name_file = ann_path.stem
        video_path = resolve_video_path(name_file, index)
        if video_path is None:
            print(f'skip: missing video for {name_file}')
            continue
        tasks.append(
            (
                str(ann_path),
                str(video_path),
                name_file,
                subset_name,
                str(output_root),
                win_max_length,
                win_shift,
                win_min_length,
                s2s_frames,
                missing_value,
                skip_existing,
                show_progress,
            )
        )
        if max_videos > 0 and len(tasks) >= max_videos:
            break

    if workers <= 1:
        for task in tqdm(tasks, desc=f'VA {subset_name}'):
            status, name_file, seg_meta, full_meta = process_single_video(*task)
            if status != 'ok' or seg_meta is None or full_meta is None:
                print(f'skip: bad annotation file {name_file}')
                continue
            segment_metadata.extend(seg_meta)
            video_metadata.append(full_meta)
    else:
        global ACTIVE_POOL
        old_handler = signal.signal(signal.SIGINT, _handle_sigint)
        ctx = mp.get_context('spawn') if os.name == 'nt' else mp.get_context('fork')
        pool = ctx.Pool(processes=workers, initializer=_init_worker)
        ACTIVE_POOL = pool
        try:
            for status, name_file, seg_meta, full_meta in tqdm(
                pool.imap_unordered(process_task, tasks),
                total=len(tasks),
                desc=f'VA {subset_name}',
            ):
                if status != 'ok' or seg_meta is None or full_meta is None:
                    print(f'skip: bad annotation file {name_file}')
                    continue
                segment_metadata.extend(seg_meta)
                video_metadata.append(full_meta)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise SystemExit(1)
        else:
            pool.close()
            pool.join()
        finally:
            ACTIVE_POOL = None
            signal.signal(signal.SIGINT, old_handler)

    df_full = pd.DataFrame(
        video_metadata,
        columns=['full_video_name', 'video_name', 'text', 'valence', 'arousal'],
    )

    segment_cols = [
        'full_video_name', 'video_name', 'text',
        'start_frame', 'end_frame', 'start_time', 'end_time',
    ]

    for i in range(s2s_frames):
        segment_cols += [f'valence_s{i}', f'arousal_s{i}']
    
    segment_cols += ['valence', 'arousal']
    df_segment = pd.DataFrame(segment_metadata, columns=segment_cols)
    df_segment.to_csv(output_root / f'{subset_name}_segment.csv', index=False)


def process_test_subset(
    subset_name: str,
    test_ann_file: Path,
    videos_dir: Path,
    output_root: Path,
    win_max_length: float,
    win_shift: float,
    win_min_length: float,
    s2s_frames: int,
    missing_value: float,
    max_videos: int,
    skip_existing: bool,
    workers: int,
    show_segment_progress: bool,
    test_fill_mode: str,
) -> None:
    if not test_ann_file.exists():
        raise FileNotFoundError(f'Missing test annotation file: {test_ann_file}')

    df = pd.read_csv(test_ann_file)
    if 'image_location' not in df.columns:
        raise ValueError('Test annotation file must contain image_location column')

    video_names_raw = df['image_location'].astype(str).tolist()
    seen = set()
    video_names: List[str] = []
    for item in video_names_raw:
        name = item.split('/', 1)[0].strip()
        if name and name not in seen:
            seen.add(name)
            video_names.append(name)

    index = build_video_index(videos_dir)
    segment_metadata: List[List[object]] = []
    video_metadata: List[List[object]] = []

    chunks_dir = output_root / subset_name
    chunks_dir.mkdir(parents=True, exist_ok=True)

    show_progress = show_segment_progress and workers <= 1
    tasks: List[Tuple[str, str]] = []
    for name_file in video_names:
        video_path = resolve_video_path(name_file, index)
        if video_path is None:
            print(f'skip: missing video for {name_file}')
            continue
        tasks.append((name_file, str(video_path)))
        if max_videos > 0 and len(tasks) >= max_videos:
            break

    for name_file, video_path_str in tqdm(tasks, desc=f'VA {subset_name}'):
        video_path = Path(video_path_str)
        fps, duration, _ = get_video_info(video_path)
        output_dir = chunks_dir / name_file
        seg_meta, full_meta = segment_video_test(
            video_path,
            output_dir,
            name_file,
            fps,
            duration,
            win_max_length,
            win_shift,
            win_min_length,
            s2s_frames,
            skip_existing,
            show_progress,
            test_fill_mode,
            missing_value,
        )
        segment_metadata.extend(seg_meta)
        video_metadata.append(full_meta)

    df_full = pd.DataFrame(
        video_metadata,
        columns=['full_video_name', 'video_name', 'text', 'valence', 'arousal'],
    )

    segment_cols = [
        'full_video_name', 'video_name', 'text',
        'start_frame', 'end_frame', 'start_time', 'end_time',
    ]

    for i in range(s2s_frames):
        segment_cols += [f'valence_s{i}', f'arousal_s{i}']
    
    segment_cols += ['valence', 'arousal']
    df_segment = pd.DataFrame(segment_metadata, columns=segment_cols)
    df_segment.to_csv(output_root / f'{subset_name}_segment.csv', index=False)


def main() -> None:
    if not DEFAULT_CONFIG.exists():
        raise FileNotFoundError(f'Missing config: {DEFAULT_CONFIG}')
    cfg = load_yaml_config(DEFAULT_CONFIG)

    videos_dir = Path(cfg.get('videos_dir', ''))
    ann_root = Path(cfg.get('ann_root', ''))
    output_root = Path(cfg.get('output_root', ''))
    subsets = cfg.get('subsets', ['train', 'val'])
    if isinstance(subsets, str):
        subsets = [s.strip() for s in subsets.split(',') if s.strip()]

    win_max_length = float(cfg.get('win_max_length', 4))
    win_shift = float(cfg.get('win_shift', 2))
    win_min_length = float(cfg.get('win_min_length', 1))
    missing_value = float(cfg.get('missing_value', -5))
    max_videos = int(cfg.get('max_videos', 0))
    skip_existing = bool(cfg.get('skip_existing', False))
    workers = int(cfg.get('workers', 1))
    show_segment_progress = bool(cfg.get('show_segment_progress', False))
    test_ann_file = Path(cfg.get('test_ann_file', ''))
    test_fill_mode = str(cfg.get('test_fill_mode', 'empty')).strip().lower()
    if test_fill_mode not in ('empty', 'minus5'):
        raise ValueError("test_fill_mode must be 'empty' or 'minus5'")

    s2s_frames = int(cfg.get('s2s_frames', 4))
    if s2s_frames not in (4, 8):
        raise ValueError("s2s_frames must be 4 or 8")
    
    if not videos_dir.exists():
        raise FileNotFoundError(f'Missing videos dir: {videos_dir}')
    if not ann_root.exists():
        raise FileNotFoundError(f'Missing ann root: {ann_root}')
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    subset_map = {
        'train': ann_root / 'Train_Set',
        'val': ann_root / 'Validation_Set',
        'test': None,
    }

    for subset in subsets:
        if subset not in subset_map:
            raise ValueError(f'Unsupported subset: {subset}')
        if subset == 'test':
            process_test_subset(
                subset,
                test_ann_file,
                videos_dir,
                output_root,
                win_max_length,
                win_shift,
                win_min_length,
                s2s_frames,
                missing_value,
                max_videos,
                skip_existing,
                workers,
                show_segment_progress,
                test_fill_mode,
            )
        else:
            process_subset(
                subset,
                subset_map[subset],
                videos_dir,
                output_root,
                win_max_length,
                win_shift,
                win_min_length,
                s2s_frames,
                missing_value,
                max_videos,
                skip_existing,
                workers,
                show_segment_progress,
            )


if __name__ == '__main__':
    main()
