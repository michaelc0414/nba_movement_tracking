from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

"""run multiple trackers on the same clip(s) and log metrics.

Swaps the tracker stage while holding the detector constant so we can
isolate ID-switching behavior. Writes:
"""

# boxmot exposes each tracker by class name
def build_tracker(name: str, reid_weights: Path, device: int | str = 0):
    """Instantiate a boxmot tracker by friendly name. Raises on unknown."""
    from boxmot import BotSort, DeepOcSort, BoostTrack, StrongSort, ByteTrack

    name = name.lower()
    common = dict(device=device, half=False)
    if name in ("botsort", "bot_sort"):
        return BotSort(
            reid_weights=reid_weights,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=150,
            match_thresh=0.8,
            **common,
        )
    if name in ("deepocsort", "deep_oc_sort"):
        return DeepOcSort(
            reid_weights=reid_weights,
            det_thresh=0.5,
            max_age=150,
            min_hits=2,
            iou_threshold=0.3,
            **common,
        )
    if name == "boosttrack":
        return BoostTrack(
            reid_weights=reid_weights,
            det_thresh=0.5,
            max_age=150,
            min_hits=2,
            **common,
        )
    if name == "strongsort":
        return StrongSort(
            reid_weights=reid_weights,
            max_dist=0.2,
            max_iou_dist=0.7,
            max_age=150,
            n_init=3,
            nn_budget=100,
            **common,
        )
    if name == "bytetrack":
        # ByteTrack has no ReID; signature differs
        return ByteTrack(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=150,
        )
    raise ValueError(
        f"unknown tracker '{name}'. supported: botsort, deepocsort, "
        f"boosttrack, strongsort, bytetrack"
    )

def detect_frame(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    min_box_height: int,
    y_max_frac: float,
) -> np.ndarray:
    """Return Nx6 [x1, y1, x2, y2, conf, cls] after height+y filters."""
    h = frame.shape[0]
    y_limit = h * y_max_frac
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        c = float(box.conf)
        cl = float(box.cls)
        if (y2 - y1) < min_box_height:
            continue
        if y1 >= y_limit:
            continue
        dets.append([x1, y1, x2, y2, c, cl])
    return np.array(dets) if dets else np.empty((0, 6))

def run_tracker_on_clip(
    clip_path: Path,
    tracker_name: str,
    model: YOLO,
    reid_weights: Path,
    device: int | str,
    out_dir: Path,
    write_video: bool,
    conf: float,
    imgsz: int,
    min_box_height: int,
    y_max_frac: float,
) -> dict:
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = build_tracker(tracker_name, reid_weights, device=device)

    writer = None
    if write_video:
        video_dir = out_dir / "annotated"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{tracker_name}_{clip_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    rows = []
    det_counts = []
    t0 = time.perf_counter()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detect_frame(
            model, frame, conf=conf, imgsz=imgsz,
            min_box_height=min_box_height, y_max_frac=y_max_frac,
        )
        det_counts.append(len(dets))
        tracks = tracker.update(dets, frame)
        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            track_id = int(t[4])
            c = float(t[5]) if len(t) > 5 else None
            rows.append(
                {
                    "clip": clip_path.name,
                    "tracker": tracker_name,
                    "frame": frame_idx,
                    "track_id": track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "conf": c,
                }
            )
            if writer is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
        if writer is not None:
            cv2.putText(
                frame,
                f"{tracker_name} | frame {frame_idx} | tracks {len(tracks)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            )
            writer.write(frame)
        frame_idx += 1

    runtime = time.perf_counter() - t0
    cap.release()
    if writer is not None:
        writer.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"tracks_{tracker_name}_{clip_path.stem}.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    stats = compute_stats(df, frame_idx)
    stats.update(
        {
            "clip": clip_path.name,
            "tracker": tracker_name,
            "total_frames": frame_idx,
            "mean_detections": float(np.mean(det_counts)) if det_counts else 0.0,
            "runtime_s": round(runtime, 2),
            "parquet": str(parquet_path.relative_to(out_dir.parent.parent))
                if parquet_path.is_relative_to(out_dir.parent.parent) else str(parquet_path),
        }
    )
    return stats

def compute_stats(df: pd.DataFrame, total_frames: int) -> dict:
    if df.empty:
        return dict(
            unique_ids=0, mean_active=0.0, max_active=0,
            short_tracks=0, mean_track_len=0.0, switches_proxy=0,
        )
    unique_ids = df["track_id"].nunique()
    active = df.groupby("frame")["track_id"].nunique()
    mean_active = float(active.mean())
    max_active = int(active.max())
    durations = df.groupby("track_id")["frame"].agg(
        lambda s: s.max() - s.min() + 1
    )
    short_tracks = int((durations < 30).sum())
    mean_track_len = float(durations.mean())

    # Cheap "switches proxy": deaths + births in the middle 80% of the clip.
    # Genuine HOTA/IDF1 needs ground truth; this is only for cross-tracker ranking.
    start_trim = int(total_frames * 0.1)
    end_trim = int(total_frames * 0.9)
    mids = df.groupby("track_id")["frame"].agg(["min", "max"])
    mid_births = int(((mids["min"] >= start_trim) & (mids["min"] <= end_trim)).sum())
    mid_deaths = int(((mids["max"] >= start_trim) & (mids["max"] <= end_trim)).sum())
    switches_proxy = mid_births + mid_deaths

    return dict(
        unique_ids=unique_ids,
        mean_active=round(mean_active, 2),
        max_active=max_active,
        short_tracks=short_tracks,
        mean_track_len=round(mean_track_len, 1),
        switches_proxy=switches_proxy,
    )

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--clip", help="Path to a single clip mp4")
    g.add_argument("--clips", nargs="+", help="Glob or list of clip mp4 paths")
    p.add_argument(
        "--trackers",
        nargs="+",
        default=["botsort", "deepocsort", "boosttrack", "strongsort"],
        help="Trackers to bake off (subset of: botsort, deepocsort, boosttrack, strongsort, bytetrack)",
    )
    p.add_argument(
        "--weights",
        default="runs/detect/models/player_detector5/weights/best.pt",
        help="YOLO detection weights",
    )
    p.add_argument(
        "--reid",
        default="models/osnet_x0_25_msmt17.pt",
        help="ReID weights for appearance-based trackers",
    )
    p.add_argument("--device", default=0, help="Torch device; int for cuda, 'cpu' for cpu")
    p.add_argument("--out-root", default="data/bakeoff", help="Output root")
    p.add_argument("--run-id", default=None, help="Run folder name (default: timestamp)")
    p.add_argument("--write-video", action="store_true", help="Also render annotated mp4s")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size (6GB VRAM supports 1280)")
    p.add_argument("--min-box-height", type=int, default=150)
    p.add_argument("--y-max-frac", type=float, default=0.85)
    return p.parse_args(argv)


def expand_clips(args: argparse.Namespace) -> list[Path]:
    if args.clip:
        return [Path(args.clip)]
    paths = []
    for c in args.clips:
        p = Path(c)
        if "*" in c or "?" in c:
            paths.extend(sorted(p.parent.glob(p.name)))
        else:
            paths.append(p)
    return paths


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    clips = expand_clips(args)
    if not clips:
        print("no clips to process", file=sys.stderr)
        return 2

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bakeoff] run_id={run_id} out={out_dir}")
    print(f"[bakeoff] clips={len(clips)} trackers={args.trackers}")
    print(f"[bakeoff] weights={args.weights} reid={args.reid} device={args.device}")
    print(f"[bakeoff] conf={args.conf} imgsz={args.imgsz}")

    # Load YOLO once and reuse across all (tracker, clip) combinations
    model = YOLO(args.weights)
    reid = Path(args.reid)

    stats_rows: list[dict] = []
    for clip in clips:
        if not clip.exists():
            print(f"[bakeoff] skip missing {clip}")
            continue
        for tracker_name in args.trackers:
            print(f"[bakeoff] running {tracker_name} on {clip.name}")
            try:
                s = run_tracker_on_clip(
                    clip_path=clip,
                    tracker_name=tracker_name,
                    model=model,
                    reid_weights=reid,
                    device=args.device,
                    out_dir=out_dir,
                    write_video=args.write_video,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    min_box_height=args.min_box_height,
                    y_max_frac=args.y_max_frac,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[bakeoff]   ERROR {tracker_name} / {clip.name}: {e}")
                continue
            stats_rows.append(s)
            print(
                f"[bakeoff]   unique_ids={s['unique_ids']:>3d}  "
                f"mean_active={s['mean_active']:>5.2f}  "
                f"short={s['short_tracks']:>3d}  "
                f"switches~={s['switches_proxy']:>3d}  "
                f"runtime={s['runtime_s']:>6.1f}s"
            )

    stats_path = out_dir / "stats.csv"
    if stats_rows:
        fieldnames = list(stats_rows[0].keys())
        with open(stats_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(stats_rows)
        print(f"[bakeoff] wrote {stats_path}")
    else:
        print("[bakeoff] no stats produced (all runs failed?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
