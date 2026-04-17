"""Diagnose exactly where and why ID switches happen.

Outputs:
1. Table of every track: ID, first/last frame, duration, start/end pixel positions
2. Candidate merge pairs: tracks that look like fragments of the same player
3. Per-frame birth/death events: which frames have new IDs appearing or old IDs disappearing
4. Gap analysis: for each track that dies, what track is born nearest to it?
"""
import pandas as pd
import numpy as np
import sys

parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data/parquet/test_clip_trackingNEW.parquet"
df = pd.read_parquet(parquet_path)

print(f"{'='*80}")
print(f"ID SWITCHING DIAGNOSTIC — {parquet_path}")
print(f"{'='*80}")
print(f"Total frames: {df['frame'].max() - df['frame'].min() + 1}")
print(f"Unique track IDs: {df['track_id'].nunique()}")
print()

# Build track summary table
tracks = []
for tid, group in df.groupby("track_id"):
    group_sorted = group.sort_values("frame")
    tracks.append({
        "track_id": tid,
        "first_frame": group_sorted["frame"].min(),
        "last_frame": group_sorted["frame"].max(),
        "duration": group_sorted["frame"].max() - group_sorted["frame"].min() + 1,
        "num_records": len(group_sorted),
        "start_px": (group_sorted.iloc[0]["pixel_x"], group_sorted.iloc[0]["pixel_y"]),
        "end_px": (group_sorted.iloc[-1]["pixel_x"], group_sorted.iloc[-1]["pixel_y"]),
    })

tracks_df = pd.DataFrame(tracks).sort_values("first_frame")

print("ALL TRACKS (sorted by birth frame):")
print("-" * 80)
for _, t in tracks_df.iterrows():
    tag = "SHORT" if t["duration"] < 30 else ""
    print(f"  ID {t['track_id']:>3d}  |  frames {t['first_frame']:>3d}-{t['last_frame']:>3d}  "
          f"|  duration {t['duration']:>3d}f  |  start ({t['start_px'][0]:>4d},{t['start_px'][1]:>4d})  "
          f"end ({t['end_px'][0]:>4d},{t['end_px'][1]:>4d})  {tag}")

print()
print(f"{'='*80}")
print("CANDIDATE MERGE PAIRS (tracks that might be the same player):")
print("Criteria: A ends before B starts, gap < 90 frames, distance < 150px, no overlap")
print("-" * 80)

tracks_list = tracks_df.to_dict("records")
track_frame_sets = {t["track_id"]: set(df[df["track_id"] == t["track_id"]]["frame"]) for t in tracks_list}

candidates = []
for i, a in enumerate(tracks_list):
    for j, b in enumerate(tracks_list):
        if a["track_id"] == b["track_id"]:
            continue
        gap = b["first_frame"] - a["last_frame"]
        if gap < 1 or gap > 90:
            continue
        # No temporal overlap
        if track_frame_sets[a["track_id"]] & track_frame_sets[b["track_id"]]:
            continue
        dist = np.sqrt((a["end_px"][0] - b["start_px"][0])**2 +
                       (a["end_px"][1] - b["start_px"][1])**2)
        candidates.append({
            "track_a": a["track_id"],
            "track_b": b["track_id"],
            "a_ends": a["last_frame"],
            "b_starts": b["first_frame"],
            "frame_gap": gap,
            "pixel_dist": round(dist, 1),
            "a_end_pos": a["end_px"],
            "b_start_pos": b["start_px"],
        })

candidates.sort(key=lambda x: (x["frame_gap"], x["pixel_dist"]))

if candidates:
    for c in candidates:
        flag = "*** LIKELY SAME PLAYER ***" if c["pixel_dist"] < 75 else ""
        flag = flag or ("  POSSIBLE" if c["pixel_dist"] < 150 else "")
        print(f"  ID {c['track_a']:>3d} (ends f{c['a_ends']:>3d}) -> ID {c['track_b']:>3d} (starts f{c['b_starts']:>3d})  "
              f"|  gap {c['frame_gap']:>2d}f  |  dist {c['pixel_dist']:>6.1f}px  {flag}")
else:
    print("  No candidate pairs found")

print()
print(f"{'='*80}")
print("FRAME-BY-FRAME BIRTHS AND DEATHS:")
print("-" * 80)

births = tracks_df.groupby("first_frame")["track_id"].apply(list)
deaths = tracks_df.groupby("last_frame")["track_id"].apply(list)
all_events = sorted(set(births.index) | set(deaths.index))

for frame in all_events:
    born = births.get(frame, [])
    died = deaths.get(frame, [])
    if born or died:
        parts = []
        if born:
            parts.append(f"BORN: {born}")
        if died:
            parts.append(f"DIED: {died}")
        active = df[df["frame"] == frame]["track_id"].nunique()
        print(f"  Frame {frame:>3d}  |  active: {active:>2d}  |  {' | '.join(parts)}")
