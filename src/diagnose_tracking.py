import pandas as pd
import matplotlib.pyplot as plt
import sys

parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data/parquet/test_clip_trackingNEW.parquet"
df = pd.read_parquet(parquet_path)

has_on_court = "on_court" in df.columns

print(f"Parquet: {parquet_path}")
print(f"Total records: {len(df)}")
print(f"Unique track IDs: {df['track_id'].nunique()}")
if has_on_court:
    print(f"On-court records: {df['on_court'].sum()}")
    print(f"Unique track IDs (on-court): {df[df['on_court']]['track_id'].nunique()}")
print(f"Frames: {df['frame'].min()} to {df['frame'].max()}")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. Per-frame active track count
tracks_per_frame = df.groupby("frame")["track_id"].nunique()
axes[0].plot(tracks_per_frame.index, tracks_per_frame.values, linewidth=0.8, label="All tracks")
if has_on_court:
    court_df = df[df["on_court"]]
    court_tracks_per_frame = court_df.groupby("frame")["track_id"].nunique()
    axes[0].plot(court_tracks_per_frame.index, court_tracks_per_frame.values,
                 linewidth=0.8, color="orange", label="On-court only")
axes[0].axhline(y=13, color="r", linestyle="--", alpha=0.5, label="Expected (~13)")
axes[0].set_xlabel("Frame")
axes[0].set_ylabel("Active Tracks")
axes[0].set_title("Per-Frame Active Track Count")
axes[0].legend()

# 2. Track lifetime histogram
track_lifetimes = df.groupby("track_id")["frame"].agg(["min", "max"])
track_lifetimes["duration"] = track_lifetimes["max"] - track_lifetimes["min"] + 1
axes[1].hist(track_lifetimes["duration"], bins=30, edgecolor="black", alpha=0.7)
axes[1].axvline(x=30, color="r", linestyle="--", alpha=0.5, label="30-frame threshold")
short = (track_lifetimes["duration"] < 30).sum()
long = (track_lifetimes["duration"] >= 30).sum()
axes[1].set_xlabel("Track Lifetime (frames)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Track Lifetime Histogram — {short} short (<30f), {long} long (>=30f)")
axes[1].legend()

# 3. Track birth/death timeline
track_lifetimes = track_lifetimes.sort_values("min")
for i, (track_id, row) in enumerate(track_lifetimes.iterrows()):
    color = "steelblue" if row["duration"] >= 30 else "salmon"
    axes[2].barh(i, row["duration"], left=row["min"], height=0.8, color=color, alpha=0.7)
axes[2].set_xlabel("Frame")
axes[2].set_ylabel("Track (sorted by birth)")
axes[2].set_title(f"Track Timeline — {df['track_id'].nunique()} unique IDs")
axes[2].set_yticks([])

plt.tight_layout()
plt.savefig("data/output/tracking_diagnostics.png", dpi=150)
plt.show()
print(f"\nSaved to data/output/tracking_diagnostics.png")
