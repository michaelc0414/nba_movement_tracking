"""
this pulls event mp4 clips from stats.nba.com and writes a json for each clip with 
game metadata, event metadata, and the player_ids
involved in the play. what the output should lookl like:

    data/clips/{game_id}/{event_id:03d}.mp4
    data/clips/{game_id}/{event_id:03d}.json

examples to use:
    python src/fetch_clips.py --game-id 0021500431
    python src/fetch_clips.py --game-id 0021500431 --event-types 1,2,3 --limit 20
    python src/fetch_clips.py --date 12/25/2015 --home LAL --away GSW
    python src/fetch_clips.py --game-id 0021500431 --resolution large --sleep 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
from nba_api.stats.endpoints import (
    leaguegamefinder,
    playbyplayv2,
    videoeventsasset,
)

#stats.nba.com needs a user-agent
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
}

EVENT_TYPES = {
    1: "made_shot",
    2: "missed_shot",
    3: "free_throw",
    4: "rebound",
    5: "turnover",
    6: "foul",
    # 7 violation, 8 substitution, 9 timeout, 10 jumpball, 11 ejection,
    # 12 period_start, 13 period_end, 18 instant_replay
}


def get_game_id(date: str, home: str, away: str, timeout: int = 30) -> str:
    """get game_id from a date (MM/DD/YYYY) + team tricodes.

    Uses LeagueGameFinder and filters to the home team's matchup row
    containing the away team tricode. Returns the 10-char game_id.
    """
    gf = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=date,
        date_to_nullable=date,
        timeout=timeout,
    )
    df = gf.get_data_frames()[0]
    # Each game shows up twice (once per team); pick the home row.
    mask = df["MATCHUP"].str.contains(f"vs. {away}", regex=False) & (
        df["TEAM_ABBREVIATION"] == home
    )
    hits = df[mask]
    if hits.empty:
        # Try the inverse in case the user swapped home/away
        mask = df["MATCHUP"].str.contains(f"vs. {home}", regex=False) & (
            df["TEAM_ABBREVIATION"] == away
        )
        hits = df[mask]
    if hits.empty:
        raise RuntimeError(
            f"No game found for {date} {away}@{home}. "
            f"Available matchups on that date: "
            f"{sorted(set(df['MATCHUP'].tolist()))}"
        )
    return str(hits.iloc[0]["GAME_ID"])

#get play by play data

def fetch_pbp(game_id: str, timeout: int = 30) -> list[dict[str, Any]]:
    """Return PlayByPlayV2 rows as list of dicts."""
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=timeout)
    df = pbp.get_data_frames()[0]
    return df.to_dict(orient="records")

#get video links

def fetch_video_urls(
    game_id: str, event_id: int, timeout: int = 30
) -> dict[str, str] | None:
    """Return mp4 URLs for a given (game_id, event_id) or None if missing.

    Response shape (observed):
        resultSets[0].name == 'Meta' with rowSet[0][0] = {videoUrls: [...]}
        resultSets[1].name == 'playlist'
    """
    asset = videoeventsasset.VideoEventsAsset(
        game_id=game_id, game_event_id=event_id, timeout=timeout
    )
    raw = asset.nba_response.get_dict()
    result_sets = raw.get("resultSets") or raw.get("resultSet") or {}
    # The meta block carries videoUrls
    meta = None
    if isinstance(result_sets, list):
        for rs in result_sets:
            if rs.get("name") == "Meta":
                meta = rs
                break
    elif isinstance(result_sets, dict) and result_sets.get("name") == "Meta":
        meta = result_sets

    if not meta:
        return None
    rows = meta.get("rowSet") or []
    if not rows or not rows[0]:
        return None
    cell = rows[0][0] if isinstance(rows[0], list) else rows[0]
    video_urls = cell.get("videoUrls") if isinstance(cell, dict) else None
    if not video_urls:
        return None
    v = video_urls[0]
    out = {}
    if v.get("lurl"):
        out["large"] = v["lurl"]
    if v.get("murl"):
        out["medium"] = v["murl"]
    if v.get("surl"):
        out["small"] = v["surl"]
    if v.get("uuid"):
        out["uuid"] = v["uuid"]
    return out or None

#downloads the video

def download_mp4(url: str, dest: Path, chunk: int = 1 << 15) -> int:
    """Stream mp4 to disk. Returns bytes written."""
    with requests.get(
        url, stream=True, headers=DOWNLOAD_HEADERS, timeout=60
    ) as r:
        r.raise_for_status()
        n = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for buf in r.iter_content(chunk_size=chunk):
                if buf:
                    f.write(buf)
                    n += len(buf)
        return n

#main bulk of work

def harvest(
    game_id: str,
    out_root: Path,
    event_types: set[int] | None,
    resolution: str,
    limit: int | None,
    sleep: float,
    skip_existing: bool,
) -> None:
    out_dir = out_root / game_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pbp] {game_id} -> fetching play-by-play")
    rows = fetch_pbp(game_id)
    print(f"[pbp] {len(rows)} events total")

    if event_types:
        rows = [r for r in rows if r.get("EVENTMSGTYPE") in event_types]
        print(f"[pbp] {len(rows)} events after type filter {sorted(event_types)}")

    if limit is not None:
        rows = rows[:limit]
        print(f"[pbp] limited to {len(rows)}")

    ok = 0
    missing = 0
    errors = 0

    for i, row in enumerate(rows, 1):
        event_id = int(row["EVENTNUM"])
        mp4_path = out_dir / f"{event_id:03d}.mp4"
        json_path = out_dir / f"{event_id:03d}.json"

        if skip_existing and mp4_path.exists() and json_path.exists():
            print(f"  [{i}/{len(rows)}] event {event_id} cached, skip")
            continue

        try:
            urls = fetch_video_urls(game_id, event_id)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(rows)}] event {event_id} asset error: {e}")
            errors += 1
            time.sleep(sleep)
            continue

        if not urls:
            print(f"  [{i}/{len(rows)}] event {event_id} no video available")
            missing += 1
            time.sleep(sleep)
            continue

        chosen = urls.get(resolution) or urls.get("large") or urls.get("medium") or urls.get("small")
        if not chosen:
            print(f"  [{i}/{len(rows)}] event {event_id} no usable url: {urls}")
            missing += 1
            time.sleep(sleep)
            continue

        try:
            n_bytes = download_mp4(chosen, mp4_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(rows)}] event {event_id} download error: {e}")
            errors += 1
            time.sleep(sleep)
            continue

        sidecar = {
            "game_id": game_id,
            "event_id": event_id,
            "event_msg_type": row.get("EVENTMSGTYPE"),
            "event_msg_type_name": EVENT_TYPES.get(
                row.get("EVENTMSGTYPE"), "other"
            ),
            "event_msg_action_type": row.get("EVENTMSGACTIONTYPE"),
            "period": row.get("PERIOD"),
            "pctimestring": row.get("PCTIMESTRING"),
            "wctimestring": row.get("WCTIMESTRING"),
            "homedescription": row.get("HOMEDESCRIPTION"),
            "neutraldescription": row.get("NEUTRALDESCRIPTION"),
            "visitordescription": row.get("VISITORDESCRIPTION"),
            "player1_id": row.get("PLAYER1_ID"),
            "player1_name": row.get("PLAYER1_NAME"),
            "player1_team_id": row.get("PLAYER1_TEAM_ID"),
            "player2_id": row.get("PLAYER2_ID"),
            "player2_name": row.get("PLAYER2_NAME"),
            "player2_team_id": row.get("PLAYER2_TEAM_ID"),
            "player3_id": row.get("PLAYER3_ID"),
            "player3_name": row.get("PLAYER3_NAME"),
            "player3_team_id": row.get("PLAYER3_TEAM_ID"),
            "score": row.get("SCORE"),
            "scoremargin": row.get("SCOREMARGIN"),
            "video_urls": urls,
            "downloaded_url": chosen,
            "downloaded_bytes": n_bytes,
        }
        json_path.write_text(json.dumps(sidecar, indent=2, default=str))
        print(
            f"  [{i}/{len(rows)}] event {event_id} period {row.get('PERIOD')} "
            f"{row.get('PCTIMESTRING')} {EVENT_TYPES.get(row.get('EVENTMSGTYPE'), 'other')} "
            f"-> {n_bytes/1024:.0f} KB"
        )
        ok += 1
        time.sleep(sleep)

    print()
    print(f"[done] {game_id}: ok={ok} missing={missing} errors={errors}")
    print(f"[done] output: {out_dir}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--game-id", help="10-char NBA game_id, e.g. 0021500431")
    g.add_argument("--date", help="MM/DD/YYYY, used with --home and --away")
    p.add_argument("--home", help="Home team tricode (e.g. LAL)")
    p.add_argument("--away", help="Away team tricode (e.g. GSW)")
    p.add_argument(
        "--event-types",
        default="1,2,3,4,5",
        help="Comma-separated EVENTMSGTYPE filter. Default 1,2,3,4,5 (shots/FT/rebounds/TO)."
             " Use 'all' for no filter.",
    )
    p.add_argument("--limit", type=int, default=None, help="Max events to fetch")
    p.add_argument(
        "--resolution",
        default="large",
        choices=["large", "medium", "small"],
        help="mp4 resolution (large=1280x720)",
    )
    p.add_argument("--sleep", type=float, default=0.6, help="Seconds between API calls")
    p.add_argument(
        "--out-root",
        default="data/clips",
        help="Output root directory (under project root)",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download clips that already exist on disk",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.game_id:
        game_id = args.game_id
    else:
        if not (args.date and args.home and args.away):
            print("--date requires --home and --away", file=sys.stderr)
            return 2
        game_id = get_game_id(args.date, args.home, args.away)
        print(f"[lookup] resolved {args.date} {args.away}@{args.home} -> {game_id}")

    if args.event_types.lower() == "all":
        event_types = None
    else:
        event_types = {int(x) for x in args.event_types.split(",") if x.strip()}

    harvest(
        game_id=game_id,
        out_root=Path(args.out_root),
        event_types=event_types,
        resolution=args.resolution,
        limit=args.limit,
        sleep=args.sleep,
        skip_existing=not args.no_skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
