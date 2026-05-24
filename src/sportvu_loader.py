"""this is to look at the sportvu data and compare data to
 try and match with own scraping
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_game(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def events_to_long_df(game: dict, events_filter: set[int] | None = None) -> pd.DataFrame:
    """Explode moments into long format. One row per (moment, entity)."""
    game_id = game.get("gameid")
    gamedate = game.get("gamedate")
    rows: list[dict] = []
    for event in game.get("events", []):
        event_id = int(event.get("eventId"))
        if events_filter and event_id not in events_filter:
            continue
        # Build jersey/team lookup from visitor + home player rosters
        roster = {}
        for side in ("visitor", "home"):
            team = event.get(side) or {}
            tid = team.get("teamid")
            for p in team.get("players", []):
                roster[p["playerid"]] = {
                    "team_id": tid,
                    "jersey": p.get("jersey"),
                    "firstname": p.get("firstname"),
                    "lastname": p.get("lastname"),
                    "position": p.get("position"),
                }
        for mi, moment in enumerate(event.get("moments", [])):
            # moment = [period, wall_clock_ms, game_clock_s, shot_clock_s, unused, entities]
            period = moment[0]
            game_clock = moment[2]
            shot_clock = moment[3]
            entities = moment[5]
            for ent in entities:
                team_id, player_id, x, y, radius = ent
                # Ball has player_id -1, team_id -1 per SportVU convention
                is_ball = player_id == -1
                info = roster.get(player_id, {})
                rows.append(
                    {
                        "game_id": game_id,
                        "game_date": gamedate,
                        "event_id": event_id,
                        "period": period,
                        "game_clock": game_clock,
                        "shot_clock": shot_clock,
                        "moment_idx": mi,
                        "team_id": team_id,
                        "player_id": player_id,
                        "jersey": None if is_ball else info.get("jersey"),
                        "name": "BALL"
                        if is_ball
                        else f"{info.get('firstname','')} {info.get('lastname','')}".strip(),
                        "x_loc": x,
                        "y_loc": y,
                        "radius": radius,
                        "is_ball": is_ball,
                    }
                )
    return pd.DataFrame(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("json_path", help="Path to unpacked SportVU game json")
    p.add_argument("--events", default=None, help="Comma-separated event_ids to keep")
    p.add_argument("--out", default=None, help="Optional parquet output path")
    p.add_argument("--peek", action="store_true", help="Print a small sample")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    events_filter = None
    if args.events:
        events_filter = {int(x) for x in args.events.split(",")}
    game = load_game(Path(args.json_path))
    df = events_to_long_df(game, events_filter=events_filter)
    print(f"game_id={game.get('gameid')} date={game.get('gamedate')}")
    print(f"events_in_log={len(game.get('events', []))}")
    print(f"rows emitted: {len(df):,}")
    print(f"unique events: {df['event_id'].nunique() if len(df) else 0}")
    print(f"unique players (non-ball): {df[~df['is_ball']]['player_id'].nunique() if len(df) else 0}")
    if args.peek:
        print(df.head(10).to_string())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
