#!/usr/bin/env python3
"""Batch-estimate player ranks for an SGF directory with KataGo human SL MLE.

The script is restartable at two layers:
  * completed CSV rows are skipped by gameid on restart
  * per-SGF rank MLE caches are written next to each SGF in .rank_mle_cache/
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import signal
import sys
import time
from pathlib import Path

from rankmle.katago_client import KataGoClient, KataGoConfig
from rankmle.rank_mle import analyze_path, predict_per_player
from rankmle.sgf_loader import load_sgf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SGF_DIR = (
    REPO_ROOT / "data" / "sample-100k-medium-ranked-19x19-human-150moves-sgfs"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "sample_100k_150moves_rank_mle.csv"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "analysis_config.optimized.cfg"
FALLBACK_CONFIG = REPO_ROOT / "configs" / "analysis_config.cfg"
DEFAULT_KATAGO = "/opt/homebrew/bin/katago"
DEFAULT_MODEL = "~/.katrain/kata1-b28c512nbt-s12704148736-d5790336910.bin.gz"
DEFAULT_HUMAN_MODEL = "~/.katrain/b18c384nbt-humanv0.bin.gz"
CSV_FIELDS = [
    "gameid",
    "game_date",
    "black_player_id",
    "black_ogs_rank",
    "black_estimated_rank",
    "white_player_id",
    "white_ogs_rank",
    "white_estimated_rank",
]
ERROR_FIELDS = ["gameid", "sgf_path", "error_type", "error"]

_STOP = False


def _handle_stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def _game_id(path: Path) -> str:
    match = re.match(r"^(\d+)", path.name)
    if match:
        return match.group(1)
    return path.stem


def _completed_gameids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as handle:
        return {
            row["gameid"]
            for row in csv.DictReader(handle)
            if row.get("gameid")
        }


def _errored_gameids(error_path: Path) -> set[str]:
    if not error_path.exists():
        return set()
    with error_path.open(newline="") as handle:
        return {
            row["gameid"]
            for row in csv.DictReader(handle)
            if row.get("gameid")
        }


def _ensure_csv_schema(csv_path: Path, sgf_by_gameid: dict[str, Path]) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if fieldnames == CSV_FIELDS:
        return

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            gameid = row.get("gameid") or ""
            if not row.get("game_date") and gameid in sgf_by_gameid:
                row["game_date"] = load_sgf(str(sgf_by_gameid[gameid])).date
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    os.replace(tmp_path, csv_path)


def _iter_sgfs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.sgf"))


def _ordered_sgfs(
    paths: list[Path],
    *,
    order: str,
    random_seed: int | None = None,
) -> list[Path]:
    ordered = list(paths)
    if order == "chronological":
        return ordered
    if order == "reverse":
        return list(reversed(ordered))
    if order == "random":
        rng = random.Random(random_seed)
        rng.shuffle(ordered)
        return ordered
    raise ValueError(f"unsupported order: {order}")


def _row_for_sgf(path: Path, data: dict) -> dict[str, str]:
    game = load_sgf(str(path))
    prediction = predict_per_player(data)
    black = game.players.get("B", {})
    white = game.players.get("W", {})
    return {
        "gameid": _game_id(path),
        "game_date": game.date,
        "black_player_id": black.get("name") or "",
        "black_ogs_rank": black.get("rating") or "",
        "black_estimated_rank": prediction.get("B", {}).get("rank") or "",
        "white_player_id": white.get("name") or "",
        "white_ogs_rank": white.get("rating") or "",
        "white_estimated_rank": prediction.get("W", {}).get("rank") or "",
    }


def _print_problem_sgf(path: Path, error: BaseException) -> None:
    print(f"Error while analyzing SGF: {path}", file=sys.stderr)
    print(f"Exception: {type(error).__name__}: {error}", file=sys.stderr)
    print("----- begin problematic SGF -----", file=sys.stderr)
    try:
        print(path.read_text(encoding="utf-8", errors="replace").rstrip(), file=sys.stderr)
    except OSError as read_error:
        print(f"<could not read SGF: {read_error}>", file=sys.stderr)
    print("----- end problematic SGF -----", file=sys.stderr)


def _write_error_row(
    writer: csv.DictWriter,
    path: Path,
    error: BaseException,
) -> None:
    writer.writerow(
        {
            "gameid": _game_id(path),
            "sgf_path": str(path),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )


def _expand(path: str | Path) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def _sleep_interruptibly(seconds: float) -> None:
    deadline = time.time() + seconds
    while not _STOP and time.time() < deadline:
        time.sleep(min(1.0, deadline - time.time()))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgf-dir", type=Path, default=DEFAULT_SGF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--katago", default=DEFAULT_KATAGO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--human-model", default=DEFAULT_HUMAN_MODEL)
    parser.add_argument(
        "--home-data-dir",
        default="~/.katrain",
        help="passed to KataGo as -override-config homeDataDir=...",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else FALLBACK_CONFIG),
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, help="process at most this many pending SGFs")
    parser.add_argument(
        "--order",
        choices=("chronological", "reverse", "random"),
        default="chronological",
        help="order for pending SGFs before applying --limit",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="seed for reproducible --order random runs",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print progress every N completed games",
    )
    parser.add_argument(
        "--sleep-every",
        type=int,
        default=0,
        help="sleep after every N completed games in this run; 0 disables",
    )
    parser.add_argument(
        "--sleep-minutes",
        type=float,
        default=5.0,
        help="minutes to sleep when --sleep-every triggers",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    sgf_dir = args.sgf_dir.resolve()
    if not sgf_dir.is_dir():
        raise SystemExit(f"SGF directory not found: {sgf_dir}")

    output = args.output.resolve()
    error_output = output.with_suffix(output.suffix + ".error")
    output.parent.mkdir(parents=True, exist_ok=True)
    all_sgfs = _iter_sgfs(sgf_dir)
    sgf_by_gameid = {_game_id(path): path for path in all_sgfs}
    _ensure_csv_schema(output, sgf_by_gameid)
    completed = _completed_gameids(output)
    errored = _errored_gameids(error_output)
    already_seen = completed | errored
    pending = [path for path in all_sgfs if _game_id(path) not in already_seen]
    pending = _ordered_sgfs(
        pending,
        order=args.order,
        random_seed=args.random_seed,
    )
    if args.limit is not None:
        pending = pending[: args.limit]

    write_header = not output.exists() or output.stat().st_size == 0
    write_error_header = (
        not error_output.exists() or error_output.stat().st_size == 0
    )
    cfg = KataGoConfig(
        katago=_expand(args.katago),
        model=_expand(args.model),
        human_model=_expand(args.human_model),
        config=_expand(args.config),
        extra_args=["-override-config", f"homeDataDir={_expand(args.home_data_dir)}"],
    )

    print(
        f"Found {len(all_sgfs)} SGFs, {len(completed)} already in CSV, "
        f"{len(errored)} already in error log, {len(pending)} pending.",
        file=sys.stderr,
    )
    if not pending:
        return 0

    started = time.time()
    client = KataGoClient(cfg)
    client.start()
    processed = 0
    failed = 0
    try:
        with (
            output.open("a", newline="") as handle,
            error_output.open("a", newline="") as error_handle,
        ):
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            error_writer = csv.DictWriter(error_handle, fieldnames=ERROR_FIELDS)
            if write_header:
                writer.writeheader()
                handle.flush()
            if write_error_header:
                error_writer.writeheader()
                error_handle.flush()
            for index, sgf_path in enumerate(pending, start=1):
                if _STOP:
                    print("Stop requested; exiting after last completed row.", file=sys.stderr)
                    break
                gameid = _game_id(sgf_path)
                print(f"[{index}/{len(pending)}] analyzing game {gameid}", file=sys.stderr)
                try:
                    data = analyze_path(
                        client,
                        str(sgf_path),
                        use_cache=not args.no_cache,
                    )
                except Exception as error:
                    _print_problem_sgf(sgf_path, error)
                    _write_error_row(error_writer, sgf_path, error)
                    error_handle.flush()
                    os.fsync(error_handle.fileno())
                    failed += 1
                    continue
                writer.writerow(_row_for_sgf(sgf_path, data))
                handle.flush()
                os.fsync(handle.fileno())
                processed += 1
                if processed % max(1, args.progress_every) == 0:
                    elapsed = time.time() - started
                    rate = processed * 60.0 / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Completed {processed} this run "
                        f"({rate:.2f} games/min elapsed-rate).",
                        file=sys.stderr,
                    )
                if (
                    args.sleep_every > 0
                    and processed % args.sleep_every == 0
                    and index < len(pending)
                ):
                    sleep_seconds = max(0.0, args.sleep_minutes * 60.0)
                    if sleep_seconds:
                        print(
                            f"Cooling down for {args.sleep_minutes:g} minutes "
                            f"after {processed} games.",
                            file=sys.stderr,
                        )
                        _sleep_interruptibly(sleep_seconds)
    finally:
        client.shutdown()

    print(
        f"Wrote {processed} rows to {output} and {failed} errors to {error_output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
