#!/usr/bin/env python3
"""Benchmark a few KataGo analysis config settings and write the fastest one."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rankmle.katago_client import (
    HUMAN_RANKS,
    KataGoClient,
    KataGoConfig,
    build_rank_policy_queries,
)
from rankmle.sgf_loader import gtp_to_index, load_sgf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "analysis_config.cfg"
DEFAULT_OUTPUT_CONFIG = REPO_ROOT / "configs" / "analysis_config.optimized.cfg"
DEFAULT_RESULTS = REPO_ROOT / "results" / "katago_config_benchmark.json"
DEFAULT_SGF_DIR = (
    REPO_ROOT / "data" / "sample-100k-medium-ranked-19x19-human-150moves-sgfs"
)
DEFAULT_KATAGO = "/opt/homebrew/bin/katago"
DEFAULT_MODEL = "~/.katrain/kata1-b28c512nbt-s12704148736-d5790336910.bin.gz"
DEFAULT_HUMAN_MODEL = "~/.katrain/b18c384nbt-humanv0.bin.gz"


@dataclass(frozen=True)
class Candidate:
    num_analysis_threads: int
    num_search_threads: int
    nn_max_batch_size: int
    nn_cache_size_power_of_two: int = 20

    @property
    def name(self) -> str:
        return (
            f"a{self.num_analysis_threads}-s{self.num_search_threads}-"
            f"b{self.nn_max_batch_size}"
        )


def _expand(path: str | Path) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def _replace_or_append(text: str, key: str, value: int | str) -> str:
    line = f"{key} = {value}"
    pattern = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=.*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(line, text)
    return text.rstrip() + "\n" + line + "\n"


def render_config(base_text: str, candidate: Candidate) -> str:
    text = base_text
    updates = {
        "numAnalysisThreads": candidate.num_analysis_threads,
        "numSearchThreads": candidate.num_search_threads,
        "nnMaxBatchSize": candidate.nn_max_batch_size,
        "nnCacheSizePowerOfTwo": candidate.nn_cache_size_power_of_two,
        "nnRandomize": "false",
        "maxVisits": 1,
    }
    for key, value in updates.items():
        text = _replace_or_append(text, key, value)
    return text


def default_candidates() -> list[Candidate]:
    return [
        Candidate(8, 1, 16),
        Candidate(16, 1, 32),
        Candidate(24, 1, 32),
        Candidate(32, 1, 64),
        Candidate(16, 2, 64),
        Candidate(24, 2, 96),
    ]


def _sample_queries(sgf_dir: Path, max_moves: int, ranks_per_move: int) -> list[dict]:
    sgfs = sorted(sgf_dir.rglob("*.sgf"))
    if not sgfs:
        raise SystemExit(f"No SGFs found under {sgf_dir}")
    game = load_sgf(str(sgfs[0]))
    ranks = HUMAN_RANKS[:: max(1, len(HUMAN_RANKS) // ranks_per_move)][:ranks_per_move]
    queries = []
    for move_idx, (_player, gtp) in enumerate(game.moves[:max_moves]):
        played_idx = gtp_to_index(gtp, game.board_size)
        for query in build_rank_policy_queries(
            game.moves[:move_idx],
            initial_stones=game.initial_stones,
            board_size=game.board_size,
            komi=game.komi,
            rules=game.rules,
            initial_player=game.initial_player,
            ranks=ranks,
        ):
            query["_played_idx"] = played_idx
            queries.append(query)
    return queries


def benchmark_candidate(
    candidate: Candidate,
    *,
    base_text: str,
    args: argparse.Namespace,
    queries: list[dict],
) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as handle:
        handle.write(render_config(base_text, candidate))
        config_path = handle.name

    cfg = KataGoConfig(
        katago=_expand(args.katago),
        model=_expand(args.model),
        human_model=_expand(args.human_model),
        config=config_path,
        extra_args=["-override-config", f"homeDataDir={_expand(args.home_data_dir)}"],
    )
    done = 0
    errors = 0
    latencies = []
    lock = threading.Lock()
    finished = threading.Event()
    started = time.time()

    def mark_done(sent_at: float, ok: bool) -> None:
        nonlocal done, errors
        with lock:
            done += 1
            errors += 0 if ok else 1
            latencies.append(time.time() - sent_at)
            if done >= len(queries):
                finished.set()

    client = KataGoClient(cfg)
    try:
        client.start()
        for query in queries:
            sent_at = time.time()
            played_idx = query.pop("_played_idx")

            def callback(msg: dict, sent_at=sent_at, played_idx=played_idx) -> None:
                policy = msg.get("humanPolicy") or []
                _ = policy[played_idx] if played_idx < len(policy) else None
                mark_done(sent_at, True)

            def error_callback(_msg: dict, sent_at=sent_at) -> None:
                mark_done(sent_at, False)

            client.send_query(query, callback, error_callback)
        timeout = max(args.timeout, len(queries) * 2)
        while not finished.wait(timeout=0.25):
            if not client.is_alive():
                raise RuntimeError("KataGo exited during benchmark")
            if time.time() - started > timeout:
                raise TimeoutError(f"benchmark timed out after {timeout}s")
    finally:
        client.shutdown()
        try:
            os.unlink(config_path)
        except OSError:
            pass

    elapsed = time.time() - started
    return {
        "candidate": candidate.__dict__,
        "name": candidate.name,
        "queries": len(queries),
        "errors": errors,
        "seconds": elapsed,
        "queries_per_second": len(queries) / elapsed if elapsed > 0 else 0.0,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--sgf-dir", type=Path, default=DEFAULT_SGF_DIR)
    parser.add_argument("--katago", default=DEFAULT_KATAGO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--human-model", default=DEFAULT_HUMAN_MODEL)
    parser.add_argument(
        "--home-data-dir",
        default="~/.katrain",
        help="passed to KataGo as -override-config homeDataDir=...",
    )
    parser.add_argument("--max-moves", type=int, default=12)
    parser.add_argument("--ranks-per-move", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base_text = args.base_config.read_text()
    queries = _sample_queries(args.sgf_dir.resolve(), args.max_moves, args.ranks_per_move)
    print(f"Benchmarking {len(queries)} queries per candidate.", file=sys.stderr)

    results = []
    for candidate in default_candidates():
        print(f"Testing {candidate.name}", file=sys.stderr)
        try:
            result = benchmark_candidate(
                candidate,
                base_text=base_text,
                args=args,
                queries=[dict(query) for query in queries],
            )
        except Exception as exc:
            result = {
                "candidate": candidate.__dict__,
                "name": candidate.name,
                "error": repr(exc),
                "queries_per_second": 0.0,
            }
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        results.append(result)

    best = max(results, key=lambda item: item.get("queries_per_second", 0.0))
    if best.get("queries_per_second", 0.0) <= 0:
        raise SystemExit("No benchmark candidate completed successfully")

    best_candidate = Candidate(**best["candidate"])
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(render_config(base_text, best_candidate))
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps({"best": best, "results": results}, indent=2))

    print(f"Wrote best config to {args.output_config}", file=sys.stderr)
    print(f"Wrote benchmark results to {args.results_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
