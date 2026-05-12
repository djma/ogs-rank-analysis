#!/usr/bin/env python3
"""Human-rank maximum-likelihood estimator for one SGF.

Adapted from https://github.com/djma/rankmle/blob/master/rank_mle.py for this
workspace. It queries KataGo's human policy model for every legal human rank
profile and selects the rank with the highest mean log-likelihood per player.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time

try:
    from .katago_client import HUMAN_RANKS, KataGoClient, build_rank_policy_queries
    from .sgf_loader import LoadedGame, gtp_to_index, load_sgf
except ImportError:
    from katago_client import HUMAN_RANKS, KataGoClient, build_rank_policy_queries
    from sgf_loader import LoadedGame, gtp_to_index, load_sgf

CACHE_VERSION = 2
CACHE_DIRNAME = ".rank_mle_cache"
EPS = 1e-7


def _sha_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _cache_path(sgf_path: str, sha: str) -> str:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(sgf_path)), CACHE_DIRNAME)
    return os.path.join(cache_dir, f"{os.path.basename(sgf_path)}.{sha}.json")


def _load_cache(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("version") == CACHE_VERSION else None


def _save_cache(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(data, handle, separators=(",", ":"))
    os.replace(tmp, path)


def _empty_stats(n_ranks: int) -> dict:
    return {
        player: {"n_moves": 0, "loglik_sums": [0.0] * n_ranks}
        for player in ("B", "W")
    }


def _analyze_game(
    game: LoadedGame,
    client: KataGoClient,
    progress_cb=None,
    ranks: list[str] = HUMAN_RANKS,
) -> dict:
    n_moves = len(game.moves)
    n_ranks = len(ranks)
    if n_moves == 0:
        return {
            "version": CACHE_VERSION,
            "ranks": ranks,
            "stats": _empty_stats(n_ranks),
            "total_moves": 0,
            "players": game.players,
        }

    move_players = [move[0] for move in game.moves]
    fallback_log_p = math.log(EPS)
    stats = _empty_stats(n_ranks)
    for player in move_players:
        stats[player]["n_moves"] += 1
        stats[player]["loglik_sums"] = [
            value + fallback_log_p for value in stats[player]["loglik_sums"]
        ]

    lock = threading.Lock()
    done_count = [0]
    error_count = [0]
    first_error: list[str | None] = [None]
    expected = n_moves * n_ranks
    done_event = threading.Event()
    last_log = [time.time()]
    rank_indices = {rank: idx for idx, rank in enumerate(ranks)}

    def make_cb(move_idx: int, rank: str, played_idx: int):
        def cb(msg: dict) -> None:
            policy = msg.get("humanPolicy") or []
            prob = float(policy[played_idx]) if 0 <= played_idx < len(policy) else 0.0
            log_p = math.log(max(prob, EPS))
            with lock:
                done_count[0] += 1
                rank_idx = rank_indices[rank]
                player = move_players[move_idx]
                stats[player]["loglik_sums"][rank_idx] += log_p - fallback_log_p
                now = time.time()
                if now - last_log[0] > 2.0:
                    print(f"  {done_count[0]}/{expected} rank queries done", file=sys.stderr)
                    last_log[0] = now
                if progress_cb is not None:
                    progress_cb(done_count[0], expected)
                if done_count[0] >= expected:
                    done_event.set()

        return cb

    def err_cb(msg: dict) -> None:
        err = msg.get("error", "unknown error")
        print(f"  katago error: {err}", file=sys.stderr)
        with lock:
            error_count[0] += 1
            if first_error[0] is None:
                first_error[0] = err
            done_count[0] += 1
            if done_count[0] >= expected:
                done_event.set()

    for move_idx, (_player, gtp) in enumerate(game.moves):
        prefix = game.moves[:move_idx]
        played_idx = gtp_to_index(gtp, game.board_size)
        queries = build_rank_policy_queries(
            prefix,
            initial_stones=game.initial_stones,
            board_size=game.board_size,
            komi=game.komi,
            rules=game.rules,
            initial_player=game.initial_player,
            ranks=ranks,
        )
        for rank, query in zip(ranks, queries):
            client.send_query(query, make_cb(move_idx, rank, played_idx), err_cb)

    while not done_event.is_set():
        if not client.is_alive():
            raise RuntimeError("KataGo process died during rank analysis")
        done_event.wait(timeout=0.5)

    if error_count[0] == expected:
        raise RuntimeError(f"KataGo rejected all {expected} queries: {first_error[0]!r}")

    return {
        "version": CACHE_VERSION,
        "ranks": ranks,
        "stats": stats,
        "total_moves": n_moves,
        "players": game.players,
    }


def analyze_path(
    client: KataGoClient,
    sgf_path: str,
    *,
    use_cache: bool = True,
    progress_cb=None,
    ranks: list[str] = HUMAN_RANKS,
) -> dict:
    sha = _sha_file(sgf_path)
    cache_path = _cache_path(sgf_path, sha)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None and cached.get("ranks") == ranks:
            if progress_cb is not None:
                progress_cb(1, 1)
            return cached
    data = _analyze_game(load_sgf(sgf_path), client, progress_cb=progress_cb, ranks=ranks)
    _save_cache(cache_path, data)
    return data


def predict_per_player(data: dict) -> dict:
    ranks = data["ranks"]
    n_ranks = len(ranks)
    out = {}
    for player in ("B", "W"):
        player_stats = (data.get("stats") or {}).get(player, {})
        n_moves = player_stats.get("n_moves", 0)
        if not n_moves:
            out[player] = {"rank": None, "n_moves": 0}
            continue
        sums = player_stats.get("loglik_sums", [])[:n_ranks]
        sums = sums + [math.log(EPS) * n_moves] * (n_ranks - len(sums))
        means = [value / n_moves for value in sums]
        best = max(range(n_ranks), key=lambda idx: means[idx])
        out[player] = {
            "rank": ranks[best].replace("rank_", ""),
            "n_moves": n_moves,
            "mean_loglik": means[best],
        }
    return out

