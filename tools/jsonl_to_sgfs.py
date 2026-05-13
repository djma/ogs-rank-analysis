#!/usr/bin/env python3
import argparse
import datetime as dt
import gzip
import json
import math
import random
import re
import unicodedata
from pathlib import Path


SGF_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
SGF_ESCAPE_TABLE = str.maketrans({"]": r"\]", "\\": r"\\", "\0": "", "[": r"\["})
UPSTREAM_BOT_USERNAMES = {
    "unknown",
    "Natsu (Fuego)",
    "Billy (GnuGo lvl10)",
    "TheKid (GnuGo lvl1)",
    "Random Bot",
    "GNU Go Level 1",
    "Kugutsu",
}
BLACKLISTED_GAME_IDS = {
    427508,
    428481,
    429705,
    455926,
    459488,
    474293,
    526904,
    552120,
    657172,
    40163786,
    40179045,
    40307714,
    40460301,
    40886435,
    40907427,
    41234623,
    41477368,
    42032919,
    42036884,
    42283009,
    44168992,
    44168998,
    49397680,
    49833763,
    59445457,
    64514119,
    65190381,
    70640302,
    71822605,
}
BOT_USERNAME_MARKERS = (
    "bot",
    "fuego",
    "gnugo",
    "gnu go",
    "katago",
    "leela",
    "pachi",
    "kugutsu",
)


class SgfException(Exception):
    pass


def rankstr(rank):
    if rank >= 30.0:
        return f"{min(9, int(math.floor(rank) - 29))}d"
    return f"{min(30, int(30 - math.floor(rank)))}k"


def sgfescape(value):
    return str(value).translate(SGF_ESCAPE_TABLE)


def param(key, value):
    return f"{key}[{sgfescape(value)}]"


def get(obj, field, default=None, required=False, game_id=None):
    if field in obj:
        return obj[field]
    if required:
        game_id = game_id or obj.get("game_id") or "Unknown"
        raise SgfException(f"JSON field not found for game {game_id}: {field}")
    return default


def get_sgf(game_id, sgf, field):
    match = re.search(field + r"\[([^][]+)\]", sgf)
    if match:
        return match.group(1)
    return None


def normalized_date_from_sgf_date(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            parsed = dt.datetime.strptime(value[: len(fmt)], fmt)
            return parsed.date()
        except ValueError:
            pass
    return None


def is_medium_ranked_19x19_game(ogsdata):
    """Return True for 19x19 ranked byoyomi games in the target time band."""
    time_control = ogsdata.get("time_control") or {}
    system = time_control.get("time_control") or time_control.get("system")

    return (
        ogsdata.get("width") == 19
        and ogsdata.get("height") == 19
        and ogsdata.get("ranked") is True
        and ogsdata.get("handicap", 0) <= 2
        and system == "byoyomi"
        and 15 * 60 <= time_control.get("main_time", -1) <= 40 * 60
        and 3 <= time_control.get("periods", -1) <= 8
        and time_control.get("period_time") == 30
    )


def is_not_blacklisted_game(ogsdata):
    try:
        game_id = int(ogsdata.get("game_id"))
    except (TypeError, ValueError):
        return True
    return game_id not in BLACKLISTED_GAME_IDS


def is_bot_player(player):
    """Best-effort bot detection for OGS dump player objects.

    The sample dump does not include a reliable bot flag. This checks any
    explicit bot-like booleans if present, then falls back to the upstream
    za3k/ogs bot username blacklist plus common OGS bot name markers.
    """
    if not player:
        return True

    for field in ("bot", "is_bot", "robot"):
        if player.get(field) is True:
            return True

    username = player.get("username")
    if not username:
        return True
    if username in UPSTREAM_BOT_USERNAMES:
        return True

    normalized = username.casefold()
    return any(marker in normalized for marker in BOT_USERNAME_MARKERS)


def is_human_vs_human_game(ogsdata):
    players = ogsdata.get("players") or {}
    return not is_bot_player(players.get("black")) and not is_bot_player(players.get("white"))


def is_medium_ranked_19x19_human_game(ogsdata):
    return is_medium_ranked_19x19_game(ogsdata) and is_human_vs_human_game(ogsdata) and is_not_blacklisted_game(ogsdata)


def has_at_least_moves(ogsdata, minimum):
    return len(ogsdata.get("moves") or []) >= minimum


def starts_on_or_after(ogsdata, start_date):
    start_time = ogsdata.get("start_time")
    if start_time is None:
        return False
    return dt.datetime.utcfromtimestamp(start_time).date() >= start_date


def construct_sgf(ogsdata):
    out = "(;FF[4]CA[UTF-8]GM[1]US[za3k/ogstosgf.py]"
    extra_info = []

    game_id = get(ogsdata, "game_id", required=True)
    out += param("PC", f"OGS: https://online-go.com/game/{game_id}")

    game_date = None
    start_time = get(ogsdata, "start_time")
    if start_time is not None:
        game_date = dt.datetime.utcfromtimestamp(start_time).date()
        out += param("DT", game_date.strftime("%Y-%m-%d"))

    black_username = None
    white_username = None
    players = get(ogsdata, "players", required=True)
    black = get(players, "black", required=True)
    white = get(players, "white", required=True)

    if black:
        black_username = get(black, "username", required=True, game_id=game_id)
        out += param("PB", black_username)
        black_rank = get(black, "rank")
        if black_rank is not None:
            out += param("BR", rankstr(black_rank))

    if white:
        white_username = get(white, "username", required=True, game_id=game_id)
        out += param("PW", white_username)
        white_rank = get(white, "rank")
        if white_rank is not None:
            out += param("WR", rankstr(white_rank))

    original_sgf = get(ogsdata, "original_sgf")
    if original_sgf is not None:
        if original_sgf == 0:
            raise SgfException("Invalid uploaded SGF")
        sgf_date = normalized_date_from_sgf_date(get_sgf(game_id, original_sgf, "DT"))
        game_date = sgf_date or game_date
        black_username = get_sgf(game_id, original_sgf, "PB") or black_username
        white_username = get_sgf(game_id, original_sgf, "PW") or white_username
        return True, ([black_username, get(ogsdata, "black_player_id")], [white_username, get(ogsdata, "white_player_id")], game_date, game_id), original_sgf.rstrip() + "\n"

    game_name = get(ogsdata, "game_name")
    if game_name is not None:
        out += param("GN", game_name)

    time_control = get(ogsdata, "time_control")
    if time_control is not None:
        system = get(time_control, "time_control") or get(time_control, "system")
        if system == "byoyomi":
            main_time = get(time_control, "main_time", required=True, game_id=game_id)
            period_time = get(time_control, "period_time", required=True, game_id=game_id)
            periods = get(time_control, "periods", required=True, game_id=game_id)
            out += param("TM", main_time)
            out += param("OT", f"{periods}x{period_time} byo-yomi")
        elif system == "fischer":
            initial_time = get(time_control, "initial_time", required=True, game_id=game_id)
            time_increment = get(time_control, "time_increment", required=True, game_id=game_id)
            out += param("TM", initial_time)
            out += param("OT", f"{time_increment} fischer")
        elif system == "simple":
            per_move = get(time_control, "per_move", required=True, game_id=game_id)
            out += param("TM", 0)
            out += param("OT", f"{per_move} simple")
        elif system == "canadian":
            main_time = get(time_control, "main_time", required=True, game_id=game_id)
            period_time = get(time_control, "period_time", required=True, game_id=game_id)
            stones_per_period = get(time_control, "stones_per_period", required=True, game_id=game_id)
            out += param("TM", main_time)
            out += param("OT", f"{stones_per_period}/{period_time} canadian")
        elif system == "absolute":
            out += param("TM", get(time_control, "total_time", required=True, game_id=game_id))
        elif system == "none":
            pass
        else:
            raise SgfException(f"Unknown time control for game {game_id}: {system}")

        speed = get(time_control, "speed")
        extra_info.append(speed or ("none" if system == "none" else "unknown"))

    winner = get(ogsdata, "winner")
    outcome = get(ogsdata, "outcome")
    white_player_id = get(ogsdata, "white_player_id", required=True)
    black_player_id = get(ogsdata, "black_player_id", required=True)
    if winner is not None or outcome is not None:
        if outcome == "0 points":
            out += param("RE", "Draw")
        elif outcome and not winner:
            out += param("RE", "Void")
        else:
            if winner == white_player_id:
                winner_sgf = "W"
            elif winner == black_player_id:
                winner_sgf = "B"
            else:
                raise SgfException(f"Unknown winner '{winner}' for game {game_id}")
            if outcome.endswith(" points"):
                out += param("RE", f"{winner_sgf}+{outcome[:-7]}")
            elif outcome == "1 point":
                out += param("RE", f"{winner_sgf}+1")
            elif outcome == "Resignation":
                out += param("RE", f"{winner_sgf}+R")
            elif outcome == "Timeout":
                out += param("RE", f"{winner_sgf}+T")
            elif outcome in (
                "Cancellation",
                "Disconnection",
                "Game not started",
                "Moderator Decision",
                "Decision",
                "Disqualification",
                "Ladder Withdrawn",
                "TD Decision",
                "Player's account was removed",
                "Opponent's account was removed",
                "Banned user",
                "Server Decision",
            ):
                out += param("RE", f"{winner_sgf}+F")
            elif "Server Decision" in outcome:
                match = re.match(r"Server Decision \(\d+\.\d+% ([WB]+.*)\)", outcome)
                out += param("RE", match.group(1) if match else f"{winner_sgf}+F")
            else:
                raise SgfException(f"Unknown outcome '{outcome}' for game {game_id}")
    else:
        out += param("RE", "?")

    width = get(ogsdata, "width", default=19, required=True)
    height = get(ogsdata, "height", default=19, required=True)
    out += param("SZ", f"{width}:{height}" if width != height else width)

    komi = get(ogsdata, "komi", required=True)
    out += param("KM", komi)

    rules = get(ogsdata, "rules", required=True)
    rules_map = {
        "chinese": "Chinese",
        "japanese": "Japanese",
        "korean": "Korean",
        "nz": "NZ",
        "aga": "AGA",
        "ing": "Ing",
    }
    out += param("RU", rules_map.get(str(rules).lower(), rules))

    handicap = get(ogsdata, "handicap", default=0)
    if handicap > 0:
        out += param("HA", handicap)

    extra_info.append("ranked" if get(ogsdata, "ranked", default=False, required=True) else "unranked")
    out += param("GC", ",".join(extra_info))

    initial_player = get(ogsdata, "initial_player") or "black"
    if initial_player == "white":
        out += param("PL", "W")
    elif initial_player != "black":
        raise SgfException(f"Unknown initial player for game {game_id}")

    initial_state = get(ogsdata, "initial_state")
    if initial_state is not None:
        for key, sgf_key in (("black", "AB"), ("white", "AW")):
            state = get(initial_state, key, required=True, game_id=game_id)
            if state:
                out += sgf_key + "".join(f"[{state[i:i + 2]}]" for i in range(0, len(state), 2))

    moves = get(ogsdata, "moves", required=True)
    blacknext = initial_player == "black"
    for idx, move in enumerate(moves):
        if idx < handicap:
            if idx == 0:
                out += "AB"
            out += f"[{SGF_LETTERS[move[0]]}{SGF_LETTERS[move[1]]}]"
            if idx == handicap - 1:
                blacknext = False
            continue

        out += ";B[" if blacknext else ";W["
        if move[0] >= 0 and move[1] >= 0:
            out += SGF_LETTERS[move[0]] + SGF_LETTERS[move[1]]
        out += "]"
        blacknext = not blacknext

    return False, ([black_username, black_player_id], [white_username, white_player_id], game_date, game_id), out + ")\n"


def name_ok(name):
    if name is None:
        return False
    name = str(name).strip()
    if name in {".", "..", ""} or len(name) > 100:
        return False
    for char in name:
        if char in "/\\\0" or unicodedata.category(char) == "Cc":
            return False
    return True


def filesystem_name(name, player_id):
    if name_ok(name):
        return str(name).strip()
    if player_id is None:
        return "unknown"
    return str(player_id)


def write_sgf(output_dir, is_export, black, white, game_date, game_id, sgf_content):
    black_name = filesystem_name(*black)
    white_name = filesystem_name(*white)
    base = "sgfs-uploaded" if is_export else "sgfs-by-date"
    if game_date:
        rel = Path(base) / str(game_date.year) / f"{game_date.month:02d}" / f"{game_date.day:02d}"
    else:
        rel = Path(base) / "unknown"
    directory = output_dir / rel
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{game_id}-{black_name}-{white_name}.sgf"
    path.write_text(sgf_content, encoding="utf-8")
    return path


def convert(input_path, output_dir, limit=None, record_filter=None, sample_rate=None, sample_seed=0):
    rng = random.Random(sample_seed)
    converted = 0
    filtered = 0
    failed = 0
    sampled_out = 0
    failure_log = output_dir / "failures.jsonl"
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    def print_progress(line_no):
        print(
            f"processed={line_no} sampled_out={sampled_out} converted={converted} filtered={filtered} failed={failed}",
            flush=True,
        )

    with gzip.open(input_path, "rt", encoding="utf-8") as infile, failure_log.open("w", encoding="utf-8") as failures:
        for line_no, line in enumerate(infile, 1):
            if limit is not None and line_no > limit:
                break
            if not line.strip():
                continue
            if sample_rate is not None and rng.random() >= sample_rate:
                sampled_out += 1
                if line_no % 10000 == 0:
                    print_progress(line_no)
                continue
            try:
                obj = json.loads(line)
                if record_filter is not None and not record_filter(obj):
                    filtered += 1
                    continue
                is_export, (black, white, game_date, game_id), sgf_content = construct_sgf(obj)
                write_sgf(output_dir, is_export, black, white, game_date, game_id, sgf_content)
                converted += 1
            except Exception as exc:
                failed += 1
                failures.write(json.dumps({"line": line_no, "error": str(exc), "record": line.rstrip()}, ensure_ascii=False) + "\n")
            if line_no % 10000 == 0:
                print_progress(line_no)

    print(f"done converted={converted} sampled_out={sampled_out} filtered={filtered} failed={failed} output={output_dir}")


def parse_date(value):
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def main():
    parser = argparse.ArgumentParser(description="Convert za3k OGS JSONL gzip dumps to SGF files.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--medium-ranked-19x19",
        action="store_true",
        help="Only convert 19x19 ranked byoyomi games with handicap <= 2, 15-40m main time, and 3-8x30s overtime.",
    )
    parser.add_argument(
        "--human-vs-human",
        action="store_true",
        help="Only convert games where both players pass best-effort human player detection.",
    )
    parser.add_argument("--min-moves", type=int, help="Only convert games with at least this many moves.")
    parser.add_argument("--start-date", type=parse_date, help="Only convert games with UTC start_time date on or after YYYY-MM-DD. e.g. 2021-09-23")
    parser.add_argument("--sample-rate", type=float, help="Randomly keep this fraction of input records before filters.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for reproducible --sample-rate sampling.")
    args = parser.parse_args()

    if args.sample_rate is not None and not 0 < args.sample_rate <= 1:
        parser.error("--sample-rate must be in (0, 1]")

    filters = []
    if args.medium_ranked_19x19:
        filters.append(is_medium_ranked_19x19_game)
    if args.human_vs_human:
        filters.append(is_human_vs_human_game)
    filters.append(is_not_blacklisted_game)
    if args.min_moves is not None:
        filters.append(lambda obj: has_at_least_moves(obj, args.min_moves))
    if args.start_date is not None:
        filters.append(lambda obj: starts_on_or_after(obj, args.start_date))

    record_filter = None
    if filters:
        record_filter = lambda obj: all(filter_func(obj) for filter_func in filters)
    convert(
        args.input,
        args.output,
        args.limit,
        record_filter=record_filter,
        sample_rate=args.sample_rate,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
