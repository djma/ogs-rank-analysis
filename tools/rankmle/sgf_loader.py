#!/usr/bin/env python3
"""Small SGF metadata and main-line move parser for OGS SGFs."""

from __future__ import annotations

from dataclasses import dataclass

_GTP_COLS = "ABCDEFGHJKLMNOPQRST"
_RULES_MAP = {
    "aga": "aga",
    "bga": "bga",
    "chinese": "chinese",
    "cn": "chinese",
    "jp": "japanese",
    "japanese": "japanese",
    "korean": "korean",
    "kr": "korean",
    "new-zealand": "new-zealand",
    "nz": "new-zealand",
    "stone-scoring": "stone-scoring",
    "tromp-taylor": "tromp-taylor",
    "tt": "tromp-taylor",
}


@dataclass
class LoadedGame:
    sgf_path: str
    date: str
    board_size: tuple[int, int]
    komi: float
    rules: str
    initial_player: str
    initial_stones: list[tuple[str, str]]
    moves: list[tuple[str, str]]
    players: dict


def _split_nodes(text: str) -> list[str]:
    nodes = []
    start = None
    in_prop = False
    escape = False
    for idx, char in enumerate(text):
        if in_prop:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "]":
                in_prop = False
        elif char == "[":
            in_prop = True
        elif char == ";":
            if start is not None:
                nodes.append(text[start:idx])
            start = idx + 1
    if start is not None:
        nodes.append(text[start:])
    return nodes


def _parse_props(node: str) -> dict[str, list[str]]:
    props: dict[str, list[str]] = {}
    idx = 0
    while idx < len(node):
        while idx < len(node) and not node[idx].isalpha():
            idx += 1
        key_start = idx
        while idx < len(node) and node[idx].isalpha():
            idx += 1
        if idx == key_start or idx >= len(node) or node[idx] != "[":
            idx += 1
            continue
        key = node[key_start:idx].upper()
        values = []
        while idx < len(node) and node[idx] == "[":
            idx += 1
            out = []
            escape = False
            while idx < len(node):
                char = node[idx]
                idx += 1
                if escape:
                    if char not in "\r\n":
                        out.append(char)
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == "]":
                    break
                else:
                    out.append(char)
            values.append("".join(out))
        props.setdefault(key, []).extend(values)
    return props


def _first(props: dict[str, list[str]], key: str, default: str = "") -> str:
    values = props.get(key)
    return values[0] if values else default


def _sgf_coord_to_gtp(coord: str, board_size: int) -> str:
    if coord == "" or coord.lower() == "tt":
        return "pass"
    if len(coord) < 2:
        return "pass"
    col = ord(coord[0].lower()) - ord("a")
    row_from_top = ord(coord[1].lower()) - ord("a")
    if not (0 <= col < board_size and 0 <= row_from_top < board_size):
        return "pass"
    row = board_size - row_from_top
    return f"{_GTP_COLS[col]}{row}"


def gtp_to_index(gtp: str, board_size: tuple[int, int]) -> int:
    bx, by = board_size
    if gtp == "pass":
        return bx * by
    col = _GTP_COLS.index(gtp[0])
    row = int(gtp[1:]) - 1
    return (by - 1 - row) * bx + col


def normalize_rules(rules: str) -> str:
    return _RULES_MAP.get((rules or "").lower().strip(), "japanese")


def load_sgf(path: str) -> LoadedGame:
    body = open(path, "rb").read()
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("latin-1")

    nodes = _split_nodes(text)
    if not nodes:
        raise ValueError(f"{path} has no SGF nodes")
    root = _parse_props(nodes[0])
    board_size = int(_first(root, "SZ", "19").split(":")[0] or 19)
    try:
        komi = float(_first(root, "KM", "6.5"))
    except ValueError:
        komi = 6.5
    if abs(komi) > 50:
        komi = 6.5

    initial_stones = []
    for coord in root.get("AB", []):
        initial_stones.append(("B", _sgf_coord_to_gtp(coord, board_size)))
    for coord in root.get("AW", []):
        initial_stones.append(("W", _sgf_coord_to_gtp(coord, board_size)))

    initial_player = _first(root, "PL", "B")
    if initial_player not in ("B", "W"):
        initial_player = "B"

    moves = []
    for node in nodes[1:]:
        props = _parse_props(node)
        for player in ("B", "W"):
            if player in props:
                moves.append((player, _sgf_coord_to_gtp(props[player][0], board_size)))
                break

    return LoadedGame(
        sgf_path=path,
        date=_first(root, "DT"),
        board_size=(board_size, board_size),
        komi=komi,
        rules=normalize_rules(_first(root, "RU", "japanese")),
        initial_player=initial_player,
        initial_stones=initial_stones,
        moves=moves,
        players={
            "B": {"name": _first(root, "PB"), "rating": _first(root, "BR")},
            "W": {"name": _first(root, "PW"), "rating": _first(root, "WR")},
        },
    )
