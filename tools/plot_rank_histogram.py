#!/usr/bin/env python3
"""Plot estimated KGS rank distributions by OGS rank bucket as an SVG chart."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results" / "sample_100k_150moves_rank_mle.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "ogs_rank_histogram.svg"
ADJUST2021 = datetime(2021, 1, 28, 9, 30)


@dataclass(frozen=True)
class DateFilters:
    begin: datetime | None = None
    end: datetime | None = None


@dataclass(frozen=True)
class Sample:
    buckets: dict[int, list[int]]
    start: date | datetime | None = None
    end: date | datetime | None = None

    def describe_dates(self) -> str:
        if self.start is None or self.end is None:
            return ""
        return f"{format_sample_date(self.start)} to {format_sample_date(self.end)}"


def rank_to_strength(rank: str) -> int | None:
    """Map ranks onto a contiguous scale: 30k=1, ..., 1k=30, 1d=31, 2d=32."""
    rank = (rank or "").strip().lower()
    if len(rank) < 2:
        return None
    suffix = rank[-1]
    try:
        value = int(rank[:-1])
    except ValueError:
        return None
    if suffix == "k" and 1 <= value <= 30:
        return 31 - value
    if suffix == "d" and 1 <= value <= 9:
        return 30 + value
    return None


def strength_to_rank(strength: int) -> str:
    if strength <= 30:
        return f"{31 - strength}k"
    return f"{strength - 30}d"


KGS_PREDICTED_MIN_STRENGTH = 11
KGS_PREDICTED_MAX_STRENGTH = 36


def predicted_strength(strength: int) -> int:
    return min(KGS_PREDICTED_MAX_STRENGTH, max(KGS_PREDICTED_MIN_STRENGTH, strength))


def predicted_rank_label(strength: int | float) -> str:
    rounded = round(strength)
    if rounded <= KGS_PREDICTED_MIN_STRENGTH:
        return "≤20k"
    if rounded >= KGS_PREDICTED_MAX_STRENGTH:
        return "≥6d"
    return strength_to_rank(rounded)


def median(values: list[int]) -> float:
    return float(statistics.median(values))


def parse_filter_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() == "adjust2021":
        return ADJUST2021
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"invalid date {value!r}; use YYYY-MM-DD, YYYY-MM-DD HH:MM, or adjust2021"
    )


def parse_row_date(value: str) -> date | datetime | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            parsed = datetime.strptime(normalized, fmt)
            return parsed.date() if fmt == "%Y-%m-%d" else parsed
        except ValueError:
            pass
    return None


def date_in_range(value: date | datetime | None, filters: DateFilters) -> bool:
    if value is None:
        return False
    if isinstance(value, datetime):
        if filters.begin and value < filters.begin:
            return False
        if filters.end and value > filters.end:
            return False
        return True

    if filters.begin and value < filters.begin.date():
        return False
    if filters.end and value > filters.end.date():
        return False
    return True


def date_sort_key(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime(value.year, value.month, value.day)


def format_sample_date(value: date | datetime) -> str:
    if isinstance(value, datetime):
        return format_filter_date(value)
    return value.strftime("%Y-%m-%d")


def format_filter_date(value: datetime) -> str:
    if value.time() == datetime.min.time():
        return value.strftime("%Y-%m-%d")
    return value.strftime("%Y-%m-%d %H:%M")


def collect_points(input_path: Path, filters: DateFilters) -> Sample:
    buckets: dict[int, list[int]] = defaultdict(list)
    sample_start: date | datetime | None = None
    sample_end: date | datetime | None = None
    with input_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row_date = parse_row_date(row.get("game_date", ""))
            if not date_in_range(row_date, filters):
                continue
            pairs = (
                ("black_ogs_rank", "black_estimated_rank"),
                ("white_ogs_rank", "white_estimated_rank"),
            )
            row_added = False
            for ogs_column, estimate_column in pairs:
                ogs_strength = rank_to_strength(row.get(ogs_column, ""))
                estimate_strength = rank_to_strength(row.get(estimate_column, ""))
                if ogs_strength is not None and estimate_strength is not None:
                    buckets[ogs_strength].append(predicted_strength(estimate_strength))
                    row_added = True
            if row_added and row_date is not None:
                if sample_start is None or date_sort_key(row_date) < date_sort_key(sample_start):
                    sample_start = row_date
                if sample_end is None or date_sort_key(row_date) > date_sort_key(sample_end):
                    sample_end = row_date
    return Sample(
        buckets=dict(sorted(buckets.items())),
        start=sample_start,
        end=sample_end,
    )


def collect_histogram(
    buckets: dict[int, list[int]],
) -> tuple[dict[tuple[int, int], int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    max_count = 0
    for ogs_strength, values in buckets.items():
        for estimate_strength in values:
            key = (ogs_strength, estimate_strength)
            counts[key] += 1
            max_count = max(max_count, counts[key])
    return dict(counts), max_count


def y_axis_ticks(min_strength: int, max_strength: int) -> list[int]:
    ticks = list(range(min_strength, max_strength + 1, 2))
    if 31 >= min_strength and 31 <= max_strength:
        ticks.append(31)
    if max_strength not in ticks:
        ticks.append(max_strength)
    return sorted(set(ticks))


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(
    sample: Sample,
    output_path: Path,
    filters: DateFilters,
) -> None:
    buckets = sample.buckets
    if not buckets:
        raise SystemExit("No rank pairs found in the CSV.")

    all_estimates = [value for values in buckets.values() for value in values]
    min_y = KGS_PREDICTED_MIN_STRENGTH
    max_y = KGS_PREDICTED_MAX_STRENGTH
    min_x = min(buckets)
    max_x = max(buckets)
    counts, _max_count = collect_histogram(buckets)

    width = max(1000, 44 * len(buckets) + 180)
    height = 760
    left = 86
    right = 32
    top = 48
    bottom = 104
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_bucket_count = max_x - min_x + 1
    y_bucket_count = max_y - min_y + 1
    cell_width = plot_width / x_bucket_count
    cell_height = plot_height / y_bucket_count

    def x_pos(strength: float) -> float:
        return left + (strength - min_x) * cell_width + cell_width / 2

    def y_pos(strength: float) -> float:
        return top + (max_y - strength) * cell_height + cell_height / 2

    subtitle = "Bars are normalized within each OGS rank; orange marks the median."
    sample_dates = sample.describe_dates()
    if sample_dates:
        subtitle += f" Sample: {sample_dates}."

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2933; }",
        ".axis { stroke: #52606d; stroke-width: 1; }",
        ".grid { stroke: #d9e2ec; stroke-width: 1; }",
        ".bar { fill: #2f80a0; fill-opacity: 0.72; }",
        ".bar-zero { stroke: #d9e2ec; stroke-width: 0.8; }",
        ".identity { stroke: #475569; stroke-width: 1.5; stroke-dasharray: 2 6; stroke-linecap: round; opacity: 0.42; }",
        ".median { fill: none; stroke: #f97316; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }",
        ".median-point { fill: #f97316; stroke: #ffffff; stroke-width: 1.5; }",
        ".count { fill: #627d98; font-size: 10px; }",
        ".label { font-size: 12px; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { fill: #627d98; font-size: 13px; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text class="title" x="{left}" y="28">Estimated KGS rank by OGS rank bucket</text>',
        f'<text class="subtitle" x="{left}" y="47">{svg_escape(subtitle)}</text>',
    ]

    for tick in y_axis_ticks(min_y, max_y):
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(
            f'<text class="label" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">'
            f'{svg_escape(predicted_rank_label(tick))}</text>'
        )
        lines.append(
            f'<text class="label" x="{width - right + 10}" y="{y + 4:.2f}" text-anchor="start">'
            f'{svg_escape(predicted_rank_label(tick))}</text>'
        )

    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>')
    lines.append(f'<line class="axis" x1="{width - right}" y1="{top}" x2="{width - right}" y2="{height - bottom}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>')

    for ogs_strength in range(min_x, max_x + 1):
        if ogs_strength not in buckets:
            continue
        column_max = max(
            counts.get((ogs_strength, estimate_strength), 0)
            for estimate_strength in range(min_y, max_y + 1)
        )
        center_x = x_pos(ogs_strength)
        max_bar_width = cell_width * 0.84
        zero_y1 = top
        zero_y2 = top + plot_height
        lines.append(
            f'<line class="bar-zero" x1="{center_x:.2f}" y1="{zero_y1:.2f}" '
            f'x2="{center_x:.2f}" y2="{zero_y2:.2f}"/>'
        )
        for estimate_strength in range(min_y, max_y + 1):
            count = counts.get((ogs_strength, estimate_strength), 0)
            if count <= 0 or column_max <= 0:
                continue
            bar_width = max_bar_width * count / column_max
            x = center_x - bar_width / 2
            y = top + (max_y - estimate_strength) * cell_height + cell_height * 0.16
            bar_height = cell_height * 0.68
            lines.append(
                f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" rx="1.5">'
                f'<title>{strength_to_rank(ogs_strength)}: {count} estimated as {svg_escape(predicted_rank_label(estimate_strength))}</title>'
                "</rect>"
            )

    identity_start = max(min_x, min_y)
    identity_end = min(max_x, max_y)
    if identity_start <= identity_end:
        lines.append(
            f'<line class="identity" x1="{x_pos(identity_start):.2f}" '
            f'y1="{y_pos(identity_start):.2f}" x2="{x_pos(identity_end):.2f}" '
            f'y2="{y_pos(identity_end):.2f}"><title>x = y</title></line>'
        )

    median_points = [
        (x_pos(strength), y_pos(median(values)))
        for strength, values in buckets.items()
    ]
    median_path = " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(median_points)
    )
    lines.append(f'<path class="median" d="{median_path}"/>')
    for x, y in median_points:
        lines.append(f'<circle class="median-point" cx="{x:.2f}" cy="{y:.2f}" r="3.5"/>')

    for strength in range(min_x, max_x + 1):
        if strength not in buckets:
            continue
        x = x_pos(strength)
        lines.append(
            f'<line class="axis" x1="{x:.2f}" y1="{height - bottom}" '
            f'x2="{x:.2f}" y2="{height - bottom + 5}"/>'
        )
        label = strength_to_rank(strength)
        lines.append(
            f'<text class="label" x="{x:.2f}" y="{height - bottom + 20}" '
            f'text-anchor="middle" transform="rotate(-45 {x:.2f} {height - bottom + 20})">'
            f'{svg_escape(label)}</text>'
        )
        lines.append(
            f'<text class="count" x="{x:.2f}" y="{height - bottom + 38}" text-anchor="middle">'
            f'n={len(buckets[strength])}</text>'
        )

    y_label_x = 22
    y_label_y = top + plot_height / 2
    lines.append(
        f'<text class="label" x="{y_label_x}" y="{y_label_y:.2f}" '
        f'text-anchor="middle" transform="rotate(-90 {y_label_x} {y_label_y:.2f})">'
        "Estimated KGS rank</text>"
    )
    lines.append(
        f'<text class="label" x="{left + plot_width / 2:.2f}" y="{height - 16}" text-anchor="middle">'
        "OGS rank bucket</text>"
    )
    lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--begin-date",
        type=parse_filter_date,
        help=(
            "inclusive game_date lower bound; use YYYY-MM-DD, YYYY-MM-DD HH:MM, "
            "or adjust2021 for 2021-01-28 09:30 SGT"
        ),
    )
    parser.add_argument(
        "--end-date",
        type=parse_filter_date,
        help=(
            "inclusive game_date upper bound; use YYYY-MM-DD, YYYY-MM-DD HH:MM, "
            "or adjust2021 for 2021-01-28 09:30 SGT"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    filters = DateFilters(begin=args.begin_date, end=args.end_date)
    if filters.begin and filters.end and filters.begin > filters.end:
        raise SystemExit("--begin-date must be earlier than or equal to --end-date")
    sample = collect_points(args.input, filters)
    render_svg(sample, args.output, filters)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
