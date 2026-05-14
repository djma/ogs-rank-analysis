#!/usr/bin/env python3
"""Plot yearly median predicted rank for selected OGS ranks."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from plot_rank_histogram import (
    ADJUST2021,
    KGS_PREDICTED_MAX_STRENGTH,
    KGS_PREDICTED_MIN_STRENGTH,
    parse_row_date,
    predicted_rank_label,
    predicted_strength,
    rank_to_strength,
    strength_to_rank,
    svg_escape,
    y_axis_ticks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results" / "sample_100k_150moves_rank_mle.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "ogs_rank_yearly_medians.svg"
DEFAULT_RANKS = ("20k", "15k", "10k", "5k", "1d")


@dataclass(frozen=True, order=True)
class YearBucket:
    start_year: int

    @property
    def start(self) -> date:
        return date(self.start_year, ADJUST2021.month, ADJUST2021.day)

    @property
    def end(self) -> date:
        return date(self.start_year + 1, ADJUST2021.month, ADJUST2021.day)

    def label(self) -> str:
        return f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"

    def short_label(self) -> str:
        return f"{self.start_year}"


def bucket_for_date(value: date) -> YearBucket:
    anchor = ADJUST2021.date()
    candidate_year = value.year
    candidate_start = date(candidate_year, anchor.month, anchor.day)
    if value < candidate_start:
        candidate_year -= 1
    return YearBucket(candidate_year)


def median(values: list[int]) -> float:
    return float(statistics.median(values))


def percentile(values: list[int], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_rank_list(values: list[str] | None) -> list[int]:
    labels = values or list(DEFAULT_RANKS)
    strengths = []
    for label in labels:
        strength = rank_to_strength(label)
        if strength is None:
            raise argparse.ArgumentTypeError(f"invalid rank: {label}")
        strengths.append(strength)
    return strengths


def collect_yearly_buckets(
    input_path: Path,
    target_strengths: set[int],
) -> dict[int, dict[YearBucket, list[int]]]:
    series: dict[int, dict[YearBucket, list[int]]] = {
        strength: defaultdict(list)
        for strength in sorted(target_strengths)
    }
    with input_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed_date = parse_row_date(row.get("game_date", ""))
            if parsed_date is None:
                continue
            row_date = parsed_date.date() if hasattr(parsed_date, "date") else parsed_date
            bucket = bucket_for_date(row_date)
            pairs = (
                ("black_ogs_rank", "black_estimated_rank"),
                ("white_ogs_rank", "white_estimated_rank"),
            )
            for ogs_column, estimate_column in pairs:
                ogs_strength = rank_to_strength(row.get(ogs_column, ""))
                if ogs_strength not in target_strengths:
                    continue
                estimate_strength = rank_to_strength(row.get(estimate_column, ""))
                if estimate_strength is not None:
                    series[ogs_strength][bucket].append(predicted_strength(estimate_strength))
    return {
        strength: dict(sorted(buckets.items()))
        for strength, buckets in series.items()
    }


def all_buckets(series: dict[int, dict[YearBucket, list[int]]]) -> list[YearBucket]:
    observed = sorted({bucket for buckets in series.values() for bucket in buckets})
    if not observed:
        return []
    return [
        YearBucket(year)
        for year in range(observed[0].start_year, observed[-1].start_year + 1)
    ]


def color_for_rank(strength: int, min_strength: int, max_strength: int) -> str:
    """Ordered blue-to-red palette where stronger ranks are warmer."""
    if max_strength == min_strength:
        ratio = 0.5
    else:
        ratio = (strength - min_strength) / (max_strength - min_strength)
    stops = [
        (0.00, (37, 99, 235)),    # weaker: blue
        (0.28, (20, 184, 166)),   # teal
        (0.52, (34, 197, 94)),    # green
        (0.76, (245, 158, 11)),   # amber
        (1.00, (220, 38, 38)),    # stronger: red
    ]
    for (left_at, left_color), (right_at, right_color) in zip(stops, stops[1:]):
        if ratio <= right_at:
            span = right_at - left_at
            local = 0.0 if span == 0 else (ratio - left_at) / span
            rgb = tuple(
                round(left + (right - left) * local)
                for left, right in zip(left_color, right_color)
            )
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    red = stops[-1][1]
    return f"#{red[0]:02x}{red[1]:02x}{red[2]:02x}"


def contiguous_segments(
    points: list[tuple[int, float, float, float, YearBucket, int]],
) -> list[list[tuple[int, float, float, float, YearBucket, int]]]:
    if not points:
        return []
    segments = [[points[0]]]
    for point in points[1:]:
        previous = segments[-1][-1]
        if point[4].start_year == previous[4].start_year + 1:
            segments[-1].append(point)
        else:
            segments.append([point])
    return segments


def render_svg(
    series: dict[int, dict[YearBucket, list[int]]],
    bucket_order: list[YearBucket],
    target_strengths: list[int],
    output_path: Path,
) -> None:
    if not bucket_order:
        raise SystemExit("No rank pairs found in the CSV for the requested ranks.")

    all_values = [
        value
        for buckets in series.values()
        for values in buckets.values()
        for value in values
    ]
    min_y = KGS_PREDICTED_MIN_STRENGTH
    max_y = KGS_PREDICTED_MAX_STRENGTH

    width = max(1000, 90 * len(bucket_order) + 240)
    height = 720
    left = 86
    right = 188
    top = 56
    bottom = 96
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_count = max(1, len(bucket_order) - 1)
    y_bucket_count = max_y - min_y + 1

    def x_pos(index: int) -> float:
        return left + (index / x_count) * plot_width

    def y_pos(strength: float) -> float:
        return top + (max_y - strength) / y_bucket_count * plot_height + plot_height / y_bucket_count / 2

    bucket_index = {bucket: index for index, bucket in enumerate(bucket_order)}

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2933; }",
        ".axis { stroke: #52606d; stroke-width: 1; }",
        ".grid { stroke: #d9e2ec; stroke-width: 1; }",
        ".series-line { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }",
        ".iqr-whisker { stroke-width: 2; stroke-linecap: round; opacity: 0.55; }",
        ".iqr-cap { stroke-width: 2; stroke-linecap: round; opacity: 0.55; }",
        ".series-point { stroke: #ffffff; stroke-width: 1.4; }",
        ".label { font-size: 12px; }",
        ".xcount { fill: #627d98; font-size: 10px; }",
        ".legend { font-size: 12px; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { fill: #627d98; font-size: 13px; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text class="title" x="{left}" y="30">Yearly median predicted KGS rank by OGS rank</text>',
        f'<text class="subtitle" x="{left}" y="49">Year buckets run Jan 28 to Jan 28, aligned to the 2021-01-28 adjustment date.</text>',
    ]

    for tick in y_axis_ticks(min_y, max_y):
        y = y_pos(tick)
        label = svg_escape(predicted_rank_label(tick))
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        lines.append(
            f'<text class="label" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{label}</text>'
        )
        lines.append(
            f'<text class="label" x="{width - right + 10}" y="{y + 4:.2f}" text-anchor="start">{label}</text>'
        )

    for index, bucket in enumerate(bucket_order):
        x = x_pos(index)
        sample_size = sum(
            len(series.get(strength, {}).get(bucket, []))
            for strength in target_strengths
        )
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}"/>')
        lines.append(
            f'<text class="label" x="{x:.2f}" y="{height - bottom + 22}" text-anchor="middle" '
            f'transform="rotate(-45 {x:.2f} {height - bottom + 22})">{svg_escape(bucket.short_label())}</text>'
        )
        lines.append(
            f'<text class="xcount" x="{x:.2f}" y="{height - bottom + 52}" text-anchor="middle">'
            f'n={sample_size}</text>'
        )

    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>')
    lines.append(f'<line class="axis" x1="{width - right}" y1="{top}" x2="{width - right}" y2="{height - bottom}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>')

    min_target = min(target_strengths)
    max_target = max(target_strengths)

    for strength in target_strengths:
        color = color_for_rank(strength, min_target, max_target)
        points = [
            (
                bucket_index[bucket],
                median(values),
                percentile(values, 0.25),
                percentile(values, 0.75),
                bucket,
                len(values),
            )
            for bucket, values in sorted(series.get(strength, {}).items())
            if values
        ]
        for segment in contiguous_segments(points):
            if len(segment) < 2:
                continue
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x_pos(bucket_i):.2f} {y_pos(med):.2f}"
                for index, (bucket_i, med, _p25, _p75, _bucket, _count) in enumerate(segment)
            )
            lines.append(
                f'<path class="series-line" d="{path}" stroke="{color}">'
                f'<title>{strength_to_rank(strength)}</title></path>'
            )
        cap_width = 10
        for bucket_i, _med, p25, p75, bucket, count in points:
            x = x_pos(bucket_i)
            y_low = y_pos(p25)
            y_high = y_pos(p75)
            lines.append(
                f'<line class="iqr-whisker" x1="{x:.2f}" y1="{y_low:.2f}" '
                f'x2="{x:.2f}" y2="{y_high:.2f}" stroke="{color}">'
                f'<title>{strength_to_rank(strength)}, {bucket.label()}, 25th={svg_escape(predicted_rank_label(p25))}, 75th={svg_escape(predicted_rank_label(p75))}</title>'
                "</line>"
            )
            lines.append(
                f'<line class="iqr-cap" x1="{x - cap_width / 2:.2f}" y1="{y_low:.2f}" '
                f'x2="{x + cap_width / 2:.2f}" y2="{y_low:.2f}" stroke="{color}"/>'
            )
            lines.append(
                f'<line class="iqr-cap" x1="{x - cap_width / 2:.2f}" y1="{y_high:.2f}" '
                f'x2="{x + cap_width / 2:.2f}" y2="{y_high:.2f}" stroke="{color}"/>'
            )
        for bucket_i, med, _p25, _p75, bucket, count in points:
            lines.append(
                f'<circle class="series-point" cx="{x_pos(bucket_i):.2f}" cy="{y_pos(med):.2f}" '
                f'r="4" fill="{color}"><title>{strength_to_rank(strength)}, {bucket.label()}, n={count}</title></circle>'
            )

    legend_x = width - right + 58
    legend_y = top + 10
    lines.append(
        f'<text class="legend" x="{legend_x}" y="{legend_y - 18}" font-weight="700">OGS rank IQR</text>'
    )
    for index, strength in enumerate(target_strengths):
        y = legend_y + index * 22
        color = color_for_rank(strength, min_target, max_target)
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        lines.append(
            f'<text class="legend" x="{legend_x + 36}" y="{y + 4}">{strength_to_rank(strength)}</text>'
        )

    y_label_x = 22
    y_label_y = top + plot_height / 2
    lines.append(
        f'<text class="label" x="{y_label_x}" y="{y_label_y:.2f}" '
        f'text-anchor="middle" transform="rotate(-90 {y_label_x} {y_label_y:.2f})">'
        "Median predicted KGS rank</text>"
    )
    lines.append(
        f'<text class="label" x="{left + plot_width / 2:.2f}" y="{height - 14}" text-anchor="middle">'
        "Year bucket start</text>"
    )
    lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ranks",
        nargs="+",
        default=list(DEFAULT_RANKS),
        help="OGS rank lines to plot, default: 20k 15k 10k 5k 1d",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_strengths = parse_rank_list(args.ranks)
    series = collect_yearly_buckets(args.input, set(target_strengths))
    render_svg(series, all_buckets(series), target_strengths, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
