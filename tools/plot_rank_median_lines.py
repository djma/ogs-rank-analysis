#!/usr/bin/env python3
"""Plot median estimated rank by OGS rank for two-year date buckets."""

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
    parse_row_date,
    rank_to_strength,
    strength_to_rank,
    svg_escape,
    y_axis_ticks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results" / "sample_100k_150moves_rank_mle.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "ogs_rank_median_lines_2yr.svg"
YEARS_PER_BUCKET = 2


@dataclass(frozen=True, order=True)
class Period:
    start_year: int

    @property
    def start(self) -> date:
        return date(self.start_year, ADJUST2021.month, ADJUST2021.day)

    @property
    def end(self) -> date:
        return date(self.start_year + YEARS_PER_BUCKET, ADJUST2021.month, ADJUST2021.day)

    def label(self) -> str:
        return f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"


def period_for_date(value: date) -> Period:
    anchor = ADJUST2021.date()
    years_delta = value.year - anchor.year
    period_offset = years_delta // YEARS_PER_BUCKET
    candidate_year = anchor.year + period_offset * YEARS_PER_BUCKET
    candidate_start = date(candidate_year, anchor.month, anchor.day)
    if value < candidate_start:
        candidate_year -= YEARS_PER_BUCKET
    return Period(candidate_year)


def median(values: list[int]) -> float:
    return float(statistics.median(values))


def collect_period_buckets(input_path: Path) -> dict[Period, dict[int, list[int]]]:
    periods: dict[Period, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    with input_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed_date = parse_row_date(row.get("game_date", ""))
            if parsed_date is None:
                continue
            row_date = parsed_date.date() if hasattr(parsed_date, "date") else parsed_date
            period = period_for_date(row_date)
            pairs = (
                ("black_ogs_rank", "black_estimated_rank"),
                ("white_ogs_rank", "white_estimated_rank"),
            )
            for ogs_column, estimate_column in pairs:
                ogs_strength = rank_to_strength(row.get(ogs_column, ""))
                estimate_strength = rank_to_strength(row.get(estimate_column, ""))
                if ogs_strength is not None and estimate_strength is not None:
                    periods[period][ogs_strength].append(estimate_strength)
    return {
        period: dict(sorted(buckets.items()))
        for period, buckets in sorted(periods.items())
    }


def palette(index: int) -> str:
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#7f7f7f",
    ]
    return colors[index % len(colors)]


def render_svg(periods: dict[Period, dict[int, list[int]]], output_path: Path) -> None:
    if not periods:
        raise SystemExit("No rank pairs found in the CSV.")

    all_ogs = [
        ogs_strength
        for buckets in periods.values()
        for ogs_strength in buckets
    ]
    all_estimates = [
        estimate_strength
        for buckets in periods.values()
        for values in buckets.values()
        for estimate_strength in values
    ]
    min_x = min(all_ogs)
    max_x = max(all_ogs)
    min_y = max(1, min(min(all_estimates), min_x))
    max_y = max(max(all_estimates), max_x)

    width = max(1000, 44 * (max_x - min_x + 1) + 240)
    height = 760
    left = 86
    right = 188
    top = 56
    bottom = 96
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

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2933; }",
        ".axis { stroke: #52606d; stroke-width: 1; }",
        ".grid { stroke: #d9e2ec; stroke-width: 1; }",
        ".identity { stroke: #475569; stroke-width: 1.5; stroke-dasharray: 2 6; stroke-linecap: round; opacity: 0.38; }",
        ".median-line { fill: none; stroke-width: 2.7; stroke-linecap: round; stroke-linejoin: round; }",
        ".median-point { stroke: #ffffff; stroke-width: 1.2; }",
        ".label { font-size: 12px; }",
        ".legend { font-size: 12px; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { fill: #627d98; font-size: 13px; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text class="title" x="{left}" y="30">Median estimated rank by OGS rank</text>',
        f'<text class="subtitle" x="{left}" y="49">Each line is a two-year sample bucket; bucket cutoffs are aligned to adjust2021 ({ADJUST2021:%Y-%m-%d %H:%M} SGT).</text>',
    ]

    for tick in y_axis_ticks(min_y, max_y):
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        label = strength_to_rank(tick)
        lines.append(
            f'<text class="label" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{label}</text>'
        )
        lines.append(
            f'<text class="label" x="{width - right + 10}" y="{y + 4:.2f}" text-anchor="start">{label}</text>'
        )

    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>')
    lines.append(f'<line class="axis" x1="{width - right}" y1="{top}" x2="{width - right}" y2="{height - bottom}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>')

    identity_start = max(min_x, min_y)
    identity_end = min(max_x, max_y)
    if identity_start <= identity_end:
        lines.append(
            f'<line class="identity" x1="{x_pos(identity_start):.2f}" '
            f'y1="{y_pos(identity_start):.2f}" x2="{x_pos(identity_end):.2f}" '
            f'y2="{y_pos(identity_end):.2f}"><title>x = y</title></line>'
        )

    for index, (period, buckets) in enumerate(periods.items()):
        color = palette(index)
        points = [
            (x_pos(ogs_strength), y_pos(median(values)), ogs_strength, len(values))
            for ogs_strength, values in buckets.items()
            if values
        ]
        if not points:
            continue
        path = " ".join(
            ("M" if point_index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for point_index, (x, y, _ogs_strength, _count) in enumerate(points)
        )
        lines.append(
            f'<path class="median-line" d="{path}" stroke="{color}">'
            f'<title>{svg_escape(period.label())}</title></path>'
        )
        for x, y, ogs_strength, count in points:
            lines.append(
                f'<circle class="median-point" cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}">'
                f'<title>{svg_escape(period.label())}, {strength_to_rank(ogs_strength)}, n={count}</title>'
                "</circle>"
            )

    for strength in range(min_x, max_x + 1):
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

    legend_x = width - right + 58
    legend_y = top + 10
    for index, period in enumerate(periods):
        y = legend_y + index * 22
        color = palette(index)
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        lines.append(
            f'<text class="legend" x="{legend_x + 36}" y="{y + 4}">{svg_escape(period.label())}</text>'
        )

    y_label_x = 22
    y_label_y = top + plot_height / 2
    lines.append(
        f'<text class="label" x="{y_label_x}" y="{y_label_y:.2f}" '
        f'text-anchor="middle" transform="rotate(-90 {y_label_x} {y_label_y:.2f})">'
        "Median estimated rank</text>"
    )
    lines.append(
        f'<text class="label" x="{left + plot_width / 2:.2f}" y="{height - 14}" text-anchor="middle">'
        "OGS rank bucket</text>"
    )
    lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    periods = collect_period_buckets(args.input)
    render_svg(periods, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
