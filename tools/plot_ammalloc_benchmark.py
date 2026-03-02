#!/usr/bin/env python3
"""
Plot ammalloc benchmark comparison charts from TSV summary.

Input TSV format (tab-separated):
benchmark    am_mean_ns    std_mean_ns    speedup(std/am)    am_cv_pct    std_cv_pct    am_ci95_ns    std_ci95_ns    am_Bps    std_Bps
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

HAS_MPL = True
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    HAS_MPL = False


@dataclass
class Row:
    benchmark: str
    am_mean_ns: float
    std_mean_ns: float
    speedup: float
    am_cv_pct: float
    std_cv_pct: float
    am_ci95_ns: float
    std_ci95_ns: float
    am_bps: float
    std_bps: float


def _parse_float(raw: str) -> float:
    raw = raw.strip()
    if not raw:
        return 0.0
    return float(raw)


def load_rows(tsv_path: str) -> list[Row]:
    rows: list[Row] = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for rec in reader:
            rows.append(
                Row(
                    benchmark=rec["benchmark"],
                    am_mean_ns=_parse_float(rec["am_mean_ns"]),
                    std_mean_ns=_parse_float(rec["std_mean_ns"]),
                    speedup=_parse_float(rec["speedup(std/am)"]),
                    am_cv_pct=_parse_float(rec["am_cv_pct"]),
                    std_cv_pct=_parse_float(rec["std_cv_pct"]),
                    am_ci95_ns=_parse_float(rec["am_ci95_ns"]),
                    std_ci95_ns=_parse_float(rec["std_ci95_ns"]),
                    am_bps=_parse_float(rec["am_Bps"]),
                    std_bps=_parse_float(rec["std_Bps"]),
                )
            )
    return rows


def short_name(name: str) -> str:
    n = name
    n = n.replace("BM_", "")
    n = n.replace("am_malloc", "am")
    n = n.replace("std::malloc", "std")
    n = n.replace("am_free", "afree")
    n = n.replace("std::free", "sfree")
    n = n.replace("real_time/", "")
    n = n.replace("/threads:", "/t")
    return n


def save_speedup(rows: list[Row], out_path: str) -> None:
    labels = [short_name(r.benchmark) for r in rows]
    vals = [r.speedup for r in rows]
    y = list(range(len(rows)))
    fig_h = max(6, 0.34 * len(rows))
    plt.figure(figsize=(14, fig_h))
    bars = plt.barh(y, vals, color="#2a9d8f")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Speedup = std / am")
    plt.title("ammalloc vs std::malloc Speedup")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for b, v in zip(bars, vals):
        plt.text(v + 0.03, b.get_y() + b.get_height() * 0.15, f"{v:.2f}x", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_latency(rows: list[Row], out_path: str) -> None:
    labels = [short_name(r.benchmark) for r in rows]
    am_vals = [r.am_mean_ns for r in rows]
    std_vals = [r.std_mean_ns for r in rows]
    x = list(range(len(rows)))
    width = 0.42
    fig_h = max(6, 0.32 * len(rows))
    plt.figure(figsize=(14, fig_h))
    plt.barh([i - width / 2 for i in x], am_vals, height=width, color="#264653", label="am")
    plt.barh([i + width / 2 for i in x], std_vals, height=width, color="#e76f51", label="std")
    plt.yticks(x, labels, fontsize=8)
    plt.xlabel("Mean Latency (ns)")
    plt.title("Mean Latency Comparison")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _to_gib_per_s(bps: float) -> float:
    if bps <= 0:
        return 0.0
    return bps / float(1024**3)


def save_throughput(rows: list[Row], out_path: str) -> None:
    mt_rows = [r for r in rows if ("multithread" in r.benchmark and r.am_bps > 0 and r.std_bps > 0)]
    if not mt_rows:
        return
    labels = [short_name(r.benchmark) for r in mt_rows]
    am_vals = [_to_gib_per_s(r.am_bps) for r in mt_rows]
    std_vals = [_to_gib_per_s(r.std_bps) for r in mt_rows]
    x = list(range(len(mt_rows)))
    width = 0.38
    plt.figure(figsize=(14, 7))
    plt.bar([i - width / 2 for i in x], am_vals, width=width, color="#1d3557", label="am")
    plt.bar([i + width / 2 for i in x], std_vals, width=width, color="#e63946", label="std")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Throughput (GiB/s)")
    plt.title("Multithread Throughput")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_cv(rows: list[Row], out_path: str) -> None:
    labels = [short_name(r.benchmark) for r in rows]
    am_vals = [r.am_cv_pct for r in rows]
    std_vals = [r.std_cv_pct for r in rows]
    x = list(range(len(rows)))
    width = 0.42
    fig_h = max(6, 0.32 * len(rows))
    plt.figure(figsize=(14, fig_h))
    plt.barh([i - width / 2 for i in x], am_vals, height=width, color="#457b9d", label="am")
    plt.barh([i + width / 2 for i in x], std_vals, height=width, color="#f4a261", label="std")
    plt.yticks(x, labels, fontsize=8)
    plt.xlabel("CV (%)")
    plt.title("Run-to-Run Variability (CV)")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _svg_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _write_svg(path: str, width: int, height: int, body: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
        f.write('<rect width="100%" height="100%" fill="white"/>')
        for line in body:
            f.write(line)
        f.write("</svg>")


def save_speedup_svg(rows: list[Row], out_path: str) -> None:
    width = 1600
    row_h = 24
    left = 520
    top = 50
    bar_h = 14
    h = top + len(rows) * row_h + 60
    max_v = max(r.speedup for r in rows) * 1.1
    plot_w = width - left - 80
    body: list[str] = [f'<text x="20" y="28" font-size="20">ammalloc vs std::malloc Speedup</text>']
    for i, r in enumerate(rows):
        y = top + i * row_h
        label = _svg_escape(short_name(r.benchmark))
        body.append(f'<text x="10" y="{y + 12}" font-size="11">{label}</text>')
        w = 0 if max_v <= 0 else (r.speedup / max_v) * plot_w
        body.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="#2a9d8f"/>')
        body.append(f'<text x="{left + w + 6:.1f}" y="{y + 11}" font-size="10">{r.speedup:.2f}x</text>')
    _write_svg(out_path, width, h, body)


def save_latency_svg(rows: list[Row], out_path: str) -> None:
    width = 1600
    row_h = 24
    left = 520
    top = 50
    bar_h = 6
    h = top + len(rows) * row_h + 70
    max_v = max(max(r.am_mean_ns, r.std_mean_ns) for r in rows) * 1.1
    plot_w = width - left - 80
    body: list[str] = [f'<text x="20" y="28" font-size="20">Mean Latency Comparison (ns)</text>']
    for i, r in enumerate(rows):
        y = top + i * row_h
        label = _svg_escape(short_name(r.benchmark))
        body.append(f'<text x="10" y="{y + 12}" font-size="11">{label}</text>')
        am_w = 0 if max_v <= 0 else (r.am_mean_ns / max_v) * plot_w
        std_w = 0 if max_v <= 0 else (r.std_mean_ns / max_v) * plot_w
        body.append(f'<rect x="{left}" y="{y}" width="{am_w:.1f}" height="{bar_h}" fill="#264653"/>')
        body.append(f'<rect x="{left}" y="{y + bar_h + 2}" width="{std_w:.1f}" height="{bar_h}" fill="#e76f51"/>')
    body.append('<text x="20" y="48" font-size="11" fill="#264653">am</text>')
    body.append('<text x="60" y="48" font-size="11" fill="#e76f51">std</text>')
    _write_svg(out_path, width, h, body)


def save_throughput_svg(rows: list[Row], out_path: str) -> None:
    mt_rows = [r for r in rows if ("multithread" in r.benchmark and r.am_bps > 0 and r.std_bps > 0)]
    if not mt_rows:
        return
    width = 1600
    height = 700
    left = 80
    top = 80
    bottom = 160
    plot_w = width - left - 40
    plot_h = height - top - bottom
    am_vals = [_to_gib_per_s(r.am_bps) for r in mt_rows]
    std_vals = [_to_gib_per_s(r.std_bps) for r in mt_rows]
    max_v = max(max(am_vals), max(std_vals)) * 1.1
    n = len(mt_rows)
    group_w = plot_w / max(1, n)
    bar_w = group_w * 0.35
    body: list[str] = [f'<text x="20" y="32" font-size="20">Multithread Throughput (GiB/s)</text>']
    body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#999"/>')
    for i, r in enumerate(mt_rows):
        gx = left + i * group_w
        am_h = 0 if max_v <= 0 else (am_vals[i] / max_v) * plot_h
        std_h = 0 if max_v <= 0 else (std_vals[i] / max_v) * plot_h
        ax = gx + group_w * 0.12
        sx = ax + bar_w + group_w * 0.08
        ay = top + plot_h - am_h
        sy = top + plot_h - std_h
        body.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bar_w:.1f}" height="{am_h:.1f}" fill="#1d3557"/>')
        body.append(f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{bar_w:.1f}" height="{std_h:.1f}" fill="#e63946"/>')
        label = _svg_escape(short_name(r.benchmark))
        body.append(f'<text x="{gx + 2:.1f}" y="{top + plot_h + 14}" font-size="8" transform="rotate(45 {gx + 2:.1f},{top + plot_h + 14})">{label}</text>')
    body.append('<text x="20" y="52" font-size="11" fill="#1d3557">am</text>')
    body.append('<text x="50" y="52" font-size="11" fill="#e63946">std</text>')
    _write_svg(out_path, width, height, body)


def save_cv_svg(rows: list[Row], out_path: str) -> None:
    width = 1600
    row_h = 24
    left = 520
    top = 50
    bar_h = 6
    h = top + len(rows) * row_h + 70
    max_v = max(max(r.am_cv_pct, r.std_cv_pct) for r in rows) * 1.1
    plot_w = width - left - 80
    body: list[str] = [f'<text x="20" y="28" font-size="20">Run-to-Run Variability (CV %)</text>']
    for i, r in enumerate(rows):
        y = top + i * row_h
        label = _svg_escape(short_name(r.benchmark))
        body.append(f'<text x="10" y="{y + 12}" font-size="11">{label}</text>')
        am_w = 0 if max_v <= 0 else (r.am_cv_pct / max_v) * plot_w
        std_w = 0 if max_v <= 0 else (r.std_cv_pct / max_v) * plot_w
        body.append(f'<rect x="{left}" y="{y}" width="{am_w:.1f}" height="{bar_h}" fill="#457b9d"/>')
        body.append(f'<rect x="{left}" y="{y + bar_h + 2}" width="{std_w:.1f}" height="{bar_h}" fill="#f4a261"/>')
    body.append('<text x="20" y="48" font-size="11" fill="#457b9d">am</text>')
    body.append('<text x="60" y="48" font-size="11" fill="#f4a261">std</text>')
    _write_svg(out_path, width, h, body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ammalloc benchmark comparison charts from TSV.")
    parser.add_argument(
        "--input",
        default="/tmp/ammalloc_rigorous_pairwise_20260303.tsv",
        help="Input TSV file path.",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/figures/ammalloc_bench_20260303",
        help="Output directory for PNG charts.",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No benchmark rows found in: {args.input}")

    os.makedirs(args.out_dir, exist_ok=True)
    if HAS_MPL:
        save_speedup(rows, os.path.join(args.out_dir, "speedup_std_over_am.png"))
        save_latency(rows, os.path.join(args.out_dir, "latency_mean_ns.png"))
        save_throughput(rows, os.path.join(args.out_dir, "throughput_multithread_gibs.png"))
        save_cv(rows, os.path.join(args.out_dir, "cv_variability_pct.png"))
        print(f"Generated PNG charts in: {args.out_dir}")
    else:
        save_speedup_svg(rows, os.path.join(args.out_dir, "speedup_std_over_am.svg"))
        save_latency_svg(rows, os.path.join(args.out_dir, "latency_mean_ns.svg"))
        save_throughput_svg(rows, os.path.join(args.out_dir, "throughput_multithread_gibs.svg"))
        save_cv_svg(rows, os.path.join(args.out_dir, "cv_variability_pct.svg"))
        print(f"Generated SVG charts in: {args.out_dir} (matplotlib not found)")


if __name__ == "__main__":
    main()
