#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class BenchValue:
    name: str
    time_ns: float
    is_multithread: bool


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_values(doc: dict) -> Dict[str, BenchValue]:
    rows = doc.get("benchmarks", [])
    mean_rows = {
        r["name"]: r
        for r in rows
        if r.get("aggregate_name") == "mean" and r.get("run_type") == "aggregate"
    }

    selected: Dict[str, dict] = {}
    if mean_rows:
        selected = mean_rows
    else:
        for r in rows:
            if r.get("run_type") == "iteration":
                selected.setdefault(r["name"], r)

    result: Dict[str, BenchValue] = {}
    for name, row in selected.items():
        unit = row.get("time_unit", "ns")
        value = row.get("real_time", row.get("cpu_time"))
        if value is None:
            continue
        if unit != "ns":
            scale = {
                "s": 1_000_000_000.0,
                "ms": 1_000_000.0,
                "us": 1_000.0,
                "ns": 1.0,
            }.get(unit)
            if scale is None:
                continue
            value *= scale
        result[name] = BenchValue(
            name=name,
            time_ns=float(value),
            is_multithread=("/threads:" in name),
        )
    return result


def _fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Google Benchmark JSON outputs."
    )
    parser.add_argument(
        "--baseline", required=True, help="Baseline benchmark JSON path"
    )
    parser.add_argument(
        "--candidate", required=True, help="Candidate benchmark JSON path"
    )
    parser.add_argument(
        "--name-filter",
        default="",
        help="Optional substring filter on benchmark names (e.g. PageCache|PageMap is not regex, use one substring)",
    )
    parser.add_argument(
        "--single-thread-threshold",
        type=float,
        default=5.0,
        help="Regression threshold in %% for non-threaded benchmarks",
    )
    parser.add_argument(
        "--multi-thread-threshold",
        type=float,
        default=8.0,
        help="Regression threshold in %% for benchmarks containing /threads:",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 2 when regression exceeds threshold",
    )
    args = parser.parse_args()

    baseline = _select_values(_load_json(Path(args.baseline)))
    candidate = _select_values(_load_json(Path(args.candidate)))

    names = sorted(set(baseline) & set(candidate))
    if args.name_filter:
        names = [n for n in names if args.name_filter in n]

    if not names:
        print("No overlapping benchmark names found.")
        return 1

    print("Benchmark Comparison (candidate vs baseline)")
    print("-" * 120)
    print(f"{'name':70} {'base(ns)':>12} {'cand(ns)':>12} {'delta':>10} {'status':>10}")
    print("-" * 120)

    regressions = 0
    improved = 0
    unchanged = 0

    for name in names:
        b = baseline[name]
        c = candidate[name]
        if b.time_ns <= 0:
            continue
        delta_pct = (c.time_ns - b.time_ns) / b.time_ns * 100.0
        threshold = (
            args.multi_thread_threshold
            if c.is_multithread
            else args.single_thread_threshold
        )
        status = "OK"
        if delta_pct > threshold:
            status = "REGRESS"
            regressions += 1
        elif delta_pct < -0.5:
            status = "IMPROVE"
            improved += 1
        else:
            unchanged += 1

        print(
            f"{name:70} {b.time_ns:12.2f} {c.time_ns:12.2f} {_fmt_pct(delta_pct):>10} {status:>10}"
        )

    print("-" * 120)
    print(
        f"Summary: total={len(names)} regressions={regressions} improved={improved} unchanged={unchanged}"
    )
    print(
        f"Thresholds: single-thread={args.single_thread_threshold:.2f}% multi-thread={args.multi_thread_threshold:.2f}%"
    )

    if args.fail_on_regression and regressions > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
