#!/usr/bin/env python3
"""Summarize benchmark search results for a dataset as a speed-vs-precision table.

Reads ``results/*-<dataset>-search-*.json`` and prints rps alongside
mean_precisions and tail latencies, so engines are compared at comparable
recall rather than on raw throughput alone.

    python tools/summarize_comparison.py [DATASET]
"""

import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "glove-100-angular"
    pattern = os.path.join(RESULTS_DIR, f"*-{dataset}-search-*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found matching {pattern}")
        return 1

    rows = []
    for path in files:
        with open(path) as fh:
            data = json.load(fh)
        params = data.get("params", {})
        results = data.get("results", {})
        rows.append(
            {
                "experiment": params.get("experiment", "?"),
                "engine": params.get("engine", "?"),
                "parallel": params.get("parallel", "?"),
                "rps": results.get("rps"),
                "precision": results.get("mean_precisions"),
                "p95_ms": _ms(results.get("p95_time")),
                "p99_ms": _ms(results.get("p99_time")),
            }
        )

    # Highest throughput first; precision shown alongside so it is not ignored.
    rows.sort(key=lambda r: (r["rps"] is None, -(r["rps"] or 0)))

    header = f"{'experiment':<34}{'engine':<16}{'par':>4}{'rps':>10}{'precision':>11}{'p95_ms':>9}{'p99_ms':>9}"
    print(f"dataset: {dataset}\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['experiment']:<34}{r['engine']:<16}{str(r['parallel']):>4}"
            f"{_fmt(r['rps'], 1):>10}{_fmt(r['precision'], 4):>11}"
            f"{_fmt(r['p95_ms'], 3):>9}{_fmt(r['p99_ms'], 3):>9}"
        )
    print("\nNote: compare rps at comparable precision; pgturbohybrid-dense-default")
    print("uses a 4-bit quantized index (lower precision, higher speed).")
    return 0


def _ms(seconds):
    return None if seconds is None else seconds * 1000.0


def _fmt(value, digits):
    return "n/a" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    sys.exit(main())
