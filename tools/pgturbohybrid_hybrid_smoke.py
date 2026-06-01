#!/usr/bin/env python3
"""Standalone smoke check for pgturbohybrid hybrid query-text extraction.

Runs with a bare Python interpreter (no database, numpy, or psycopg needed):
it loads the dependency-light ``hybrid.py`` directly by path and prints how
query text is extracted from representative benchmark ``Query.meta_conditions``
shapes.

    python tools/pgturbohybrid_hybrid_smoke.py
"""

import importlib.util
import os
import sys

HYBRID_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "engine",
    "clients",
    "pgturbohybrid",
    "hybrid.py",
)

spec = importlib.util.spec_from_file_location("pgturbohybrid_hybrid", HYBRID_PATH)
hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid)
extract_query_text = hybrid.extract_query_text


# (description, meta_conditions, query_text_key, expected)
CASES = [
    ("dense query (no conditions)", None, "a", None),
    (
        "keyword match (AnnCompoundReader)",
        {"and": [{"a": {"match": {"value": "running-shoes"}}}]},
        "a",
        "running-shoes",
    ),
    (
        "match in OR group",
        {"or": [{"a": {"match": {"value": "boots"}}}]},
        "a",
        "boots",
    ),
    (
        "match_text on text field",
        {"and": [{"t": {"match_text": {"text": "postgres hybrid search"}}}]},
        "t",
        "postgres hybrid search",
    ),
    (
        "match_any joined",
        {"and": [{"a": {"match_any": {"any": ["red", "blue"]}}}]},
        "a",
        "red blue",
    ),
    ("flat custom mapping", {"a": "free text query"}, "a", "free text query"),
    (
        "field absent -> None (fails loudly upstream)",
        {"and": [{"b": {"match": {"value": "x"}}}]},
        "a",
        None,
    ),
]


def main() -> int:
    failures = 0
    for description, meta, key, expected in CASES:
        got = extract_query_text(meta, key)
        ok = got == expected
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {description}: got={got!r} expected={expected!r}")
    print()
    if failures:
        print(f"SMOKE FAILED: {failures} case(s)")
        return 1
    print(f"SMOKE OK: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
