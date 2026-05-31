"""Helpers for pgturbohybrid's optional hybrid (dense + BM25) search mode.

This module is intentionally dependency-light (standard library only) so the
query-text extraction logic can be unit-tested and smoke-tested without a
database, numpy, or psycopg installed.

The benchmark ``Query`` model exposes no dedicated query-text field. The only
place text can live is ``Query.meta_conditions``, which uses the canonical
filter structure documented in ``engine/base_client/parser.py``::

    {
        "and": [ {"<field>": {"<filter_type>": <criteria>}}, ... ],
        "or":  [ {"<field>": {"<filter_type>": <criteria>}}, ... ],
    }

For hybrid mode, ``query_text_key`` names the field whose lexical value should
be used as the BM25 query text. ``extract_query_text`` pulls that value out of
the condition structure, supporting ``match`` (value), ``match_text`` (text),
and ``match_any`` (any), and also a flat ``{key: text}`` mapping for custom
readers that provide query text directly.
"""

from typing import Optional


def _text_from_criteria(criteria: Optional[dict]) -> Optional[str]:
    """Extract a lexical string from a single field's filter criteria."""
    if not isinstance(criteria, dict):
        return None

    # Plain string criteria (custom readers): {"<field>": "some text"}
    match = criteria.get("match")
    if isinstance(match, dict) and match.get("value") is not None:
        return str(match["value"])

    match_text = criteria.get("match_text")
    if isinstance(match_text, dict) and match_text.get("text") is not None:
        return str(match_text["text"])

    match_any = criteria.get("match_any")
    if isinstance(match_any, dict) and match_any.get("any"):
        values = match_any["any"]
        if isinstance(values, (list, tuple)) and len(values) > 0:
            return " ".join(str(v) for v in values)

    return None


def extract_query_text(
    meta_conditions: Optional[dict], query_text_key: Optional[str]
) -> Optional[str]:
    """Return the BM25 query text for ``query_text_key``, or ``None``.

    Returns ``None`` when no text can be found so the caller can fail loudly
    rather than silently degrade to a dense-only query.
    """
    if not query_text_key or not isinstance(meta_conditions, dict):
        return None

    # Flat mapping fallback: {"<key>": "some text"} (custom readers).
    direct = meta_conditions.get(query_text_key)
    if isinstance(direct, str) and direct != "":
        return direct

    # Canonical condition structure: walk both boolean groups.
    for group in ("and", "or"):
        entries = meta_conditions.get(group)
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            criteria = entry.get(query_text_key)
            text = _text_from_criteria(criteria)
            if text:
                return text

    return None
