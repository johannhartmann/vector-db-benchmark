"""Unit tests for pgturbohybrid hybrid query-text extraction.

These exercise the dependency-light helper in
``engine.clients.pgturbohybrid.hybrid`` so they run without a database.
"""

import pytest

from engine.clients.pgturbohybrid.hybrid import extract_query_text


def test_returns_none_without_key():
    conditions = {"and": [{"a": {"match": {"value": "shoes"}}}]}
    assert extract_query_text(conditions, None) is None


def test_returns_none_for_dense_query():
    # Dense datasets (h5, jsonl) carry no conditions.
    assert extract_query_text(None, "a") is None


def test_extracts_keyword_match_value():
    # Canonical structure produced by AnnCompoundReader for keyword datasets.
    conditions = {"and": [{"a": {"match": {"value": "running-shoes"}}}]}
    assert extract_query_text(conditions, "a") == "running-shoes"


def test_extracts_match_value_from_or_group():
    conditions = {"or": [{"a": {"match": {"value": "boots"}}}]}
    assert extract_query_text(conditions, "a") == "boots"


def test_extracts_match_text():
    conditions = {"and": [{"t": {"match_text": {"text": "postgres hybrid search"}}}]}
    assert extract_query_text(conditions, "t") == "postgres hybrid search"


def test_extracts_match_any_joined():
    conditions = {"and": [{"a": {"match_any": {"any": ["red", "blue"]}}}]}
    assert extract_query_text(conditions, "a") == "red blue"


def test_supports_flat_custom_mapping():
    # A custom reader may expose query text directly as {key: text}.
    conditions = {"a": "free text query"}
    assert extract_query_text(conditions, "a") == "free text query"


def test_returns_none_when_field_absent():
    conditions = {"and": [{"b": {"match": {"value": "x"}}}]}
    assert extract_query_text(conditions, "a") is None


def test_coerces_non_string_match_value():
    conditions = {"and": [{"a": {"match": {"value": 80}}}]}
    assert extract_query_text(conditions, "a") == "80"


@pytest.mark.parametrize("meta", [None, {}, [], "x", 42])
def test_tolerates_malformed_conditions(meta):
    assert extract_query_text(meta, "a") is None
