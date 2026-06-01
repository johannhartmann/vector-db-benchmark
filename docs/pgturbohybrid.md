# pgturbohybrid

[`pgturbohybrid`](https://github.com/mayflower/pgturbohybrid) is a PostgreSQL
extension that adds a `turbohybrid` index access method on top of pgvector,
combining dense-vector and BM25 lexical retrieval.

> **Dense-vector mode is the default.** The engine always builds the mandatory
> two-column `turbohybrid` index (vector + `tsvector`) and, by default, queries
> dense-only via `turbohybrid_query(vector_query => ...)`. An opt-in true hybrid
> (dense + BM25 text) mode is also implemented, but it only works with datasets
> that supply both document text and query text — see
> [Hybrid mode](#hybrid-mode). `SPARSE_VECTOR_SUPPORT` remains `False`; sparse
> vectors are not supported.

## Start the server

```bash
cd engine/servers/pgturbohybrid-single-node
docker compose up -d
```

This uses the published image
`ghcr.io/mayflower/pgturbohybrid:${PGTURBOHYBRID_TAG:-0.1.0-pg17}` and creates
both the `vector` and `pgturbohybrid` extensions on first start.

## Environment variables

The client connection is configured through these variables (defaults match the
server image above):

| Variable | Default |
| --- | --- |
| `PGTURBOHYBRID_PORT` | `5432` |
| `PGTURBOHYBRID_DB` | `pgturbohybrid` |
| `PGTURBOHYBRID_USER` | `postgres` |
| `PGTURBOHYBRID_PASSWORD` | `postgres` |

## Run a dense benchmark

```bash
poetry run python run.py --engines "pgturbohybrid-dense-default" --datasets "glove-25-angular" --timeout 3600
```

A larger run:

```bash
poetry run python run.py --engines "pgturbohybrid-dense-default" --datasets "dbpedia-openai-100K-1536-angular" --timeout 7200
```

**Recommended default: `pgturbohybrid-high-recall`.** This is the exact-free,
high-recall operating point. It selects the extension's `high_recall` profile,
which bakes in `dense_heap_rescore=band` (exact rescore of the final candidate
band from the heap), `dense_adaptive_widening=off`, and a heuristic graph
topology (`graph_ef_construction=256`, `graph_ef_search=192`,
`graph_oversampling=12`) — all with `exact_storage=off`. The config additionally
pins those graph reloptions and `native_segments=1`, and builds with
`dense_build_neighbor_select=heuristic` / `dense_build_distance=code`. It also
pins `dense_heap_rescore=band` / `dense_adaptive_widening=off` explicitly in
`search_params.runtime_settings`: the harness sets session GUCs on each worker
connection *before* the extension library is loaded, so the profile's cascading
defaults (which need the library loaded to fire) are pinned explicitly rather
than relied upon. Interactive `SET turbohybrid.profile = 'high_recall'` does not
need this. On `dbpedia-openai-100K-1536-angular` it reaches ~0.99 recall while
staying faster than pgvector and qdrant:

```bash
poetry run python run.py --engines "pgturbohybrid-high-recall" --datasets "dbpedia-openai-100K-1536-angular" --timeout 7200
```

`pgturbohybrid-dense-default` (latency profile, ~0.86 recall) is the
max-throughput point. `pgturbohybrid-dense-quality` is also available (exact
storage on, `quality` profile) but `exact_storage=on` is generally not the
goal — prefer `matched-recall-band`. Use `--engines "pgturbohybrid-*"` to run
all configurations.

## Configuration

Experiments live in
[`experiments/configurations/pgturbohybrid-single-node.json`](../experiments/configurations/pgturbohybrid-single-node.json).
The files are kept as pure JSON (the repo does not use JSON comments), so the
engine-specific fields are documented here:

- `collection_params.ts_config` — text-search configuration for the generated
  `tsvector` column (default `english`). Validated as a safe identifier.
- `upload_params.index.quantization_bits` — index quantization (`2` or `4`,
  default `4`).
- `upload_params.index.exact_storage` — keep exact vectors alongside the
  quantized index (`true`/`false`). `false` is faster and smaller; `true`
  trades size for quality.
- `search_params.profile` — session profile: `latency`, `balanced`,
  `matched_recall`, `high_recall`, or `quality`. `high_recall` is the
  exact-free near-exact-recall point (heap-band rescore; see the recommended
  config above).
- `search_params.config.dense_k` — dense candidate budget per query.
- `search_params.config.final_k` — final result count requested from the index.
- `search_params.runtime_settings` — arbitrary `turbohybrid.*` GUCs set per
  search session (bound via `set_config`), e.g.
  `{"turbohybrid.dense_heap_rescore": "band"}` to enable exact heap rescore of
  the candidate band, or `"off"` to disable it.
- `upload_params.index` graph/build options forwarded into the index `WITH`
  clause: `graph_ef_construction`, `graph_ef_search`, `graph_oversampling`,
  `native_segments` (ints), and `residual_rerank` (bool) /
  `residual_rerank_bytes` (int).
- `upload_params.index.build_settings` — `turbohybrid.*` GUCs set on the build
  connection before `CREATE INDEX` (e.g. `turbohybrid.profile`,
  `dense_build_neighbor_select`, `dense_build_distance`).

Hybrid-only fields are described under [Hybrid mode](#hybrid-mode).

The shipped configurations:

| Experiment | mode | quantization_bits | exact_storage | profile | dense_k | final_k |
| --- | --- | --- | --- | --- | --- | --- |
| `pgturbohybrid-dense-default` | dense | 4 | false | latency | 100 | 10 |
| `pgturbohybrid-dense-quality` | dense | 4 | true | quality | 200 | 10 |
| `pgturbohybrid-hybrid-kw-small-vocab` | hybrid | 4 | false | — | 100 | 10 |

> **`text_field` lives in `upload_params`, not `collection_params`.** Document
> text is copied into `items.text` during the parallel upload stage, and the
> benchmark only forwards `upload_params` (not `collection_params`) to the
> upload worker processes. Placing `text_field` in `collection_params` would be
> silently ignored and produce empty text — so it must be set in
> `upload_params`. In hybrid mode the searcher additionally checks that
> `items.text` is non-empty and fails loudly if it is not.

## Hybrid mode

Hybrid mode (`search_params.mode = "hybrid"`, default `"dense"`) fuses dense
vector search with BM25 lexical ranking. It calls:

```sql
turbohybrid_query(
    vector_query => %s::vector,
    text_query => plainto_tsquery(<ts_config>, %s),
    dense_k => %s, bm25_k => %s, rrf_k => %s,
    final_k => %s, require_bm25_match => %s
)
```

Hybrid fields:

- `upload_params.text_field` — metadata field copied into `items.text`.
- `search_params.query_text_key` — field whose lexical value becomes the BM25
  query text (see extraction below).
- `search_params.ts_config` — text-search config for `plainto_tsquery`; **must
  match** `collection_params.ts_config` used to build the column.
- `search_params.config.bm25_k` — BM25 candidate budget.
- `search_params.config.rrf_k` — reciprocal-rank-fusion constant (default `60`).
- `search_params.config.require_bm25_match` — require a lexical match.

### Query-text availability

The benchmark `Query` model exposes **no dedicated query-text field**; the only
place text can live is `Query.meta_conditions`, which is the nested filter
structure from `engine/base_client/parser.py`
(`{"and"|"or": [{<field>: {match|match_text|match_any: …}}]}`). Most datasets
(h5, sparse, plain jsonl) carry no usable text at all.

The `random-*-filters` tar datasets (read by `AnnCompoundReader`) are the
exception: they expose payload fields in `record.metadata` and per-field
conditions in `meta_conditions`. `extract_query_text` pulls the lexical value
for `query_text_key` from `match` / `match_text` / `match_any` criteria (and
from a flat `{key: text}` mapping for custom readers). If no text is found, the
search **fails loudly** instead of silently running dense-only.

### Example

`pgturbohybrid-hybrid-kw-small-vocab` targets
`random-100-match-kw-small-vocab-filters` (schema `{"a": "keyword", "b":
"keyword"}`, ~10-keyword vocabulary). The keyword field `a` provides both the
indexed document text (`upload_params.text_field = "a"`) and the query text
(`query_text_key = "a"`, extracted from the `match` condition on `a`):

```bash
poetry run python run.py --engines "pgturbohybrid-hybrid-kw-small-vocab" --datasets "random-100-match-kw-small-vocab-filters" --timeout 3600
```

This keyword text is single-token, so BM25 behaves like keyword matching — a
valid but simple hybrid exercise. Natural-language hybrid search needs a dataset
and reader that supply real document and query text.

### Verifying extraction

The extraction logic lives in the dependency-light
`engine/clients/pgturbohybrid/hybrid.py`. Run the unit tests or the standalone
smoke check (no database required):

```bash
poetry run pytest tests/engine/clients/pgturbohybrid/test_pgturbohybrid_hybrid.py
python tools/pgturbohybrid_hybrid_smoke.py
```
