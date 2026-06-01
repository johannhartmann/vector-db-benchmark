#!/usr/bin/env bash
#
# Local speed comparison: pgturbohybrid vs pgvector vs qdrant.
#
# Runs each engine sequentially on the same dataset and host, bringing the
# engine's server up before the run and tearing it down afterwards (pgvector and
# pgturbohybrid both bind port 5432, so they must not run at the same time).
#
# Postgres versions are aligned: the pgvector server is pinned to a pg17 image
# (via PGVECTOR_IMAGE) to match the pgturbohybrid pg17 image, so the comparison
# isolates the engine rather than the Postgres major version.
#
# Requires a uv-managed virtualenv with the client dependencies, e.g.:
#   uv venv --python 3.12 .venv
#   uv pip install --python .venv "git+https://github.com/qdrant/qdrant-client.git@dev" \
#       "weaviate-client>=4.5,<4.7" "elasticsearch>=8.10,<9" "pymilvus>=2.5,<3" \
#       "redis>=5.0.1,<6" "opensearch-py>=2.3.2,<3" "psycopg[binary]>=3.1.17" \
#       "pgvector>=0.2.4" "h5py>=3.7" "typer>=0.15" "tqdm>=4.66.1" "jsons>=1.6.3" "ipdb"
#
# Usage:
#   tools/compare_pgturbohybrid.sh [DATASET]
#
# Env overrides:
#   DATASET         dataset name (default: glove-100-angular)
#   HOST            client target host (default: localhost)
#   TIMEOUT         per-experiment timeout in seconds (default: 7200)
#   PGVECTOR_IMAGE  pgvector image used for alignment (default: pgvector/pgvector:pg17)
#   STARTUP_WAIT    seconds to wait after `compose up` (default: 30)
#
# Results land in ./results/. Summarize with:
#   tools/summarize_comparison.py        (see bottom of this file)

# No `set -e`: engine failures are handled explicitly so teardown always runs
# and one engine's failure does not silently abort (or mask) the rest.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-${DATASET:-glove-100-angular}}"
HOST="${HOST:-localhost}"
TIMEOUT="${TIMEOUT:-7200}"
STARTUP_WAIT="${STARTUP_WAIT:-30}"
# Align Postgres major version with the pgturbohybrid image (pg17).
export PGVECTOR_IMAGE="${PGVECTOR_IMAGE:-pgvector/pgvector:pg17}"

# Host port for the Postgres-based engines. Override (e.g. PG_PORT=5433) when
# 5432 is already taken on the host. Both the published container port and the
# client connection port are set from it.
PG_PORT="${PG_PORT:-5432}"
export PGVECTOR_HOST_PORT="$PG_PORT" PGVECTOR_PORT="$PG_PORT"
export PGTURBOHYBRID_HOST_PORT="$PG_PORT" PGTURBOHYBRID_PORT="$PG_PORT"

# Postgres memory tuning. The committed compose defaults target a 25GB+ host;
# these laptop-friendly defaults keep Postgres inside a small Docker VM (a
# 1.18M-row COPY OOM-kills the backend otherwise). Raise them on a big host.
export PG_SHARED_BUFFERS="${PG_SHARED_BUFFERS:-2GB}"
export PG_MAINTENANCE_WORK_MEM="${PG_MAINTENANCE_WORK_MEM:-2GB}"
export PG_SHM_SIZE="${PG_SHM_SIZE:-4g}"

# engine-config fnmatch patterns (run.py treats --engines as a single glob per
# value, so use one pattern rather than space-separated names).
PGVECTOR_ENGINES="pgvector-default"
QDRANT_ENGINES="qdrant-default"
# Default to the recommended exact-free, high-recall operating point: the
# `high_recall` profile (heap-band exact rescore + heuristic graph topology, no
# exact_storage). On dbpedia-openai-100K-1536-angular it reaches ~0.99 recall
# while staying faster than pgvector/qdrant. Override for other points, e.g.
# PGTURBOHYBRID_ENGINES="pgturbohybrid-dense-default" (max throughput, ~0.86
# recall) or "pgturbohybrid-*" for the full set.
PGTURBOHYBRID_ENGINES="${PGTURBOHYBRID_ENGINES:-pgturbohybrid-high-recall}"

FAILED=()

run_engine() {
    local server_dir="$1" engines="$2"
    local compose="engine/servers/${server_dir}/docker-compose.yaml"

    echo ">>> [$server_dir] bringing server up"
    docker compose -f "$compose" up -d
    sleep "$STARTUP_WAIT"

    echo ">>> [$server_dir] running: $engines on $DATASET"
    local rc=0
    uv run --no-project python run.py \
        --engines "$engines" \
        --datasets "$DATASET" \
        --host "$HOST" \
        --timeout "$TIMEOUT" || rc=$?

    echo ">>> [$server_dir] tearing server down"
    docker compose -f "$compose" down || true

    if [ "$rc" -ne 0 ]; then
        echo "!!! [$server_dir] run.py FAILED (exit $rc)"
        FAILED+=("$server_dir")
    fi
}

echo "=== comparison on dataset: $DATASET (host: $HOST) ==="
echo "=== pgvector image (aligned): $PGVECTOR_IMAGE | shared_buffers=$PG_SHARED_BUFFERS ==="

run_engine "pgvector-single-node"     "$PGVECTOR_ENGINES"
run_engine "qdrant-single-node"       "$QDRANT_ENGINES"
run_engine "pgturbohybrid-single-node" "$PGTURBOHYBRID_ENGINES"

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "=== FAILED engines: ${FAILED[*]} ==="
    echo "=== partial results; summarize with: tools/summarize_comparison.py \"$DATASET\" ==="
    exit 1
fi
echo "=== done. summarize with: tools/summarize_comparison.py \"$DATASET\" ==="
