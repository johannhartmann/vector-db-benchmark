import re

import pgvector.psycopg
import psycopg

from benchmark.dataset import Dataset
from engine.base_client import IncompatibilityError
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.pgturbohybrid.config import get_db_config

# A pgturbohybrid index requires exactly two key columns: one pgvector
# ``vector`` column followed by one ``tsvector`` column (enforced by the
# extension's index access method). For dense-only benchmarking the lexical
# column is kept but left empty.
TABLE_NAME = "items"

# Text search configuration names are interpolated into the GENERATED column
# DDL (DDL cannot be parameterized), so the value is validated against a strict
# identifier pattern instead of being passed through untrusted.
_TS_CONFIG_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def validate_ts_config(ts_config: str) -> str:
    if not isinstance(ts_config, str) or not _TS_CONFIG_PATTERN.match(ts_config):
        raise IncompatibilityError(
            f"Unsafe text search configuration name: {ts_config!r}"
        )
    return ts_config


class PgturboHybridConfigurator(BaseConfigurator):
    SPARSE_VECTOR_SUPPORT = False

    # Internal, trusted mappings from the benchmark distance enum to the
    # pgturbohybrid operator class / order-by operator. Never interpolate
    # user-provided strings into SQL identifiers.
    DISTANCE_OPCLASS = {
        Distance.COSINE: "vector_cosine_turbohybrid_ops",
        Distance.L2: "vector_l2_turbohybrid_ops",
        Distance.DOT: "vector_ip_turbohybrid_ops",
    }
    DISTANCE_OPERATOR = {
        Distance.COSINE: "<~>",
        Distance.L2: "<~->",
        Distance.DOT: "<~#>",
    }

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        self.conn = psycopg.connect(**get_db_config(host, connection_params))
        print("configure connection created")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS pgturbohybrid;")
        pgvector.psycopg.register_vector(self.conn)

    def clean(self):
        self.conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE;")

    def recreate(self, dataset: Dataset, collection_params):
        if dataset.config.type == "sparse" or dataset.config.vector_size is None:
            raise IncompatibilityError(
                "pgturbohybrid MVP only supports dense vectors with a known vector_size"
            )

        vector_size = int(dataset.config.vector_size)
        ts_config = validate_ts_config(collection_params.get("ts_config", "english"))

        self.conn.execute(
            f"""CREATE TABLE {TABLE_NAME} (
                id integer PRIMARY KEY,
                embedding vector({vector_size}) NOT NULL,
                text text NOT NULL DEFAULT '',
                text_tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('{ts_config}', text)
                ) STORED
            );"""
        )
        self.conn.execute(
            f"ALTER TABLE {TABLE_NAME} ALTER COLUMN embedding SET STORAGE PLAIN"
        )

    def execution_params(self, distance, vector_size) -> dict:
        return {}

    def delete_client(self):
        if self.conn:
            self.conn.close()
            self.conn = None
