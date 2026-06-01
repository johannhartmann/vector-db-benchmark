from typing import List, Tuple

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from dataset_reader.base_reader import Query
from engine.base_client import IncompatibilityError
from engine.base_client.distances import Distance
from engine.base_client.search import BaseSearcher
from engine.clients.pgturbohybrid.config import get_db_config
from engine.clients.pgturbohybrid.configure import TABLE_NAME, validate_ts_config
from engine.clients.pgturbohybrid.hybrid import extract_query_text


class PgturboHybridSearcher(BaseSearcher):
    # Trusted mapping from the benchmark distance enum to the pgturbohybrid
    # order-by operator. Never interpolate user-provided strings as operators.
    DISTANCE_OPERATOR = {
        Distance.COSINE: "<~>",
        Distance.L2: "<~->",
        Distance.DOT: "<~#>",
    }

    conn = None
    cur = None
    distance = None
    search_params = {}
    mode = "dense"
    ts_config = "english"
    query_text_key = None
    dense_query = None
    hybrid_query = None

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        cls.conn = psycopg.connect(**get_db_config(host, connection_params))
        register_vector(cls.conn)
        cls.cur = cls.conn.cursor()
        cls.distance = distance
        cls.search_params = search_params

        try:
            operator = cls.DISTANCE_OPERATOR[distance]
        except KeyError:
            raise IncompatibilityError(f"Unsupported distance metric: {distance}")

        # Optional session profile (e.g. "latency" / "quality"). The value is
        # bound, never interpolated.
        profile = search_params.get("profile")
        if profile is not None:
            cls.cur.execute(
                "SELECT set_config('turbohybrid.profile', %s, false)", (str(profile),)
            )

        # Arbitrary GUCs. Both the name and value are bound through set_config,
        # so nothing is interpolated into SQL.
        for key, value in search_params.get("runtime_settings", {}).items():
            cls.cur.execute("SELECT set_config(%s, %s, false)", (key, str(value)))

        cls.mode = search_params.get("mode", "dense")
        cls.ts_config = validate_ts_config(search_params.get("ts_config", "english"))
        cls.query_text_key = search_params.get("query_text_key")

        # The operator comes from the trusted mapping above; the validated
        # ts_config is the only other interpolated value (DDL/SQL functions
        # cannot take a regconfig as a bound parameter). All per-query values
        # (vector, candidate budgets, text) are bound.
        # The benchmark only consumes ids (for recall), so SELECT only id and
        # let the turbohybrid operator be evaluated once, in ORDER BY. (Selecting
        # the score too would re-evaluate the operator per row for no benefit.)
        cls.dense_query = f"""
            SELECT id
            FROM {TABLE_NAME}
            ORDER BY embedding {operator} turbohybrid_query(
                       vector_query => %s::vector,
                       dense_k => %s,
                       final_k => %s
                   )
            LIMIT %s
        """

        cls.hybrid_query = f"""
            SELECT id
            FROM {TABLE_NAME}
            ORDER BY embedding {operator} turbohybrid_query(
                       vector_query => %s::vector,
                       text_query => plainto_tsquery('{cls.ts_config}', %s),
                       dense_k => %s,
                       bm25_k => %s,
                       rrf_k => %s,
                       final_k => %s,
                       require_bm25_match => %s
                   )
            LIMIT %s
        """

    @classmethod
    def search_one(cls, query: Query, top) -> List[Tuple[int, float]]:
        config = cls.search_params.get("config", {})
        dense_k = config.get("dense_k", top)
        final_k = config.get("final_k", top)
        vector = np.array(query.vector)

        if cls.mode == "hybrid":
            query_text = extract_query_text(query.meta_conditions, cls.query_text_key)
            if not query_text:
                raise IncompatibilityError(
                    "Hybrid search mode requires query text, but none could be "
                    f"extracted for query_text_key={cls.query_text_key!r} from "
                    "query.meta_conditions. Use a dataset whose conditions carry a "
                    "match/match_text/match_any value on that field, or set "
                    "search_params.query_text_key correctly."
                )
            bm25_k = config.get("bm25_k", top)
            rrf_k = config.get("rrf_k")
            require_bm25_match = bool(config.get("require_bm25_match", False))
            args = (
                vector,
                query_text,
                dense_k,
                bm25_k,
                rrf_k,
                final_k,
                require_bm25_match,
            )
            cls.cur.execute(
                cls.hybrid_query, args + (top,), binary=True, prepare=True
            )
        else:
            args = (vector, dense_k, final_k)
            cls.cur.execute(
                cls.dense_query, args + (top,), binary=True, prepare=True
            )

        return cls.cur.fetchall()

    def setup_search(self):
        # Loud guard (runs once in the parent process): hybrid ranking is
        # meaningless if no document text was indexed into items.text. Failing
        # here avoids silently degrading to dense-only because text_field was
        # omitted from upload_params or the metadata field was empty.
        cls = self.__class__
        if cls.mode != "hybrid":
            return
        row = cls.cur.execute(
            f"SELECT count(*) FROM {TABLE_NAME} WHERE text <> ''"
        ).fetchone()
        if not row or not row[0]:
            raise IncompatibilityError(
                "Hybrid search mode requires indexed document text, but items.text "
                "is empty. Set upload_params.text_field to a metadata field present "
                "in the dataset and re-run the upload stage."
            )

    @classmethod
    def delete_client(cls):
        if cls.cur:
            cls.cur.close()
            cls.cur = None
        if cls.conn:
            cls.conn.close()
            cls.conn = None
