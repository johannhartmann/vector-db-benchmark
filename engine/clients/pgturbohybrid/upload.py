import json
from typing import List

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from dataset_reader.base_reader import Record
from engine.base_client import IncompatibilityError
from engine.base_client.distances import Distance
from engine.base_client.upload import BaseUploader
from engine.clients.pgturbohybrid.config import get_db_config
from engine.clients.pgturbohybrid.configure import TABLE_NAME

INDEX_NAME = "items_turbohybrid_idx"


class PgturboHybridUploader(BaseUploader):
    # Trusted mapping from the benchmark distance enum to the pgturbohybrid
    # vector operator class used to build the index.
    DISTANCE_MAPPING = {
        Distance.L2: "vector_l2_turbohybrid_ops",
        Distance.COSINE: "vector_cosine_turbohybrid_ops",
        Distance.DOT: "vector_ip_turbohybrid_ops",
    }
    conn = None
    cur = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        cls.conn = psycopg.connect(**get_db_config(host, connection_params))
        register_vector(cls.conn)
        cls.cur = cls.conn.cursor()
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(cls, batch: List[Record]):
        text_field = cls.upload_params.get("text_field")

        ids, vectors, texts = [], [], []
        for record in batch:
            ids.append(record.id)
            vectors.append(record.vector)

            text_value = ""
            if (
                text_field
                and isinstance(record.metadata, dict)
                and record.metadata.get(text_field) is not None
            ):
                text_value = str(record.metadata[text_field])
            texts.append(text_value)

        vectors = np.array(vectors)
        # COPY is faster than INSERT. The generated text_tsv column is derived
        # automatically and is therefore not part of the column list.
        with cls.cur.copy(
            f"COPY {TABLE_NAME} (id, embedding, text) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["integer", "vector", "text"])
            for i, embedding, text_value in zip(ids, vectors, texts):
                copy.write_row((i, embedding, text_value))

    @classmethod
    def post_upload(cls, distance):
        try:
            opclass = cls.DISTANCE_MAPPING[distance]
        except KeyError:
            raise IncompatibilityError(f"Unsupported distance metric: {distance}")

        index_params = cls.upload_params.get("index", {})
        quantization_bits = int(index_params.get("quantization_bits", 4))
        exact_storage = "on" if index_params.get("exact_storage", False) else "off"

        # opclass / exact_storage / quantization_bits are all derived from
        # trusted internal mappings or coerced to int / on|off, so they are safe
        # to interpolate into the DDL.
        cls.conn.execute(
            f"""CREATE INDEX {INDEX_NAME} ON {TABLE_NAME}
            USING turbohybrid (
                embedding {opclass},
                text_tsv bm25_tsvector_turbohybrid_ops
            )
            WITH (quantization_bits = {quantization_bits}, exact_storage = {exact_storage})"""
        )
        cls.conn.execute(f"ANALYZE {TABLE_NAME}")

        try:
            row = cls.conn.execute(
                "SELECT turbohybrid_index_stats(%s::regclass)::text", (INDEX_NAME,)
            ).fetchone()
            if row and row[0]:
                return {"index_stats": json.loads(row[0])}
        except Exception as e:
            print(f"Could not fetch turbohybrid_index_stats: {e}")

        return {}

    @classmethod
    def delete_client(cls):
        if cls.cur:
            cls.cur.close()
            cls.cur = None
        if cls.conn:
            cls.conn.close()
            cls.conn = None
