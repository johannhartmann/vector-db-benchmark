import functools
import time
from multiprocessing import get_context
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tqdm

from dataset_reader.base_reader import Query

DEFAULT_TOP = 10


class BaseSearcher:
    MP_CONTEXT = None

    def __init__(self, host, connection_params, search_params):
        self.host = host
        self.connection_params = connection_params
        self.search_params = search_params

    @classmethod
    def init_client(
        cls, host: str, distance, connection_params: dict, search_params: dict
    ):
        raise NotImplementedError()

    @classmethod
    def get_mp_start_method(cls):
        return None

    @classmethod
    def search_one(cls, query: Query, top: Optional[int]) -> List[Tuple[int, float]]:
        raise NotImplementedError()

    @classmethod
    def _search_one(cls, query: Query, top: Optional[int] = None):
        if top is None:
            top = (
                len(query.expected_result)
                if query.expected_result is not None and len(query.expected_result) > 0
                else DEFAULT_TOP
            )

        start = time.perf_counter()
        search_res = cls.search_one(query, top)
        end = time.perf_counter()

        precision = 1.0
        if query.expected_result:
            ids = set(x[0] for x in search_res)
            precision = len(ids.intersection(query.expected_result[:top])) / top

        return precision, end - start

    def search_all(
        self,
        distance,
        queries: Iterable[Query],
    ):
        parallel = self.search_params.get("parallel", 1)
        top = self.search_params.get("top", None)
        # Run an untimed warm-up pass before timing so every engine is measured
        # at steady state. Engines differ in where their working set lives:
        # pgvector/qdrant/etc keep the index in shared buffers (already warm from
        # the build that precedes search), while engines with a separate scan
        # cache (e.g. pgturbohybrid's native mmap cache) start cold after build
        # and would otherwise pay one-time cache population INSIDE the timed
        # window. A warm-up pass through the same persistent workers makes the
        # comparison fair (engine vs engine, not warm-cache vs cold-cache). On by
        # default; set search_params.warmup=false to measure cold-start instead.
        warmup = self.search_params.get("warmup", True)

        # Materialize (and thereby parse) all queries up front, OUTSIDE the
        # timed region. Reading/decoding a query from the dataset costs ~0.5ms
        # here (JSON parse + ndarray build); leaving it inside the loop made the
        # single parent process the throughput bottleneck for fast engines — it
        # could only feed/parse queries at ~1-2k/s no matter how quickly the
        # engine answered. Search RPS should measure the engine, not the dataset
        # reader.
        queries = list(queries)

        # setup_search may require initialized client
        self.init_client(
            self.host, distance, self.connection_params, self.search_params
        )
        self.setup_search()

        search_one = functools.partial(self.__class__._search_one, top=top)

        if parallel == 1:
            if warmup:
                for query in tqdm.tqdm(queries, desc="warmup"):
                    search_one(query)
            start = time.perf_counter()
            precisions, latencies = list(
                zip(*[search_one(query) for query in tqdm.tqdm(queries)])
            )
        else:
            ctx = get_context(self.get_mp_start_method())

            # Hand each worker a batch of queries per IPC round-trip instead of
            # one at a time (imap's default chunksize=1). With single-query
            # chunks the parent does one pipe send + one result recv per query,
            # and that per-task orchestration — not the engine — capped
            # throughput at ~1k/s for sub-millisecond engines. Chunking keeps
            # the parent off the critical path while still load-balancing across
            # workers (each worker pulls many small chunks over the run).
            chunksize = max(1, len(queries) // (parallel * 16))

            with ctx.Pool(
                processes=parallel,
                initializer=self.__class__.init_client,
                initargs=(
                    self.host,
                    distance,
                    self.connection_params,
                    self.search_params,
                ),
            ) as pool:
                if parallel > 10:
                    time.sleep(15)  # Wait for all processes to start
                if warmup:
                    # Untimed warm-up through the SAME persistent pool workers,
                    # so each worker's connection and engine-side cache are hot
                    # before the timed pass.
                    list(
                        pool.imap_unordered(
                            search_one,
                            iterable=tqdm.tqdm(queries, desc="warmup"),
                            chunksize=chunksize,
                        )
                    )
                start = time.perf_counter()
                precisions, latencies = list(
                    zip(
                        *pool.imap_unordered(
                            search_one,
                            iterable=tqdm.tqdm(queries),
                            chunksize=chunksize,
                        )
                    )
                )

        total_time = time.perf_counter() - start

        self.__class__.delete_client()

        return {
            "total_time": total_time,
            "mean_time": np.mean(latencies),
            "mean_precisions": np.mean(precisions),
            "std_time": np.std(latencies),
            "min_time": np.min(latencies),
            "max_time": np.max(latencies),
            "rps": len(latencies) / total_time,
            "p95_time": np.percentile(latencies, 95),
            "p99_time": np.percentile(latencies, 99),
            "precisions": precisions,
            "latencies": latencies,
        }

    def setup_search(self):
        pass

    def post_search(self):
        pass

    @classmethod
    def delete_client(cls):
        pass
