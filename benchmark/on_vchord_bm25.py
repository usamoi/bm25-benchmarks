import json
import os
from pathlib import Path
import time
from typing import List, LiteralString, cast, Any

import psycopg
from tqdm import tqdm

import beir.util  # pyright: ignore[reportMissingTypeStubs]
from beir.datasets.data_loader import GenericDataLoader  # pyright: ignore[reportMissingTypeStubs]
from beir.retrieval.evaluation import EvaluateRetrieval  # pyright: ignore[reportMissingTypeStubs]

from utils.benchmark import get_max_memory_usage, Timer
from utils.beir import merge_cqa_dupstack, clean_results_keys  # pyright: ignore[reportUnknownVariableType]


def create_table(
    client: psycopg.Connection,
    corpus: List[tuple[str, str]],
    queries: List[tuple[str, str]],
) -> None:
    with client.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE corpus (
                id text primary key,
                text TEXT,
                embedding tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED)
            """)
        cursor.execute("CREATE TABLE queries (id text primary key, text TEXT)")
        with cursor.copy(
            "COPY corpus (id, text) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["text", "text"])
            for id, text in tqdm(corpus):
                copy.write_row((id, text))
        with cursor.copy(
            "COPY queries (id, text) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["text", "text"])
            for id, text in tqdm(queries):
                copy.write_row((id, text))


def create_index(
    client: psycopg.Connection,
) -> None:
    with client.cursor() as cursor:
        cursor.execute("""
            CREATE INDEX corpus_embedding_bm25 ON corpus
            USING bm25 (embedding bm25_ops);
            """)


def select(client: psycopg.Connection, top_k: int) -> dict[str, dict[str, float]]:
    with client.cursor() as cursor:
        cursor.execute(cast(LiteralString, f'SET "bm25.limit" = {top_k};'))
        cursor.execute(
            cast(
                LiteralString,
                f"""
            SELECT q.id, c.id, c.distance FROM queries q, LATERAL (
                SELECT id, corpus.embedding <&> to_bm25query(to_tsvector('english', q.text), 'corpus_embedding_bm25') AS distance
                FROM corpus
                ORDER BY distance
                LIMIT {top_k}
            ) c;
            """,
            )
        )
        rows = cursor.fetchall()
        results: dict[str, dict[str, float]] = {}
        for qid, cid, distance in rows:
            key = str(qid)
            if key not in results:
                results[key] = {}
            results[key][str(cid)] = -float(distance)
        return results


def main(
    dataset: str,
    top_k: int = 1000,
    save_dir: str = "datasets",
    result_dir: str = "results",
):
    data_path = beir.util.download_and_unzip(
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
        save_dir,
    )

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    if dataset == "msmarco":
        split = "dev"
    else:
        split = "test"

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    num_docs = len(corpus)
    num_queries = len(queries)

    corpusl: List[tuple[str, str]] = []
    for key, value in corpus.items():
        id = key
        text = (value["title"] + " " + value["text"]).replace("\u0000", "")
        corpusl.append((id, text))

    queriesl: List[tuple[str, str]] = []
    for key, value in queries.items():
        id = key
        text = value
        queriesl.append((id, text))

    print("=" * 50)
    print("Dataset: ", dataset)
    print(f"Corpus Size: {num_docs:,}")
    print(f"Queries Size: {num_queries:,}")

    client = psycopg.connect(
        "postgresql://usamoi@localhost:5432/usamoi", autocommit=True
    )

    timer = Timer("[vchord-bm25]")

    t_insert = timer.start("Insert")  # pyright: ignore[reportUnknownMemberType]
    create_table(client, corpusl, queriesl)
    timer.stop(t_insert, show=True, n_total=num_docs)  # pyright: ignore[reportUnknownMemberType]

    t_index = timer.start("Index")  # pyright: ignore[reportUnknownMemberType]
    create_index(client)
    timer.stop(t_index, show=True, n_total=num_docs)  # pyright: ignore[reportUnknownMemberType]

    t_query = timer.start("Query")  # pyright: ignore[reportUnknownMemberType]
    results = select(client, top_k)
    timer.stop(t_query, show=True, n_total=num_queries)  # pyright: ignore[reportUnknownMemberType]

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [1, 10, 100, 1000]
    )

    max_mem_gb = get_max_memory_usage("GB")

    print("=" * 50)
    print(f"Max Memory Usage: {max_mem_gb:.4f} GB")
    print("-" * 50)
    print(ndcg)
    print(recall)
    print("=" * 50)

    # Save everything to json
    save_dict: dict[str, Any] = {
        "model": "vchord-bm25",
        "dataset": dataset,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": top_k,
        "max_mem_gb": max_mem_gb,
        "stats": {
            "num_docs": num_docs,
            "num_queries": num_queries,
        },
        "timing": timer.to_dict(underscore=True, lowercase=True),
        "scores": {
            "ndcg": clean_results_keys(ndcg),
            "map": clean_results_keys(_map),
            "recall": clean_results_keys(recall),
            "precision": clean_results_keys(precision),
        },
    }

    dir = Path(result_dir) / Path("vchord-bm25")
    dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(result_dir) / f"{dataset}-{os.urandom(8).hex()}.json"
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark vchord-bm25 on a dataset.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="fiqa",
        help="Dataset to benchmark on.",
    )

    parser.add_argument(
        "-n", "--num_runs", type=int, default=1, help="Number of runs to repeat main."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of top-k documents to retrieve.",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets",
        help="Directory to save datasets.",
    )

    kwargs = vars(parser.parse_args())
    print(kwargs)
    profile = kwargs.pop("profile")
    num_runs = kwargs.pop("num_runs")

    if profile:
        import cProfile
        import pstats

        if num_runs > 1:
            raise ValueError("Cannot profile with multiple runs.")

        cProfile.run("main(**kwargs)", filename="vchord-bm25.prof")
        p = pstats.Stats("vchord-bm25.prof")
        p.sort_stats("time").print_stats(50)
    else:
        for _ in range(num_runs):
            main(**kwargs)
