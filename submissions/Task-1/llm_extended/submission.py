#!/usr/bin/env python3
"""UniAgent 2026 Task 1 (TIRA entrypoint): one retrieval system per document collection.

Compact version of ../retrieval-collection-routed/: same routing, same three systems, same runs.

1. Recognise the collection from the dataset itself (never from its name, judgments or narratives):
       query language en                            -> university-en
       query language de, document ids mostly UUIDs  -> university-de
       query language de, other ids (LARE190000005)  -> hessian-law-de
2. Run its system; every query starts from title + up to 12 description keywords:
       university-de   domain dictionary terms (weight 0.3) + BM25 over the official index;
                       top 100 re-ranked with 70 % BM25 + 30 % TF-IDF cosine
       university-en   DPH over the official index
       hessian-law-de  BM25F over body / title / body keywords (weights 1 / 3 / 2);
                       top 100 re-ranked with 70 % BM25F + 30 % TF-IDF cosine
3. Write run.txt.gz, run-trace.jsonl.log.gz, routing.json, retrieval_config.json, language.txt
   and the tracker metadata (index-ir-metadata.yml, retrieval-ir-metadata.yml).

Event log per query (../../event-logging-contract), model "-":
    input -> route_collection -> formulate_query -> [expand_with_dictionary] -> pyterrier_retrieve
          -> [cosine_rerank] -> observation -> decision
A failing step is logged with status "error", followed by an error decision, before the
exception propagates. The first stage is unchanged. Optional local Qwen3 reranking uses a neural model,
with local-only weights and no relevance judgments or runtime network requests.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import pandas as pd
import pyterrier as pt
from tirex_tracker import tracking
from tira.third_party_integrations import ir_datasets

from baseline import create_index, detect_query_language
from components import (CANDIDATE_DEPTH, DICTIONARY_FILE, FIELD_NORMALISATION, FIELD_WEIGHTS, MAX_DOCUMENT_KEYWORDS,
                        MAX_QUERY_KEYWORDS, TRACE_DEPTH, TRACE_FILE, DomainDictionary, EventLog, TermPipeline,
                        TfIdf, bm25, bm25f, bm25f_controls, build_field_index, cosine_scores, document_texts, dph,
                        fuse, query_rows, terrier_queries, weighted_query)

TOP_K = 1000
SYNONYM_WEIGHT = 0.3
ALPHA = 0.7  # weight of the first-stage score in the cosine fusion
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
SAMPLE_SIZE = 200     # documents inspected to recognise the collection
UUID_THRESHOLD = 0.5  # share of UUID document ids that marks a university collection
RULE = {"en": "university-en", f"de, uuid_share > {UUID_THRESHOLD}": "university-de",
        f"de, uuid_share <= {UUID_THRESHOLD}": "hessian-law-de"}
NO_RANKING = {"decision": "no_ranking", "answer": [], "evidence": []}


# ============================================================== routing
def recognise_collection(dataset) -> dict:
    """The collection and the evidence for it (query language, share of UUID document ids)."""
    language = detect_query_language(dataset)  # official; raises on missing or mixed languages
    ids = [str(document.doc_id) for _, document in zip(range(SAMPLE_SIZE), dataset.docs_iter())]
    if not ids:
        raise ValueError("Cannot recognise the collection: the dataset has no documents.")
    uuid_share = sum(bool(UUID.match(i)) for i in ids) / len(ids)
    if language == "en":
        collection = "university-en"
    elif language == "de":
        collection = "university-de" if uuid_share > UUID_THRESHOLD else "hessian-law-de"
    else:
        raise ValueError(f"No system for query language {language!r}; expected 'de' or 'en'.")
    return {"collection": collection, "language": language, "sampled_documents": len(ids),
            "uuid_share": round(uuid_share, 4), "example_document_ids": ids[:3]}


# ============================================================== event log
def top_documents(frame: pd.DataFrame | None, qid: str, columns=("score",), digits: int | None = 6) -> list[dict]:
    if frame is None or frame.empty:
        return []
    hits = frame[frame["qid"] == qid].head(TRACE_DEPTH)
    rnd = (lambda x: round(float(x), digits)) if digits is not None else float
    return [{"source_id": str(row.docno), "rank": int(row.rank), **{c: rnd(getattr(row, c)) for c in columns}}
            for row in hits.itertuples()]


def write_trace(output: Path, queries: list[dict], route: dict, steps: list[tuple[str, Callable, Callable]],
                final: Callable | None, failed: str | None = None, error: Exception | None = None) -> None:
    """One causal chain per query.

    `steps`: (tool, input(row), output(row)) after formulate_query. `failed` names the step that
    raised `error`; it is logged with status "error" and the chain ends with an error decision.
    `final(row)` returns (top documents, relevance text, ranked documents) for observation/decision.
    """
    with EventLog(output / TRACE_FILE) as log:
        for row in queries:
            qid = row["qid"]
            parent = log.log(qid, "input", input={"title": row["title"], "description": row["description"]})
            parent = log.log(qid, "tool_call", parent=parent, tool="route_collection",
                             input={"rule": RULE, "evidence": {k: v for k, v in route.items() if k != "collection"}},
                             output={"collection": route["collection"], "system": SYSTEMS[route["collection"]][0],
                                     "settings": SYSTEMS[route["collection"]][2]})
            parent = log.log(qid, "tool_call", parent=parent, tool="formulate_query",
                             input={"title": row["title"], "description": row["description"],
                                    "use_description_keywords": True},
                             output={"description_keywords": row["keywords"], "query": row["expanded"]})
            for tool, tool_input, tool_output in steps:
                if tool == failed:
                    parent = log.log(qid, "tool_call", parent=parent, tool=tool, input=tool_input(row),
                                     status="error", error=str(error))
                    log.log(qid, "decision", parent=parent, output=NO_RANKING, status="error", error=str(error))
                    break
                parent = log.log(qid, "tool_call", parent=parent, tool=tool, input=tool_input(row), output=tool_output(row))
            else:
                top, relevance, ranked = final(row)
                parent = log.log(qid, "observation", parent=parent, output={"top_documents": top})
                log.log(qid, "decision", parent=parent, output={
                    "decision": "ranking", "answer": [d["source_id"] for d in top],
                    "evidence": [{**d, "relevance": relevance} for d in top], "ranked_documents": ranked})


def write_routing_error_trace(dataset, output: Path, error: Exception) -> None:
    """Routing failed before any system ran: one error chain per query."""
    with EventLog(output / TRACE_FILE) as log:
        for query in dataset.queries_iter():
            qid = str(query.query_id)
            original = query.original_query if isinstance(query.original_query, dict) else {}
            parent = log.log(qid, "input", input={"title": query.default_text(),
                                                  "description": original.get("description", "")})
            parent = log.log(qid, "tool_call", parent=parent, tool="route_collection", input={"rule": RULE, "evidence": {}},
                             status="error", error=str(error))
            log.log(qid, "decision", parent=parent, output=NO_RANKING, status="error", error=str(error))


def count(frame: pd.DataFrame | None, qid: str) -> int:
    return 0 if frame is None or frame.empty else int((frame["qid"] == qid).sum())


# ============================================================== the three systems
def university_de(ds, language: str, queries: list[dict], output: Path, route: dict) -> dict:
    """Dictionary-expanded query, BM25, top 100 fused with TF-IDF cosine (70 / 30)."""
    sent_base = terrier_queries(queries, language)
    pipeline = TermPipeline(language)
    dictionary = DomainDictionary(language, pipeline)
    expansions = {row["qid"]: dictionary.expand(row["expanded"]) for row in queries}
    sent = {qid: weighted_query(sent_base[qid], expansions[qid].added, SYNONYM_WEIGHT) for qid in sent_base}
    index = create_index(ds, language, output)
    stages, timings = {}, {}
    steps = [
        ("expand_with_dictionary",
         lambda r: {"query": r["expanded"], "dictionary": DICTIONARY_FILE.name, "enabled": True, "weight": SYNONYM_WEIGHT},
         lambda r: {"matches": expansions[r["qid"]].matches, "added_terms": expansions[r["qid"]].added,
                    "terrier_query": sent.get(r["qid"], "")}),
        ("pyterrier_retrieve",
         lambda r: {"query": sent.get(r["qid"], ""), "wmodel": "BM25", "bm25.b": 0.75, "bm25.k_1": 1.2, "top_k": TOP_K},
         lambda r: {"retrieved": count(stages.get("bm25"), r["qid"]), "top_documents": top_documents(stages.get("bm25"), r["qid"])}),
        ("cosine_rerank",
         lambda r: {"enabled": True, "alpha": ALPHA, "depth": CANDIDATE_DEPTH, "vectors": "TF-IDF (1+ln tf)*ln(N/df)"},
         lambda r: {"reranked": int(min(CANDIDATE_DEPTH, count(stages.get("run"), r["qid"]))),
                    "top_documents": top_documents(stages.get("run"), r["qid"], ("score", "bm25", "cosine"))}),
    ]
    final = lambda r: (top_documents(stages["run"], r["qid"]),
                       "BM25 (expanded + dictionary query) fused with TF-IDF cosine", count(stages["run"], r["qid"]))
    step = "pyterrier_retrieve"
    try:
        start = time.perf_counter()
        with tracking(export_file_path=output / "retrieval-ir-metadata.yml"):
            stages["bm25"] = bm25(index, sent, TOP_K)
        timings["bm25_seconds"] = time.perf_counter() - start
        step, start = "cosine_rerank", time.perf_counter()
        tfidf = TfIdf(index, pipeline)
        vectors = {r["qid"]: tfidf.query_vector(r["expanded"], expansions[r["qid"]].added, SYNONYM_WEIGHT) for r in queries}
        texts = document_texts(ds, set(stages["bm25"].groupby("qid", sort=False).head(CANDIDATE_DEPTH)["docno"]))
        stages["run"] = fuse(stages["bm25"], cosine_scores(stages["bm25"], vectors, texts, tfidf), ALPHA)
        timings["cosine_seconds"] = time.perf_counter() - start
    except Exception as error:
        write_trace(output, queries, route, steps, None, step, error)
        raise
    pt.io.write_results(stages["run"][["qid", "docno", "score", "rank"]], output / "run.txt.gz")
    write_trace(output, queries, route, steps, final)
    return {"approach": "qe+dictionary+bm25+tfidf-cosine", "dictionary": True, "synonym_weight": SYNONYM_WEIGHT,
            "cosine": True, "alpha": ALPHA, "depth": CANDIDATE_DEPTH,
            "queries_with_dictionary_terms": sum(bool(e.added) for e in expansions.values()),
            "retrieved_rows": len(stages["run"]), **{k: round(v, 3) for k, v in timings.items()}}


def university_en(ds, language: str, queries: list[dict], output: Path, route: dict) -> dict:
    """Description-keyword query, DPH."""
    index = create_index(ds, language, output)
    sent = terrier_queries(queries, language)
    stages = {}
    settings = {"implementation": "components.dph", "index": "Original BM25 index", "wmodel": "DPH", "controls": {},
                "top_k": TOP_K}
    steps = [("pyterrier_retrieve", lambda r: {"query": sent[r["qid"]], **settings},
              lambda r: {"retrieved": count(stages.get("run"), r["qid"])})]
    final = lambda r: (top_documents(stages["run"], r["qid"], digits=None),
                       f"DPH score for query '{sent[r['qid']]}'", count(stages["run"], r["qid"]))
    try:
        with tracking(export_file_path=output / "retrieval-ir-metadata.yml"):
            stages["run"] = dph(index, sent, TOP_K)
        pt.io.write_results(stages["run"], output / "run.txt.gz")
    except Exception as error:
        write_trace(output, queries, route, steps, None, "pyterrier_retrieve", error)
        raise
    write_trace(output, queries, route, steps, final)
    return {"approach": "query-expansion+lexical-ranker", "query": "title + description keywords",
            "index": "Original BM25 index", "wmodel": "DPH", "controls": {}, "retrieved_rows": len(stages["run"])}


def hessian_law_de(ds, language: str, queries: list[dict], output: Path, route: dict) -> dict:
    """BM25F field index, top 100 fused with TF-IDF cosine (70 / 30)."""
    index, fields, field_stats = build_field_index(ds, language, output)
    sent = terrier_queries(queries, language)
    controls = bm25f_controls(fields)
    stages, timings = {}, {}
    steps = [
        ("pyterrier_retrieve",
         lambda r: {"query": sent.get(r["qid"], ""), "wmodel": "BM25F", "controls": controls, "top_k": TOP_K},
         lambda r: {"retrieved": count(stages.get("bm25f"), r["qid"]), "top_documents": top_documents(stages.get("bm25f"), r["qid"])}),
        ("cosine_rerank",
         lambda r: {"alpha": ALPHA, "depth": CANDIDATE_DEPTH, "vectors": "TF-IDF (1+ln tf)*ln(N/df), body text"},
         lambda r: {"reranked": int(min(CANDIDATE_DEPTH, count(stages.get("run"), r["qid"]))),
                    "top_documents": top_documents(stages.get("run"), r["qid"], ("score", "bm25f", "cosine"))}),
    ]
    final = lambda r: (top_documents(stages["run"], r["qid"]),
                       f"BM25F + TF-IDF cosine (alpha={ALPHA}) for query '{sent.get(r['qid'], '')}'",
                       count(stages["run"], r["qid"]))
    step = "pyterrier_retrieve"
    try:
        start = time.perf_counter()
        with tracking(export_file_path=output / "retrieval-ir-metadata.yml"):
            stages["bm25f"] = bm25f(index, sent, fields, TOP_K)
        timings["bm25f_seconds"] = time.perf_counter() - start
        step, start = "cosine_rerank", time.perf_counter()
        tfidf = TfIdf(index, TermPipeline(language))
        vectors = {r["qid"]: tfidf.vector(r["expanded"]) for r in queries}
        texts = document_texts(ds, set(stages["bm25f"].groupby("qid", sort=False).head(CANDIDATE_DEPTH)["docno"]))
        stages["run"] = fuse(stages["bm25f"], cosine_scores(stages["bm25f"], vectors, texts, tfidf), ALPHA,
                             first_stage="bm25f")
        timings["cosine_seconds"] = time.perf_counter() - start
    except Exception as error:
        write_trace(output, queries, route, steps, None, step, error)
        raise
    pt.io.write_results(stages["run"][["qid", "docno", "score", "rank"]], output / "run.txt.gz")
    write_trace(output, queries, route, steps, final)
    return {"approach": "bm25f-combined+tfidf-cosine", "alpha": ALPHA, "depth": CANDIDATE_DEPTH,
            "field_weights": FIELD_WEIGHTS, "field_normalisation": FIELD_NORMALISATION,
            "max_document_keywords": MAX_DOCUMENT_KEYWORDS, "field_statistics": field_stats,
            "retrieved_rows": len(stages["run"]), **{k: round(v, 3) for k, v in timings.items()}}


SYSTEMS = {  # collection -> (description, function, settings)
    "university-de": ("domain dictionary (w=0.3) + BM25 + 30% TF-IDF cosine", university_de,
                      {"synonym_weight": SYNONYM_WEIGHT, "alpha": ALPHA}),
    "university-en": ("query expansion + DPH", university_en, {"wmodel": "DPH"}),
    "hessian-law-de": ("BM25F + keywords + 30% TF-IDF cosine", hessian_law_de, {"alpha": ALPHA}),
}


# ============================================================== command line
@click.command()
@click.option("--dataset", required=True, help="Dataset directory or TIRA dataset ID")
@click.option("--output", required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option("--reranker", type=click.Choice(["none", "qwen3"]), default="none", show_default=True)
@click.option("--model-path", default="models/qwen3-reranker", show_default=True)
@click.option("--rerank-depth", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default="cpu", show_default=True)
@click.option("--max-length", type=click.IntRange(min=256,max=32768), default=1536, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=1, show_default=True)
def main(dataset: str, output: Path, reranker: str, model_path: str,
         rerank_depth: int, device: str, max_length: int, batch_size: int) -> None:
    """Recognise the collection and run its system."""
    if reranker == "qwen3" and (output / "run.first-stage.txt.gz").exists():
        raise click.ClickException("Use a fresh output folder for a full pipeline run; to rerank an existing source use neural_reranking.py.")
    ds = ir_datasets.load(dataset)
    output.mkdir(parents=True, exist_ok=True)
    try:
        route = recognise_collection(ds)
    except ValueError as error:
        write_routing_error_trace(ds, output, error)
        raise click.ClickException(str(error)) from error
    name, system, settings = SYSTEMS[route["collection"]]
    click.echo(f"Collection {route['collection']} (language {route['language']}, UUID share {route['uuid_share']}): {name}")
    (output / "routing.json").write_text(json.dumps(
        {**route, "system": name, "settings": settings,
         "decided_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}, indent=2), encoding="utf-8")
    language = route["language"]
    (output / "language.txt").write_text(language + "\n", encoding="utf-8")
    queries = query_rows(ds, language, MAX_QUERY_KEYWORDS)
    config = system(ds, language, queries, output, route)
    config = {"collection": route["collection"], "language": language, "top_k": TOP_K,
              "max_query_keywords": MAX_QUERY_KEYWORDS, **config, "uses_narrative": False,
              "uses_relevance_judgments": False, "uses_llm": False, "uses_embeddings": False}
    (output / "retrieval_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    click.echo(f"{name}: {config['retrieved_rows']} results -> {output / 'run.txt.gz'}")
    if reranker == "qwen3":
        from neural_reranking import rerank_dataset
        click.echo("Starting offline Qwen3 reranking; original ranks will be preserved.")
        rerank_dataset(dataset, output, model_path, depth=rerank_depth,
                       device=device, max_length=max_length, batch_size=batch_size)
        click.echo(f"Reranked results -> {output / 'run.txt.gz'}")


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    print("Starting retrieval submission", flush=True)
    main()
