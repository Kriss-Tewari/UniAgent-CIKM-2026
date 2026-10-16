#!/usr/bin/env python3
"""UniAgent 2026 Task 1: official PyTerrier BM25 baseline + description-keyword query expansion.

The official baseline.py is imported, never edited: its index (create_index) and
language detection are reused unchanged. The only difference to the baseline is the
query: the topic title is extended with unique non-stopword terms from
original_query.description. Narratives and qrels are never read. No LLM, neural model
or embedding is used.

Outputs in --output: run.txt.gz, run-trace.jsonl.log.gz (see
../../event-logging-contract/README.md), language.txt, index-ir-metadata.yml,
retrieval-ir-metadata.yml and retrieval_config.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
import pandas as pd
import pyterrier as pt
from tirex_tracker import tracking
from tira.third_party_integrations import ir_datasets

# REUSE the official baseline instead of copying/changing its implementation.
from baseline import RETRIEVAL_CONFIGURATION, create_index, detect_query_language
from event_logging import TRACE_FILE, EventLog

WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
EN_STOP = set("""a an and are as at be been by can could did do does for from had has have
how i if in into is it its may more my no not of on or our should some such than that
the their them there these they this those to under up us was were what when where which
who why will with would you your""".split())
DE_QUESTION_STOP = set("""bitte erklaere erklären erkläre welche welcher welches welchen warum
wann wer wen wessen wofür womit woher wohin information informationen""".split())
EN_QUESTION_STOP = {"please", "explain", "information", "details"}
TRACE_DEPTH = 10  # retrieved documents per query listed as observation/decision evidence


def stopwords(language: str) -> set[str]:
    """Use the official German stopword file; small local English supplement."""
    if language == "de":
        path = Path(__file__).with_name("german-stopwords.txt")
        return set(path.read_text(encoding="utf-8").lower().splitlines()) | DE_QUESTION_STOP
    if language == "en":
        return EN_STOP | EN_QUESTION_STOP
    raise ValueError(f"Unsupported language: {language}")


def tokens(text: str, language: str) -> list[str]:
    stops = stopwords(language)
    return [word.lower() for word in WORD_RE.findall(text or "")
            if len(word) >= 3 and not word.isdigit() and word.lower() not in stops]


def query_keywords(description: str, title: str, language: str, limit: int = 12) -> list[str]:
    """Select extra terms from DESCRIPTION only. No narrative, model, or qrels."""
    existing = {w.lower() for w in WORD_RE.findall(title)}
    result = []
    for word in tokens(description, language):
        if word not in existing:
            result.append(word)
            existing.add(word)
        if len(result) >= limit:
            break
    return result


def query_rows(dataset, language: str, limit: int) -> list[dict]:
    """Read title and description; intentionally never access narrative."""
    rows = []
    for query in dataset.queries_iter():
        title = query.default_text()
        original = query.original_query
        description = original.get("description", "") if isinstance(original, dict) else ""
        if not isinstance(description, str):
            description = ""
        kws = query_keywords(description, title, language, limit)
        rows.append({"qid": str(query.query_id), "title": title,
                     "description": description, "keywords": kws,
                     "expanded": " ".join([title, *kws]).strip()})
    return rows


def terrier_queries(queries: list[dict], language: str) -> dict[str, str]:
    """qid -> expanded query as sent to Terrier (same tokenisation as official baseline.retrieve)."""
    tokeniser = pt.TerrierTokeniser.java_tokeniser(RETRIEVAL_CONFIGURATION[language]["tokeniser"])
    return {row["qid"]: " ".join(tokeniser.getTokens(row["expanded"])) for row in queries}


def search(index, sent: dict[str, str], output: Path, top_k: int) -> pd.DataFrame:
    """Original tokenize -> BM25 Retriever -> run.txt.gz flow, with the expanded query."""
    topics = pd.DataFrame([{"qid": qid, "query": query} for qid, query in sent.items()],
                          columns=["qid", "query"])
    if topics.empty:
        raise ValueError("Dataset has no queries")

    with tracking(export_file_path=output / "retrieval-ir-metadata.yml"):
        results = pt.terrier.Retriever(index, wmodel="BM25", verbose=True).transform(topics)
    if not results.empty:
        results = (results.sort_values(["qid", "rank"])
                   .groupby("qid", sort=False).head(top_k).copy())
    pt.io.write_results(results, output / "run.txt.gz")
    return results


def retrieval_settings(top_k: int) -> dict:
    return {"implementation": "query_expansion.search", "index": "Original BM25 index",
            "wmodel": "BM25", "top_k": top_k}


def write_trace(queries: list[dict], sent: dict[str, str], results: pd.DataFrame | None,
                settings: dict, output: Path, error: Exception | None = None) -> None:
    """One event chain per query: input -> query formulation -> retrieval -> observation -> decision.

    No model is called, so every event has model "-". With `error`, the failed
    retrieval is logged with status "error" for each query.
    """
    hits_by_qid = {}
    if results is not None and not results.empty:
        hits_by_qid = {str(qid): hits.sort_values("rank") for qid, hits in results.groupby("qid")}
    with EventLog(output / TRACE_FILE) as log:
        for row in queries:
            qid = row["qid"]
            parent = log.log(qid, "input", input={"title": row["title"], "description": row["description"]})
            parent = log.log(qid, "tool_call", parent=parent, tool="formulate_query",
                             input={"title": row["title"], "description": row["description"],
                                    "use_description_keywords": True},
                             output={"description_keywords": row["keywords"], "query": row["expanded"]})
            if error is not None:
                parent = log.log(qid, "tool_call", parent=parent, tool="pyterrier_retrieve",
                                 input={"query": sent[qid], **settings}, status="error", error=str(error))
                log.log(qid, "decision", parent=parent, output={"decision": "no_ranking", "answer": [],
                                                                  "evidence": []},
                        status="error", error=str(error))
                continue
            hits = hits_by_qid.get(qid, pd.DataFrame(columns=["docno", "rank", "score"]))
            parent = log.log(qid, "tool_call", parent=parent, tool="pyterrier_retrieve",
                             input={"query": sent[qid], **settings}, output={"retrieved": len(hits)})
            top = [{"source_id": str(doc), "rank": int(rank), "score": float(score)}
                   for doc, rank, score in zip(hits["docno"], hits["rank"], hits["score"])][:TRACE_DEPTH]
            parent = log.log(qid, "observation", parent=parent, output={"top_documents": top})
            log.log(qid, "decision", parent=parent, output={
                "decision": "ranking",
                "answer": [doc["source_id"] for doc in top],
                "evidence": [{**doc, "relevance": f"BM25 score for query '{sent[qid]}'"} for doc in top],
                "ranked_documents": len(hits),
            })


@click.command()
@click.option("--dataset", required=True, help="Dataset directory or TIRA dataset ID")
@click.option("--output", required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option("--top-k", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--max-query-keywords", type=click.IntRange(min=1), default=12, show_default=True)
def main(dataset: str, output: Path, top_k: int, max_query_keywords: int) -> None:
    """BM25 retrieval with the title expanded by description keywords."""
    ds = ir_datasets.load(dataset)
    try:
        language = detect_query_language(ds)  # Official function
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    queries = query_rows(ds, language, max_query_keywords)
    output.mkdir(parents=True, exist_ok=True)
    (output / "language.txt").write_text(language + "\n", encoding="utf-8")
    click.echo(f"Detected query language: {language}")

    index = create_index(ds, language, output)  # Official, unchanged index
    settings = retrieval_settings(top_k)
    sent = terrier_queries(queries, language)
    try:
        result = search(index, sent, output, top_k)
    except Exception as error:
        # Contract: errors are logged before the exception propagates.
        write_trace(queries, sent, None, settings, output, error)
        raise
    write_trace(queries, sent, result, settings, output)
    config = {
        "approach": "query", "language": language, "top_k": top_k,
        "uses_narrative": False, "uses_llm": False, "uses_embeddings": False,
        "query": "title + description keywords", "index": "Original BM25 index",
        "max_query_keywords": max_query_keywords, "retrieved_rows": len(result),
    }
    (output / "retrieval_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    click.echo(f"query: {len(result)} results -> {output / 'run.txt.gz'}")


if __name__ == "__main__":
    main()
