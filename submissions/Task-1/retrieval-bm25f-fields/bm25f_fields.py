#!/usr/bin/env python3
"""UniAgent 2026 Task 1: official PyTerrier baseline query + BM25F over three document fields.

The official baseline.py is imported, never edited: its language detection, Terrier
configuration (stemmer, stopwords, tokeniser) and query tokenisation are reused. The only
difference to the baseline is the index: every document is indexed with three fields

    text      - the original document body (as in the baseline),
    title     - the stored document title, if the dataset provides one (never made up),
    keywords  - the 20 most frequent informative body words,

and ranked with BM25F using field weights text/title/keywords = 1/3/2. The query is the
topic title, exactly as in the baseline. Narratives and qrels are never read. No LLM,
neural model or embedding is used.

Outputs in --output: run.txt.gz, run-trace.jsonl.log.gz (see
../../event-logging-contract/README.md), language.txt, index-ir-metadata.yml,
retrieval-ir-metadata.yml and retrieval_config.json.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import click
import pandas as pd
import pyterrier as pt
from tirex_tracker import tracking
from tqdm import tqdm
from tira.third_party_integrations import ir_datasets

# REUSE the official baseline instead of copying/changing its implementation.
from baseline import RETRIEVAL_CONFIGURATION, detect_query_language
from event_logging import TRACE_FILE, EventLog

FIELD_WEIGHTS = {"text": 1.0, "title": 3.0, "keywords": 2.0}
FIELD_NORMALISATION = {"text": 0.75, "title": 0.5, "keywords": 0.5}
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


def document_keywords(text: str, language: str, limit: int = 20) -> list[str]:
    """Most frequent informative BODY words, breaking ties by first occurrence."""
    words = tokens(text[:200_000], language)
    counts = Counter(words)
    first = {}
    for i, word in enumerate(words):
        first.setdefault(word, i)
    return sorted(counts, key=lambda word: (-counts[word], first[word]))[:limit]


def explicit_title(document) -> str:
    """Only use a stored document title; never make up a title from the body."""
    for key in ("title", "page_title"):
        value = getattr(document, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("title", "page_title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def query_rows(dataset) -> list[dict]:
    """Read title (the query) and description (logged only); never access narrative."""
    rows = []
    for query in dataset.queries_iter():
        original = query.original_query
        description = original.get("description", "") if isinstance(original, dict) else ""
        if not isinstance(description, str):
            description = ""
        rows.append({"qid": str(query.query_id), "title": query.default_text(), "description": description})
    return rows


def build_field_index(dataset, language: str, output: Path, max_keywords: int):
    """Same IterDictIndexer style as the official baseline; adds title and keyword fields.

    Each document keeps its original docno; query terms never enter the index.
    """
    index_dir = Path(tempfile.mkdtemp(prefix="uniagent-bm25f-fields_"))
    fields = ["text", "title", "keywords"]
    indexer = pt.IterDictIndexer(
        str(index_dir.resolve()), overwrite=True, meta={"docno": 100},
        text_attrs=fields, fields=True, verbose=True,
        **RETRIEVAL_CONFIGURATION[language],
    )
    title_count = 0
    document_count = 0

    def docs():
        nonlocal title_count, document_count
        for document in dataset.docs_iter():
            text = document.default_text() or ""
            title = explicit_title(document)
            document_count += 1
            title_count += bool(title)
            yield {"docno": str(document.doc_id), "text": text,
                   "title": title, "keywords": " ".join(document_keywords(text, language, max_keywords))}

    # Same resource tracker as the official baseline.
    with tracking(export_file_path=output / "index-ir-metadata.yml"):
        ref = indexer.index(tqdm(docs(), desc="Index document fields"))
    index = pt.IndexFactory.of(ref)
    actual = [str(field) for field in index.getCollectionStatistics().getFieldNames()]
    if set(actual) != set(fields):
        raise RuntimeError(f"Unexpected BM25F fields: {actual}")
    print(f"Indexed {document_count} documents; {title_count} have explicit titles.")
    return index, actual, {"documents": document_count, "documents_with_titles": title_count}


def bm25f_controls(actual_fields: list[str]) -> dict[str, float]:
    """Field indices come from Terrier, not from assumed dictionary ordering."""
    controls = {}
    for i, name in enumerate(actual_fields):
        controls[f"w.{i}"] = FIELD_WEIGHTS[name]
        controls[f"c.{i}"] = FIELD_NORMALISATION[name]
    return controls


def terrier_queries(queries: list[dict], language: str) -> dict[str, str]:
    """qid -> title query as sent to Terrier (same tokenisation as official baseline.retrieve)."""
    tokeniser = pt.TerrierTokeniser.java_tokeniser(RETRIEVAL_CONFIGURATION[language]["tokeniser"])
    return {row["qid"]: " ".join(tokeniser.getTokens(row["title"])) for row in queries}


def search(index, sent: dict[str, str], fields: list[str], output: Path, top_k: int) -> pd.DataFrame:
    """Original tokenize -> Retriever -> run.txt.gz flow, with BM25F."""
    topics = pd.DataFrame([{"qid": qid, "query": query} for qid, query in sent.items()],
                          columns=["qid", "query"])
    if topics.empty:
        raise ValueError("Dataset has no queries")

    with tracking(export_file_path=output / "retrieval-ir-metadata.yml"):
        results = pt.terrier.Retriever(index, wmodel="BM25F", controls=bm25f_controls(fields),
                                       verbose=True).transform(topics)
    if not results.empty:
        results = (results.sort_values(["qid", "rank"])
                   .groupby("qid", sort=False).head(top_k).copy())
    pt.io.write_results(results, output / "run.txt.gz")
    return results


def retrieval_settings(top_k: int, fields: list[str]) -> dict:
    return {"implementation": "bm25f_fields.search", "index": "BM25F field index",
            "wmodel": "BM25F", "controls": bm25f_controls(fields), "top_k": top_k}


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
                                    "use_description_keywords": False},
                             output={"description_keywords": [], "query": row["title"]})
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
                "evidence": [{**doc, "relevance": f"BM25F score for query '{sent[qid]}'"} for doc in top],
                "ranked_documents": len(hits),
            })


@click.command()
@click.option("--dataset", required=True, help="Dataset directory or TIRA dataset ID")
@click.option("--output", required=True, type=click.Path(path_type=Path, file_okay=False))
@click.option("--top-k", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--max-document-keywords", type=click.IntRange(min=1), default=20, show_default=True)
def main(dataset: str, output: Path, top_k: int, max_document_keywords: int) -> None:
    """BM25F retrieval over body, stored title and body-keyword fields."""
    ds = ir_datasets.load(dataset)
    try:
        language = detect_query_language(ds)  # Official function
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    queries = query_rows(ds)
    output.mkdir(parents=True, exist_ok=True)
    (output / "language.txt").write_text(language + "\n", encoding="utf-8")
    click.echo(f"Detected query language: {language}")

    index, fields, field_stats = build_field_index(ds, language, output, max_document_keywords)
    settings = retrieval_settings(top_k, fields)
    sent = terrier_queries(queries, language)
    try:
        result = search(index, sent, fields, output, top_k)
    except Exception as error:
        # Contract: errors are logged before the exception propagates.
        write_trace(queries, sent, None, settings, output, error)
        raise
    write_trace(queries, sent, result, settings, output)
    config = {
        "approach": "fields", "language": language, "top_k": top_k,
        "uses_narrative": False, "uses_llm": False, "uses_embeddings": False,
        "query": "title", "index": "BM25F field index", "field_weights": FIELD_WEIGHTS,
        "field_normalisation": FIELD_NORMALISATION, "max_document_keywords": max_document_keywords,
        "field_statistics": field_stats, "retrieved_rows": len(result),
    }
    (output / "retrieval_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    click.echo(f"fields: {len(result)} results -> {output / 'run.txt.gz'}")


if __name__ == "__main__":
    main()
