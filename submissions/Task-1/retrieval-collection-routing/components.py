"""Shared building blocks for submission.py (compact version of ../retrieval-collection-routed/).

Sections:
  1. Event log            - the UniAgent'26 event-logging contract (run-trace.jsonl.log.gz)
  2. Query formulation    - title + up to 12 description keywords (never the narrative)
  3. Index terms          - the index's own tokeniser, stopword list and stemmer
  4. Domain dictionary    - related terms added to the query as weighted Terrier terms
  5. Retrieval            - BM25, DPH and BM25F (field index) through Terrier
  6. TF-IDF cosine fusion - re-ranks the top 100 of a run

Every function is copied, unchanged in behaviour, from the systems the full version routes to:
../retrieval-query-expansion, ../retrieval-qe-dictionary-cosine, ../retrieval-lexical-ranking,
../retrieval-bm25f-combined and ../retrieval-bm25f-combined-cosine.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyterrier as pt
from tirex_tracker import tracking
from tqdm import tqdm

from baseline import RETRIEVAL_CONFIGURATION

# ============================================================== 1. Event log
NO_MODEL = "-"  # no model is called: the contract asks for "-" on every event
TRACE_FILE = "run-trace.jsonl.log.gz"
EVENT_TYPES = ("input", "model_call", "tool_call", "observation", "decision", "error")
PREVIEW_LIMIT = 1000  # long strings are whitespace-normalised and cut to this many characters
TRACE_DEPTH = 10      # documents listed per query in observation/decision events


def preview(value: Any, limit: int = PREVIEW_LIMIT) -> Any:
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"
    if isinstance(value, dict):
        return {str(key): preview(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [preview(item, limit) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return preview(str(value), limit)


class EventLog:
    """Writes contract-compliant events; use as `with EventLog(path) as log:`."""

    def __init__(self, path: Path, model: str = NO_MODEL):
        if not model:
            raise ValueError("The contract requires a model on every event ('-' if none).")
        self.model = model
        self._file = gzip.open(path, "wt", encoding="utf-8")
        self._count = 0

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self._file.close()

    def log(self, case_id: str, event_type: str, *, parent: Optional[str] = None,
            tool: Optional[str] = None, input: Any = None, output: Any = None,
            status: str = "ok", error: Optional[str] = None) -> str:
        """Write one event and return its event_id (to pass as `parent` of the next one)."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {event_type}")
        if (event_type == "tool_call") != (tool is not None):
            raise ValueError("`tool` is required for tool_call events and must be null otherwise.")
        if status not in ("ok", "error") or (status == "error") != (error is not None):
            raise ValueError("status must be 'ok' with error=None, or 'error' with an error message.")
        self._count += 1
        event_id = f"evt-{self._count:04d}"
        entry = {"case_id": str(case_id), "event_id": event_id, "parent_event_id": parent,
                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 "event_type": event_type, "model": self.model, "tool": tool,
                 "input": preview(input), "output": preview(output), "status": status, "error": error}
        print(json.dumps(entry, ensure_ascii=False), file=self._file, flush=True)
        return event_id


# ============================================================== 2. Query formulation
WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
EN_STOP = set("""a an and are as at be been by can could did do does for from had has have
how i if in into is it its may more my no not of on or our should some such than that
the their them there these they this those to under up us was were what when where which
who why will with would you your""".split())
DE_QUESTION_STOP = set("""bitte erklaere erklären erkläre welche welcher welches welchen warum
wann wer wen wessen wofür womit woher wohin information informationen""".split())
EN_QUESTION_STOP = {"please", "explain", "information", "details"}
MAX_QUERY_KEYWORDS = 12


def stopwords(language: str) -> set[str]:
    """The official German stopword file; a small local English list."""
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


def query_keywords(description: str, title: str, language: str, limit: int = MAX_QUERY_KEYWORDS) -> list[str]:
    """Up to `limit` description words not already in the title, in order of first occurrence."""
    existing = {w.lower() for w in WORD_RE.findall(title)}
    result = []
    for word in tokens(description, language):
        if word not in existing:
            result.append(word)
            existing.add(word)
        if len(result) >= limit:
            break
    return result


def query_rows(dataset, language: str, limit: int = MAX_QUERY_KEYWORDS) -> list[dict]:
    """Title and description of every query; the narrative is never read."""
    rows = []
    for query in dataset.queries_iter():
        title = query.default_text()
        original = query.original_query
        description = original.get("description", "") if isinstance(original, dict) else ""
        if not isinstance(description, str):
            description = ""
        kws = query_keywords(description, title, language, limit)
        rows.append({"qid": str(query.query_id), "title": title, "description": description,
                     "keywords": kws, "expanded": " ".join([title, *kws]).strip()})
    return rows


def terrier_queries(queries: list[dict], language: str) -> dict[str, str]:
    """qid -> expanded query, tokenised as the official baseline.retrieve does."""
    tokeniser = pt.TerrierTokeniser.java_tokeniser(RETRIEVAL_CONFIGURATION[language]["tokeniser"])
    return {row["qid"]: " ".join(tokeniser.getTokens(row["expanded"])) for row in queries}


# ============================================================== 3. Index terms
class TermPipeline:
    """Text -> indexed terms with the index's tokeniser, stopword list and stemmer (cached).

    Note: Terrier's tokeniser drops words longer than 20 characters (max.term.length).
    """

    def __init__(self, language: str):
        if language not in RETRIEVAL_CONFIGURATION:
            raise ValueError(f"Unsupported language: {language}")
        if not pt.java.started():
            pt.java.init()
        autoclass = pt.java.autoclass
        if language == "de":
            stops = {w.strip().lower() for w in RETRIEVAL_CONFIGURATION["de"]["stopwords"]}
            self._is_stopword = stops.__contains__
            self._stemmer = autoclass("org.terrier.terms.GermanSnowballStemmer")(None)
        else:  # pt.TerrierStopwords.terrier + Porter, as configured for the English index
            terrier_stopwords = autoclass("org.terrier.terms.Stopwords")(None)
            self._is_stopword = lambda term: bool(terrier_stopwords.isStopword(term))
            self._stemmer = autoclass("org.terrier.terms.PorterStemmer")(None)
        self.language = language
        self._tokeniser = pt.TerrierTokeniser.java_tokeniser(RETRIEVAL_CONFIGURATION[language]["tokeniser"])
        self._cache: dict[str, str | None] = {}

    def tokens(self, text: str) -> list[str]:
        return [str(t).lower() for t in self._tokeniser.getTokens(text or "") if str(t)]

    def term(self, token: str) -> str | None:
        if token not in self._cache:
            term = None
            if token and not self._is_stopword(token):
                term = str(self._stemmer.stem(token)) or None
            self._cache[token] = term
        return self._cache[token]

    def terms(self, text: str) -> list[str]:
        return [t for t in (self.term(tok) for tok in self.tokens(text)) if t]


# ============================================================== 4. Domain dictionary
DICTIONARY_FILE = Path(__file__).with_name("domain_dictionary.json")
MAX_ADDED_TERMS = 20
COMPOUND_MIN_EXTRA = 3  # a compound must be >= 3 characters longer than the member it contains


@dataclass
class Expansion:
    added: list[str] = field(default_factory=list)     # tokens added, in order
    matches: list[dict] = field(default_factory=list)  # {"trigger", "group", "via", "added"}


class DomainDictionary:
    """A query word matching one member of a group adds the group's other members.

    Matching uses the index's stemmer (exact stem match); a German compound also matches a
    member it starts or ends with (Dienstreiseantrag -> dienstreise, antrag), which adds that
    member itself. Multi-word members match as phrases.
    """

    def __init__(self, language: str, pipeline: TermPipeline, path: Path = DICTIONARY_FILE):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if language not in data:
            raise ValueError(f"No dictionary for language {language}")
        self.language, self.pipeline = language, pipeline
        self.groups = {name: [m.lower() for m in members] for name, members in data[language].items()}
        self._member_stem = {m: pipeline.term(m) for ms in self.groups.values() for m in ms
                             if " " not in m and len(pipeline.tokens(m)) == 1}

    def expand(self, base_query: str, max_added: int = MAX_ADDED_TERMS) -> Expansion:
        tokens = self.pipeline.tokens(base_query)
        present = {self.pipeline.term(t) for t in tokens} - {None}
        phrase_text = " " + " ".join(tokens) + " "
        result, added_terms = Expansion(), set()
        for name, members in self.groups.items():
            trigger = self._trigger(members, tokens, present, phrase_text)
            if trigger is None:
                continue
            word, member = trigger
            new = []
            for candidate in members:
                for token in self.pipeline.tokens(candidate):
                    term = self.pipeline.term(token)
                    if term and term not in present and term not in added_terms and len(result.added) < max_added:
                        added_terms.add(term)
                        result.added.append(token)
                        new.append(token)
            if new:
                result.matches.append({"trigger": word, "group": name, "via": member, "added": new})
        return result

    def _trigger(self, members, tokens, present, phrase_text):
        for member in members:
            if " " in member or member not in self._member_stem:
                if " " + " ".join(self.pipeline.tokens(member)) + " " in phrase_text:
                    return member, member
                continue
            stem = self._member_stem[member]
            if stem and stem in present:
                return member, member
        if self.language == "de":
            for token in tokens:
                for member in members:
                    if " " not in member and len(token) >= len(member) + COMPOUND_MIN_EXTRA and len(member) >= 4 \
                            and (token.startswith(member) or token.endswith(member)):
                        return token, member
        return None


def weighted_query(base_tokens: str, added: list[str], weight: float) -> str:
    """Terrier query: base tokens unchanged, added tokens as term^weight."""
    if not 0 < weight <= 1:
        raise ValueError("synonym weight must be in (0, 1]")
    return " ".join([base_tokens, *(f"{t}^{weight:g}" for t in added)]).strip()


# ============================================================== 5. Retrieval
RESULT_COLUMNS = ["qid", "docno", "score", "rank"]
BM25_CONTROLS = {"bm25.b": "0.75", "bm25.k_1": "1.2"}  # Terrier defaults, always explicit (copied per call:
                                                      # the Retriever adds entries to the dict it is given)
FIELD_WEIGHTS = {"text": 1.0, "title": 3.0, "keywords": 2.0}
FIELD_NORMALISATION = {"text": 0.75, "title": 0.5, "keywords": 0.5}
MAX_DOCUMENT_KEYWORDS = 20


def bm25(index, queries: dict[str, str], top_k: int = 1000) -> pd.DataFrame:
    """BM25 (b=0.75, k_1=1.2), ordered by qid then rank."""
    topics = pd.DataFrame([{"qid": q, "query": s} for q, s in queries.items()], columns=["qid", "query"])
    if topics.empty:
        raise ValueError("Dataset has no queries")
    results = pt.terrier.Retriever(index, wmodel="BM25", controls=dict(BM25_CONTROLS), num_results=top_k).transform(topics)
    if results.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = results[RESULT_COLUMNS].copy()
    results["qid"] = results["qid"].astype(str)
    return results.sort_values(["qid", "rank"], kind="mergesort").groupby("qid", sort=False).head(top_k).reset_index(drop=True)


def dph(index, queries: dict[str, str], top_k: int = 1000) -> pd.DataFrame:
    """DPH (parameter-free). Empty queries are skipped, as Terrier does; ties keep Terrier's order."""
    topics = pd.DataFrame([{"qid": q, "query": s} for q, s in queries.items() if s.strip()], columns=["qid", "query"])
    if topics.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = pt.terrier.Retriever(index, wmodel="DPH", controls={}, num_results=top_k, verbose=False).transform(topics)
    if results.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = results[RESULT_COLUMNS].copy()
    results["qid"] = results["qid"].astype(str)
    results = results.sort_values(["qid", "rank"], kind="mergesort").groupby("qid", sort=False).head(top_k).reset_index(drop=True)
    if results.duplicated(["qid", "docno"]).any():
        raise RuntimeError("Duplicate docno within a query ranking")
    if (results.groupby("qid", sort=False)["rank"].diff().dropna() != 1).any():
        raise RuntimeError("Terrier ranks are not consecutive")
    return results


def document_keywords(text: str, language: str, limit: int = MAX_DOCUMENT_KEYWORDS) -> list[str]:
    """Most frequent informative body words, breaking ties by first occurrence."""
    words = tokens(text[:200_000], language)
    counts = Counter(words)
    first = {}
    for i, word in enumerate(words):
        first.setdefault(word, i)
    return sorted(counts, key=lambda word: (-counts[word], first[word]))[:limit]


def explicit_title(document) -> str:
    """Only a stored document title; never one made up from the body."""
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


def build_field_index(dataset, language: str, output: Path, max_keywords: int = MAX_DOCUMENT_KEYWORDS):
    """Official indexing configuration with three fields: body text, stored title, body keywords."""
    index_dir = Path(tempfile.mkdtemp(prefix="uniagent-bm25f-fields_"))
    fields = ["text", "title", "keywords"]
    indexer = pt.IterDictIndexer(str(index_dir.resolve()), overwrite=True, meta={"docno": 100},
                                 text_attrs=fields, fields=True, verbose=True, **RETRIEVAL_CONFIGURATION[language])
    stats = {"documents": 0, "documents_with_titles": 0}

    def docs():
        for document in dataset.docs_iter():
            text = document.default_text() or ""
            title = explicit_title(document)
            stats["documents"] += 1
            stats["documents_with_titles"] += bool(title)
            yield {"docno": str(document.doc_id), "text": text, "title": title,
                   "keywords": " ".join(document_keywords(text, language, max_keywords))}

    with tracking(export_file_path=output / "index-ir-metadata.yml"):
        ref = indexer.index(tqdm(docs(), desc="Index document fields"))
    index = pt.IndexFactory.of(ref)
    actual = [str(f) for f in index.getCollectionStatistics().getFieldNames()]
    if set(actual) != set(fields):
        raise RuntimeError(f"Unexpected BM25F fields: {actual}")
    return index, actual, stats


def bm25f_controls(actual_fields: list[str]) -> dict[str, float]:
    """Field indices come from Terrier, not from an assumed ordering."""
    controls = {}
    for i, name in enumerate(actual_fields):
        controls[f"w.{i}"] = FIELD_WEIGHTS[name]
        controls[f"c.{i}"] = FIELD_NORMALISATION[name]
    return controls


def bm25f(index, queries: dict[str, str], fields: list[str], top_k: int = 1000) -> pd.DataFrame:
    """BM25F over the field index, ordered by qid then rank."""
    topics = pd.DataFrame([{"qid": q, "query": s} for q, s in queries.items()], columns=["qid", "query"])
    if topics.empty:
        raise ValueError("Dataset has no queries")
    results = pt.terrier.Retriever(index, wmodel="BM25F", controls=bm25f_controls(fields), verbose=True).transform(topics)
    if results.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = results.sort_values(["qid", "rank"]).groupby("qid", sort=False).head(top_k)
    out = results[RESULT_COLUMNS].copy()
    out["qid"] = out["qid"].astype(str)
    return out.reset_index(drop=True)


# ============================================================== 6. TF-IDF cosine fusion
CANDIDATE_DEPTH = 100


class TfIdf:
    """TF-IDF vectors over the index's terms, (1 + ln tf) * ln(N / df); df and N from the Terrier index."""

    def __init__(self, index, pipeline: TermPipeline):
        self.pipeline = pipeline
        self._lexicon = index.getLexicon()
        self.n_docs = int(index.getCollectionStatistics().getNumberOfDocuments())
        self._idf: dict[str, float] = {}

    def idf(self, term: str) -> float:
        if term not in self._idf:
            entry = self._lexicon.getLexiconEntry(term)
            df = int(entry.getDocumentFrequency()) if entry is not None else 0
            self._idf[term] = math.log(self.n_docs / df) if df > 0 else 0.0
        return self._idf[term]

    def vector(self, text: str) -> dict[str, float]:
        counts = Counter(self.pipeline.terms(text))
        return {t: (1 + math.log(c)) * self.idf(t) for t, c in counts.items() if self.idf(t) > 0}

    def query_vector(self, base_query: str, added: list[str], weight: float) -> dict[str, float]:
        """Base terms weighted idf, dictionary terms weight * idf (as in the BM25 query)."""
        vector: dict[str, float] = {}
        for term in self.pipeline.terms(base_query):
            vector[term] = max(vector.get(term, 0.0), self.idf(term))
        for token in added:
            term = self.pipeline.term(token)
            if term and term not in vector:
                vector[term] = weight * self.idf(term)
        return {t: v for t, v in vector.items() if v > 0}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(t, 0.0) for t, v in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def document_texts(dataset, docnos: set[str]) -> dict[str, str]:
    texts = {str(d.doc_id): d.default_text() for d in dataset.docs_iter() if str(d.doc_id) in docnos}
    missing = docnos - texts.keys()
    if missing:
        raise ValueError(f"{len(missing)} retrieved documents not found in the dataset, e.g. {sorted(missing)[:3]}")
    return texts


def cosine_scores(results: pd.DataFrame, query_vectors: dict[str, dict], texts: dict[str, str], tfidf: TfIdf,
                  depth: int = CANDIDATE_DEPTH) -> pd.DataFrame:
    """qid, docno, cosine for the top `depth` documents of each query."""
    top = results.groupby("qid", sort=False).head(depth)
    vectors: dict[str, dict] = {}
    rows = []
    for qid, docno in zip(top["qid"], top["docno"]):
        if docno not in vectors:
            vectors[docno] = tfidf.vector(texts[docno])
        rows.append((qid, docno, cosine(query_vectors[qid], vectors[docno])))
    return pd.DataFrame(rows, columns=["qid", "docno", "cosine"])


def _minmax(values: pd.Series) -> pd.Series:
    low, high = values.min(), values.max()
    return (values - low) / (high - low) if high > low else values * 0.0


def fuse(results: pd.DataFrame, cosines: pd.DataFrame, alpha: float, depth: int = CANDIDATE_DEPTH,
         first_stage: str = "bm25") -> pd.DataFrame:
    """alpha * minmax(first stage) + (1 - alpha) * minmax(cosine) in the top `depth`; the rest keeps its order.

    Written scores: fused score + 1 for the top `depth` (in [1, 2]); below it, the first-stage score
    rescaled into [0, 1), so scores decrease with rank in every query. Ties keep the first-stage order.
    The first-stage score is kept in a column named `first_stage` (e.g. bm25 or bm25f).
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if results.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS + [first_stage, "cosine"])
    out = results[["qid", "docno", "score"]].copy()
    out["r_first"] = out.groupby("qid", sort=False).cumcount()
    out = out.merge(cosines, on=["qid", "docno"], how="left")
    head = out["r_first"] < depth
    if out.loc[head, "cosine"].isna().any():
        raise ValueError("Missing cosine score for a top-depth document")
    fused = pd.Series(0.0, index=out.index)
    for _, part in out[head].groupby("qid", sort=False):
        fused[part.index] = 1.0 + alpha * _minmax(part["score"]) + (1 - alpha) * _minmax(part["cosine"])
    for _, part in out[~head].groupby("qid", sort=False):
        fused[part.index] = 0.999 * _minmax(part["score"]) if len(part) > 1 else 0.5
    out["fused"] = fused
    out["_qorder"] = out["qid"].map({q: i for i, q in enumerate(results["qid"].drop_duplicates())})
    out = out.sort_values(["_qorder", "fused", "r_first"], ascending=[True, False, True], kind="mergesort")
    out["rank"] = out.groupby("qid", sort=False).cumcount()
    out = out.rename(columns={"score": first_stage, "fused": "score"})
    return out[RESULT_COLUMNS + [first_stage, "cosine"]].reset_index(drop=True)
