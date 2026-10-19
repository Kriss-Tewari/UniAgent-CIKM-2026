"""Tests for submission.py and components.py; the official test_baseline.py stays untouched."""
import ast
import gzip
import json
import math
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import pyterrier as pt
import pytest
from click.testing import CliRunner
from tira.third_party_integrations import ir_datasets

import components as c
import submission as s
from baseline import RETRIEVAL_CONFIGURATION

HERE = Path(__file__).parent
REQUIRED_FIELDS = {"case_id", "event_id", "parent_event_id", "timestamp", "event_type",
                   "model", "tool", "input", "output", "status", "error"}
TOOLS = {"university-de": ["route_collection", "formulate_query", "expand_with_dictionary", "pyterrier_retrieve", "cosine_rerank"],
         "university-en": ["route_collection", "formulate_query", "pyterrier_retrieve"],
         "hessian-law-de": ["route_collection", "formulate_query", "pyterrier_retrieve", "cosine_rerank"]}
CONTENT = {
    "de": ("Reisekosten", "Wie werden Kosten einer Dienstreise erstattet?",
           ["Reisekosten einer Dienstreise werden erstattet.", "Bibliothek und Ausleihe von Büchern.",
            "Die Reiseabrechnung reichen Sie mit Belegen ein.", "Übernachtung im Hotel wird erstattet."]),
    "en": ("Travel expenses", "How are business trip costs reimbursed?",
           ["Travel expenses of a business trip are reimbursed.", "Library loans and books.",
            "Submit the expense report with receipts.", "Hotel accommodation is refunded."]),
}


# ---------------------------------------------------------------- helpers
def write_dataset(directory: Path, language: str, id_style: str = "uuid", languages=None) -> Path:
    title, description, texts = CONTENT[language]
    ids = ([str(uuid.UUID(int=i + 1)) for i in range(len(texts))] if id_style == "uuid"
           else [f"LARE19000000{i}" for i in range(len(texts))])
    queries = [{"qid": str(i + 1), "query": title, "original_query": {
        "language": lang, "description": description, "narrative": "NARRATIVE-MUST-NOT-BE-USED"}}
        for i, lang in enumerate(languages or [language, language])]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8")
    (directory / "documents.jsonl").write_text(
        "".join(json.dumps({"doc_id": d, "text": t}) + "\n" for d, t in zip(ids, texts)), encoding="utf-8")
    return directory


def read_trace(path: Path) -> list[dict]:
    with gzip.open(path / c.TRACE_FILE, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check_contract(events: list[dict]) -> dict:
    """Required fields, model '-', one chain per query from input via route_collection to decision."""
    assert len({e["event_id"] for e in events}) == len(events)
    by_case = {}
    for event in events:
        assert set(event) == REQUIRED_FIELDS and event["model"] == "-"
        assert (event["tool"] is not None) == (event["event_type"] == "tool_call")
        assert (event["status"] == "error") == (event["error"] is not None)
        by_case.setdefault(event["case_id"], []).append(event)
    for case in by_case.values():
        assert case[0]["event_type"] == "input" and case[0]["parent_event_id"] is None
        assert case[1]["tool"] == "route_collection"
        for previous, event in zip(case, case[1:]):
            assert event["parent_event_id"] == previous["event_id"]
        assert case[-1]["event_type"] == "decision"
    return by_case


def toy_index(docs, language="de"):
    ref = pt.IterDictIndexer(tempfile.mkdtemp(), meta={"docno": 20}, **RETRIEVAL_CONFIGURATION[language]).index(docs)
    return pt.IndexFactory.of(ref)


# ---------------------------------------------------------------- routing
@pytest.mark.parametrize("language,id_style,expected", [
    ("en", "uuid", "university-en"), ("de", "uuid", "university-de"), ("de", "lare", "hessian-law-de")])
def test_collection_is_recognised_from_language_and_document_ids(tmp_path, language, id_style, expected):
    route = s.recognise_collection(ir_datasets.load(str(write_dataset(tmp_path / "d", language, id_style))))
    assert route["collection"] == expected and route["uuid_share"] == (1.0 if id_style == "uuid" else 0.0)


def test_routing_error_is_logged_before_raising(tmp_path):
    data = write_dataset(tmp_path / "data", "de", languages=["de", "en"])
    result = CliRunner().invoke(s.main, ["--dataset", str(data), "--output", str(tmp_path / "o")])
    assert result.exit_code != 0 and "one language" in result.output
    for case in check_contract(read_trace(tmp_path / "o")).values():
        assert [e["status"] for e in case] == ["ok", "error", "error"]
    assert not (tmp_path / "o" / "run.txt.gz").exists()


# ---------------------------------------------------------------- end to end, one per collection
@pytest.mark.parametrize("language,id_style,collection", [
    ("en", "uuid", "university-en"), ("de", "uuid", "university-de"), ("de", "lare", "hessian-law-de")])
def test_end_to_end_run_metadata_and_event_log(tmp_path, language, id_style, collection):
    out = tmp_path / "out"
    result = CliRunner().invoke(s.main, ["--dataset", str(write_dataset(tmp_path / "data", language, id_style)),
                                         "--output", str(out)])
    assert result.exit_code == 0, result.output
    rows = [line.split() for line in gzip.open(out / "run.txt.gz", "rt").read().splitlines()]
    assert rows and all(len(r) == 6 for r in rows)
    for qid in {r[0] for r in rows}:
        hits = [r for r in rows if r[0] == qid]
        assert [int(r[3]) for r in hits] == list(range(len(hits)))
        assert all(float(a[4]) >= float(b[4]) for a, b in zip(hits, hits[1:]))
        assert len({r[2] for r in hits}) == len(hits)
    events = read_trace(out)
    for case in check_contract(events).values():
        assert [e["tool"] for e in case if e["event_type"] == "tool_call"] == TOOLS[collection]
        assert case[1]["output"]["collection"] == collection
        assert [e["event_type"] for e in case[-2:]] == ["observation", "decision"]
    assert "NARRATIVE" not in json.dumps(events)
    assert json.loads((out / "routing.json").read_text())["collection"] == collection
    config = json.loads((out / "retrieval_config.json").read_text())
    assert config["collection"] == collection and config["uses_relevance_judgments"] is False
    for name in ("language.txt", "index-ir-metadata.yml", "retrieval-ir-metadata.yml"):
        assert (out / name).is_file()


@pytest.mark.parametrize("language,id_style,target,tool", [
    ("de", "uuid", "bm25", "pyterrier_retrieve"), ("de", "uuid", "cosine_scores", "cosine_rerank"),
    ("en", "uuid", "dph", "pyterrier_retrieve"),
    ("de", "lare", "bm25f", "pyterrier_retrieve"), ("de", "lare", "cosine_scores", "cosine_rerank")])
def test_errors_are_logged_at_the_failing_step(tmp_path, monkeypatch, language, id_style, target, tool):
    def boom(*args, **kwargs):
        raise RuntimeError("failed")
    monkeypatch.setattr(s, target, boom)
    out = tmp_path / "o"
    result = CliRunner().invoke(s.main, ["--dataset", str(write_dataset(tmp_path / "data", language, id_style)),
                                         "--output", str(out)])
    assert result.exit_code != 0 and isinstance(result.exception, RuntimeError)
    for case in check_contract(read_trace(out)).values():
        assert [e["tool"] for e in case if e["status"] == "error"] == [tool, None]
    assert not (out / "run.txt.gz").exists()


# ---------------------------------------------------------------- query formulation
def test_description_keywords_skip_title_words_stopwords_and_duplicates():
    assert c.query_keywords("Travel reimbursement deadline deadline", "Travel", "en") == ["reimbursement", "deadline"]
    assert len(c.query_keywords(" ".join(f"word{i}abc" for i in range(30)), "x", "en")) == 12


def test_query_rows_never_read_the_narrative():
    class Query:
        query_id, original_query = 1, {"description": "hotel costs", "narrative": "SECRET"}
        def default_text(self):
            return "Travel"
    class Dataset:
        def queries_iter(self):
            return iter([Query()])
    row = c.query_rows(Dataset(), "en")[0]
    assert row["expanded"] == "Travel hotel costs" and "SECRET" not in json.dumps(row)


# ---------------------------------------------------------------- dictionary
@pytest.fixture(scope="module")
def german():
    pipeline = c.TermPipeline("de")
    return pipeline, c.DomainDictionary("de", pipeline)


def test_dictionary_stem_compound_and_phrase_matching(german):
    pipeline, dictionary = german
    expansion = dictionary.expand("Reisekosten Erstattung")
    assert "reiseabrechnung" in expansion.added and "rückerstattung" in expansion.added
    assert not {pipeline.term("reisekosten"), pipeline.term("erstattung")} & {pipeline.term(t) for t in expansion.added}
    assert {"dienstreise", "antrag"} <= set(dictionary.expand("Dienstreiseantrag").added)  # compound
    assert dictionary.expand("Preise Kultur").added == []  # 'preise' is no compound of 'reise'
    english = c.DomainDictionary("en", c.TermPipeline("en"))
    assert {"subsistence", "daily", "allowance"} <= set(english.expand("per diem rates").added)  # phrase
    assert len(dictionary.expand("reisekosten dienstreise beschaffung vergabe angebot rechnung erstattung auslagen hotel flug bahn").added) == c.MAX_ADDED_TERMS


@pytest.mark.parametrize("language", ["de", "en"])
def test_every_dictionary_word_is_indexable(language):
    """Terrier drops words longer than 20 characters, so such entries could never match."""
    pipeline = c.TermPipeline(language)
    for members in json.loads(c.DICTIONARY_FILE.read_text(encoding="utf-8"))[language].values():
        for member in members:
            assert len(pipeline.tokens(member)) == len(member.replace("-", " ").split()), member


def test_weighted_synonyms_find_new_documents_without_lowering_the_originals():
    index = toy_index([{"docno": "orig", "text": "Reisekosten werden erstattet"},
                       {"docno": "syn", "text": "Übernachtung im Hotel"}, {"docno": "other", "text": "Bibliothek"}])
    plain = c.bm25(index, {"1": "reisekosten"})
    expanded = c.bm25(index, {"1": c.weighted_query("reisekosten", ["übernachtung", "hotel"], 0.3)})
    assert list(plain["docno"]) == ["orig"] and list(expanded["docno"]) == ["orig", "syn"]
    assert expanded.iloc[0]["score"] == pytest.approx(plain.iloc[0]["score"])
    assert c.BM25_CONTROLS == {"bm25.b": "0.75", "bm25.k_1": "1.2"}  # not mutated by the Retriever
    with pytest.raises(ValueError):
        c.weighted_query("a", ["b"], 1.5)


# ---------------------------------------------------------------- retrieval models
def test_dph_and_bm25f_rank_the_matching_document_first(tmp_path):
    index = toy_index([{"docno": "a", "text": "Reisekosten werden erstattet"}, {"docno": "b", "text": "Bibliothek"}])
    assert list(c.dph(index, {"1": "reisekosten", "2": "   "})["docno"]) == ["a"]  # empty query skipped
    ds = ir_datasets.load(str(write_dataset(tmp_path / "d", "de", "lare")))
    field_index, fields, stats = c.build_field_index(ds, "de", tmp_path)
    assert set(fields) == {"text", "title", "keywords"} and stats["documents"] == 4
    assert c.bm25f(field_index, {"1": "reisekosten"}, fields).iloc[0]["docno"] == "LARE190000000"


# ---------------------------------------------------------------- TF-IDF cosine fusion
def test_tfidf_uses_index_statistics():
    pipeline = c.TermPipeline("de")
    tfidf = c.TfIdf(toy_index([{"docno": "a", "text": "Reisekosten Hotel"}, {"docno": "b", "text": "Reisekosten Bahn"},
                               {"docno": "c", "text": "Bibliothek"}]), pipeline)
    hotel, kosten = pipeline.terms("Hotel")[0], pipeline.terms("Reisekosten")[0]
    assert tfidf.idf(kosten) == pytest.approx(math.log(3 / 2)) and tfidf.idf("doesnotexist") == 0.0
    assert tfidf.vector("Hotel Hotel")[hotel] == pytest.approx((1 + math.log(2)) * math.log(3))
    assert tfidf.query_vector("Reisekosten", ["hotel"], 0.3)[hotel] == pytest.approx(0.3 * math.log(3))
    assert c.cosine({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 4.0}) == pytest.approx(1.0) and c.cosine({"a": 1.0}, {"b": 1.0}) == 0


def test_fusion_reorders_the_top_depth_only_and_keeps_scores_consistent():
    results = pd.DataFrame({"qid": "1", "docno": ["a", "b", "c", "d"], "score": [4.0, 3.0, 2.0, 1.0], "rank": range(4)})
    cos = pd.DataFrame({"qid": "1", "docno": ["a", "b", "c"], "cosine": [0.0, 0.1, 0.9]})
    assert list(c.fuse(results, cos, 1.0, depth=3)["docno"]) == ["a", "b", "c", "d"]
    fused = c.fuse(results, cos, 0.0, depth=3, first_stage="bm25f")
    assert list(fused["docno"]) == ["c", "b", "a", "d"] and "bm25f" in fused
    assert fused["score"].is_monotonic_decreasing and list(fused["rank"]) == [0, 1, 2, 3]
    assert c.fuse(results, cos, 0.7, depth=3).iloc[0]["score"] == pytest.approx(1 + 0.7)
    with pytest.raises(ValueError):
        c.fuse(results, cos[cos["docno"] != "b"], 0.5, depth=3)


# ---------------------------------------------------------------- inference code reads no judgments
@pytest.mark.parametrize("name", ["submission.py", "components.py"])
def test_code_reads_no_judgments_narrative_or_network(name):
    tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    names = {n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    strings = {n.value.lower() for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings}
    modules = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
              {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {x for x in names if "qrels" in x or "narrative" in x}
    assert not {x for x in strings if "qrels" in x or x == "narrative"}
    assert not modules & {"requests", "urllib", "socket", "http", "httpx"}
