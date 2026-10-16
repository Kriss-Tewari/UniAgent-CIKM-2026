"""Tests for query_expansion.py; the official test_baseline.py remains untouched."""
import gzip
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from event_logging import EventLog
from query_expansion import query_keywords, query_rows, retrieval_settings, terrier_queries, write_trace

REQUIRED_FIELDS = {"case_id", "event_id", "parent_event_id", "timestamp", "event_type",
                   "model", "tool", "input", "output", "status", "error"}
SENT = {"1": "travel allowance hotel reimbursement", "2": "parking"}


def trace_rows():
    return [{"qid": "1", "title": "Travel allowance", "description": "Hotel reimbursement",
             "keywords": ["hotel", "reimbursement"], "expanded": "Travel allowance hotel reimbursement"},
            {"qid": "2", "title": "Parking", "description": "", "keywords": [], "expanded": "Parking"}]


def read_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check_contract(events):
    ids = [e["event_id"] for e in events]
    assert len(ids) == len(set(ids))
    for event in events:
        assert set(event) == REQUIRED_FIELDS
        assert event["model"] == "-"
        assert (event["tool"] is not None) == (event["event_type"] == "tool_call")
        assert (event["status"] == "error") == (event["error"] is not None)
    by_case = {}
    for event in events:
        by_case.setdefault(event["case_id"], []).append(event)
    for case in by_case.values():
        assert case[0]["parent_event_id"] is None
        for previous, event in zip(case, case[1:]):
            assert event["parent_event_id"] == previous["event_id"]
        assert case[-1]["event_type"] == "decision"
    return by_case


def test_expansion_keeps_new_terms_only():
    assert query_keywords("Travel reimbursement deadline deadline", "Travel", "en") == ["reimbursement", "deadline"]


def test_description_only_no_narrative():
    query = SimpleNamespace(
        query_id="q1", default_text=lambda: "Travel allowance",
        original_query={"language": "en", "description": "Hotel reimbursement after travel",
                        "narrative": "SECRET_DO_NOT_READ"},
    )
    ds = SimpleNamespace(queries_iter=lambda: iter([query]))
    rows = query_rows(ds, "en", 12)
    assert rows[0]["expanded"] == "Travel allowance hotel reimbursement after"
    assert "SECRET" not in str(rows)


def test_german_stopwords_from_official_file():
    assert query_keywords("Welche Regeln gelten für die Dienstreise", "Dienstreise", "de") == ["regeln", "gelten"]


def test_terrier_queries_match_official_tokenisation():
    sent = terrier_queries(trace_rows(), "en")
    assert sent["1"].lower().split() == ["travel", "allowance", "hotel", "reimbursement"]
    assert sent["2"].lower() == "parking"


def test_trace_follows_event_logging_contract(tmp_path):
    results = pd.DataFrame({"qid": ["1", "1"], "docno": ["b", "a"], "rank": [1, 0],
                            "score": [1.5, 2.5], "query": [SENT["1"]] * 2})
    write_trace(trace_rows(), SENT, results, retrieval_settings(1000), tmp_path)
    by_case = check_contract(read_trace(tmp_path / "run-trace.jsonl.log.gz"))
    assert [e["event_type"] for e in by_case["1"]] == ["input", "tool_call", "tool_call", "observation", "decision"]
    assert by_case["1"][1]["output"]["description_keywords"] == ["hotel", "reimbursement"]
    assert by_case["1"][2]["input"]["query"] == SENT["1"]
    decision = by_case["1"][-1]["output"]
    assert decision["answer"] == ["a", "b"]
    assert decision["evidence"][0]["source_id"] == "a"
    assert by_case["2"][-1]["output"]["answer"] == []


def test_trace_logs_retrieval_errors(tmp_path):
    write_trace(trace_rows(), SENT, None, retrieval_settings(1000), tmp_path, RuntimeError("index broken"))
    by_case = check_contract(read_trace(tmp_path / "run-trace.jsonl.log.gz"))
    for case in by_case.values():
        assert case[-2]["tool"] == "pyterrier_retrieve" and case[-2]["error"] == "index broken"
        assert case[-1]["status"] == "error"


def test_event_log_rejects_invalid_events(tmp_path):
    with EventLog(tmp_path / "t.gz") as log:
        with pytest.raises(ValueError):
            log.log("1", "tool_call")
        with pytest.raises(ValueError):
            log.log("1", "unknown")
        with pytest.raises(ValueError):
            log.log("1", "decision", status="error")
