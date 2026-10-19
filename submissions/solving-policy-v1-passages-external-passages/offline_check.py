"""Check packaged dependencies without calling any hosted decision model."""
from pathlib import Path
import gzip
import hashlib
import importlib
import json
import math
import subprocess
import tempfile
import uuid

import pyterrier as pt

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / "submission-config.json").read_text())
expected = json.loads((ROOT / "source-checksums.json").read_text())

for name, checksum in expected.items():
    data = (ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == checksum, name
    if name.endswith(".py"):
        compile(data, str(ROOT / name), "exec")

subprocess.run(["pdftotext", "-v"], check=True)

for name in (
    "smolagents", "openai", "case_tools", "event_logging",
    "passage_focus", "external_passages", "external_focus",
    "task1_lexical", "task1_qwen3", "predict",
):
    importlib.import_module(name)

pt.terrier.set_version("5.11")
pt.terrier.set_helper_version("0.0.8")
pt.java.mavenresolver.offline()
pt.java.set_memory_limit(2048)
pt.java.init()
assert str(pt.terrier.version()) == "5.11"

from external_retrieval import ExternalRetriever

with tempfile.TemporaryDirectory(prefix="uniagent-container-check-") as temporary:
    temporary = Path(temporary)
    inputs = temporary / "inputs"

    texts = [
        (
            "Procurement framework agreements",
            "Beschaffung Universität Rahmenvertrag Vergleichsangebote. "
            "University procurement framework agreements and comparison offers.",
        ),
        (
            "Business travel",
            "Dienstreise Reiseantrag Hotel Übernachtung. "
            "Business travel requests and hotel accommodation.",
        ),
        (
            "Expense reimbursement",
            "Auslagenerstattung Rechnung Zahlungsbeleg Erstattung. "
            "Expense reimbursement receipts and payment evidence.",
        ),
        (
            "Library opening hours",
            "Bibliothek Öffnungszeiten Bücher Ausleihe. "
            "Library books opening hours and borrowing.",
        ),
    ]

    for corpus in (
        "hessian-law-de",
        "university-kassel-public-de",
        "university-kassel-public-en",
    ):
        folder = inputs / "retrieval-corpora" / corpus
        folder.mkdir(parents=True)

        with gzip.open(
            folder / "documents.jsonl.gz", "wt", encoding="utf-8"
        ) as stream:
            for number, (title, content) in enumerate(texts):
                doc_id = (
                    str(uuid.uuid5(uuid.NAMESPACE_URL, f"{corpus}/{number}"))
                    if corpus.startswith("university")
                    else f"TEST-{number}"
                )
                stream.write(json.dumps({
                    "doc_id": doc_id,
                    "title": title,
                    "content": content,
                    "language": "en" if corpus.endswith("-en") else "de",
                }, ensure_ascii=False) + "\n")

    retriever = ExternalRetriever(inputs, temporary / "indices")
    result = retriever.retrieve(
        {
            "de": {"probe": "Beschaffung Rahmenvertrag Vergleichsangebote"},
            "en": {"probe": "university procurement framework agreements"},
        },
        document_limit=12,
        per_query=3,
    )
    assert result["documents"], "Synthetic retrieval returned no documents."
    assert {item["corpus"] for item in result["documents"]} == {
        "hessian-law-de",
        "university-kassel-public-de",
        "university-kassel-public-en",
    }
    print("PASS: Offline indexing and retrieval across all three routes")

if config["external_mode"] == "qwen3":
    import torch
    from task1_qwen3 import QwenScorer

    assert torch.__version__.split("+")[0] == "2.13.0"
    assert torch.version.cuda is None, "Expected the CPU-only PyTorch build."

    scorer = QwenScorer(
        str(ROOT / "models/qwen3-reranker"),
        device="cpu",
        max_length=1536,
        batch_size=1,
        threads=4,
    )
    scores, usage = scorer.score(
        "Muss bei einer Beschaffung ein bestehender Rahmenvertrag genutzt werden?",
        [
            "Bei Beschaffungen ist ein bestehender verbindlicher Rahmenvertrag zu nutzen.",
            "Die Universitätsbibliothek ist montags bis freitags geöffnet.",
        ],
    )
    assert len(scores) == 2
    assert all(math.isfinite(score) for score in scores)
    assert scores[0] > scores[1], scores

    provenance = json.loads(
        (ROOT / "models/qwen3-reranker/download_source.json").read_text()
    )
    assert provenance["resolved_commit"] == config["reranker_revision"]
    print("PASS: Offline CPU reranking; scores:", scores)

print("PASS: Source checksums, imports, PDF utility, and Java")
print("OFFLINE CONTAINER CHECK PASSED:", config["variant"])
