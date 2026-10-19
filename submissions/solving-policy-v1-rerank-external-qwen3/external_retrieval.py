"""Task 1 lexical retrieval adapted to Task 2's supplied corpora."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import pyterrier as pt
import task1_lexical as t1

UUID = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.I)


def init_java():
    if pt.java.started():
        return
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        jvm = Path(prefix) / "lib/jvm/lib/server/libjvm.so"
        configured = Path(os.environ.get("JVM_PATH", "/missing"))
        if jvm.is_file() and not configured.is_file():
            os.environ["JVM_PATH"] = str(jvm)
            os.environ["JAVA_HOME"] = str(jvm.parents[2])
    pt.java.init()


class Corpus:
    def __init__(self, path, cache):
        self.name = path.parent.name
        self.documents = {}
        content_hash = hashlib.sha256()
        self.empty_documents = 0
        sample_language = None

        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                content_hash.update(line.encode("utf-8"))
                if not line.strip():
                    continue
                record = json.loads(line)
                doc_id = str(record["doc_id"])
                text = record.get("content") or record.get("text") or ""
                if not isinstance(text, str):
                    raise TypeError(f"Non-string text: {self.name}/{doc_id}")
                if not text.strip():
                    self.empty_documents += 1
                    continue
                if doc_id in self.documents:
                    raise ValueError(f"Duplicate ID: {self.name}/{doc_id}")

                metadata = record.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                self.documents[doc_id] = {
                    "corpus": self.name,
                    "doc_id": doc_id,
                    "text": text,
                    "title": (
                        record.get("title") or record.get("page_title")
                        or metadata.get("title") or ""
                    ),
                    "url": record.get("url") or metadata.get("url"),
                }
                sample_language = sample_language or record.get("language")

        if not self.documents:
            raise ValueError(f"No nonempty documents: {path}")

        self.language = (
            "de" if self.name.endswith("-de") else
            "en" if self.name.endswith("-en") else sample_language
        )
        if self.language not in {"de", "en"}:
            raise ValueError(f"Unsupported language: {self.name}")

        sample = list(self.documents)[:200]
        uuid_share = sum(bool(UUID.fullmatch(x)) for x in sample) / len(sample)
        self.route = (
            "university-en" if self.language == "en" else
            "university-de" if uuid_share > .5 else "hessian-law-de"
        )

        fingerprint = hashlib.sha256()
        fingerprint.update(content_hash.digest())
        fingerprint.update((
            "field-index-v1:" + self.route + ":"
            + str(getattr(pt, "__version__", "unknown"))
        ).encode())
        for name in ("task1_lexical.py", "german-stopwords.txt"):
            fingerprint.update(Path(__file__).with_name(name).read_bytes())

        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / (
            self.name + "-" + fingerprint.hexdigest()[:24]
        )

        with (cache / (destination.name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not destination.exists():
                print(f"BUILDING INDEX: {self.name}", flush=True)
                with tempfile.TemporaryDirectory(
                    prefix="building-", dir=cache
                ) as temporary:
                    temporary = Path(temporary)
                    options = dict(
                        overwrite=True,
                        verbose=False,
                        meta={
                            "docno": max(
                                100, max(map(len, self.documents)) + 1
                            )
                        },
                        **t1.RETRIEVAL_CONFIGURATION[self.language],
                    )
                    if self.route == "hessian-law-de":
                        options.update(
                            text_attrs=["text", "title", "keywords"],
                            fields=True,
                        )
                    indexer = pt.IterDictIndexer(
                        str(temporary / "index"), **options
                    )
                    reference = indexer.index(self._index_rows())
                    built = pt.IndexFactory.of(reference)
                    built.close()
                    (temporary / "manifest.json").write_text(json.dumps({
                        "corpus": self.name,
                        "route": self.route,
                        "documents": len(self.documents),
                        "empty_documents_skipped": self.empty_documents,
                    }, indent=2))
                    os.rename(temporary, destination)
            else:
                print(f"REUSING INDEX: {self.name}", flush=True)

        properties = destination / "index/data.properties"
        if not properties.is_file():
            raise RuntimeError(f"Incomplete index cache: {destination}")

        self.index = pt.IndexFactory.of(str(properties))
        self.fields = [
            str(x)
            for x in self.index.getCollectionStatistics().getFieldNames()
        ]
        self.pipeline = t1.TermPipeline(self.language)
        self.dictionary = (
            t1.DomainDictionary(self.language, self.pipeline)
            if self.route == "university-de" else None
        )
        self.tfidf = t1.TfIdf(self.index, self.pipeline)
        self.tokeniser = pt.TerrierTokeniser.java_tokeniser(
            t1.RETRIEVAL_CONFIGURATION[self.language]["tokeniser"]
        )
        print(
            f"READY: {self.name} | {self.route} | "
            f"{len(self.documents)} documents | "
            f"empty skipped={self.empty_documents}",
            flush=True,
        )

    def _index_rows(self):
        for document in self.documents.values():
            row = {
                "docno": document["doc_id"],
                "text": document["text"],
            }
            if self.route == "hessian-law-de":
                row.update(
                    title=str(document["title"]),
                    keywords=" ".join(t1.document_keywords(
                        document["text"], self.language
                    )),
                )
            yield row

    def search(self, queries, per_query=5):
        raw = {
            str(key): str(text)
            for key, text in queries.items() if str(text).strip()
        }
        sent = {
            key: " ".join(self.tokeniser.getTokens(text))
            for key, text in raw.items()
        }
        sent = {key: text for key, text in sent.items() if text.strip()}
        if not sent:
            return {}

        expansions = {}
        if self.route == "university-de":
            expansions = {
                key: self.dictionary.expand(raw[key]) for key in sent
            }
            sent = {
                key: t1.weighted_query(text, expansions[key].added, .3)
                for key, text in sent.items()
            }
            frame = t1.bm25(self.index, sent, top_k=100)
        elif self.route == "university-en":
            frame = t1.dph(self.index, sent, top_k=100)
        else:
            frame = t1.bm25f(
                self.index, sent, self.fields, top_k=100
            )

        if not frame.empty and self.route != "university-en":
            vectors = {
                key: (
                    self.tfidf.query_vector(
                        raw[key], expansions[key].added, .3
                    )
                    if key in expansions
                    else self.tfidf.vector(raw[key])
                )
                for key in sent
            }
            texts = {
                str(docno): self.documents[str(docno)]["text"]
                for docno in frame["docno"].unique()
            }
            cosines = t1.cosine_scores(
                frame, vectors, texts, self.tfidf, depth=100
            )
            frame = t1.fuse(
                frame, cosines, .7, depth=100,
                first_stage=(
                    "bm25f" if self.route == "hessian-law-de" else "bm25"
                ),
            )

        groups = {}
        for key, group in frame.groupby("qid", sort=False):
            groups[str(key)] = [{
                "corpus": self.name,
                "doc_id": str(row.docno),
                "policy_id": str(key),
                "query": raw[str(key)],
                "rank": int(row.rank),
                "score": float(row.score),
                "route": self.route,
            } for row in (
                group.sort_values("rank").head(per_query).itertuples()
            )]
        return groups


class ExternalRetriever:
    def __init__(self, input_root, cache_root=None):
        paths = sorted(
            (Path(input_root) / "retrieval-corpora")
            .glob("*/documents.jsonl.gz")
        )
        if not paths:
            raise FileNotFoundError(
                f"No supplied retrieval corpora below {input_root}"
            )
        init_java()
        cache = (
            cache_root
            or os.environ.get("UNIAGENT_RETRIEVAL_CACHE")
            or Path(tempfile.gettempdir()) / "uniagent-external-indices"
        )
        self.corpora = [Corpus(path, cache) for path in paths]

    def retrieve(
        self, queries_by_language, document_limit=24, per_query=5
    ):
        if not 1 <= document_limit <= 100 or not 1 <= per_query <= 100:
            raise ValueError("Candidate limits must be in [1, 100].")

        queues, audit = [], []
        by_corpus = {corpus.name: corpus for corpus in self.corpora}

        for corpus in self.corpora:
            groups = corpus.search(
                queries_by_language.get(corpus.language, {}), per_query
            )
            queues.extend(groups.values())
            audit.extend(
                hit for group in groups.values() for hit in group
            )

        selected, seen = [], set()
        for depth in range(per_query):
            for queue in queues:
                if depth >= len(queue):
                    continue
                hit = queue[depth]
                identity = hit["corpus"], hit["doc_id"]
                if identity in seen:
                    continue
                seen.add(identity)
                document = by_corpus[hit["corpus"]].documents[hit["doc_id"]]
                selected.append(dict(document))
                if len(selected) == document_limit:
                    break
            if len(selected) == document_limit:
                break

        return {
            "documents": selected,
            "retrieval_hits": audit,
            "queries": queries_by_language,
            "document_limit": document_limit,
            "per_query": per_query,
        }
