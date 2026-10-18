import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock, patch

from click.testing import CliRunner
from tira.third_party_integrations import ir_datasets

from baseline import create_index, detect_query_language, main, retrieve


@contextmanager
def persisted_dataset(
    queries: list[dict],
    documents: list[dict],
) -> Iterator:
    with tempfile.TemporaryDirectory() as temporary_directory:
        dataset_directory = Path(temporary_directory)
        (dataset_directory / "queries.jsonl").write_text(
            "".join(json.dumps(query) + "\n" for query in queries),
            encoding="utf-8",
        )
        (dataset_directory / "documents.jsonl").write_text(
            "".join(json.dumps(document) + "\n" for document in documents),
            encoding="utf-8",
        )
        yield ir_datasets.load(str(dataset_directory))


def dataset_with_languages(*languages: str) -> Mock:
    queries = []
    for index, language in enumerate(languages, start=1):
        query = Mock()
        query.query_id = str(index)
        query.original_query = {"language": language}
        queries.append(query)

    dataset = Mock()
    dataset.queries_iter.return_value = iter(queries)
    return dataset


def dataset_with_documents(*texts: str) -> Mock:
    documents = []
    for index, text in enumerate(texts, start=1):
        document = Mock()
        document.doc_id = str(index)
        document.default_text.return_value = text
        documents.append(document)

    dataset = Mock()
    dataset.docs_iter.return_value = iter(documents)
    return dataset


class DetectQueryLanguageTest(unittest.TestCase):
    def test_detects_english_dataset(self) -> None:
        queries = [
            {
                "qid": "1",
                "query": "Travel reimbursement deadline",
                "original_query": {"language": "en"},
            },
            {
                "qid": "2",
                "query": "Required conference documents",
                "original_query": {"language": "en"},
            },
        ]
        documents = [{"doc_id": "1", "text": "A document"}]

        with persisted_dataset(queries, documents) as dataset:
            self.assertEqual(detect_query_language(dataset), "en")

    def test_detects_german_dataset(self) -> None:
        queries = [
            {
                "qid": "1",
                "query": "Frist für Reisekostenerstattung",
                "original_query": {"language": "de"},
            },
            {
                "qid": "2",
                "query": "Erforderliche Konferenzunterlagen",
                "original_query": {"language": "de"},
            },
        ]
        documents = [{"doc_id": "1", "text": "Ein Dokument"}]

        with persisted_dataset(queries, documents) as dataset:
            self.assertEqual(detect_query_language(dataset), "de")

    def test_rejects_mixed_query_languages(self) -> None:
        with self.assertRaisesRegex(ValueError, "found: de, en"):
            detect_query_language(dataset_with_languages("de", "en"))

    def test_rejects_missing_original_query(self) -> None:
        dataset = dataset_with_languages("en")
        query = next(dataset.queries_iter())
        query.original_query = None
        dataset.queries_iter.return_value = iter([query])

        with self.assertRaisesRegex(ValueError, "original_query dictionary"):
            detect_query_language(dataset)

    def test_rejects_missing_language(self) -> None:
        dataset = dataset_with_languages("en")
        query = next(dataset.queries_iter())
        query.original_query = {}
        dataset.queries_iter.return_value = iter([query])

        with self.assertRaisesRegex(ValueError, "valid language"):
            detect_query_language(dataset)


class CreateIndexTest(unittest.TestCase):
    def test_creates_german_index_with_stopwords_and_stemming(self) -> None:
        dataset = dataset_with_documents("Häuser und laufen schnell")

        index = create_index(dataset, "de")
        terms = {entry.getKey() for entry in index.getLexicon()}

        self.assertEqual({"haus", "lauf", "schnell"}, terms)

    def test_creates_english_index_with_stopwords_and_stemming(self) -> None:
        dataset = dataset_with_documents("Houses and running quickly")

        index = create_index(dataset, "en")
        terms = {entry.getKey() for entry in index.getLexicon()}

        self.assertNotIn("and", terms)
        self.assertIn("run", terms)

    def test_rejects_unsupported_language(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported language: fr"):
            create_index(dataset_with_documents("Du texte"), "fr")


class RetrieveTest(unittest.TestCase):
    def test_retrieves_documents_for_all_queries_de(self) -> None:
        queries = [
            {
                "qid": "houses",
                "query": "laufende Häuser",
                "original_query": {"language": "de"},
            },
            {
                "qid": "travel",
                "query": "Reisekosten Erstattung",
                "original_query": {"language": "de"},
            },
        ]
        documents = [
            {"doc_id": "house-document", "text": "Die Häuser laufen schnell."},
            {
                "doc_id": "travel-document",
                "text": "Die Erstattung der Reisekosten erfolgt monatlich.",
            },
            {"doc_id": "unrelated-document", "text": "Allgemeine Universität."},
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "de")
            run = retrieve(dataset, index, "de")

        top_documents = {
            qid: group.sort_values("rank").iloc[0]["docno"]
            for qid, group in run.groupby("qid")
        }
        self.assertEqual(
            {
                "houses": "house-document",
                "travel": "travel-document",
            },
            top_documents,
        )
        self.assertTrue({"qid", "docno", "rank", "score"}.issubset(run.columns))

    def test_retrieves_documents_for_all_queries_de_with_special_characters(
        self,
    ) -> None:
        queries = [
            {
                "qid": "houses",
                "query": "wo kann ich ein haus kaufen?",
                "original_query": {"language": "de"},
            },
            {
                "qid": "travel",
                "query": "Wie erstatte ich reisekosten?!",
                "original_query": {"language": "de"},
            },
        ]
        documents = [
            {"doc_id": "house-document", "text": "Die Häuser laufen schnell."},
            {
                "doc_id": "travel-document",
                "text": "Die Erstattung der Reisekosten erfolgt monatlich.",
            },
            {"doc_id": "unrelated-document", "text": "Allgemeine Universität."},
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "de")
            run = retrieve(dataset, index, "de")

        top_documents = {
            qid: group.sort_values("rank").iloc[0]["docno"]
            for qid, group in run.groupby("qid")
        }
        self.assertEqual(
            {
                "houses": "house-document",
                "travel": "travel-document",
            },
            top_documents,
        )
        self.assertTrue({"qid", "docno", "rank", "score"}.issubset(run.columns))

    def test_retrieves_german_inflections(self) -> None:
        queries = [
            {
                "qid": "travel",
                "query": "Dienstreisen Unterlagen",
                "original_query": {"language": "de"},
            },
        ]
        documents = [
            {
                "doc_id": "relevant-document",
                "text": "Für eine Dienstreise ist diese Unterlage erforderlich.",
            },
            {
                "doc_id": "unrelated-document",
                "text": "Informationen zur Einschreibung und zu Prüfungen.",
            },
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "de")
            run = retrieve(dataset, index, "de")

        self.assertEqual("relevant-document", run.sort_values("rank").iloc[0]["docno"])

    def test_retrieves_german_umlauts_and_sharp_s(self) -> None:
        queries = [
            {
                "qid": "garden",
                "query": "größere Gärten",
                "original_query": {"language": "de"},
            },
        ]
        documents = [
            {
                "doc_id": "garden-document",
                "text": "Ein großer Garten mit vielen Pflanzen.",
            },
            {
                "doc_id": "unrelated-document",
                "text": "Ein modernes Labor mit neuen Geräten.",
            },
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "de")
            run = retrieve(dataset, index, "de")

        self.assertEqual("garden-document", run.sort_values("rank").iloc[0]["docno"])

    def test_ranks_document_matching_more_german_query_terms_first(self) -> None:
        queries = [
            {
                "qid": "travel",
                "query": "Erstattung Reisekosten",
                "original_query": {"language": "de"},
            },
        ]
        documents = [
            {
                "doc_id": "complete-document",
                "text": "Die Erstattung der Reisekosten erfolgt monatlich.",
            },
            {
                "doc_id": "partial-document",
                "text": "Die Erstattung erfolgt monatlich.",
            },
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "de")
            run = retrieve(dataset, index, "de")

        self.assertEqual("complete-document", run.sort_values("rank").iloc[0]["docno"])

    def test_german_stopword_only_query_returns_no_results(self) -> None:
        queries = [
            {
                "qid": "stopwords",
                "query": "wie und wo",
                "original_query": {"language": "de"},
            },
        ]
        documents = [
            {"doc_id": "document", "text": "Ein Dokument über Reisekosten."},
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "de")
            run = retrieve(dataset, index, "de")

        self.assertTrue(run.empty)

    def test_retrieves_english_inflections(self) -> None:
        queries = [
            {
                "qid": "houses",
                "query": "running houses",
                "original_query": {"language": "en"},
            },
        ]
        documents = [
            {
                "doc_id": "relevant-document",
                "text": "The house runs efficiently.",
            },
            {
                "doc_id": "unrelated-document",
                "text": "Information about conference registration.",
            },
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "en")
            run = retrieve(dataset, index, "en")

        self.assertEqual("relevant-document", run.sort_values("rank").iloc[0]["docno"])

    def test_retrieves_english_case_and_special_characters(self) -> None:
        queries = [
            {
                "qid": "house",
                "query": "WHERE can I BUY a HOUSE?!",
                "original_query": {"language": "en"},
            },
        ]
        documents = [
            {
                "doc_id": "house-document",
                "text": "You can buy a house through the university.",
            },
            {
                "doc_id": "unrelated-document",
                "text": "Laboratory safety instructions.",
            },
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "en")
            run = retrieve(dataset, index, "en")

        self.assertEqual("house-document", run.sort_values("rank").iloc[0]["docno"])

    def test_ranks_document_matching_more_english_query_terms_first(self) -> None:
        queries = [
            {
                "qid": "travel",
                "query": "travel reimbursement",
                "original_query": {"language": "en"},
            },
        ]
        documents = [
            {
                "doc_id": "complete-document",
                "text": "Travel reimbursement is processed monthly.",
            },
            {
                "doc_id": "partial-document",
                "text": "Reimbursement is processed monthly.",
            },
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "en")
            run = retrieve(dataset, index, "en")

        self.assertEqual("complete-document", run.sort_values("rank").iloc[0]["docno"])

    def test_english_stopword_only_query_returns_no_results(self) -> None:
        queries = [
            {
                "qid": "stopwords",
                "query": "the and",
                "original_query": {"language": "en"},
            },
        ]
        documents = [
            {"doc_id": "document", "text": "A document about travel expenses."},
        ]

        with persisted_dataset(queries, documents) as dataset:
            index = create_index(dataset, "en")
            run = retrieve(dataset, index, "en")

        self.assertTrue(run.empty)

    def test_rejects_dataset_without_queries(self) -> None:
        dataset = Mock()
        dataset.queries_iter.return_value = iter([])

        with self.assertRaisesRegex(ValueError, "dataset has no queries"):
            retrieve(dataset, Mock(), "en")


class MainTest(unittest.TestCase):
    @patch("baseline.retrieve")
    @patch("baseline.create_index")
    @patch("baseline.ir_datasets.load")
    def test_loads_dataset_and_writes_detected_language(
        self,
        load: Mock,
        create_index_mock: Mock,
        retrieve_mock: Mock,
    ) -> None:
        dataset = dataset_with_languages("en", "en")
        index = object()
        load.return_value = dataset
        create_index_mock.return_value = index
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(
                main,
                ["--dataset", "local-dataset", "--output", "output"],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            load.assert_called_once_with("local-dataset")
            self.assertEqual(
                Path("output/language.txt").read_text(encoding="utf-8"),
                "en\n",
            )
            self.assertIn("Detected query language: en", result.output)
            create_index_mock.assert_called_once_with(
                dataset,
                "en",
                Path("output"),
            )
            retrieve_mock.assert_called_once_with(
                dataset,
                index,
                "en",
                "BM25",
                Path("output"),
            )


if __name__ == "__main__":
    unittest.main()
