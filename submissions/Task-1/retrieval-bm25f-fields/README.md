# Retrieval: BM25F over Body, Title and Keyword Fields

UniAgent 2026 (CIKM'26), Task 1 (retrieval). The official PyTerrier BM25 baseline with a richer
index: each document is split into three fields that are weighted differently. The query is the
same as in the baseline (the title).

## What happens

1. **Load the dataset and detect its language** (`de`/`en`) with the official
   `detect_query_language()`.
2. **Build a field index** with the official language-specific stemmer, stopwords and tokeniser:
   - `text`: the document body, as in the baseline
   - `title`: the document title stored in the dataset, if any (never made up from the body)
   - `keywords`: the document's 20 most frequent informative words
3. **Rank the documents** with BM25F for the query title and write the top 1000 per query to
   `run.txt.gz`.
4. **Log every step** per query in `run-trace.jsonl.log.gz`, following the competition's
   event-logging contract:
   ```
   input → formulate_query → pyterrier_retrieve → observation → decision
   ```

## Methodology

- **Field weights:** BM25F scores each field separately and combines them with weights
  text / title / keywords = 1 / 3 / 2. A match in the title or in the document's main keywords
  therefore counts more than a match anywhere in the body.
- **Length normalisation** is 0.75 / 0.5 / 0.5 per field.
- **Keywords:** non-stopword words of at least 3 characters, excluding numbers; ties are broken
  by first occurrence.
- **Not used:** the narrative, relevance judgments, LLMs, embeddings and network access.

## Files

| File | Content |
|---|---|
| `bm25f_fields.py` | Entrypoint: field index, BM25F retrieval, event log |
| `event_logging.py` | Writer for `run-trace.jsonl.log.gz` |
| `test_bm25f_fields.py` | Tests, run during `docker build` |
| `baseline.py`, `test_baseline.py`, `german-stopwords.*` | Official PyTerrier baseline, unchanged |
| `Dockerfile` | Official baseline image plus the files above |
