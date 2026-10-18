# Retrieval: BM25F Fields + Description-Keyword Query Expansion

UniAgent 2026 (CIKM'26), Task 1 (retrieval). Combines the two extensions of the official
PyTerrier BM25 baseline:
- the field index of [`../retrieval-bm25f-fields/`](../retrieval-bm25f-fields/);
- the expanded query of [`../retrieval-query-expansion/`](../retrieval-query-expansion/).

## What happens

1. **Load the dataset and detect its language** (`de`/`en`) with the official
   `detect_query_language()`.
2. **Build a field index** with the official language-specific stemmer, stopwords and tokeniser:
   - `text`: the document body
   - `title`: the document title stored in the dataset, if any
   - `keywords`: the document's 20 most frequent informative words
3. **Build the query:** the title plus up to 12 words from the description.
4. **Rank the documents** with BM25F and write the top 1000 per query to `run.txt.gz`.
5. **Log every step** per query in `run-trace.jsonl.log.gz`, following the competition's
   event-logging contract:
   ```
   input → formulate_query → pyterrier_retrieve → observation → decision
   ```

## Methodology

- **Index side:** BM25F combines the three fields with weights text / title / keywords =
  1 / 3 / 2 (length normalisation 0.75 / 0.5 / 0.5), so matches in the title or in the
  document's main keywords count more.
- **Query side:** description words are added in order of occurrence, skipping stopwords,
  title words, repeats, numbers and words shorter than 3 characters.
  - German uses the official stopword list plus a few question words.
  - English uses a small built-in stopword list.
- **Not used:** the narrative, relevance judgments, LLMs, embeddings and network access.

## Files

| File | Content |
|---|---|
| `bm25f_combined.py` | Entrypoint: field index, query expansion, BM25F retrieval, event log |
| `event_logging.py` | Writer for `run-trace.jsonl.log.gz` |
| `test_bm25f_combined.py` | Tests, run during `docker build` |
| `baseline.py`, `test_baseline.py`, `german-stopwords.*` | Official PyTerrier baseline, unchanged |
| `Dockerfile` | Official baseline image plus the files above |
