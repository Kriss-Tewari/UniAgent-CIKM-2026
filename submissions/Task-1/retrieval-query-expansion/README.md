# Retrieval: BM25 with Description-Keyword Query Expansion

UniAgent 2026 (CIKM'26), Task 1 (retrieval). The official PyTerrier BM25 baseline, with a
richer query: the title is extended with keywords from the query's description.

## What happens

1. **Load the dataset and detect its language** (`de`/`en`) with the official
   `detect_query_language()`.
2. **Build the index** with the official `create_index()` (language-specific stemmer, stopwords
   and tokeniser).
3. **Build the query:** the title plus up to 12 words from the description.
4. **Rank the documents** with BM25 and write the top 1000 per query to `run.txt.gz`.
5. **Log every step** per query in `run-trace.jsonl.log.gz`, following the competition's
   event-logging contract:
   ```
   input → formulate_query → pyterrier_retrieve → observation → decision
   ```

## Methodology

- **Description words:** taken in order of occurrence, skipping stopwords, title words,
  repeats, numbers and words shorter than 3 characters.
  - German uses the official stopword list plus a few question words.
  - English uses a small built-in stopword list.
- **Scoring:** the expanded query is tokenised exactly as in the official baseline and scored
  with BM25 (Terrier defaults).
- **Not used:** the narrative, relevance judgments, LLMs, embeddings and network access.

## Files

| File | Content |
|---|---|
| `query_expansion.py` | Entrypoint: query expansion, retrieval, event log |
| `event_logging.py` | Writer for `run-trace.jsonl.log.gz` |
| `test_query_expansion.py` | Tests, run during `docker build` |
| `baseline.py`, `test_baseline.py`, `german-stopwords.*` | Official PyTerrier baseline, unchanged |
| `Dockerfile` | Official baseline image plus the files above |
