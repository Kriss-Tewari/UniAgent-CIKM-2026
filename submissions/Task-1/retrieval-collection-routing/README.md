# Retrieval: Collection-Routed Submission

UniAgent 2026 (CIKM'26), Task 1 (retrieval). The submission recognises which document
collection it is given and ranks the documents with the system that suits that collection.

## What happens

1. **Recognise the collection** from the dataset itself:
   - query language `en` → **University of Kassel, English**
   - query language `de` and mostly UUID document IDs → **University of Kassel, German**
   - query language `de` and other document IDs (e.g. `LARE190000005`) → **Hessian law**
2. **Build the query:** the title plus up to 12 words from the query's description, in order of
   occurrence. Stopwords, title words and repeats are skipped. The narrative is never used.
3. **Rank the documents** with the collection's system (below), up to 1000 per query.
4. **Log every step** per query in `run-trace.jsonl.log.gz`, following the competition's
   event-logging contract:
   ```
   input → route_collection → formulate_query → [expand_with_dictionary] → pyterrier_retrieve
         → [cosine_rerank] → observation → decision
   ```

## Methodology

| Collection | Method |
|---|---|
| University, German | **Domain dictionary + BM25 + cosine.** Related terms from `domain_dictionary.json` (business trips, travel expenses, reimbursement, procurement, administration) are added to the query at weight 0.3, so the original words keep full weight. BM25 ranks the documents; the top 100 are re-ranked by 70% BM25 + 30% TF-IDF cosine similarity between query and document. |
| University, English | **DPH.** Terrier's parameter-free DPH model ranks the documents for the expanded query. |
| Hessian law | **BM25F + cosine.** Each document is indexed with three fields: body (weight 1), stored title (3) and its 20 most frequent words (2). BM25F ranks the documents; the top 100 are re-ranked by 70% BM25F + 30% TF-IDF cosine. |

**Shared details:**
- Indexing uses the official baseline's stemmer, stopwords and tokeniser.
- Cosine similarity uses TF-IDF weights, (1 + ln tf) · ln(N/df), over the index's own terms.
- Below the re-ranked top 100, documents keep their first-stage order.

No LLM, embedding, neural model, relevance judgments or network access is used.

## Files

| File | Content |
|---|---|
| `submission.py` | Entrypoint: collection recognition, the three systems, the event log |
| `components.py` | Building blocks: event log, query keywords, index terms, dictionary, BM25 / DPH / BM25F retrieval, TF-IDF cosine fusion |
| `domain_dictionary.json` | German and English related-term groups |
| `test_submission.py` | Tests, run during `docker build` |
| `baseline.py`, `test_baseline.py`, `german-stopwords.*` | Official PyTerrier baseline, unchanged |
| `Dockerfile` | Official baseline image plus the files above |
