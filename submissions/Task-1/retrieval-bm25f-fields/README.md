# Retrieval: BM25F over Body, Title and Body-Keyword Fields

UniAgent 2026 (CIKM'26), Task 1 (retrieval) submission. It extends the official
[PyTerrier BM25 baseline](../retrieval-baseline-pyterrier/) in one way: it changes
the index. The query is the same as in the baseline.

1. Loads the retrieval dataset through `ir_datasets` (TIRA integration).
2. Detects the query language (`de`/`en`) with the official `detect_query_language()`.
3. Builds a PyTerrier `fields=True` index with the official language-specific
   configuration (stemmer, stopwords, tokeniser). Each document has three fields:
   - `text`: the original document body, as in the baseline
   - `title`: the document title stored in the dataset, if any (never generated
     from the body)
   - `keywords`: the 20 most frequent informative body words (non-stopword, at
     least 3 characters, non-numeric; ties broken by first occurrence)
4. Tokenises the topic title (`query.default_text()`) exactly as the official
   `retrieve()` does, ranks with BM25F (field weights text/title/keywords =
   1/3/2, length normalisation 0.75/0.5/0.5), and writes the top 1000 documents
   per query to `run.txt.gz`.

The topic `narrative` and the qrels are never read. No LLM, neural model,
embedding or network access is used.

`baseline.py`, `test_baseline.py`, `german-stopwords.txt` and
`german-stopwords.NOTICE` are unchanged copies of
`../retrieval-baseline-pyterrier/`. `bm25f_fields.py` imports them and does not
edit them.

## Files

| File | Purpose |
|---|---|
| `bm25f_fields.py` | Entry point of the submission |
| `event_logging.py` | Writer for `run-trace.jsonl.log.gz` (event-logging contract) |
| `test_bm25f_fields.py` | Tests for keywords, titles, BM25F controls and the trace (run during `docker build`) |
| `baseline.py`, `test_baseline.py`, `german-stopwords.*` | Official baseline, unchanged |
| `Dockerfile` | Official baseline image recipe plus the files above |

## Output (`$outputDir`)

- `run.txt.gz`: TREC run, top 1000 documents per query
- `run-trace.jsonl.log.gz`: event trace (see below)
- `language.txt`, `retrieval_config.json`: detected language, settings and field statistics
- `index-ir-metadata.yml`, `retrieval-ir-metadata.yml`: tirex-tracker resource metadata

## Event-logging contract

Each run writes `run-trace.jsonl.log.gz` as required by
[`../../event-logging-contract/README.md`](../../event-logging-contract/README.md).
No model is called, so `model` is `"-"` on every event. Each query
(`case_id` = query id) gets one causal chain:

1. `input`: the title and the description (never the narrative)
2. `tool_call` `formulate_query`: the query (the title; no description keywords)
3. `tool_call` `pyterrier_retrieve`: the tokenised query sent to Terrier, the
   BM25F controls and the number of retrieved documents
4. `observation`: the top 10 document ids with rank and score
5. `decision`: the final ranking, with the top 10 as `answer` and their scores as `evidence`

If retrieval fails, steps 3 and 5 are logged with `status: "error"` before the
exception propagates.

## Build and test

The image build runs all tests (official and new):

```bash
docker build -t uniagent-bm25f-fields .
```

## Run locally

```bash
docker run --rm --network none \
  -v "$PWD/../../datasets/retrieval-de-spot-check:/data:ro" -v "$PWD/out:/out" \
  --entrypoint python3 uniagent-bm25f-fields \
  /app/bm25f_fields.py --dataset /data --output /out/retrieval-de-spot-check
```

Write outputs **outside** this git repository. TIRA requires a clean working
tree for the real submission.

## Submit to TIRA

Run from this folder. Remove `--dry-run` only for the real submission, which
also requires the whole git repository to be clean (everything committed,
including untracked files).

```bash
tira-cli code-submission \
  --path . \
  --task uniagent-2026 \
  --dataset retrieval-de-spot-check-20260816-training \
  --command 'python3 /app/bm25f_fields.py --dataset $inputDataset --output $outputDir' \
  --dry-run
```

**Windows note:** tira-cli 0.0.205 builds the Docker tag (and the uploaded
source-code copy) from the repo-relative path and expects `/`. On Windows the
path has backslashes, so `docker build` fails with `invalid reference format`
and the source-code copy comes out empty. Run tira-cli from WSL/Linux, or use a
wrapper that converts the backslashes.
