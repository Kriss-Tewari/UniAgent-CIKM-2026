# Collection-routed retrieval + optional Qwen3 reranking

A separate experimental copy of the user's supplied code. No claim of improved
scores is possible without the dataset and qrels. The original German/English/law
routes remain available; Qwen3 is an explicit second-stage option.

## 1. Prepare an environment

From the extracted uniagent_qwen3 folder. Reuse your working UniAgent environment,
or clone it to avoid changing its dependencies:

    conda create -n UniAgentQwen --clone UniAgent
    conda activate UniAgentQwen
    python -m pip install -r requirements-neural.txt

The clone assumes UniAgent exists and has the working lexical dependencies.
For a fresh environment, also install:

    python -m pip install python-terrier tira tirex-tracker pandas click tqdm pytest

Java is required for PyTerrier. Check `java -version`. No Docker is required for
local experiments. Record exact installed versions with `python -m pip freeze`.
The provided ranges are not a lockfile. Do not upgrade a working environment blindly.

## 2. Download the reranker once (online preparation)

    python download_model.py --output models/qwen3-reranker

This downloads the tokenizer and full 0.6B model weights (a substantial download),
not just the tokenizer. A resolved model commit is recorded in download_source.json.
You may pass --revision COMMIT to reproduce it. No API key is normally needed for
this public model. Runtime loading uses local_files_only=True.

## 3. Run the original lexical system, then Qwen3

Replace `/actual/dataset/path` with the SAME real retrieval dataset path you used
successfully. These are not Task 2 PDF folders. Always use a fresh output folder.

    python -u submission.py --dataset /actual/dataset/path --output results-qwen3 --reranker qwen3 --model-path models/qwen3-reranker --device cpu --rerank-depth 20

For a CUDA-enabled local server with a compatible PyTorch installation:

    python -u submission.py --dataset /actual/dataset/path --output results-qwen3-gpu --reranker qwen3 --model-path models/qwen3-reranker --device cuda --rerank-depth 20 --batch-size 4

--reranker none (the default) runs only the original collection-routed approach.
The reranker default is intentionally opt-in, so the unchanged reference and
original tests still work without loading a neural model.

## 4. Rerank an EXISTING run without rebuilding its index

Copy your existing result folder (including run.txt.gz and run-trace.jsonl.log.gz)
to a new folder, for example results-existing-qwen3. Then run:

    python -u neural_reranking.py --dataset /actual/dataset/path --output results-existing-qwen3 --model-path models/qwen3-reranker --device cpu --depth 20

This also works with an earlier BM25F-combined run that has the same trace layout.
Never reuse a result directory for a different dataset. If a first-stage backup
exists, standalone reranking consistently uses that backup as its source.

## 5. Outputs and evaluation

- run.txt.gz: final ranking; its scores encode final rank, not neural probabilities.
- run.first-stage.txt.gz: untouched original ranking for comparison.
- rerank_scores.json: raw yes-minus-no model logit margins, candidate IDs, passage
  offsets and input token lengths. Margins are NOT calibrated probabilities.
- run-trace.first-stage.jsonl.log.gz: original trace.
- run-trace.jsonl.log.gz: original events plus model calls and new decisions,
  with distinct event IDs and links back to each original query chain.
- retrieval_config.json: original settings nested under first_stage_config,
  reranker settings, package versions and stage timing (excluding model load).
- rerank-error-trace.jsonl.log.gz: available if scoring fails. Original rankings
  are not replaced when scoring fails.

Compare with the SAME official development qrels:

    python evaluate_runs.py --qrels /actual/path/qrels.txt --baseline results-qwen3/run.first-stage.txt.gz --reranked results-qwen3/run.txt.gz

This reports nDCG@10, MAP, Recall@100 and MRR@10. The run is document retrieval,
not answer generation or administrative approval. Also inspect per-query failures.
Reranking only the top 20 cannot recover documents absent from those 20, and leaves
Recall@100 unchanged when at least 100 results exist. It aims at top-rank precision.
Do not select settings using hidden evaluation labels.

## Passage selection and scoring

Long documents: use a 40-word lead plus two 200-word windows with greatest lexical
query overlap (stride 160), scanning the first 200,000 characters. Source order is
preserved. This bounded heuristic can miss evidence; it is not exhaustive reading
or a trained passage selector. Inputs use the raw query title and description,
never the narrative. Max context defaults to 1536 tokens; document content is
truncated to fit after a bounded query and official model prompt.

Qwen3 follows the official causal yes/no relevance scoring format. Only final-token
logits are retained; no free-form answer or chain-of-thought is generated. The raw
logit margin orders candidates, with original rank breaking ties. Remaining
candidates keep their original order and document IDs.

## Runtime and TIRA

Official public page currently describes 4 CPU cores/40 GB RAM, no outbound network,
and time limits. https://uniagent.webis.de/cikm26/uniagent26-web/agentic-university-administration.html
A local GPU run is NOT evidence that this meets TIRA CPU limits. CPU inference may
be slow. The code defaults to CPU, batch size 1, four PyTorch threads. Measure full
runtime including indexing/model load. Confirm permission to package your own model
before choosing this for official submission; hosted Qwen2.5 is not this reranker.
Keep a tested lexical submission while evaluating this optional extension.

Dockerfile packages dependencies AND already-downloaded weights and specifies the
correct entrypoint. Weights are not included in this zip. Build only after step 2:

    docker build -t uniagent-qwen3 .

TIRA execution must explicitly include --reranker qwen3 to enable it. Docker tests
can require initial Terrier resource downloads; complete preparation before offline
runtime. Full Docker build, inference and TIRA compatibility have not been verified.
Resource tracker files from the original code measure lexical stages only; total
neural CPU/GPU/RAM resource instrumentation is not added.

## Validation and change locations

See CHANGES.md and VALIDATION.txt. Four model-free tests pass, including simulated
end-to-end stage-two orchestration. Actual downloaded-model inference and the full
original PyTerrier test suite were not run in the authoring environment.

Sources:
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
https://github.com/QwenLM/Qwen3-Embedding
Model weights: Apache-2.0 per official model card. Retain its license when packaging.
