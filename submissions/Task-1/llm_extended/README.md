# Retrieval: Collection-Routed Retrieval + Qwen3 Reranking
UniAgent 2026 (CIKM’26), Task 1. Extends collection-specific lexical retrieval with optional Qwen3-Reranker-0.6B reranking.
# What happens
1. Detect the dataset language and collection.
2. Retrieve documents using dictionary-expanded BM25, DPH or BM25F, depending on the collection.
3. Rerank the top 20 candidates with Qwen3 using the query title and description.
4. Preserve the remaining documents’ order and save the final ranking and event logs.
Original lexical tokenisation and stemming remain unchanged. Narratives and relevance judgments are excluded from retrieval. Model inference uses local weights without external API calls.
# Run
Install additional dependencies and download the model once:
python -m pip install -r requirements-neural.txt
python download_model.py --output models/qwen3-reranker

# Run from your working UniAgent environment:
python submission.py \
  --dataset "/actual/dataset/path" \
  --output results-qwen3 \
  --reranker qwen3 \
  --model-path models/qwen3-reranker \
  --device cpu \
  --rerank-depth 20

Use --device cuda for a compatible GPU, or --reranker none for lexical retrieval only.
# Main files
File	Purpose
submission.py	Collection routing and retrieval entrypoint
components.py	Lexical retrieval, expansion and fusion
neural_reranking.py	Passage selection and Qwen3 scoring
download_model.py	Download model weights
evaluate_runs.py	Compare both runs using the same qrels


# Outputs and evaluation
- run.txt.gz: final ranking.
- run.first-stage.txt.gz: original ranking.
- rerank_scores.json: raw model scores and passage details.
- run-trace.jsonl.log.gz: retrieval and reranking events.
- retrieval_config.json: execution settings.
