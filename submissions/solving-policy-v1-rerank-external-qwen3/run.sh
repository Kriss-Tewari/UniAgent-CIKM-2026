#!/bin/sh
set -eu
exec python3 /app/launcher.py "$@" --passage-mode rerank --external-mode qwen3 --max-output-tokens 4096 --reranker-model /app/models/qwen3-reranker
