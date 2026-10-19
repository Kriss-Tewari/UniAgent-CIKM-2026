#!/bin/sh
exec python3 /predict.py "$@" --passage-mode rerank --max-output-tokens 4096
