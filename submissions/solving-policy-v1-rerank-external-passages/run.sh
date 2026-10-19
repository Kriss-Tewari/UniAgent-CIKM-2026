#!/bin/sh
set -eu
exec python3 /app/launcher.py "$@" --passage-mode rerank --external-mode passages --max-output-tokens 4096
