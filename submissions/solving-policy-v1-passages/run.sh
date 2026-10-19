#!/bin/sh
exec python3 /predict.py "$@" --passage-mode passages --max-output-tokens 4096
