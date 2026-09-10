#!/usr/bin/env bash
# Run SBERT-based semantic evaluation on N test examples (default 100).
set -e

N="${1:-100}"
python -m src.evaluate --config configs/config.yaml --n "$N"
