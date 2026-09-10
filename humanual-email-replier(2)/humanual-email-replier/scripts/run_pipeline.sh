#!/usr/bin/env bash
# End-to-end: build the SBERT retrieval index, then generate a few sample replies.
set -e

python -m src.retrieval --config configs/config.yaml
python -m src.generate --config configs/config.yaml --n 3
