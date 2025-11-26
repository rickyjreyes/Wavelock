#!/usr/bin/env bash
set -e
export PYTHONPATH="$PWD"
export SEEDS="${SEEDS:-}"
python -m network.server --port "${1:-9001}"
