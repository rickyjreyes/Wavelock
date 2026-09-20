#!/usr/bin/env bash
set -e
export PYTHONPATH="$PWD"
signed_block="${1:-signed_message.json}"
python -m wavelock.chain.cli mine --signed-path "$signed_block"
