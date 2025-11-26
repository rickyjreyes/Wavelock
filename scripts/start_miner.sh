#!/usr/bin/env bash
set -e
export PYTHONPATH="$PWD"
peer="${1:-127.0.0.1:9001}"
user="${2:-ricky}"
msg="${3:-}"
if [ -z "$msg" ]; then
  python -m chain.cli mine-daemon --peer "$peer" --user "$user"
else
  python -m chain.cli mine-daemon --peer "$peer" --user "$user" --message "$msg"
fi
