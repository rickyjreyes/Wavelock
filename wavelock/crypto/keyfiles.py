"""Durable JSON writes for public artifacts and local OTS secret state."""
from __future__ import annotations

import json
from wavelock.storage.durability import atomic_write


def write_json(path, value, *, exclusive=False, private=False):
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, encoded, exclusive=exclusive, mode=0o600 if private else 0o644)


def mark_secret_used(path):
    """Preserve encrypted-at-rest encoding while durably recording use."""
    with open(path, encoding="utf-8") as stream:
        original = json.load(stream)
    original["used"] = True
    write_json(path, original, private=True)
