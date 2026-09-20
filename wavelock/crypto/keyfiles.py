"""Durable JSON writes for public artifacts and local OTS secret state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def write_json(path, value, *, exclusive=False, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def mark_secret_used(path):
    """Preserve encrypted-at-rest encoding while durably recording use."""
    with open(path, encoding="utf-8") as stream:
        original = json.load(stream)
    original["used"] = True
    write_json(path, original, private=True)
