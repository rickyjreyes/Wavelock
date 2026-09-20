"""Local filesystem durability primitives shared by keys, records and artifacts."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sqlite3
import tempfile

try:
    import fcntl
except ImportError:
    fcntl = None


def fsync_directory(path):
    """Persist namespace changes on POSIX; Windows has no portable dir fsync.

    Windows still gets atomic filesystem publication and flushed file content.
    This function makes no Windows power-loss durability claim.
    """
    if os.name == "nt":
        return False
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def ensure_directory(path):
    path = Path(path).absolute()
    if path.is_dir():
        return
    ensure_directory(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise NotADirectoryError(str(path))
    fsync_directory(path.parent)


def atomic_write(path, data: bytes, *, exclusive=False, mode=0o600):
    """Flush a complete temporary file, then publish by link or replace.

    An exclusive hard link atomically refuses an existing destination. A
    publication/fsync failure may leave a complete file, never authorizes
    key reuse, and is propagated to the caller.
    """
    path = Path(path)
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.chmod(temporary, mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
            fsync_directory(path.parent)


@contextlib.contextmanager
def exclusive_lock(path):
    """Same-filesystem process exclusion; not distributed consensus."""
    path = Path(path)
    ensure_directory(path.parent)
    if fcntl is None:
        connection = sqlite3.connect(str(path) + ".sqlite3", timeout=30)
        try:
            connection.execute("BEGIN EXCLUSIVE")
            yield
        finally:
            connection.rollback()
            connection.close()
        return
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
