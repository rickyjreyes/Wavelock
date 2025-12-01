from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import hashlib
import json
import os

from wavelock.chain.CurvaChain import CurvaChain
from wavelock.chain.Block import Block


ARTIFACT_PREFIX = "ARTIFACT:"


@dataclass
class ArtifactNode:
    """
    A node in the research artifact DAG.

    - artifact_id: content hash (Merkle-style identifier)
    - parents: list of parent artifact_ids
    - path: optional local path (for convenience; not trusted)
    - meta: extra metadata (title, author, etc.)
    """
    artifact_id: str
    parents: List[str]
    path: Optional[str]
    meta: Dict[str, Any]


def _sha256_stream(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_artifact_file(path: str) -> str:
    """
    Content hash for a research artifact on disk.
    """
    return _sha256_stream(path)


def make_artifact_node(
    path: Optional[str],
    parents: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    content_bytes: Optional[bytes] = None,
) -> ArtifactNode:
    """
    Construct an ArtifactNode from either a file path or raw bytes.

    Exactly one of (path, content_bytes) should be provided.
    """
    if path is None and content_bytes is None:
        raise ValueError("must provide either path or content_bytes")

    if path is not None and content_bytes is not None:
        raise ValueError("provide either path or content_bytes, not both")

    parents = parents or []
    meta = meta or {}

    if content_bytes is not None:
        h = hashlib.sha256(content_bytes).hexdigest()
    else:
        h = hash_artifact_file(path)

    return ArtifactNode(
        artifact_id=h,
        parents=list(parents),
        path=path,
        meta=meta,
    )


def encode_artifact_message(node: ArtifactNode) -> str:
    """
    Encode an ArtifactNode as a single-chain message string.

    This is what gets Merkle-hashed inside the Block.
    """
    payload = {
        "artifact_id": node.artifact_id,
        "parents": node.parents,
        "path": node.path,
        "meta": node.meta,
    }
    return ARTIFACT_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_artifact_message(msg: str) -> Optional[ArtifactNode]:
    """
    Parse a chain message back into an ArtifactNode, or None if not an artifact.
    """
    if not msg.startswith(ARTIFACT_PREFIX):
        return None
    body = msg[len(ARTIFACT_PREFIX):]
    data = json.loads(body)
    return ArtifactNode(
        artifact_id=data["artifact_id"],
        parents=list(data.get("parents", [])),
        path=data.get("path"),
        meta=data.get("meta", {}),
    )


def add_artifact_block(
    chain: CurvaChain,
    node: ArtifactNode,
) -> Block:
    """
    Append a RESEARCH_TX block containing a single artifact node.
    The DAG edges are encoded via node.parents.

    The block's messages contain the encoded artifact node, and its meta
    includes a convenient copy of the artifact_id.
    """
    msg = encode_artifact_message(node)
    meta = {
        "artifact_id": node.artifact_id,
        "parents": node.parents,
        "kind": "RESEARCH_ARTIFACT",
    }
    chain.add_block(
        messages=[msg],
        block_type="RESEARCH_TX",
        meta=meta,
    )
    return chain.get_latest_block()


def build_artifact_dag(chain: CurvaChain) -> Dict[str, ArtifactNode]:
    """
    Scan the entire chain and build a mapping artifact_id -> ArtifactNode.

    This is a Merkle DAG:
    - each artifact_id is a SHA-256 content hash
    - parents[] form the edges
    """
    dag: Dict[str, ArtifactNode] = {}
    for block in chain.chain:
        for msg in block.messages:
            node = decode_artifact_message(msg)
            if node is not None:
                dag[node.artifact_id] = node
    return dag
