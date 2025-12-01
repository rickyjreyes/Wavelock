import json

from wavelock.chain.CurvaChain import CurvaChain
from wavelock.chain.artifacts import (
    make_artifact_node,
    encode_artifact_message,
    decode_artifact_message,
    add_artifact_block,
    build_artifact_dag,
)


def test_artifact_encode_decode_roundtrip():
    node = make_artifact_node(
        path=None,
        content_bytes=b"hello world",
        parents=[],
        meta={"title": "test"},
    )
    msg = encode_artifact_message(node)
    decoded = decode_artifact_message(msg)

    assert decoded is not None
    assert decoded.artifact_id == node.artifact_id
    assert decoded.parents == node.parents
    assert decoded.meta == node.meta


def test_artifact_dag_over_chain():
    chain = CurvaChain(difficulty=2)

    # root artifact
    a0 = make_artifact_node(
        path=None,
        content_bytes=b"root",
        parents=[],
        meta={"label": "root"},
    )
    add_artifact_block(chain, a0)

    # child artifact
    a1 = make_artifact_node(
        path=None,
        content_bytes=b"child",
        parents=[a0.artifact_id],
        meta={"label": "child"},
    )
    add_artifact_block(chain, a1)

    dag = build_artifact_dag(chain)

    assert a0.artifact_id in dag
    assert a1.artifact_id in dag
    assert dag[a1.artifact_id].parents == [a0.artifact_id]
