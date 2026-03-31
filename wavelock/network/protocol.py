import json

# String-based message types (used by CLI and legacy server paths)
GET_CHAIN = "GET_CHAIN"
SEND_BLOCK = "SEND_BLOCK"
GET_HASH = "GET_HASH"
SEND_HASH = "SEND_HASH"
GET_PEERS = "GET_PEERS"
SEND_PEERS = "SEND_PEERS"
VERIFY_SIGNATURE = "VERIFY_SIGNATURE"
SEND_VERIFICATION = "SEND_VERIFICATION"

# Extended opcodes for binary P2P protocol
INV = "INV"
GET_BLOCK = "GET_BLOCK"
SEND_BLOCKS = "SEND_BLOCKS"


def encode_message(type_, data=None):
    return json.dumps({
        "type": type_,
        "data": data or {}
    }).encode()


def decode_message(raw_bytes):
    """Decode a JSON-framed message.

    Returns (type, data).
    """
    try:
        msg = json.loads(raw_bytes.decode())
        return msg.get("type"), msg.get("data")
    except Exception:
        return None, None
