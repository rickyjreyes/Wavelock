import json

# Define message types
GET_CHAIN = "GET_CHAIN"
SEND_BLOCK = "SEND_BLOCK"
GET_HASH = "GET_HASH"
SEND_HASH = "SEND_HASH"
GET_PEERS = "GET_PEERS"
SEND_PEERS = "SEND_PEERS"
VERIFY_SIGNATURE = "VERIFY_SIGNATURE"
SEND_VERIFICATION = "SEND_VERIFICATION"

# add/extend message type constants (keep your existing ones)
INV         = 0x20  # announce new block hash(es)
GET_BLOCK   = 0x21  # ask for block by hash
SEND_BLOCKS = 0x22  # send block objects
GET_PEERS   = 0x30
SEND_PEERS  = 0x31



def encode_message(type_, data=None):
    return json.dumps({
        "type": type_,
        "data": data or {}
    }).encode()

def decode_message(raw_bytes):
    try:
        msg = json.loads(raw_bytes.decode())
        return msg.get("type"), msg.get("data")
    except Exception:
        return None, None
