# network/client.py
import os, sys, socket

# --- robust path so this file can run as a script or a module ---
BASE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    # when run as a module: python -m network.client
    from .protocol import encode_message, decode_message, GET_CHAIN, SEND_BLOCK
except Exception:
    # when run as a script: python network/client.py
    from protocol import encode_message, decode_message, GET_CHAIN, SEND_BLOCK


def request_chain(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"🔌 Connected to {host}:{port}")
        s.sendall(encode_message(GET_CHAIN))
        response = s.recv(65536)
        msg_type, data = decode_message(response)
        if msg_type == SEND_BLOCK:
            print(f"📦 Received {len(data)} blocks from peer")
            for block in data:
                print(f"🧱 Block #{block['index']} | Hash: {block['hash'][:12]}...")
        else:
            print("⚠️ Unexpected or empty response")

if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--host" or sys.argv[3] != "--port":
        print("Usage: python -m network.client --host <IP> --port <PORT>")
        sys.exit(1)
    request_chain(sys.argv[2], int(sys.argv[4]))
