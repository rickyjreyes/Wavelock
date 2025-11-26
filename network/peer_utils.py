# chain/peer_utils.py
import json, os, random

PEERS_PATH = "peers.json"

def _parse_peer(ent):
    # Accept "host:port" OR [host, port] OR (host, port)
    if isinstance(ent, str):
        if ":" in ent:
            h, p = ent.split(":", 1)
            return (h.strip(), int(p))
        # fallback: treat bare string as host with default port (not ideal)
        return (ent.strip(), 9001)
    if isinstance(ent, (list, tuple)) and len(ent) == 2:
        return (str(ent[0]), int(ent[1]))
    raise ValueError(f"Unrecognized peer format: {ent!r}")

def load_peers():
    if not os.path.exists(PEERS_PATH):
        return []
    try:
        raw = json.load(open(PEERS_PATH))
        peers = []
        for ent in raw:
            try:
                peers.append(_parse_peer(ent))
            except Exception:
                pass
        # dedupe
        seen = set(); out = []
        for h,p in peers:
            k = (h, int(p))
            if k not in seen:
                seen.add(k); out.append(k)
        return out
    except Exception:
        return []

def save_peers(peers):
    # persist as "host:port" strings to avoid ambiguity
    out = [f"{h}:{int(p)}" for (h,p) in load_peers() + [(h,int(p)) for h,p in peers]]
    # dedupe while preserving order
    seen = set(); kept = []
    for s in out:
        if s not in seen:
            seen.add(s); kept.append(s)
    json.dump(kept, open(PEERS_PATH, "w"), indent=2)

def add_peer(host, port):
    cur = load_peers()
    cur.append((host, int(port)))
    save_peers(cur)

def remove_peer(host, port):
    peers = [(h,p) for (h,p) in load_peers() if not (h == host and int(p) == int(port))]
    # rewrite
    json.dump([f"{h}:{p}" for (h,p) in peers], open(PEERS_PATH, "w"), indent=2)

def random_peers(k=8):
    peers = load_peers()
    random.shuffle(peers)
    return peers[:k]
