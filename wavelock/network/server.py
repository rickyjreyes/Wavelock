# network/server.py
from __future__ import annotations
import os, socket, threading, time, json, hashlib, traceback
from typing import List, Tuple, Dict, Optional

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np

# --- repo imports ---
from wavelock.chain.config import load_config
from wavelock.network.peer_utils import load_peers, save_peers, add_peer, random_peers
from wavelock.network.protocol import (
    encode_message, decode_message,
    INV, GET_BLOCK, SEND_BLOCKS, GET_PEERS, SEND_PEERS,
    GET_CHAIN, SEND_BLOCK,
)
from wavelock.chain.chain_utils import load_all_blocks, save_block_to_disk
from wavelock.chain.Block import Block
from wavelock.chain.WaveLock import (
    CurvatureKeyPair, SCHEMA_V2, _serialize_commitment_v2, _canonical_json,
)

HOST = "0.0.0.0"

###############################################################################
# In-memory chain index + helpers
###############################################################################

class ChainState:
    def __init__(self):
        self._lock = threading.RLock()
        self.blocks: List[Block] = []
        self.by_hash: Dict[str, Block] = {}

    def load_from_disk(self):
        with self._lock:
            self.blocks = load_all_blocks()
            self.by_hash = {b.hash: b for b in self.blocks}
            _reconstruct_consumed_ots(self.blocks, CONSENSUS_OTS_LEDGER)

    def tip(self) -> Optional[Block]:
        with self._lock:
            return self.blocks[-1] if self.blocks else None

    def has_hash(self, h: str) -> bool:
        with self._lock:
            return h in self.by_hash

    def get_by_hash(self, h: str) -> Optional[Block]:
        with self._lock:
            return self.by_hash.get(h)

    def append(self, b: Block):
        with self._lock:
            save_block_to_disk(b)
            self.blocks.append(b)
            self.by_hash[b.hash] = b

CHAIN = ChainState()

###############################################################################
# Trust / strict verification
###############################################################################

def _load_trusted_commitments(path: str = "trusted_commitments.json") -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path, "r"))
    except Exception:
        return []

def _commitment_is_trusted(commitment: str, trusted: List[str]) -> bool:
    return str(commitment) in set(trusted)

def _load_published_psi(commitment: str):
    key = commitment.replace(":", "_").lower()
    npz_path = os.path.join("commitments", f"{key}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=False)
    psi = data["psi_star"]
    return cp.asarray(psi)

def _extract_curvature_fields(b: Block) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse message/signature/commitment from b.messages[] lines."""
    msg = sig = com = None
    for line in getattr(b, "messages", []) or []:
        if isinstance(line, str):
            if line.startswith("message: "):
                msg = line[len("message: "):]
            elif line.startswith("signature: "):
                sig = line[len("signature: "):]
            elif line.startswith("commitment: "):
                com = line[len("commitment: "):].strip()
    return msg, sig, com

def _verify_pow_and_linkage(b: Block, cfg) -> bool:
    tip = CHAIN.tip()
    expected_prev = tip.hash if tip else "0" * 64
    if b.previous_hash != expected_prev:
        print(f"Reject: prev_hash mismatch (got {b.previous_hash[:12]}..., expected {expected_prev[:12]}...)")
        return False
    try:
        target_hex = cfg.pow_target
        ok = int(b.hash, 16) <= int(target_hex, 16)
    except Exception:
        zeros = len(cfg.pow_target) - len(cfg.pow_target.lstrip("0"))
        ok = b.hash.startswith("0" * max(1, zeros))
    if not ok:
        print(f"Reject: PoW target not met ({b.hash[:12]}... > target)")
    return ok

def _verify_curvature(b: Block, cfg) -> bool:
    """Fail-closed curvature verification for incoming blocks.

    Security invariants (see WAVELOCK_THEORY_BREAK_AUDIT.md):

    * There is NO non-strict / trust-only acceptance path. Trust-list
      membership is necessary but NEVER sufficient — a valid signature is
      always required.
    * Missing commitment, missing message/signature, or unpublished proof
      material all REJECT. There is no "allow due to policy" branch.
    * Any exception during verification REJECTS.

    NOTE: this path still verifies the *legacy* WaveLock SIGv2 signature, which
    is itself a deprecated/insecure construction (its verification requires
    ψ★). It is retained only so existing legacy ledgers fail closed rather than
    fail open. New deployments should verify WaveLock-OTS signatures
    (wavelock.crypto.wavelock_ots) instead; see docs/MIGRATION_FROM_SIGV2.md.
    """
    trusted = _load_trusted_commitments()
    msg, sig, com = _extract_curvature_fields(b)

    # 1. Commitment must be present AND trusted (necessary, not sufficient).
    if not com or not _commitment_is_trusted(com, trusted):
        print("Reject: commitment missing or not trusted.")
        return False

    # 2. Message + signature fields must be present.
    if msg is None or sig is None:
        print("Reject: missing curvature message/signature fields.")
        return False

    # 3. Proof material must be published. No bypass: unpublished => reject.
    try:
        psi = _load_published_psi(com)
    except Exception as e:
        print(f"Reject: failed to load published proof material: {e}")
        return False
    if psi is None:
        print("Reject: proof material not published; cannot verify (fail closed).")
        return False

    # 4. A valid signature is ALWAYS required.
    try:
        kp = CurvatureKeyPair(n=4, test_mode=True)
        kp.psi_star = psi
        kp.psi_0 = cp.zeros_like(psi)
        kp.commitment = com
        if not kp.verify(msg, sig):
            print("Reject: curvature signature invalid.")
            return False
    except Exception as e:
        print(f"Reject: verification error (fail closed): {e}")
        return False

    return True


# Backwards-compatible alias. The old name implied a "strict" toggle existed;
# verification is now unconditionally fail-closed.
_strict_verify_curvature = _verify_curvature


###############################################################################
# WaveLock-OTS block acceptance (fail-closed) — WIRED INTO CONSENSUS
###############################################################################
#
# INTEGRATION STATUS: OTS verification + replay rejection is now wired into the
# real block-acceptance path (``try_accept_block`` below). A block is an
# "OTS-required" block when its ``block_type == "OTS"`` (or its ``meta`` declares
# ``auth_scheme == WaveLock-OTS-v1``, or the config sets ``require_ots``). On
# that path:
#
#   * the block's OTS public key + message + signature are verified with the
#     pure :func:`verify_ots` (strict canonical / Merkle / fingerprint checks);
#   * the durable :class:`PersistentOTSReplayLedger` rejects any signature whose
#     ``one_time_key_id`` or OTS leaf id was already accepted (Findings C/D);
#   * legacy SIGv2 (or any non-WaveLock-OTS scheme) is NEVER accepted — there is
#     no silent fallback to ``_verify_curvature`` on an OTS-required block.
#
# Non-OTS (legacy curvature) blocks still go through the fail-closed legacy path.
# This keeps existing ledgers working while making OTS enforceable. WaveLock-OTS
# remains experimental and is NOT production-ready; see the report and
# docs/WAVELOCK_MERKLE_ROADMAP.md.

from wavelock.crypto.wavelock_ots import (
    SCHEME as OTS_SCHEME,
    OTSReplayLedger,
    verify_ots as _verify_ots,
)
from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger, OTSLedgerError

#: Durable, reconstructable consumed-id ledger used by block acceptance. The
#: consumed set is a function of accepted chain state (see ``index_signature``),
#: so any node replaying accepted OTS blocks derives the same set. This is the
#: load-bearing control that turns OTS "one-time" into an enforced invariant on
#: this node; full Finding-D closure requires every node to run this rejection
#: against a ledger derived from agreed chain state.
CONSENSUS_OTS_LEDGER = PersistentOTSReplayLedger()

#: There is exactly ONE authoritative replay ledger (Mythos M3). ``OTS_LEDGER``
#: used to be an independent in-memory :class:`OTSReplayLedger`; keeping two
#: ledgers that each "consume" a one_time_key_id independently was a bypass risk
#: (a signature consumed on one would still be free on the other). It is now an
#: alias for the durable, inter-process-locked :data:`CONSENSUS_OTS_LEDGER`, so
#: the standalone ``verify_ots_payload`` entry point and consensus block
#: acceptance share the same consumed set. The pure in-memory ``OTSReplayLedger``
#: class is still importable for callers that want an explicit ephemeral ledger
#: (they must pass it via ``ledger=``); the module default is the durable one.
OTS_LEDGER = CONSENSUS_OTS_LEDGER


#: Bumped if the canonical OTS block-signing transcript shape changes.
OTS_BLOCK_TRANSCRIPT_VERSION = 1


def _ots_block_transcript_payload(block_type, previous_hash, messages, meta) -> dict:
    """Build the canonical, signature-free transcript object for an OTS block.

    This is the object an OTS signature actually authorizes (Mythos M1). It binds
    the consensus-stable, attacker-relevant parts of the block:

    * ``messages`` — the actual accepted body payload;
    * ``block_type`` — so an OTS auth can't be retyped onto another block kind;
    * ``previous_hash`` — the parent reference (consensus linkage);
    * ``auth_scheme`` and the **public key** carried in ``ots_auth`` — binds the
      signing identity;
    * any other ``meta`` fields a deployment adds.

    It deliberately EXCLUDES the self-referential ``ots_auth.signature`` and
    ``ots_auth.message`` (those carry / equal the transcript itself) and the
    mining outputs (``nonce``/``hash``) and ``timestamp``/``index`` (not signed:
    a node may legitimately reposition an identical body, and the durable replay
    ledger — not the index — is what prevents reuse).
    """
    m = dict(meta or {})
    auth = m.get("ots_auth")
    if isinstance(auth, dict):
        # Strip the self-referential fields so sign-time and verify-time agree.
        m = {**m, "ots_auth": {k: v for k, v in auth.items()
                               if k not in ("signature", "message")}}
    return {
        "wl_ots_block_transcript": OTS_BLOCK_TRANSCRIPT_VERSION,
        "auth_scheme": m.get("auth_scheme"),
        "block_type": str(block_type or ""),
        "previous_hash": previous_hash,
        "messages": list(messages or []),
        "meta": m,
    }


def canonical_ots_block_message(b: "Block") -> bytes:
    """Canonical signing transcript (preimage bytes) for an OTS block.

    Recomputed from the *received* block at verify time so the OTS signature is
    bound to the actual block body, not to an arbitrary ``meta.ots_auth.message``
    (Mythos M1). Sign-time and verify-time produce identical bytes because the
    self-referential signature/message fields are excluded.
    """
    return _canonical_json(_ots_block_transcript_payload(
        getattr(b, "block_type", ""),
        getattr(b, "previous_hash", None),
        getattr(b, "messages", []) or [],
        getattr(b, "meta", {}) or {},
    ))


def canonical_ots_block_digest(b: "Block") -> str:
    """Domain-separated hex digest of :func:`canonical_ots_block_message`.

    This hex string is the message an OTS signer signs and the verifier
    recomputes; it is what gets stored in ``meta.ots_auth.message``.
    """
    return hashlib.sha256(
        b"WL-OTS-BLOCK-TRANSCRIPT-v1\x00" + canonical_ots_block_message(b)
    ).hexdigest()


def build_ots_block_meta(public_key: dict, message, signature: dict) -> dict:
    """Canonical ``meta`` for an OTS-authenticated block.

    The auth material lives in ``meta`` so it is covered by the block hash
    (``Block.calculate_hash`` hashes a sorted-key JSON of ``meta``), binding the
    OTS signature into the block identity. ``message`` MUST be the canonical
    block-signing digest (:func:`canonical_ots_block_digest`) that ``signature``
    actually signs — a free-text message no longer authorizes a block (M1). Use
    :func:`build_signed_ots_block` to construct blocks correctly.
    """
    return {
        "auth_scheme": OTS_SCHEME,
        "ots_auth": {
            "public_key": public_key,
            "message": message,
            "signature": signature,
        },
    }


def build_signed_ots_block(secret_key: dict, public_key: dict, messages,
                           *, index: int = 1, previous_hash: str = "0" * 64,
                           difficulty: int = 1, block_type: str = "OTS",
                           extra_meta: Optional[dict] = None,
                           allow_reuse: bool = False) -> "Block":
    """Build a mined OTS block whose signature is bound to its body (M1).

    The signer signs :func:`canonical_ots_block_digest` of the body (messages,
    block_type, previous_hash, auth scheme, public key, extra meta) — NOT a
    free-text message — so the accepted body is exactly what was authorized.
    """
    from wavelock.crypto.wavelock_ots import sign_ots

    base_meta = dict(extra_meta or {})
    base_meta["auth_scheme"] = OTS_SCHEME
    base_meta["ots_auth"] = {"public_key": public_key}
    transcript = hashlib.sha256(
        b"WL-OTS-BLOCK-TRANSCRIPT-v1\x00" + _canonical_json(
            _ots_block_transcript_payload(block_type, previous_hash, messages, base_meta)
        )
    ).hexdigest()
    sig = sign_ots(secret_key, transcript, allow_reuse=allow_reuse)
    meta = build_ots_block_meta(public_key, transcript, sig)
    if extra_meta:
        for k, v in extra_meta.items():
            meta.setdefault(k, v)
    return Block(index=index, messages=list(messages), previous_hash=previous_hash,
                 difficulty=difficulty, block_type=block_type, meta=meta)


def _extract_ots_auth(b: "Block") -> Optional[dict]:
    """Return the ``ots_auth`` dict from a block's meta, or None if absent/malformed."""
    meta = getattr(b, "meta", None) or {}
    auth = meta.get("ots_auth")
    return auth if isinstance(auth, dict) else None


def block_requires_ots(b: "Block", cfg=None) -> bool:
    """True if this block must be authenticated with WaveLock-OTS.

    OTS is required when the block declares it (``block_type == "OTS"`` or
    ``meta.auth_scheme == WaveLock-OTS-v1``) or when the config opts in
    (``cfg.require_ots``). On an OTS-required block, legacy SIGv2 is never
    accepted (no fallback to the curvature path).
    """
    if str(getattr(b, "block_type", "") or "").upper() == "OTS":
        return True
    meta = getattr(b, "meta", None) or {}
    if str(meta.get("auth_scheme", "")) == OTS_SCHEME:
        return True
    if cfg is not None and getattr(cfg, "require_ots", False):
        return True
    return False


def _is_well_formed_ots_signature(sig) -> bool:
    """True if ``sig`` is structurally a WaveLock-OTS signature with replay ids."""
    return (isinstance(sig, dict)
            and sig.get("scheme") == OTS_SCHEME
            and sig.get("one_time_key_id") is not None
            and sig.get("public_key_fingerprint") is not None)


def _block_claims_ots(b: "Block") -> bool:
    """True if a block STRUCTURALLY declares itself OTS-authenticated.

    Independent of any current config (Mythos M2): detection is by ``block_type``
    or ``meta.auth_scheme`` only, so reconstruction recognizes OTS blocks even if
    ``cfg.require_ots`` is no longer set.
    """
    if str(getattr(b, "block_type", "") or "").upper() == "OTS":
        return True
    meta = getattr(b, "meta", None) or {}
    return str(meta.get("auth_scheme", "")) == OTS_SCHEME


def _reconstruct_consumed_ots(blocks, ledger) -> None:
    """Rebuild the consumed-OTS-identity set from accepted chain state (M2).

    The consumed set is a function of *accepted chain state*, not of the current
    config or of the side ``ots_replay.jsonl`` cache. For every accepted block
    carrying well-formed WaveLock-OTS auth material we fold its identifiers into
    the ledger via :meth:`PersistentOTSReplayLedger.index_signature`, regardless
    of whether ``cfg.require_ots`` is currently true. Deleting the side ledger and
    restarting therefore cannot resurrect a consumed OTS identity while the
    accepted chain still contains the block that consumed it.

    Fail-closed: a block that STRUCTURALLY claims to be OTS (``block_type ==
    "OTS"`` or ``meta.auth_scheme == WaveLock-OTS-v1``) but carries malformed or
    missing OTS auth raises :class:`OTSLedgerError` rather than being skipped — a
    silently-skipped OTS block would reopen the replay hole this closes.
    """
    for b in blocks:
        auth = _extract_ots_auth(b)
        sig = auth.get("signature") if isinstance(auth, dict) else None
        if _is_well_formed_ots_signature(sig):
            ledger.index_signature(sig)
        elif _block_claims_ots(b):
            raise OTSLedgerError(
                f"accepted OTS block #{getattr(b, 'index', '?')} "
                f"({str(getattr(b, 'hash', ''))[:12]}...) has malformed/missing OTS "
                "auth; refusing to reconstruct replay state (fail closed)."
            )


def _verify_ots_block(b: "Block", cfg=None, ledger: "PersistentOTSReplayLedger | None" = None) -> bool:
    """Fail-closed OTS verification + durable replay rejection for one block.

    Rejects (returns False), consuming nothing, on:

    * missing/malformed ``ots_auth`` or any missing auth field;
    * a non-WaveLock-OTS scheme on the public key or signature (legacy SIGv2 is
      refused here — there is no fallback);
    * any ``verify_ots`` failure (canonical / Merkle / fingerprint / digest);
    * a replayed ``one_time_key_id`` or OTS leaf id (durable ledger);
    * any exception.
    """
    led = ledger if ledger is not None else CONSENSUS_OTS_LEDGER
    try:
        auth = _extract_ots_auth(b)
        if auth is None:
            print("Reject: OTS-required block missing ots_auth.")
            return False
        pub = auth.get("public_key")
        msg = auth.get("message")
        sig = auth.get("signature")
        if not isinstance(pub, dict) or not isinstance(sig, dict) or not isinstance(msg, str):
            print("Reject: malformed/missing OTS auth fields (fail closed).")
            return False
        # Legacy SIGv2 (or anything not WaveLock-OTS) is NEVER accepted here.
        if pub.get("scheme") != OTS_SCHEME or sig.get("scheme") != OTS_SCHEME:
            print("Reject: non-OTS scheme on OTS-required path (legacy SIGv2 refused).")
            return False
        # --- Mythos M1: bind the signature to the actual block body. ----------
        # Recompute the canonical transcript digest from the RECEIVED block and
        # require it to be what was signed. A signature over an arbitrary message
        # (e.g. a public "hello world" signature) no longer authorizes a block
        # whose body says something else: the carried message must equal the
        # recomputed transcript, and verification runs against that transcript.
        expected = canonical_ots_block_digest(b)
        import hmac as _hmac
        if not _hmac.compare_digest(msg, expected):
            print("Reject: OTS auth message does not bind the block body (M1, fail closed).")
            return False
        if not led.accept(pub, expected, sig):
            print("Reject: OTS signature invalid or replayed (fail closed).")
            return False
        return True
    except Exception as e:
        print(f"Reject: OTS verification error (fail closed): {e}")
        return False


def verify_ots_payload(public_key: dict, message, signature: dict,
                       ledger=None) -> bool:
    """Fail-closed WaveLock-OTS verification + duplicate-key rejection.

    Standalone entry point (not block-shaped). Returns True only if the
    signature verifies AND its one_time_key_id has not been consumed before.
    Rejects (False) on:

    * any non-WaveLock-OTS scheme (legacy SIGv2 is never accepted here);
    * any verification failure (strict canonical checks in ``verify_ots``);
    * a replayed/duplicate one_time_key_id;
    * any exception (fail closed).
    """
    try:
        if not isinstance(signature, dict) or signature.get("scheme") != OTS_SCHEME:
            return False
        if not isinstance(public_key, dict) or public_key.get("scheme") != OTS_SCHEME:
            return False
        # Default to the single authoritative durable ledger (M3 unification),
        # so a signature consumed here is also consumed for consensus and vice
        # versa. Callers wanting an ephemeral ledger pass it via ``ledger=``.
        led = ledger if ledger is not None else CONSENSUS_OTS_LEDGER
        return bool(led.accept(public_key, message, signature))
    except Exception:
        return False

###############################################################################
# P2P: dedupe + broadcast
###############################################################################

_seen_hashes: Dict[str, float] = {}

def _dedupe_seen(h: str, ttl=300) -> bool:
    now = time.time()
    for k, t in list(_seen_hashes.items()):
        if now - t > ttl:
            _seen_hashes.pop(k, None)
    if h in _seen_hashes:
        return True
    _seen_hashes[h] = now
    return False

def broadcast_inv(block_hash: str):
    if _dedupe_seen(block_hash):
        return
    for host, port in load_peers():
        try:
            with socket.socket() as s:
                s.settimeout(1.0)
                s.connect((host, int(port)))
                s.sendall(encode_message(INV, [block_hash]))
        except Exception:
            pass

###############################################################################
# Accept/validate incoming blocks
###############################################################################

def has_block(h: str) -> bool:
    return CHAIN.has_hash(h)

def get_block_by_hash(h: str) -> Optional[Block]:
    return CHAIN.get_by_hash(h)

def _block_from_dict(d: dict) -> Block:
    """Reconstruct Block from a dict payload, using canonical field names."""
    return Block.from_dict(d)

def try_accept_block_dict(d: dict, cfg) -> bool:
    b = _block_from_dict(d)
    return try_accept_block(b, cfg)

def try_accept_block(b: Block, cfg) -> bool:
    if not _verify_pow_and_linkage(b, cfg):
        return False
    # Route to the correct authentication path. An OTS-required block is verified
    # with WaveLock-OTS + durable replay rejection and NEVER falls back to the
    # legacy curvature path (legacy SIGv2 is refused where OTS is expected).
    if block_requires_ots(b, cfg):
        if not _verify_ots_block(b, cfg):
            return False
    else:
        if not _verify_curvature(b, cfg):
            return False
    CHAIN.append(b)
    print(f"Accepted Block #{b.index} | {b.hash[:12]}...")
    broadcast_inv(b.hash)
    return True

###############################################################################
# Connection handler
###############################################################################

def handle_client(conn: socket.socket, addr: Tuple[str, int], cfg):
    try:
        buf = b""
        conn.settimeout(10.0)
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            # Try to decode the buffer as a complete JSON message
            try:
                opcode, data = decode_message(buf)
            except Exception:
                continue
            if opcode is None:
                continue
            # Successfully decoded — reset buffer
            buf = b""

            if opcode == INV:
                want = [h for h in data if not has_block(h)]
                if want:
                    conn.sendall(encode_message(GET_BLOCK, want))

            elif opcode == GET_BLOCK:
                blocks = []
                for h in data:
                    b = get_block_by_hash(h)
                    if b:
                        blocks.append(b.to_dict())
                if blocks:
                    conn.sendall(encode_message(SEND_BLOCKS, blocks))

            elif opcode == SEND_BLOCKS:
                for bd in data:
                    try:
                        if not has_block(bd.get("hash", "")):
                            try_accept_block_dict(bd, cfg)
                    except Exception as e:
                        print("Failed to accept remote block:", e)

            elif opcode == GET_PEERS:
                plist = [f"{h}:{p}" for (h, p) in load_peers()]
                conn.sendall(encode_message(SEND_PEERS, plist))

            elif opcode == SEND_PEERS:
                for ent in data:
                    try:
                        h, p = ent.split(":")
                        add_peer(h, int(p))
                    except Exception:
                        pass

            elif opcode == GET_CHAIN:
                ser = [b.to_dict() for b in CHAIN.blocks]
                conn.sendall(encode_message(SEND_BLOCK, ser))

    except socket.timeout:
        pass
    except Exception as e:
        print("handler error:", e)
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except Exception:
            pass

###############################################################################
# Server main
###############################################################################

def main():
    CHAIN.load_from_disk()
    cfg = load_config(os.getenv("WAVELOCK_CONFIG"))

    trusted = _load_trusted_commitments()
    print(f"Trusted commitments loaded: {len(trusted)}")
    print(f"WaveLock P2P server listening on port {cfg.port}")
    print("  Curvature verify: FAIL-CLOSED (always requires a valid signature)")
    print("  Trust-list membership alone never accepts a block.")
    print("  Unpublished proof material is rejected.")

    for seed in cfg.seeds or []:
        try:
            host, port = seed.split(":")
            add_peer(host, int(port))
        except Exception:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, cfg.port))
        srv.listen(64)
        while True:
            try:
                conn, addr = srv.accept()
                try:
                    add_peer(addr[0], cfg.port)
                except Exception:
                    pass
                threading.Thread(target=handle_client, args=(conn, addr, cfg), daemon=True).start()
            except KeyboardInterrupt:
                print("\nServer shutdown requested.")
                break
            except Exception as e:
                print("accept() error:", e)
                time.sleep(0.2)

if __name__ == "__main__":
    main()
