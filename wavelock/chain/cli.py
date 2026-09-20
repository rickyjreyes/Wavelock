"""Supported WaveLock-OTS command line workflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SIGNED_BLOCK_FORMAT = "WaveLock-SignedBlock-v1"


def _config():
    from .config import load_config
    return load_config(os.getenv("WAVELOCK_CONFIG"))


def _key_directory(user):
    if not user or user in (".", "..") or any(c in user for c in "/\\"):
        raise ValueError("user must be a local key label without path separators")
    return Path("keys") / user


def _keygen(args):
    from wavelock.crypto.ots_cli import cmd_keygen
    args.out = args.out or str(_key_directory(getattr(args, "user", None) or "default"))
    args.unsafe_export_secret_state = False
    return cmd_keygen(args)


def _load_signed(path):
    from .Block import Block
    from .ots_blocks import verify_block_integrity, verify_ots_block
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    if (not isinstance(payload, dict) or set(payload) != {"format", "block"}
            or payload["format"] != SIGNED_BLOCK_FORMAT):
        raise ValueError("expected a WaveLock-SignedBlock-v1 artifact; generate a new OTS signature")
    block = Block.from_dict(payload["block"])
    if not verify_block_integrity(block, require_pow=False) or not verify_ots_block(block):
        raise ValueError("signed block failed public verification")
    return block


def _sign(args):
    from wavelock.crypto.keyfiles import write_json, mark_secret_used
    from wavelock.crypto.ots_cli import PUBLIC_NAME, SECRET_NAME
    from wavelock.crypto.wavelock_ots import load_public_key, load_secret_key
    from .chain_utils import load_all_blocks
    from .ots_blocks import build_signed_ots_block, verify_ots_chain

    if Path(args.output).exists():
        raise FileExistsError(f"output already exists: {args.output}")
    secret_path = Path(args.secret) if args.secret else _key_directory(args.user) / SECRET_NAME
    public_path = Path(args.public) if args.public else secret_path.with_name(PUBLIC_NAME)
    secret = load_secret_key(secret_path, passphrase=args.passphrase)
    public = load_public_key(public_path)
    blocks = load_all_blocks()
    if not verify_ots_chain(blocks):
        raise ValueError("existing ledger failed OTS verification; use a fresh data directory for migration")
    block = build_signed_ots_block(
        secret, public, [args.message], index=len(blocks) + 1,
        previous_hash=blocks[-1].hash if blocks else "0" * 64,
        difficulty=args.difficulty, mine=False,
    )
    # Durable one-time state precedes publication of the signature. A failed
    # output write can waste a key, but never authorizes its reuse.
    mark_secret_used(secret_path)
    write_json(args.output, {"format": SIGNED_BLOCK_FORMAT, "block": block.to_dict()}, exclusive=True)
    print(f"Signed OTS block draft: {args.output}")
    print(f"Public-key fingerprint: {public['public_key_fingerprint']}")
    return 0


def _mine(args):
    from wavelock.network import server
    block = _load_signed(args.signed_path)
    cfg = _config()
    if not cfg.require_ots:
        raise ValueError("the supported mining workflow requires require_ots=true")
    server.CHAIN.load_from_disk(require_ots=True)
    tip = server.CHAIN.tip()
    if block.previous_hash != (tip.hash if tip else "0" * 64):
        raise ValueError("signed parent is stale; prepare a new block with a fresh one-time key")
    if server.CONSENSUS_OTS_LEDGER.is_consumed(block.meta["ots_auth"]["signature"]):
        raise ValueError("one-time identity already accepted")
    # The target and mining fields are excluded by transcript version 1.
    # Mine to both the declared difficulty and the local acceptance target.
    target = min(int(cfg.pow_target, 16), (1 << (4 * (64 - block.difficulty))) - 1)
    nonce = 0
    while True:
        digest = block.calculate_hash(nonce)
        if int(digest, 16) <= target:
            block.nonce, block.hash = nonce, digest
            break
        nonce += 1
    if not server.try_accept_block(block, cfg):
        raise ValueError("node rejected the mined OTS block")
    print(f"Mined and accepted OTS block #{block.index}: {block.hash}")
    return 0


def _verify(args):
    from .chain_utils import load_all_blocks
    from .ots_blocks import verify_ots_chain
    if args.signed_path:
        _load_signed(args.signed_path)
        print("VALID: block body authenticated using public material only (mining not required).")
        return 0
    blocks = load_all_blocks()
    if not blocks:
        raise ValueError("no blocks found")
    if not verify_ots_chain(blocks):
        raise ValueError("ledger failed OTS signature, replay, linkage, hash, or Merkle verification")
    print(f"VALID: {len(blocks)} OTS blocks; public signatures, unique keys, linkage, hashes and Merkle roots.")
    return 0


def _view(args):
    from .chain_utils import load_all_blocks
    for block in load_all_blocks():
        print(f"#{block.index} {block.hash} {block.messages}")
    return 0


def _peer(args):
    from wavelock.network.peer_utils import add_peer, load_peers
    if args.command == "peer":
        add_peer(args.host, args.port)
    for host, port in load_peers():
        print(f"{host}:{port}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="WaveLock: public OTS signing and block verification", allow_abbrev=False)
    parser.add_argument("--data-dir", help="node data directory (or set WAVELOCK_DATA_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("keygen", "add"):
        p = sub.add_parser(name, help="Generate a fresh one-time keypair")
        if name == "add":
            p.add_argument("user", help="local key label; not a reusable signing identity")
        p.add_argument("--out", default=None)
        p.add_argument("--n", type=int, choices=range(2, 13), default=4)
        p.add_argument("--entropy-bits", type=int, choices=(128, 192, 256), default=256)
        p.add_argument("--encrypt", action="store_true")
        p.add_argument("--passphrase")
        p.set_defaults(func=_keygen)
    p = sub.add_parser("sign", help="Sign a canonical block draft once")
    p.add_argument("user", nargs="?", default="default")
    p.add_argument("--secret")
    p.add_argument("--public")
    p.add_argument("--passphrase")
    p.add_argument("--message", required=True)
    p.add_argument("--output", default="signed_message.json")
    p.add_argument("--difficulty", type=int, choices=range(0, 9), default=4)
    p.set_defaults(func=_sign)
    p = sub.add_parser("mine", help="Mine and accept a signed block without re-signing")
    p.add_argument("--signed-path", "--signed_path", dest="signed_path", default="signed_message.json")
    p.set_defaults(func=_mine)
    for name in ("verify", "audit"):
        p = sub.add_parser(name, help="Verify the OTS ledger using only public material")
        p.add_argument("--signed-path", "--signed_path", dest="signed_path")
        p.set_defaults(func=_verify)
    sub.add_parser("view", help="Show accepted blocks").set_defaults(func=_view)
    p = sub.add_parser("peer", help="Add a node peer")
    p.add_argument("host")
    p.add_argument("port", type=int)
    p.set_defaults(func=_peer)
    sub.add_parser("peers", help="List peers").set_defaults(func=_peer)
    sub.add_parser("legacy", help="Historical SIGv2 tools (insecure compatibility only)")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Parse only the prefix; delegated commands own all arguments after their
    # name, including --help. Set the data directory before importing runtime.
    prefix = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    prefix.add_argument("--data-dir")
    prefix.add_argument("command", nargs="?")
    prefix.add_argument("arguments", nargs=argparse.REMAINDER)
    global_args, unknown = prefix.parse_known_args(argv)
    if global_args.data_dir:
        os.environ["WAVELOCK_DATA_DIR"] = str(Path(global_args.data_dir).resolve())
    command = global_args.command or ""
    if not unknown and (command == "ots" or command.startswith("ots-")):
        from wavelock.crypto.ots_cli import main as ots_main
        return ots_main(global_args.arguments if command == "ots" else [command] + global_args.arguments)
    if not unknown and command == "legacy":
        from . import legacy_cli
        legacy_cli._warn_legacy_sigv2()
        original = sys.argv
        try:
            sys.argv = [original[0]] + global_args.arguments
            return legacy_cli.main() or 0
        finally:
            sys.argv = original
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.data_dir:
        os.environ["WAVELOCK_DATA_DIR"] = str(Path(args.data_dir).resolve())
    from wavelock.crypto.wavelock_ots import WaveLockOTSError
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, TypeError, WaveLockOTSError) as error:
        print(f"WaveLock: {error}", file=sys.stderr)
        return 1


def miner_main():
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
