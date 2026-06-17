"""Command-line interface for WaveLock-PDE-256-v0.

    python -m wavelock.pde_hash.cli "message"
    python -m wavelock.pde_hash.cli --stdin < file
    python -m wavelock.pde_hash.cli --impl optimized --hex 616263

No cryptographic primitive is used. This is an experimental research digest.
"""

from __future__ import annotations

import argparse
import sys

from . import spec
from . import reference, optimized


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="wavelock-pde-hash",
        description=f"{spec.VERSION}: experimental hash-free PDE digest (no security claim).",
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("message", nargs="?", help="message as a UTF-8 string")
    src.add_argument("--hex", dest="hexmsg", help="message as a hex string")
    src.add_argument("--stdin", action="store_true", help="read message bytes from stdin")
    ap.add_argument("--impl", choices=["reference", "optimized"], default="reference")
    ap.add_argument("--output-bits", type=int, default=spec.DEFAULT_OUTPUT_BITS)
    args = ap.parse_args(argv)

    if args.stdin:
        data = sys.stdin.buffer.read()
    elif args.hexmsg is not None:
        data = bytes.fromhex(args.hexmsg)
    elif args.message is not None:
        data = args.message.encode("utf-8")
    else:
        data = b""

    impl = optimized if args.impl == "optimized" else reference
    digest = impl.pde_hash(data, output_bits=args.output_bits)
    print(digest.hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
