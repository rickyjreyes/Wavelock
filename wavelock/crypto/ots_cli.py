"""
Command-line interface for WaveLock-OTS.

Subcommands (exposed as ``wavelock-ots <cmd>`` and also wired into the main
``wavelock-cli`` as ``ots-<cmd>``):

    ots-keygen   --out keys/
    ots-sign     --secret keys/wl_ots_secret.json --message "..." [--sig sig.json]
    ots-verify   --public keys/wl_ots_public.json --message "..." --sig sig.json
    ots-inspect  --public keys/wl_ots_public.json
    ots-mark-used --secret keys/wl_ots_secret.json

Design rules enforced here:
  * Public artifacts never contain ψ★/ψ₀/seed/raw slices.
  * One-time keys refuse a second signature unless --unsafe-allow-reuse.
  * Verification fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from wavelock.crypto.wavelock_ots import (
    SCHEME,
    OTSKeyReuseError,
    default_params,
    generate_ots_keypair,
    sign_ots,
    verify_ots,
    export_public_key,
    export_secret_key,
    load_public_key,
    load_secret_key,
)

PUBLIC_NAME = "wl_ots_public.json"
SECRET_NAME = "wl_ots_secret.json"


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def cmd_keygen(args) -> int:
    params = default_params(n=args.n)
    kp = generate_ots_keypair(params=params, entropy_bits=args.entropy_bits)

    out_dir = args.out
    pub_path = os.path.join(out_dir, PUBLIC_NAME)
    sec_path = os.path.join(out_dir, SECRET_NAME)

    _write_json(pub_path, export_public_key(kp["public_key"]))
    _write_json(
        sec_path,
        export_secret_key(
            kp["secret_key"],
            encrypt=args.encrypt,
            passphrase=args.passphrase,
            unsafe_export_secret_state=args.unsafe_export_secret_state,
        ),
    )
    # Lock down the secret file as best we can.
    try:
        os.chmod(sec_path, 0o600)
    except OSError:
        pass

    print(f"WaveLock-OTS keypair generated ({SCHEME})")
    print(f"  public: {pub_path}")
    print(f"  secret: {sec_path}  (one-time; keep private)")
    print(f"  one_time_key_id: {kp['public_key']['one_time_key_id']}")
    print(f"  entropy: {args.entropy_bits} bits")
    return 0


def cmd_sign(args) -> int:
    secret_key = load_secret_key(args.secret, passphrase=args.passphrase)

    if secret_key.get("used") and not args.unsafe_allow_reuse:
        print(
            "REFUSING TO SIGN: this WaveLock-OTS key has already been used.\n"
            "One-time keys MUST NOT be reused. Generate a fresh key with "
            "`wavelock-ots ots-keygen`.\n"
            "(For tests only you may pass --unsafe-allow-reuse.)",
            file=sys.stderr,
        )
        return 2

    if args.unsafe_allow_reuse:
        print(
            "\n*** WARNING: --unsafe-allow-reuse ***\n"
            "Signing again with a one-time key LEAKS unrevealed secret halves "
            "and enables forgery of other messages. Tests only.\n",
            file=sys.stderr,
        )

    try:
        sig = sign_ots(secret_key, args.message,
                       allow_reuse=args.unsafe_allow_reuse)
    except OTSKeyReuseError as e:
        print(f"REFUSING TO SIGN: {e}", file=sys.stderr)
        return 2

    sig_path = args.sig or "sig.json"
    _write_json(sig_path, sig)

    # Persist the used=True flag back to the secret file (file is the
    # authoritative one-time-usage record).
    if os.path.exists(args.secret):
        on_disk = load_secret_key(args.secret, passphrase=args.passphrase)
        on_disk["used"] = True
        # Preserve encrypted-at-rest form: re-export without re-encrypting the
        # already-stored seed by writing back what we read (seed_hex present).
        _write_json(args.secret, _reserialize_secret(args.secret, on_disk))

    print(f"Signed. Signature written to {sig_path}")
    print(f"  one_time_key_id: {sig.get('one_time_key_id')}")
    return 0


def _reserialize_secret(path: str, sk: dict) -> dict:
    """Write back the secret file preserving its original at-rest encoding."""
    with open(path, "r") as fh:
        original = json.load(fh)
    original["used"] = True
    return original


def cmd_verify(args) -> int:
    public_key = load_public_key(args.public)
    with open(args.sig, "r") as fh:
        signature = json.load(fh)

    ok = verify_ots(public_key, args.message, signature)
    if ok:
        print("VALID: signature verifies against the public key.")
        return 0
    print("INVALID: signature does NOT verify (fail-closed).", file=sys.stderr)
    return 1


def cmd_inspect(args) -> int:
    public_key = load_public_key(args.public)
    print(f"scheme:          {public_key.get('scheme')}")
    print(f"hash_alg:        {public_key.get('hash_alg')}")
    print(f"one_time_key_id: {public_key.get('one_time_key_id')}")
    print(f"created_at:      {public_key.get('created_at')}")
    print(f"params_hash:     {public_key.get('params_hash')}")
    print(f"psi_commitment:  {public_key.get('psi_commitment')}")
    print(f"merkle_root:     {public_key.get('merkle_root')}")
    pk = public_key.get("pk_commitments") or []
    print(f"pk_commitments:  {len(pk)} bit positions x 2 slices")
    params = public_key.get("params", {})
    print(f"kernel_version:  {params.get('kernel_version')}")
    print(f"n_bits:          {params.get('n_bits')}")
    # Confirm no secret leakage to the operator.
    leaked = [f for f in ("seed", "seed_hex", "psi_0", "psi_star")
              if f in public_key]
    print(f"secret leakage:  {'NONE' if not leaked else leaked}")
    return 0


def cmd_mark_used(args) -> int:
    with open(args.secret, "r") as fh:
        sk = json.load(fh)
    if sk.get("scheme") != SCHEME:
        print("Not a WaveLock-OTS secret key.", file=sys.stderr)
        return 1
    sk["used"] = True
    with open(args.secret, "w") as fh:
        json.dump(sk, fh, indent=2)
    print(f"Marked {args.secret} as used=True.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavelock-ots",
        description="WaveLock-OTS: experimental asymmetric one-time signatures.",
    )
    sub = parser.add_subparsers(dest="command")

    # Accept both "keygen" and "ots-keygen" style names.
    def add(name, help_):
        return sub.add_parser(name, help=help_), sub.add_parser("ots-" + name, help=help_)

    for p in add("keygen", "Generate a one-time WaveLock-OTS keypair"):
        p.add_argument("--out", default="keys/", help="output directory")
        p.add_argument("--n", type=int, default=4, help="ψ field size param")
        p.add_argument("--entropy-bits", dest="entropy_bits", type=int,
                       default=256, help="seed entropy (>=128)")
        p.add_argument("--encrypt", action="store_true",
                       help="encrypt the seed at rest under a passphrase")
        p.add_argument("--passphrase", default=None)
        p.add_argument("--unsafe-export-secret-state",
                       dest="unsafe_export_secret_state", action="store_true",
                       help="(discouraged) embed raw ψ★ in the secret export")
        p.set_defaults(func=cmd_keygen)

    for p in add("sign", "Sign a message with a one-time key"):
        p.add_argument("--secret", default=os.path.join("keys", SECRET_NAME))
        p.add_argument("--message", required=True)
        p.add_argument("--sig", default="sig.json")
        p.add_argument("--passphrase", default=None)
        p.add_argument("--unsafe-allow-reuse", dest="unsafe_allow_reuse",
                       action="store_true",
                       help="(tests only) allow reusing a one-time key")
        p.set_defaults(func=cmd_sign)

    for p in add("verify", "Verify a signature using only public material"):
        p.add_argument("--public", default=os.path.join("keys", PUBLIC_NAME))
        p.add_argument("--message", required=True)
        p.add_argument("--sig", required=True)
        p.set_defaults(func=cmd_verify)

    for p in add("inspect", "Inspect a public key"):
        p.add_argument("--public", default=os.path.join("keys", PUBLIC_NAME))
        p.set_defaults(func=cmd_inspect)

    for p in add("mark-used", "Mark a secret key as used"):
        p.add_argument("--secret", default=os.path.join("keys", SECRET_NAME))
        p.set_defaults(func=cmd_mark_used)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
