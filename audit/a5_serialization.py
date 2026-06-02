#!/usr/bin/env python3
"""
ATTACK 5 — Serialization ambiguity of Serialize(psi*).

Serialize (Wavelock_numpy._serialize_commitment) =
    schema + b"\\0" + canonical_json(header) + psi.tobytes('C') + pack('<4d', E)

We probe whether the serialization is canonical / injective:
  S1  -0.0 vs +0.0: a field equal-valued (psi==psi') but containing -0.0 where
      the other has +0.0 serializes to DIFFERENT bytes -> same logical field,
      two commitments (verification fragility / encoding non-canonicality).
  S2  NaN payloads: distinct NaN bit-patterns (quiet/signaling, payload bits)
      are all "NaN" yet serialize to different bytes; conversely two different
      NaN-laden fields can collide on the energy block. (Does the kernel ever
      emit NaN? a1 says no for finite seeds, but float results can.)
  S3  dtype/endianness/shape: header pins float64/C/shape, but the WLv3 path
      (WaveLock.py) uses big-endian '>f8' for the SAME field -> a field has TWO
      valid serializations (LE body in v2, BE body in v3) -> two commitments.
  S4  Energy-block redundancy: the packed '<4d' energies are a deterministic
      function of psi*, so they add zero second-preimage strength beyond psi*;
      but they are an extra fragile float channel (recomputed at verify time
      with a DIFFERENT op order than any external re-implementation).

Usage:  python audit/a5_serialization.py
Writes: audit/artifacts/a5_serialization.json
"""
from __future__ import annotations
import sys, os, json, hashlib, struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audit._wl as H
from wavelock.chain import Wavelock_numpy as wl


def C(ser):
    return hashlib.sha256(ser).hexdigest()


def main():
    out = {}
    psi = H.evolve(H.psi0_xof(42, 4))  # a real terminal field

    # S1: -0.0 vs +0.0
    f = psi.copy().ravel()
    f[0] = 0.0
    field_pos = f.reshape(psi.shape)
    f2 = f.copy(); f2[0] = -0.0
    field_neg = f2.reshape(psi.shape)
    ser_pos = wl._serialize_commitment(field_pos, wl.SCHEMA_V2)
    ser_neg = wl._serialize_commitment(field_neg, wl.SCHEMA_V2)
    out["S1_signed_zero"] = {
        "fields_numerically_equal": bool(np.array_equal(field_pos, field_neg)),
        "python_equal (0.0 == -0.0)": (0.0 == -0.0),
        "serialized_bytes_equal": ser_pos == ser_neg,
        "commitment_pos": C(ser_pos),
        "commitment_neg": C(ser_neg),
        "commitments_differ": C(ser_pos) != C(ser_neg),
        "verdict": ("NON-CANONICAL: +0.0 and -0.0 are the same number but yield "
                    "different commitments. Any reimplementation/platform that "
                    "produces -0.0 vs +0.0 in a cell breaks verification."),
    }

    # S2: distinct NaN payloads serialize differently; NaN != NaN at verify
    nan_q = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]
    nan_p = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]
    fa = psi.copy().ravel(); fa[0] = nan_q
    fb = psi.copy().ravel(); fb[0] = nan_p
    sa = wl._serialize_commitment(fa.reshape(psi.shape), wl.SCHEMA_V2)
    sb = wl._serialize_commitment(fb.reshape(psi.shape), wl.SCHEMA_V2)
    out["S2_nan_payloads"] = {
        "both_are_nan": bool(np.isnan(fa[0]) and np.isnan(fb[0])),
        "serialized_equal": sa == sb,
        "commitments_differ": C(sa) != C(sb),
        "verdict": ("Distinct NaN bit-patterns are all 'NaN' yet commit "
                    "differently; and NaN-valued cells make verify() "
                    "self-inconsistent since NaN != NaN. Kernel does not emit "
                    "NaN for finite seeds (a1), so severity is conditional."),
    }

    # S3: same field, two valid schema serializations (LE v2 body vs BE v3 body)
    ser_v2_body = psi.astype("<f8").tobytes(order="C")
    ser_v3_body = psi.astype(">f8").tobytes(order="C")
    out["S3_endianness_dual_schema"] = {
        "v2_body_sha256": hashlib.sha256(ser_v2_body).hexdigest(),
        "v3_body_sha256": hashlib.sha256(ser_v3_body).hexdigest(),
        "bodies_differ": ser_v2_body != ser_v3_body,
        "verdict": ("The SAME terminal field has two project-sanctioned "
                    "serializations (WLv2 little-endian tobytes vs WLv3 "
                    "big-endian '>f8'), hence two distinct commitments. "
                    "Commitment identity depends on schema label, not just psi*."),
    }

    # S4: energy block is a pure function of psi* (no added binding)
    E1 = wl._curvature_functional(psi)
    E2 = wl._curvature_functional(psi.copy())
    out["S4_energy_block_redundancy"] = {
        "energies_deterministic_in_psistar": tuple(E1) == tuple(E2),
        "packed_bytes_len": len(struct.pack("<4d", *E1)),
        "verdict": ("The 32-byte packed energy block is a deterministic "
                    "function of psi*, adding no second-preimage strength "
                    "beyond psi* itself, while adding 4 more float64 channels "
                    "that must match bit-exactly across implementations "
                    "(amplifying the a6 reproducibility problem)."),
    }

    # Can two DIFFERENT fields serialize identically? (injectivity check)
    # tobytes is injective on bit-patterns, so only NaN/sign-zero aliasing of
    # the *logical* value matters; raw bytes are 1:1 with the field bytes.
    out["injectivity"] = {
        "raw_tobytes_is_bijective_on_bitpatterns": True,
        "logical_value_aliasing": ["+0.0/-0.0 (S1)", "NaN payloads (S2)"],
        "note": "No two distinct bit-pattern fields share a serialization; the "
                "ambiguity is the reverse — one logical field, many byte images.",
    }

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "artifacts", "a5_serialization.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== a5 serialization ===")
    print("S1 -0.0 vs +0.0 commitments differ:", out["S1_signed_zero"]["commitments_differ"])
    print("S2 NaN payloads commitments differ:", out["S2_nan_payloads"]["commitments_differ"])
    print("S3 LE-vs-BE dual-schema bodies differ:", out["S3_endianness_dual_schema"]["bodies_differ"])
    print("S4 energy block redundant (no added binding):",
          out["S4_energy_block_redundancy"]["energies_deterministic_in_psistar"])


if __name__ == "__main__":
    main()
