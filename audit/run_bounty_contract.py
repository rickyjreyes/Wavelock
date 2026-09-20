"""Reproduce the thirteen bounty attempts; --check verifies checked-in evidence.

Run from the repository root after installing the package. Only public outcomes
are saved. OTS keys and replay state live in a temporary directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from wavelock.chain import consensus_commitment as cc

ROOT = Path(__file__).resolve().parent
SEED = bytes(range(32))  # Public test vector, never production secret material.


def rejected(call):
    try:
        call()
    except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
        return {"result": "reject", "error": str(error)}
    return {"result": "accepted"}


def attempt_records():
    records = []

    def record(slug, target, attempt, expected, observed, passed):
        records.append((slug, {"target": target, "attempt": attempt,
                               "expected": expected, "observed": observed,
                               "status": "PASS" if passed else "FAIL"}))

    for slug, seed in (("01_integer_42", 42), ("02_eight_byte_seed", bytes(8))):
        observed = rejected(lambda: cc.commit_consensus_state(seed))
        record(slug, "High 4", "seed=42" if seed == 42 else "8-byte production seed",
               "reject", observed, observed["result"] == "reject")

    for slug, name, value in (("03_nan", "NaN", np.nan),
                              ("04_positive_infinity", "+Inf", np.inf),
                              ("05_negative_infinity", "-Inf", -np.inf)):
        state = np.ones((4, 4))
        state[0, 0] = value
        observed = rejected(lambda: cc.canonical_serialize(state))
        record(slug, "High 2", name + " state injection", "reject", observed,
               observed["result"] == "reject")

    positive = np.zeros((4, 4))
    mixed = positive.copy()
    mixed.flat[::2] = -0.0
    a, b = cc.canonical_serialize(positive), cc.canonical_serialize(mixed)
    observed = {"canonical_bytes_equal": a == b,
                "commitments_equal": hashlib.sha256(a).digest() == hashlib.sha256(b).digest()}
    record("06_signed_zero", "High 3", "-0.0 versus +0.0", "equal canonical bytes and commitments",
           observed, all(observed.values()))

    state = np.arange(16, dtype=np.float64).reshape(4, 4)
    raw = cc.canonical_serialize(state)
    offset = 10 + struct.unpack(">I", raw[6:10])[0]
    substituted = raw[:offset] + state.astype("<f8").tobytes(order="C") + raw[offset+128:]
    observed = {"equivalent_internal_endian_normalized":
                raw == cc.canonical_serialize(state.astype(">f8")) == cc.canonical_serialize(state.astype("<f8")),
                "alternate_wire_endian": rejected(lambda: cc.validate_canonical_bytes(substituted))}
    record("07_native_endian", "High 1", "native-endian array/wire manipulation",
           "normalize equivalent arrays; reject substituted wire bytes", observed,
           observed["equivalent_internal_endian_normalized"] and observed["alternate_wire_endian"]["result"] == "reject")

    artifact = cc.commit_consensus_state(SEED).to_dict()
    artifact["metadata"]["kernel"]["steps"] = 49
    observed = {"accepted": cc.verify_consensus_commitment(artifact, SEED)}
    record("08_metadata_substitution", "Critical 4 / High 1", "substitute kernel step count",
           "reject", observed, not observed["accepted"])

    observed = rejected(lambda: cc.commit_consensus_state(SEED, backend="cupy"))
    record("09_unsupported_backend", "High 5", "CuPy emission labeled as consensus",
           "reject", observed, observed["result"] == "reject")

    observed = rejected(lambda: cc.commit_consensus_state(SEED, profile="WLv2"))
    record("10_alternate_profile", "High 1 / High 5", "alternate schema/profile WLv2",
           "reject", observed, observed["result"] == "reject")

    with tempfile.TemporaryDirectory(prefix="wavelock-bounty-") as temporary:
        old_state = os.environ.get("WAVELOCK_OTS_STATE_DIR")
        os.environ["WAVELOCK_OTS_STATE_DIR"] = str(Path(temporary) / "signer")
        try:
            from wavelock.chain.ots_blocks import build_signed_ots_block, verify_ots_block
            from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger
            from wavelock.crypto.wavelock_ots import generate_ots_keypair

            keys = generate_ots_keypair()
            block = build_signed_ots_block(keys["secret_key"], keys["public_key"],
                                           ["authorized body"], difficulty=0)
            auth = block.meta["ots_auth"]
            ledger_path = str(Path(temporary) / "replay.jsonl")
            ledger = PersistentOTSReplayLedger(ledger_path)
            first = ledger.accept(auth["public_key"], auth["message"], auth["signature"])
            replay = PersistentOTSReplayLedger(ledger_path).accept(
                auth["public_key"], auth["message"], auth["signature"])
            record("11_ots_replay", "Critical 8 / OTS replay", "replay after ledger restart",
                   "first accepted; replay rejected", {"first_accepted": first, "replay_accepted": replay},
                   first and not replay)

            changed = copy.deepcopy(block)
            changed.messages = ["unauthorized body"]
            changed.merkle_root = changed.calculate_merkle_root()
            changed.hash = changed.calculate_hash(changed.nonce)
            observed = {"accepted": verify_ots_block(changed)}
            record("12_ots_body_mutation", "Critical 8 / OTS binding", "mutate body and recompute Merkle root/header",
                   "reject", observed, not observed["accepted"])

            substitute = generate_ots_keypair()["public_key"]
            changed = copy.deepcopy(block)
            changed.meta["ots_auth"]["public_key"] = substitute
            changed.hash = changed.calculate_hash(changed.nonce)
            observed = {"accepted": verify_ots_block(changed)}
            record("13_public_key_substitution", "Critical 8 / OTS identity", "substitute public key and rehash",
                   "reject", observed, not observed["accepted"])
        finally:
            if old_state is None:
                os.environ.pop("WAVELOCK_OTS_STATE_DIR", None)
            else:
                os.environ["WAVELOCK_OTS_STATE_DIR"] = old_state
    return records


def matrix(records):
    outcomes = {name: value["status"] for name, value in records}
    commitment = "wavelock/chain/consensus_commitment.py:"
    tests = "tests/test_bounty_contract.py:"

    def entry(target, invariant, code, regression, dry_runs=(), *, status="PASS", limitation=None):
        if any(outcomes[name] != "PASS" for name in dry_runs):
            status = "FAIL"
        return {"target": target, "claimed_invariant": invariant, "enforcing_code": code,
                "regression_tests": regression, "dry_runs": list(dry_runs),
                "status": status, "limitation": limitation}

    rows = [
        entry("High 1", "One canonical encoding; alternate wire encodings rejected",
              [commitment + "canonical_serialize", commitment + "validate_canonical_bytes"],
              [tests + "test_high1_canonical_layout_and_dictionary_order", tests + "test_high1_alternate_wire_encoding_rejected"],
              ("07_native_endian", "08_metadata_substitution", "10_alternate_profile")),
        entry("High 2", "Nonfinite state, metadata and invariants rejected before emission",
              [commitment + "_state", commitment + "_float", commitment + "_wire"],
              [tests + "test_high2_every_state_position_rejected", tests + "test_high2_every_parameter_rejected",
               tests + "test_high2_every_invariant_rejected", tests + "test_high2_derived_overflow_and_producer_injection_rejected"],
              ("03_nan", "04_positive_infinity", "05_negative_infinity")),
        entry("High 3", "Signed zeros are one logical state and one canonical representation",
              [commitment + "_state", commitment + "_float"],
              [tests + "test_high3_signed_zero_in_state_and_invariants"], ("06_signed_zero",)),
        entry("High 4", "Production input is bytes of at least 16 bytes; reject demo/int/string modes",
              [commitment + "_validate_input"], [tests + "test_high4_production_seed_threshold",
              tests + "test_high4_accepted_inputs_and_pinned_reference_parity"],
              ("01_integer_42", "02_eight_byte_seed"), limitation="Length/type enforcement cannot measure actual secret entropy."),
        entry("High 5", "Only the registered NumPy reference path can emit the normative profile",
              [commitment + "commit_consensus_state", "wavelock/chain/Wavelock_numpy.py:_serialize_commitment"],
              [tests + "test_high5_unsupported_backend_or_profile", tests + "test_historical_modes_cannot_masquerade_as_consensus"],
              ("09_unsupported_backend", "10_alternate_profile")),
        entry("Critical 1", "Distinct high-entropy inputs should not collide under the declared profile",
              [commitment + "canonical_serialize", commitment + "_digest"],
              [tests + "test_high4_accepted_inputs_and_pinned_reference_parity"],
              limitation="PASS covers distinct pinned vectors and hash binding only, not a proof of collision resistance of the evolution."),
        entry("Critical 2", "No faster-than-brute-force preimage recovery of high-entropy input",
              [commitment + "_validate_input", commitment + "CommitmentArtifact.to_dict"],
              [tests + "test_high4_production_seed_threshold", tests + "test_commitment_artifact_publishing_is_public_and_exclusive"],
              limitation="PASS covers input/output boundaries only. No preimage complexity bound is established by regression tests."),
        entry("Critical 3", "Same input gives identical commitments in supported reference configurations",
              [commitment + "evolve_reference", commitment + "canonical_serialize"],
              [tests + "test_high4_accepted_inputs_and_pinned_reference_parity", ".github/workflows/container-ci.yml:core-tests"],
              limitation="Exact vectors exercise supported CI platforms; do not prove every NumPy build or every input deterministic."),
        entry("Critical 4", "Replay rejects wrong metadata, parameters, input, state or invariants",
              [commitment + "verify_consensus_commitment"],
              [tests + "test_every_metadata_field_is_bound", tests + "test_replay_rejects_wrong_input_state_invariants_and_extra_fields"],
              ("08_metadata_substitution",)),
        entry("Critical 5", "Deeper semantic binding break after documented canonicalization rules hold",
              [commitment + "validate_canonical_bytes", commitment + "verify_consensus_commitment", "audit/BOUNTY_SCOPE.md:Critical 5"],
              [tests + "test_high1_alternate_wire_encoding_rejected", tests + "test_replay_rejects_wrong_input_state_invariants_and_extra_fields"],
              limitation="PASS records the executable boundary and severity distinction, not a proof excluding new binding attacks."),
        entry("Critical 6", "Wrong runtime/kernel/config cannot spoof machine attestation",
              [], [], status="FAIL", limitation="No remote runtime attestation verifier exists. Recomputing a commitment does not attest the originating process."),
        entry("Critical 7", "Material behavior change cannot evade drift detection",
              [], [], status="FAIL", limitation="No behavior-wide drift oracle, observation contract, or detection verifier exists."),
        entry("Critical 8", "Ledger/record/invariant tampering always triggers verification failure",
              ["wavelock/chain/ots_blocks.py:verify_ots_chain", "wavelock/crypto/ots_ledger.py:PersistentOTSReplayLedger",
               "wavelock/network/server.py:try_accept_block"],
              ["tests/test_bounty_headers.py:test_remining_header_cannot_mutate_signed_context",
               "tests/test_bounty_headers.py:test_external_tip_detects_truncation_and_unanchored_limit_is_explicit",
               "tests/test_bounty_durability.py", "tests/test_ots_mythos_break.py"],
              ("11_ots_replay", "12_ots_body_mutation", "13_public_key_substitution"), status="FAIL",
              limitation="Signed body/header mutation and replay checks pass. Unanchored rollback, arbitrary valid-format cache edits and hostile local storage are not fully detected."),
    ]
    return {"schema": 1, "baseline": "v0.2.0", "baseline_commit": "9430442ae93e26c192d9be1f5ee9137369f5df37",
            "profile": cc.PROFILE, "full_bounty_ready": False,
            "status_meaning": "PASS means the stated executable boundary/regression checks pass; it is never a cryptographic security proof. FAIL identifies missing or incomplete enforcement.",
            "targets": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="rerun attempts and compare evidence without writing")
    args = parser.parse_args()
    records = attempt_records()
    failures = [name for name, result in records if result["status"] != "PASS"]
    files = {ROOT / "artifacts" / "bounty_dry_run" / (name + ".json"): result for name, result in records}
    files[ROOT / "artifacts" / "bounty_dry_run" / "INDEX.json"] = {
        "schema": 1, "profile": cc.PROFILE, "reproduce": "python audit/run_bounty_contract.py --check",
        "count": len(records), "passed": len(records) - len(failures), "failed": len(failures),
        "scope": "Thirteen enumerated easy submissions; broader unresolved targets are in bounty_contract_matrix.json.",
        "attempts": [{"file": name + ".json", "status": result["status"]} for name, result in records]}
    files[ROOT / "artifacts" / "bounty_contract_matrix.json"] = matrix(records)
    mismatches = []
    for path, content in files.items():
        if args.check:
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != content:
                mismatches.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(records), "passed": len(records)-len(failures),
                      "failures": failures, "evidence_mismatches": mismatches,
                      "full_bounty_ready": False}))
    return 1 if failures or mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
