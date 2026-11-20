#!/usr/bin/env python3
"""Generate lean/WaveLockProofs/OTSConformance.lean from lean/ots_conformance_traces.json.

Each trace becomes one Lean theorem stating that the executable acceptance model of
lean/WaveLockProofs/OTSKeyConsumption.lean produces exactly the trace's expected outcome
(reported acceptances, final chain, final replay records, whether each file still loads).
The kernel checks it with `decide`. tests/test_ots_formal_conformance.py replays the same
traces against the Python implementation and checks the same expected outcome, and checks
that the committed Lean file is this script's output.

    python scripts/gen_ots_conformance.py --write
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "lean" / "ots_conformance_traces.json"
OUT = ROOT / "lean" / "WaveLockProofs" / "OTSConformance.lean"

#: Lean `WriteOutcome` for each outcome name used in the traces.
OUTCOMES = {
    "ok": ".ok",
    "fail_before_write": "(.fail .nothing)",
    "fail_after_write": "(.fail .line)",
    "crash_before_write": "(.crash .nothing)",
    "crash_torn": "(.crash .torn)",
    "crash_no_newline": "(.crash .noNewline)",
}

# Fixed string codes of RawKid (see OTSKeyConsumption.RawKid): "" -> 0, "None" -> 1.
EMPTY_CODE, NONE_CODE, FIRST_STRING_CODE = 0, 1, 100

#: Lean revision of the acceptance code each trace is evaluated against.
REVISIONS = {"b310ea5": ".b310ea5", "pr47": ".pr47", "current": ".current"}
#: Revisions that consume the commitment table (``ots_material_id``).
MATERIAL_REVISIONS = {"current"}


def load_traces() -> list[dict]:
    return json.loads(TRACES.read_text(encoding="utf-8"))["traces"]


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class Codes:
    """Injective naming of the strings, keys and identities of one trace."""

    def __init__(self, trace: dict):
        self.strings: dict[str, int] = {"": EMPTY_CODE, "None": NONE_CODE}
        self.arrays: dict[str, int] = {}
        self.keys: dict[str, int] = {}
        self.fps: dict[tuple[str, str], int] = {}
        self.hashes: dict[str, int] = {}
        for i, (label, spec) in enumerate(trace["blocks"].items()):
            self.hashes[label] = 1000 + i
            self.keys.setdefault(spec["key"], 500 + len(self.keys))
            kid = spec["kid"]
            if "str" in kid and kid["str"] not in self.strings:
                self.strings[kid["str"]] = FIRST_STRING_CODE + len(self.strings)
            if "arr" in kid:
                self.arrays.setdefault(_canon(kid["arr"]), len(self.arrays))
            # The fingerprint is a function of the key material, the key id and the ψ-commitment.
            self.fps.setdefault(self.fp_key(spec), 10000 + len(self.fps))

    @staticmethod
    def fp_key(spec: dict) -> tuple:
        return (spec["key"], _canon(spec["kid"]), spec.get("psi"), bool(spec.get("hex_case")))

    def kid_term(self, kid: dict) -> str:
        if "str" in kid:
            return f"(.str {self.strings[kid['str']]})"
        if "null" in kid:
            return ".null"
        if "num" in kid:
            return f"(.num {int(kid['num'])})"
        if "arr" in kid:
            return f"(.arr {self.arrays[_canon(kid['arr'])]})"
        raise ValueError(kid)

    def render(self, kid: dict) -> int:
        """RawKid.render, i.e. Python str(kid) as a code."""
        if "str" in kid:
            return self.strings[kid["str"]]
        if "null" in kid:
            return NONE_CODE
        if "num" in kid:
            return 1000000 + int(kid["num"])
        return 2000000 + self.arrays[_canon(kid["arr"])]


def lean_ident(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def block_term(codes: Codes, trace: dict, label: str) -> str:
    spec = trace["blocks"][label]
    parent = 0 if spec["parent"] is None else codes.hashes[spec["parent"]]
    fp = codes.fps[codes.fp_key(spec)]
    authentic = "false" if spec.get("tamper") else "true"
    return (f"⟨{spec['index']}, {parent}, {codes.hashes[label]}, {codes.kid_term(spec['kid'])}, "
            f"{fp}, {codes.keys[spec['key']]}, {authentic}⟩")


def event_terms(event: list) -> str:
    kind = event[0]
    if kind == "accept":
        return f"acceptAll {event[1]} b_{lean_ident(event[2])}"
    if kind == "accept_faults":
        return (f"acceptWith {event[1]} b_{lean_ident(event[2])} "
                f"{OUTCOMES[event[3]]} {OUTCOMES[event[4]]}")
    if kind == "pay":
        return f"[.payBegin {event[1]} b_{lean_ident(event[2])}, .payWrite {event[1]} .ok]"
    if kind == "crash":
        return f"[.crash {event[1]}]"
    if kind == "restart":
        return f"[.restart {event[1]}]"
    if kind == "delete_replay":
        return "[.deleteReplay]"
    raise ValueError(event)


def trace_section(trace: dict) -> str:
    codes = Codes(trace)
    name = lean_ident(trace["name"])
    rev = REVISIONS[trace["spec"]]
    ad = "true" if trace["allow_delete"] else "false"
    exp = trace["expected"]
    lines = [f"namespace {name}", ""]
    for label in trace["blocks"]:
        lines.append(f"def b_{lean_ident(label)} : CBlock := {block_term(codes, trace, label)}")
    lines.append("")
    lines.append("def trace : List (Event CBlock) :=")
    lines.append("  " + " ++\n  ".join(event_terms(e) for e in trace["events"]))
    lines.append("")

    def blocks(labels):
        return "[" + ", ".join(f"b_{lean_ident(x)}" for x in labels) + "]"

    def record(label):
        spec = trace["blocks"][label]
        atoms = [f".kid {codes.render(spec['kid'])}", f".fp {codes.fps[codes.fp_key(spec)]}"]
        if trace["spec"] in MATERIAL_REVISIONS:
            atoms.append(f".mat {codes.keys[spec['key']]}")
        return "[" + ", ".join(atoms) + "]"

    replay = "[" + ", ".join(record(x) for x in reversed(exp["replay"])) + "]"
    spec_term = f"(pySpec {rev})"
    lines += [
        f"/-- `{trace['name']}` (revision {trace['spec']}, side-file loss {'allowed' if trace['allow_delete'] else 'excluded'}). -/",
        f"theorem conformance :",
        f"    let r := runLog Discipline.implemented {spec_term} {ad} Sys.init trace",
        f"    r.2 = {blocks(exp['accepted'])} ∧",
        f"      r.1.disk.chain = {blocks(list(reversed(exp['chain'])))} ∧",
        f"      r.1.disk.replay = {replay} ∧",
        f"      {spec_term}.chainLoads r.1.disk = {str(exp['chain_loads']).lower()} ∧",
        f"      {spec_term}.replayLoads r.1.disk = {str(exp['replay_loads']).lower()} := by",
        "  decide",
        "",
        f"end {name}",
        "",
    ]
    return "\n".join(lines)


def render() -> str:
    traces = load_traces()
    header = f"""import WaveLockProofs.OTSKeyConsumption

/-!
# Lean/Python conformance traces for WaveLock-OTS acceptance (generated)

GENERATED by `scripts/gen_ots_conformance.py` from `lean/ots_conformance_traces.json`.
Do not edit by hand; `tests/test_ots_formal_conformance.py` fails if this file drifts.

Each `conformance` theorem evaluates the acceptance model of `OTSKeyConsumption` on one trace,
for the revision named in the trace, and checks the trace's expected outcome with `decide`.  The same {len(traces)} traces are replayed
against the Python implementation (real keys, signatures, blocks, replay ledger and chain files,
with failures injected at the modelled write boundaries) by the same test.  Agreement on these
finite traces is evidence that the model refines the implementation; it is not a proof of it.
-/

namespace WaveLockProofs
namespace OTS
namespace Conformance

"""
    body = "\n".join(trace_section(t) for t in traces)
    return header + body + "end Conformance\nend OTS\nend WaveLockProofs\n"


if __name__ == "__main__":
    text = render()
    if "--write" in sys.argv:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
    else:
        sys.stdout.write(text)
