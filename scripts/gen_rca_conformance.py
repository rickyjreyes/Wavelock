#!/usr/bin/env python3
"""Generate lean/WaveLockProofs/RCAConformance.lean from lean/rca_conformance_traces.json.

The traces exercise the PROPOSED reference kernel ``wavelock.rca_reference`` (issues #42 and #43)
and are evaluated against ``lean/WaveLockProofs/RCATransitions.lean`` (``RCA.runOut``).  Each trace
becomes one Lean theorem checked by ``decide``.  ``tests/test_rca_formal_conformance.py`` replays
the same traces against the Python code, checks the same expected outcome, and checks that the
committed Lean file is this script's output.

    python scripts/gen_rca_conformance.py --write
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "lean" / "rca_conformance_traces.json"
OUT = ROOT / "lean" / "WaveLockProofs" / "RCAConformance.lean"

WRITE_OUTCOMES = {
    "ok": ".ok",
    "fail_nothing": "(.fail .nothing)", "fail_line": "(.fail .line)", "fail_torn": "(.fail .torn)",
    "crash_nothing": "(.crash .nothing)", "crash_line": "(.crash .line)",
    "crash_torn": "(.crash .torn)",
}
DISCIPLINE_FIELDS = ("lock", "reload", "persist", "driftGate", "seqCheck", "monotonePolicy",
                     "bindCred")


def load_traces() -> dict:
    return json.loads(TRACES.read_text(encoding="utf-8"))


def lean_ident(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def num(label: str) -> int:
    """Trailing number of a label such as ``p0``, ``f1``, ``d0`` or ``c2``."""
    digits = "".join(c for c in label if c.isdigit())
    return int(digits)


def lean_bool(b: bool) -> str:
    return "true" if b else "false"


def lean_list(items) -> str:
    return "[" + ", ".join(items) + "]"


class Names:
    """Injective naming of the strings of one kernel trace."""

    def __init__(self):
        self.creds: dict[str, int] = {}

    def cred(self, label: str) -> int:
        return self.creds.setdefault(label, len(self.creds) + 1)


def policy_term(pol: dict) -> str:
    return (f"⟨{pol['version']}, {pol['threshold']}, "
            f"{lean_list(str(k) for k in pol['prohibited'])}⟩")


def ints(values) -> str:
    return lean_list(str(int(v)) for v in values)


def req_term(names: Names, req: list) -> str:
    kind = req[0]
    if kind == "observe":
        return f"(.observe {req[1]} {ints(req[2])})"
    if kind == "policy":
        return f"(.policy {policy_term(req[1])})"
    if kind == "execute":
        return f"(.execute {req[1]})"
    if kind == "reset":
        return f"(.reset {names.cred(req[1])})"
    raise ValueError(req)


def submit_terms(names: Names, p: str, req: list, outcome: str) -> list:
    """A request as Lean events.  A durable ``execute`` intent is followed by its ``effect`` (the
    effect runs in ``_submit`` right after the intent's ``fsync``), or by a crash before the effect
    (outcome ``ok_crash_before_effect``)."""
    if outcome == "ok_crash_before_effect":
        return [f".submit {num(p)} {req_term(names, req)} .ok", f".crash {num(p)}"]
    out = [f".submit {num(p)} {req_term(names, req)} {WRITE_OUTCOMES[outcome]}"]
    if req[0] == "execute" and outcome == "ok":
        out.append(f".effect {num(p)}")
    return out


def kernel_events(names: Names, trace: dict) -> list:
    out = []
    for ev in trace["events"]:
        kind = ev[0]
        if kind == "provision":
            out.append(f".provision {ev[1]} {ints(ev[2])} {policy_term(ev[3])}")
        elif kind in ("observe", "policy", "execute", "reset"):
            p, outcome = ev[1], ev[-1]
            out.append(f".begin {num(p)}")
            out += submit_terms(names, p, [kind, *ev[2:-1]], outcome)
        elif kind == "begin":
            out.append(f".begin {num(ev[1])}")
        elif kind == "submit":
            out += submit_terms(names, ev[1], ev[2], ev[3])
        elif kind == "issue":
            out.append(f".issue {names.cred(ev[1])}")
        elif kind == "anchor":
            out.append(".anchorNow")
        elif kind == "crash":
            out.append(f".crash {num(ev[1])}")
        elif kind == "restart":
            out.append(f".restart {num(ev[1])}")
        elif kind == "rollback":
            out.append(f".rollback {ev[1]}")
        elif kind == "direct":
            out.append(f".directEffect {ev[1]}")
        else:
            raise ValueError(ev)
    return out


def rec_term(names: Names, rec: list) -> str:
    if rec[0] == "genesis":
        return f"RCA.Rec.genesis {rec[1]} {ints(rec[2])} {policy_term(rec[3])}"
    return "RCA.Rec" + req_term(names, rec)[1:-1]


def discipline_term(disc) -> str:
    if disc == "reference":
        return "RCA.Discipline.reference"
    fields = ", ".join(f"{k} := {lean_bool(disc[k])}" for k in DISCIPLINE_FIELDS if k in disc)
    return f"({{ {fields} }} : RCA.Discipline)"


def kernel_section(trace: dict) -> str:
    names = Names()
    name = lean_ident(trace["name"])
    events = kernel_events(names, trace)
    exp = trace["expected"]
    log = lean_list(rec_term(names, r) for r in exp["log"])
    acked = lean_list(rec_term(names, r) for r in exp["acked"])
    effects = lean_list(f"({k}, {lean_bool(ok)})" for k, ok in exp["effects"])
    results = lean_list("none" if r is None else f"some {lean_bool(r)}" for r in exp["results"])
    disc = discipline_term(trace["discipline"])
    lines = [f"namespace {name}", "", "def trace : List RCA.Event :=",
             "  " + lean_list(events), "",
             f"/-- `{trace['name']}` (discipline: {json.dumps(trace['discipline'])}). -/",
             "theorem conformance :",
             f"    let r := RCA.runOut {disc} trace",
             f"    r.1.disk.log = {log} ∧",
             f"      r.1.disk.torn = {lean_bool(exp['torn'])} ∧",
             f"      (r.1.disk.view r.1.issued r.1.anchor).isSome = {lean_bool(exp['loads'])} ∧",
             "      r.1.effects.map (fun d => (d.kind, RCA.specPermits (RCA.credOK d.issued) d.log "
             f"d.kind)) = {effects} ∧",
             f"      r.1.acked = {acked} ∧",
             f"      r.2 = {results} := by",
             "  decide", "", f"end {name}", ""]
    return "\n".join(lines)


def render() -> str:
    kernel = load_traces()["kernel_traces"]
    header = f"""import WaveLockProofs.RCATransitions

/-!
# Lean/Python conformance traces for the RCA reference kernel (generated)

GENERATED by `scripts/gen_rca_conformance.py` from `lean/rca_conformance_traces.json`.
Do not edit by hand; `tests/test_rca_formal_conformance.py` fails if this file drifts.

{len(kernel)} traces of the PROPOSED reference kernel `wavelock.rca_reference` against
`RCATransitions` (`RCA.runOut`), replayed against `GuardedAgent` with write failures and crashes
injected (including a crash between a durable `execute` intent and its effect); traces of
non-reference disciplines replay an emulation of the ablated writer.  Each theorem states the
final durable memory, whether it loads, every effect with the specification's verdict on the
durable memory at that moment, the acknowledged records and every request's verdict, and is
checked by `decide`.  Agreement on finite traces is evidence that the model describes the code; it
is not a refinement proof.
-/

namespace WaveLockProofs
namespace RCAConformance

"""
    body = "\n".join(kernel_section(t) for t in kernel)
    return header + body + "end RCAConformance\nend WaveLockProofs\n"


if __name__ == "__main__":
    text = render()
    if "--write" in sys.argv:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
    else:
        sys.stdout.write(text)
