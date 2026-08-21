#!/usr/bin/env python3
"""phase-90.9 -- classify criterion SHAPE at FILING time.

    python3 scripts/qa/criteria_shape_90_9.py --verify          # the self-test (immutable cmd)
    python3 scripts/qa/criteria_shape_90_9.py --census          # both censuses + the rule
    python3 scripts/qa/criteria_shape_90_9.py --step-file X.json  # the filing-time gate

THE THESIS THIS TESTS
---------------------
The Q/A re-cycle loop is fuelled by what the criteria ASK FOR. If a large share of
phase-86..90 criteria grade the VERIFICATION APPARATUS rather than product
behaviour, and some demand an UNBOUNDED "every new guard", then each remediation
adds guards which the same criterion then demands be mutation-tested -- a fixed
point. Classifying criterion SHAPE at filing time attacks the fuel, not the fire.

IT GRADES NOTHING AND CHANGES NO VERDICT. It exits non-zero on ONE thing only:
unbounded scope, and only in --step-file mode.

TWO MODES, TWO EXIT-CODE MEANINGS -- do not conflate them:
  --step-file : the FILING-TIME GATE. exit 2 iff the proposed step carries an
                unbounded criterion; exit 0 otherwise. Classification alone NEVER
                fails. This is the mode criterion 3 speaks about.
  --verify    : the SELF-TEST. exit 1 if any internal check fails. It is a test of
                this script, not a gate on a step.

THE CORPUS IS PINNED, AND THAT IS THE POINT
-------------------------------------------
The filing figures reproduce TO THE DIGIT only at the commit that filed them.
Measured by execution:

    252090a3 (the filing commit)  155 steps / 980 criteria   <- filed: 155 / 980
    085c74e8 (+19 steps, 49 min later)  174 / 1045
    HEAD (2026-08-21)             157 / 988

The corpus has moved TWICE since filing and in BOTH directions: 085c74e8 added
steps 86.127-86.145, and those 19 ids have since left the 86..90 range entirely
(renumbered into phase-91). So criterion 1's "reproduces the filing figures by
execution on the live tree" is UNSATISFIABLE, and taking its escape hatch --
"correct the RULE" -- would corrupt a correct rule to chase a moved corpus. The
classifier takes a git-rev pin instead (house idiom:
scripts/qa/replay_changelog_rule_86_68.py:34,
scripts/qa/sweep_absent_verification_paths.py:421) and prints BOTH censuses.
The numbers are never edited to match.

WHAT DOES *NOT* REPRODUCE, STATED PLAINLY
-----------------------------------------
The filing's APPARATUS figure -- 403 / 980 = 41.1% -- is NOT reproducible,
because the filing never recorded the rule that produced it. It is absent from
the masterplan entry AND from research_brief_90.9.md, which records the
step-inclusion rule and the unbounded-count regexes but not this one. So this
script AUTHORS a rule, PRINTS it in full, and reports its own figure beside the
filed one. Criterion 1's "the RULE is corrected and the new figure printed" is
the clause that governs, and the filed number is not edited.

The sensitivity table below is why that matters: across four defensible rules the
apparatus LEVEL ranges 16.7% -> 65.2%, while the RATIO (phase-86..90 vs
project-wide) stays in 1.65x-2.21x. The level is a property of the rule; the
ratio is a property of the corpus. The filed "1.6x-1.9x" range was hiding rule
sensitivity, not corpus ambiguity.

CRITERION 7 -- THE INPUT SURFACE
--------------------------------
This classifier is never given, and never READS, a step's verdict history, round
index or remaining attempt budget. "Never reads" is the binding verb, not "never
given": the live consequence channel on the sibling Q/A rail is a SELF-read
(qa_wip.py --spawned-at), not a caller hand-off -- qa-verdict.js's prompt never
renders verdict_sequence or attempt_number at all. So the test covers the
self-read path too: the source must contain no reference to the ledger, the WIP
records, or the budget module.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MASTERPLAN = REPO / ".claude" / "masterplan.json"
VERDICT_LEDGER = REPO / "handoff" / "verdict_ledger.jsonl"

FILING_REV = "252090a3"
DRIFT_REV = "085c74e8"

# The filed figures, quoted so the comparison is visible. NEVER edited to match.
FILED = {
    "steps": 155, "criteria": 980, "apparatus": 403, "pct": 41.1,
    "terminal_apparatus": 78, "project_apparatus": 1026, "project_criteria": 4670,
    "project_pct": 22.0, "ratio_range": (1.6, 1.9), "unbounded": 44,
}

# ---------------------------------------------------------------------------
# THE CLASSIFICATION RULE -- printed beside every output, because no external
# standard carries a product/apparatus axis and the printed rule is therefore the
# entire warrant. Written to be read and disputed.
# ---------------------------------------------------------------------------

RULE_NAME = "MID"

RULE_PROSE = """\
A criterion is EVIDENCE_APPARATUS when satisfying it requires producing or
exercising VERIFICATION MACHINERY -- a mutant, a control, a hash comparison, a
re-runnable checker, a fixture, a captured artifact, an exit code, or a test that
asserts something. It is PRODUCT_BEHAVIOUR when it constrains what the SYSTEM
DOES. Criteria are compound in this codebase, so ANY apparatus demand classifies
the whole criterion as apparatus: the question is whether the criterion adds
verification work, not whether that is all it does.

This rule is AUTHORED, not adopted: research_brief_90.9.md K6 records that the
published requirements-smell taxonomies carry no product/apparatus axis, so
citing one as authority would be false. Dispute the term list below, not a
citation."""

RULE_TERMS_APPARATUS = [
    # (a) mutation outcomes
    r"\bmutant", r"\bmutation", r"\bKILLED\b", r"\bSURVIVED\b",
    # (b) control discipline
    r"control .{0,20}GREEN", r"\bred-first\b", r"\bvacuous",
    # (c) hash / byte identity
    r"\bsha256\b", r"byte-identical",
    # (d) re-runnable machinery
    r"\bre-runnable\b", r"\bchecker\b", r"\bself-test\b", r"\bcardinality floor\b",
    r"\bfixture", r"\bcell\b", r"\bregression\b", r"\bdry.run\b",
    # (e) captured artifacts as proof
    r"\bverbatim\b", r"\bprinted\b", r"\bcaptured?\b",
    # (f) exit codes
    r"\bexits? non-zero\b", r"\bexits? 0\b", r"\bexit code\b",
    # (g) the test/assert/proof family -- the swing term; see the sensitivity table
    r"\btests?\b", r"\basserted\b", r"\bassertion\b", r"\bproven\b", r"\bproof\b",
]

# Named variants, so the reader can see how much of the LEVEL is the rule's doing.
RULE_VARIANTS = {
    "NARROW": RULE_TERMS_APPARATUS[:9],
    "HOUSE": RULE_TERMS_APPARATUS[:22],
    "MID": RULE_TERMS_APPARATUS,
    "BROAD": RULE_TERMS_APPARATUS + [
        r"\bguard\b", r"\bmeasured\b", r"\breproduce", r"\bevidence\b",
        r"\bcommand\b", r"\bcontrol\b", r"\bcoverage\b", r"\blint\b", r"\bscript\b",
    ],
}

_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in RULE_VARIANTS.items()}


def classify(criterion: str, variant: str = RULE_NAME) -> str:
    """EVIDENCE_APPARATUS | PRODUCT_BEHAVIOUR. Pure; reads only the text given."""
    for rx in _COMPILED[variant]:
        if rx.search(criterion):
            return "EVIDENCE_APPARATUS"
    return "PRODUCT_BEHAVIOUR"


# ---------------------------------------------------------------------------
# THE UNBOUNDED-SCOPE DETECTOR -- the quantified NOUN CLASS, not a keyword
# ---------------------------------------------------------------------------
#
# research_brief_90.9.md I2 measured four quantifier variants at the filing pin.
# v4 reproduces the filed 44 EXACTLY -- and v3, the only variant that actually
# tests self-reference, returns 0 of 155, because the self-reference is carried by
# the word "new" plus the surrounding sentence, never by an explicit "this step
# adds". So 44 is the right number reached from the WRONG property, and shipping
# v4 while printing "44, reproduced" would be passing criterion 1 while measuring
# something else.
#
# THE PROPERTY, stated: a criterion is UNBOUNDED when a universal quantifier
# governs an artifact class THE STEP ITSELF PRODUCES. "mutation-test every new
# guard this step adds" is unbounded because the step decides how many guards
# exist, so the criterion's satisfaction condition GROWS WITH THE WORK -- that is
# the fixed point. "resolve every attempt row" is BOUNDED: the ledger has a fixed
# number of rows the step does not create.
#
# A numeral in the same clause also bounds it ("all 155 steps", "at least 20
# returns") -- an enumerated population is finite by construction.

UNIVERSAL = r"\b(every|all|each|any)\b"

# Artifact classes a STEP PRODUCES. Quantifying over these is what grows.
STEP_PRODUCED = (
    r"(guard|guards|mutation cell|mutation cells|cell|cells|probe|probes|fixture|"
    r"fixtures|assertion|assertions|invariant|invariants|mutant|mutants|"
    r"new test|new tests|check|checks|artifact|artifacts)"
)
# Populations that EXIST INDEPENDENTLY of the step. Quantifying over these is
# bounded by the corpus, not by the work.
CORPUS_POPULATION = (
    r"(row|rows|step|steps|record|records|criterion|criteria|file|files|commit|"
    r"commits|run|runs|call site|call sites|path|paths|entry|entries|line|lines|"
    r"field|fields|key|keys|return|returns|ticker|tickers|value|values)"
)

_UNB = re.compile(UNIVERSAL + r"[^.;]{0,80}?" + STEP_PRODUCED, re.I)
_CORP = re.compile(UNIVERSAL + r"[^.;]{0,40}?" + CORPUS_POPULATION, re.I)
_NUMERAL = re.compile(r"\b\d[\d,]*\b")
# The brief's v4 keyword proxy, kept ONLY so its count can be printed beside the
# property-based one. It is never the shipped detector.
_V4_PROXY = re.compile(
    r"\b(every|all)\b[^.]{0,80}(guard|mutation cell|probe|fixture|artifact|new test)", re.I)


def unbounded_reason(criterion: str) -> str | None:
    """Why this criterion is unbounded, or None. The clause is what is examined."""
    for clause in re.split(r"[.;]", criterion):
        m = _UNB.search(clause)
        if not m:
            continue
        if _NUMERAL.search(clause):
            continue  # an enumerated population is finite by construction
        cm = _CORP.search(clause)
        # COMPARE THE NOUN POSITIONS, NOT THE MATCH STARTS. Both patterns anchor on
        # the SAME universal quantifier, so their match starts are usually
        # identical and a `cm.start() < m.start()` test never discriminates --
        # "every attempt row is covered by a guard" was flagged unbounded because
        # 0 < 0 is false. Whichever NOUN sits closer to the quantifier is the one
        # it governs. Caught by the fixture written to make this branch
        # load-bearing; the earlier fixtures never reached it.
        if cm and cm.start(2) < m.start(2):
            continue  # the quantifier governs a corpus population, not the artifact
        return (f"a universal quantifier ({m.group(1)}) governs {m.group(2)!r}, an "
                f"artifact class the step itself PRODUCES, with no numeric bound in "
                f"the clause -- so satisfying the criterion grows with the work")
    return None


def is_unbounded(criterion: str) -> bool:
    return unbounded_reason(criterion) is not None


# ---------------------------------------------------------------------------
# Corpus loading. READ-ONLY BY CONSTRUCTION: git show writes nothing, and this
# module contains no write path to the plan of record (asserted below at AST
# level, because the house's dominant write idiom is Path.write_text and a
# two-literal grep for open(...,'w')/json.dump misses it entirely).
# ---------------------------------------------------------------------------

def load_plan(rev: str | None) -> dict:
    if rev is None:
        return json.loads(MASTERPLAN.read_text(encoding="utf-8"))
    out = subprocess.run(["git", "show", f"{rev}:.claude/masterplan.json"],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git show {rev}:.claude/masterplan.json failed: {out.stderr[:300]}")
    return json.loads(out.stdout)


INCLUSION_RULE = (
    "a node carrying an `id`, a dict `verification`, and a non-empty "
    "`verification.success_criteria`; step 90.9 itself EXCLUDED (its own criteria "
    "are the whole delta that made the filed figures look irreproducible)"
)


def collect_steps(plan: dict, exclude: str = "90.9") -> list[dict]:
    out: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if ("id" in node and isinstance(node.get("verification"), dict)
                    and node["verification"].get("success_criteria")):
                out.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(plan)
    return [s for s in out if str(s.get("id")) != exclude]


_PHASE_RANGE = re.compile(r"^(8[6-9]|90)\b")


def in_range(step_id) -> bool:
    return bool(_PHASE_RANGE.match(str(step_id)))


def census(plan: dict, variant: str = RULE_NAME) -> dict:
    steps = collect_steps(plan)
    sel = [s for s in steps if in_range(s["id"])]
    crits = [c for s in sel for c in s["verification"]["success_criteria"]]
    allc = [c for s in steps for c in s["verification"]["success_criteria"]]
    app = [c for c in crits if classify(c, variant) == "EVIDENCE_APPARATUS"]
    app_all = [c for c in allc if classify(c, variant) == "EVIDENCE_APPARATUS"]
    terminal = [s["verification"]["success_criteria"][-1] for s in sel]
    term_app = [c for c in terminal if classify(c, variant) == "EVIDENCE_APPARATUS"]
    unb_steps = [s for s in sel
                 if any(is_unbounded(c) for c in s["verification"]["success_criteria"])]
    v4_steps = [s for s in sel
                if any(_V4_PROXY.search(c) for c in s["verification"]["success_criteria"])]
    pct = 100.0 * len(app) / len(crits) if crits else 0.0
    pct_all = 100.0 * len(app_all) / len(allc) if allc else 0.0
    return {
        "steps": len(sel), "criteria": len(crits),
        "apparatus": len(app), "pct": pct,
        "terminal_apparatus": len(term_app), "terminal_total": len(terminal),
        "project_apparatus": len(app_all), "project_criteria": len(allc),
        "project_pct": pct_all,
        "ratio": (pct / pct_all) if pct_all else 0.0,
        "unbounded_steps": [str(s["id"]) for s in unb_steps],
        "v4_proxy_steps": [str(s["id"]) for s in v4_steps],
    }


def _fmt(label: str, c: dict) -> str:
    return (f"  {label:<34} steps={c['steps']:<5} criteria={c['criteria']:<6} "
            f"apparatus={c['apparatus']:<5} {c['pct']:5.1f}%   "
            f"terminal={c['terminal_apparatus']}/{c['terminal_total']}   "
            f"project={c['project_apparatus']}/{c['project_criteria']} "
            f"={c['project_pct']:4.1f}%   ratio={c['ratio']:.2f}x")


def print_census() -> None:
    print("=" * 78)
    print("THE CLASSIFICATION RULE (printed beside its output -- criterion 1)")
    print("=" * 78)
    print(RULE_PROSE)
    print(f"\n  variant SHIPPED: {RULE_NAME}")
    print(f"  step-inclusion rule: {INCLUSION_RULE}")
    print("  apparatus terms:")
    for t in RULE_VARIANTS[RULE_NAME]:
        print(f"    {t}")

    print("\n" + "=" * 78)
    print("CENSUS -- BOTH CORPORA PRINTED, NEITHER EDITED (criterion 1)")
    print("=" * 78)
    pinned = census(load_plan(FILING_REV))
    drift = census(load_plan(DRIFT_REV))
    live = census(load_plan(None))
    print(_fmt(f"PINNED {FILING_REV} (filing)", pinned))
    print(_fmt(f"DRIFT  {DRIFT_REV} (+49 min)", drift))
    print(_fmt("LIVE tree (today)", live))
    print(f"\n  FILED:  steps=155  criteria=980  apparatus=403  41.1%  terminal=78  "
          f"project=1026/4670 =22.0%  ratio 1.6x-1.9x  unbounded=44")
    print(f"""
  WHAT REPRODUCES, AND WHAT DOES NOT:
    steps / criteria  -- REPRODUCE EXACTLY at {FILING_REV}: {pinned['steps']} / {pinned['criteria']}
                         vs filed 155 / 980. The step-inclusion rule is recovered
                         and is printed above.
    apparatus / pct   -- DO NOT REPRODUCE: {pinned['apparatus']} / {pinned['pct']:.1f}% vs filed 403 / 41.1%.
                         The filing never recorded the rule that produced 403 --
                         it is absent from the masterplan entry AND from
                         research_brief_90.9.md. It is therefore not recoverable,
                         not merely unmatched. This script's rule is printed above
                         and its figure stands beside the filed one. Criterion 1's
                         "the RULE is corrected and the new figure printed" is the
                         clause that governs. THE FILED NUMBER IS NOT EDITED.
    corpus drift      -- the corpus moved TWICE and in BOTH directions:
                         {FILING_REV} {pinned['steps']} -> {DRIFT_REV} {drift['steps']} (+19, steps 86.127-86.145,
                         49 minutes after filing) -> live {live['steps']}, because those same
                         19 ids have since left the 86..90 range entirely. A
                         "live tree" criterion cannot bind on a corpus that moves
                         faster than the step does.""")

    print("\n" + "=" * 78)
    print("SENSITIVITY -- the LEVEL is the rule's doing; the RATIO is the corpus's")
    print("=" * 78)
    print("  (all at the filing pin, same inclusion rule, only the term list varies)")
    plan = load_plan(FILING_REV)
    ratios = []
    for name in ("NARROW", "HOUSE", "MID", "BROAD"):
        c = census(plan, name)
        ratios.append(c["ratio"])
        mark = "  <- SHIPPED" if name == RULE_NAME else ""
        print(f"    {name:<7} apparatus={c['apparatus']:<4} {c['pct']:5.1f}%   "
              f"project={c['project_pct']:4.1f}%   ratio={c['ratio']:.2f}x{mark}")
    print(f"""
  The apparatus LEVEL spans {min(census(plan, n)['pct'] for n in RULE_VARIANTS):.1f}% -> {max(census(plan, n)['pct'] for n in RULE_VARIANTS):.1f}% across four defensible rules.
  The RATIO spans only {min(ratios):.2f}x -> {max(ratios):.2f}x, and the shipped rule's {census(plan, RULE_NAME)['ratio']:.2f}x sits inside
  the filed 1.6x-1.9x range. So: "phase-86..90 criteria are ~1.7x more
  apparatus-heavy than the project average" SURVIVES the choice of rule.
  "41.1% of them are apparatus" DOES NOT -- that number is a property of a rule
  nobody wrote down. Criterion 1 asks for the range to be collapsed "only by
  fixing the inclusion rule": fixing the inclusion rule and the corpus pin
  collapses it to ONE value PER RULE ({census(plan, RULE_NAME)['ratio']:.2f}x for the shipped one). What the filed
  range was hiding is rule sensitivity, not corpus ambiguity.""")

    print("\n" + "=" * 78)
    print("UNBOUNDED SCOPE -- the quantified NOUN CLASS, not a keyword")
    print("=" * 78)
    print(f"  property-based detector : {len(pinned['unbounded_steps'])} steps at the filing pin")
    print(f"  brief's v4 keyword proxy: {len(pinned['v4_proxy_steps'])} steps (filed: {FILED['unbounded']})")
    only_prop = sorted(set(pinned["unbounded_steps"]) - set(pinned["v4_proxy_steps"]))
    only_v4 = sorted(set(pinned["v4_proxy_steps"]) - set(pinned["unbounded_steps"]))
    print(f"  property-only: {len(only_prop)} {only_prop[:12]}")
    print(f"  proxy-only:    {len(only_v4)} {only_v4[:12]}")
    print("""
  WHY THE PROXY IS NOT SHIPPED even though it reproduces 44 exactly: the brief
  measured that its literal self-reference variant returns 0 of 155 -- the
  self-reference is carried by the word "new" plus the surrounding sentence, never
  by an explicit "this step adds". A detector that lands on the right number from
  the wrong property would pass criterion 1 while measuring something else, which
  is the house rule "assert the property, not a proxy". The proxy's count is
  printed here so the difference is auditable, and it is never the gate.""")

    print("\n" + "=" * 78)
    print("THE BOUND, JUSTIFIED AGAINST THE RECORD (criterion 6)")
    print("=" * 78)
    print(f"""  PROPOSED BOUND: replace "mutation-test every new guard this step adds" with
  "mutation-test each guard a NUMBERED criterion of this step names, plus a null
  control and a real-kill control".

  How many it would have flagged: {len(pinned['unbounded_steps'])} steps at the filing pin carry a criterion
  the property-based detector calls unbounded ({len(pinned['v4_proxy_steps'])} under the brief's proxy, which
  is the filed 44). Every one of them would have had its terminal criterion
  rewritten at filing time.

  THE MOST SERIOUS REAL HISTORICAL FINDING THE BOUND WOULD HAVE DEFERRED --
  and it is not hypothetical, it happened on 2026-08-21:

    Step 90.1's cycle-5 Q/A FAILED the step by authoring three mutation cells
    (QA1/QA1b/QA1c) that rename a CALL SITE inside attempt_gate.handle_hook. They
    proved the matrix's ERROR discriminator VACUOUS: it requires a literal
    "Traceback (most recent call last)", while the production fail-open handler at
    attempt_gate.py:465 catches Exception and prints a one-line INTERNAL ERROR with
    no traceback. QA1b defeats NO guard yet fails 7 of 25 checks -- three of them
    belonging to criteria 2, 3 and 4 -- so a build that never runs was
    green-washing three criteria at once.

    NO NUMBERED CRITERION OF 90.1 NAMED THOSE CELLS. They were the evaluator's own
    probes of the discriminator. Under the proposed bound they would never have
    been written, and the vacuous guard would have shipped green.

  DISPOSITION, FILED RATHER THAN DROPPED: masterplan step 90.12, with its own
  immutable verification command. The bound is therefore NOT recommended as
  written: it must carry an explicit carve-out for cells the EVALUATOR authors,
  because the evaluator is the one party whose probes are not scoped by the
  step's own criteria. That carve-out is the difference between bounding the
  fuel and bounding the fire.""")


# ---------------------------------------------------------------------------
# Criterion 4 -- write-capability at AST level
# ---------------------------------------------------------------------------

WRITE_CALLS = {
    "write_text", "write_bytes", "writelines", "dump", "replace", "rename",
    "unlink", "copy", "copy2", "copyfile", "move", "rmtree", "mkdir", "touch",
}


def write_capable_calls(src: str) -> list[str]:
    """AST-resolved write-capable calls. Strictly stronger than criterion 4's
    literal two-pattern list: the house's dominant idiom is Path.write_text (148
    sites in scripts/qa/*.py), which `open(...,'w')` and `json.dump` both miss.
    The gap is stated so the strengthening is visible rather than silent."""
    found = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None)
        if name in WRITE_CALLS:
            found.append(f"{name} at line {node.lineno}")
        elif name == "open":
            for a in list(node.args[1:2]) + [k.value for k in node.keywords if k.arg == "mode"]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and any(m in a.value for m in "wax+"):
                    found.append(f"open(mode={a.value!r}) at line {node.lineno}")
    return found


# Criterion 7 -- the input surface. "never READS" is the binding verb, because the
# live consequence channel on the sibling Q/A rail is a SELF-read (qa_wip.py
# --spawned-at), not a caller hand-off: qa-verdict.js's prompt never renders
# verdict_sequence or attempt_number at all.
#
# THE FIRST VERSION OF THIS CHECK WAS A TEXT GREP OVER THE WHOLE SOURCE, AND IT
# FAILED AGAINST ITSELF -- the list below IS a set of those strings, so the probe
# matched its own definition. A probe that cannot distinguish its own source from
# its subject measures nothing. It is replaced by an AST scan SCOPED TO THE
# CLASSIFICATION FUNCTIONS, which is the property the criterion is actually about:
# the classifier must not read consequence, whether handed to it or fetched by it.
#
# DISCLOSED, because a whole-module ban would be a false claim: the SELF-TEST does
# read handoff/verdict_ledger.jsonl -- to hash it, twice, and prove it did not
# change. Criterion 5 REQUIRES exactly that. Hashing a file to prove it is
# unchanged is not reading verdict history as classification input, and the scan
# below is scoped so the distinction is enforced rather than asserted.
FORBIDDEN_INPUTS = [
    "verdict_ledger", "verdict_history", "qa_wip", "attempt_budget",
    "attempt_gate", "agent-memory", "verdict_wip", "harness_log",
    "consecutive_conditional", "retry_count", "attempt_number", "round_index",
    "prior_attempts", "cycle", "escalation",
]

# Every function that participates in producing a label or an exit code.
CLASSIFIER_FNS = [
    "classify", "unbounded_reason", "is_unbounded", "census", "gate_step",
    "collect_steps", "in_range", "load_plan",
]

_IO_CALLS = {"open", "read_text", "read_bytes", "run", "check_output", "getenv", "system"}


def classifier_consequence_refs(src: str, fns=None) -> list[str]:
    """Forbidden-consequence references reachable from the classification path.

    Walks ONLY the named function bodies, and looks at what the code actually
    does -- string constants, attribute names, identifiers -- rather than at the
    file's text. `load_plan` is included deliberately: it is the only classifier
    function that performs I/O, so if consequence ever enters, it enters there.
    """
    fns = CLASSIFIER_FNS if fns is None else fns
    tree = ast.parse(src)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in fns:
            continue
        for sub in ast.walk(node):
            words: list[str] = []
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                words.append(sub.value)
            elif isinstance(sub, ast.Attribute):
                words.append(sub.attr)
            elif isinstance(sub, ast.Name):
                words.append(sub.id)
            for w in words:
                low = w.lower()
                for t in FORBIDDEN_INPUTS:
                    if t.lower() in low:
                        hits.append(f"{node.name}: {t!r} in {w[:60]!r} (line {sub.lineno})")
    return hits


def classifier_io_calls(src: str, fns=None) -> list[str]:
    """Every I/O-shaped call reachable from the classification path, named so the
    reader can see the whole surface rather than trust a claim that it is small."""
    fns = CLASSIFIER_FNS if fns is None else fns
    tree = ast.parse(src)
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in fns:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else None)
                if nm in _IO_CALLS:
                    out.append(f"{node.name}: {nm}() at line {sub.lineno}")
    return out


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "ABSENT"


# ---------------------------------------------------------------------------
# --step-file : the FILING-TIME GATE
# ---------------------------------------------------------------------------

def gate_step(step: dict) -> int:
    crits = (step.get("verification") or {}).get("success_criteria") or []
    print(f"step {step.get('id')!r}: {len(crits)} criteria")
    print(f"  rule: {RULE_NAME}  (see --census for the full rule)")
    unbounded = []
    for i, c in enumerate(crits, 1):
        shape = classify(c)
        why = unbounded_reason(c)
        flag = "  <-- UNBOUNDED SCOPE" if why else ""
        print(f"  {i}. {shape}{flag}")
        if why:
            unbounded.append((i, why))
            print(f"       {why}")
    if unbounded:
        print(f"\nEXIT 2 -- {len(unbounded)} criterion(a) carry unbounded scope. "
              "This is the ONLY condition on which this tool fails a step; "
              "classification alone never fails.")
        return 2
    print("\nEXIT 0 -- no unbounded criterion. Shape is reported, never graded.")
    return 0


# ---------------------------------------------------------------------------
# --verify : the SELF-TEST
# ---------------------------------------------------------------------------

CONTROL_ALL_PRODUCT = {
    "id": "CTRL.PRODUCT",
    "verification": {"success_criteria": [
        "the endpoint returns a sector_breakdown whose weights sum to 1.0 for a "
        "portfolio holding three positions",
        "a signal emitted for a delisted ticker is suppressed before it reaches "
        "the paper broker",
        "the kill switch halts new order submission within one scheduler tick of "
        "being flipped",
    ]},
}

CONTROL_UNBOUNDED = {
    "id": "CTRL.UNBOUNDED",
    "verification": {"success_criteria": [
        "the resolver returns a normalised ticker for a dual-listed symbol",
        "mutation-test every new guard this step adds, including reverting this "
        "step's own fix",
    ]},
}


def verify() -> int:
    fails = []

    def ck(label, ok, detail=""):
        fails.append(label) if not ok else None
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    mp_before = sha256(MASTERPLAN)
    vl_before = sha256(VERDICT_LEDGER)

    print("=" * 78)
    print("A. CONTROLS OBSERVED FIRST (criterion 2)")
    print("=" * 78)
    cp = [classify(c) for c in CONTROL_ALL_PRODUCT["verification"]["success_criteria"]]
    ck("the all-product-behaviour control scores 0% evidence-class",
       cp.count("EVIDENCE_APPARATUS") == 0, f"{cp}")
    ck("...and the gate exits 0 on it", gate_step(CONTROL_ALL_PRODUCT) == 0)
    cu = CONTROL_UNBOUNDED["verification"]["success_criteria"]
    ck("the control whose terminal criterion says 'mutation-test every new guard "
       "this step adds' is flagged unbounded", is_unbounded(cu[-1]),
       str(unbounded_reason(cu[-1]))[:90])
    ck("...and its FIRST criterion is NOT flagged, so the detector discriminates "
       "within one step", not is_unbounded(cu[0]))
    ck("...and the gate exits 2 on it", gate_step(CONTROL_UNBOUNDED) == 2)

    # Without this the classifier could return a single constant label and every
    # control above would still pass -- the all-product control only proves it does
    # not OVER-classify.
    ck("classify DISCRIMINATES: a criterion demanding a mutant + a GREEN control is "
       "EVIDENCE_APPARATUS", classify(
           "mutation-tested with the control observed GREEN first: a mutant that "
           "fails to run scores ERROR") == "EVIDENCE_APPARATUS")
    ck("...while a pure behaviour statement is PRODUCT_BEHAVIOUR",
       classify("the kill switch halts new order submission within one scheduler "
                "tick of being flipped") == "PRODUCT_BEHAVIOUR")

    print("\n" + "=" * 78)
    print("B. THE DETECTOR DISCRIMINATES BOUNDED FROM UNBOUNDED")
    print("=" * 78)
    # THE SHAPE OF THESE FIXTURES IS THE POINT, AND THE FIRST VERSION GOT IT WRONG.
    # The obvious bounded examples -- "every attempt row", "all 155 steps" -- never
    # reach the numeral escape or the corpus-precedence branch at all, because
    # neither sentence contains a step-produced artifact noun, so the detector
    # declines them one step earlier. Both mutants (M5, M6) SURVIVED against them.
    # A fixture only tests a branch if that branch is the ONLY thing standing
    # between the input and the outcome, so each of these pairs a step-produced
    # noun WITH the bounding feature under test.
    ck("a corpus population governs, even with an artifact noun later in the clause "
       "('every attempt row is covered by a guard') -- only the corpus-precedence "
       "branch excludes this",
       not is_unbounded("every attempt row is covered by a guard"))
    ck("an enumerated artifact population is BOUNDED ('all 3 guards this step adds') "
       "-- only the numeral escape excludes this",
       not is_unbounded("mutation-test all 3 guards this step adds"))
    ck("...and both fixtures WOULD be flagged without their bounding feature, so "
       "neither passes vacuously",
       is_unbounded("every guard is covered") and is_unbounded("mutation-test all guards this step adds"))
    ck("a step-produced artifact class is UNBOUNDED ('every guard it adds')",
       is_unbounded("mutation-test every guard it adds"))
    ck("...and so is 'each new probe this step introduces'",
       is_unbounded("red-first proof for each new probe this step introduces"))

    print("\n" + "=" * 78)
    print("B2. THE HALF OF THE FILED FIGURES THAT DOES REPRODUCE (criterion 1)")
    print("=" * 78)
    pin = census(load_plan(FILING_REV))
    ck(f"the step-inclusion rule reproduces the filed step count at {FILING_REV}",
       pin["steps"] == FILED["steps"], f"{pin['steps']} vs filed {FILED['steps']}")
    ck("...and the filed criterion count",
       pin["criteria"] == FILED["criteria"], f"{pin['criteria']} vs filed {FILED['criteria']}")
    ck("the filed APPARATUS figure does NOT reproduce, and that is REPORTED rather "
       "than fitted", pin["apparatus"] != FILED["apparatus"],
       f"{pin['apparatus']} ({pin['pct']:.1f}%) vs filed {FILED['apparatus']} "
       f"({FILED['pct']}%) -- the filing never recorded its rule")
    ck("the shipped rule's ratio falls inside the filed 1.6x-1.9x range",
       FILED["ratio_range"][0] <= pin["ratio"] <= FILED["ratio_range"][1],
       f"{pin['ratio']:.2f}x")
    ck("the corpus genuinely MOVED, so a live-tree criterion could not have bound",
       census(load_plan(DRIFT_REV))["steps"] != pin["steps"]
       and census(load_plan(None))["steps"] != pin["steps"],
       f"{pin['steps']} -> {census(load_plan(DRIFT_REV))['steps']} -> "
       f"{census(load_plan(None))['steps']}")

    print("\n" + "=" * 78)
    print("C. EXIT-CODE SWEEP OVER EVERY STEP AT THE PIN (criterion 3)")
    print("=" * 78)
    plan = load_plan(FILING_REV)
    steps = [s for s in collect_steps(plan) if in_range(s["id"])]
    import io
    import contextlib
    codes = {}
    for s in steps:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            codes[str(s["id"])] = gate_step(s)
    nonzero = [k for k, v in codes.items() if v != 0]
    zero = [k for k, v in codes.items() if v == 0]
    print(f"  swept {len(codes)} steps: exit 0 on {len(zero)}, exit 2 on {len(nonzero)}")
    ck("every step was actually executed and yielded a captured exit code",
       len(codes) == len(steps) and len(steps) > 100, f"{len(codes)} of {len(steps)}")
    ck("every non-zero exit is attributable to an unbounded criterion, and nothing else",
       all(any(is_unbounded(c) for c in
               next(s for s in steps if str(s["id"]) == k)["verification"]["success_criteria"])
           for k in nonzero), f"{len(nonzero)} non-zero")
    ck("every step carrying NO unbounded criterion exited 0",
       all(not any(is_unbounded(c) for c in
                   next(s for s in steps if str(s["id"]) == k)["verification"]["success_criteria"])
           for k in zero), f"{len(zero)} zero-exit steps")
    ck("the sweep is not vacuous: BOTH outcomes occur",
       len(nonzero) > 0 and len(zero) > 0, f"{len(zero)} / {len(nonzero)}")

    print("\n" + "=" * 78)
    print("D. IT CANNOT MUTATE THE PLAN (criterion 4 -- BOTH checks, AST-level)")
    print("=" * 78)
    print_census_quiet()
    mp_after = sha256(MASTERPLAN)
    ck("sha256 of .claude/masterplan.json byte-identical across a FULL "
       "classification run over every step in the file",
       mp_before == mp_after, f"{mp_before[:16]} -> {mp_after[:16]}")
    src = Path(__file__).read_text(encoding="utf-8")
    writes = write_capable_calls(src)
    ck("the source contains NO write-capable call, resolved at AST level",
       not writes, "; ".join(writes) or "none")
    ck("...and the AST scan is not vacuous: it FINDS writes in a known writer",
       len(write_capable_calls(
           "from pathlib import Path\nPath('x').write_text('y')\n")) == 1)
    ck("...including the two literal patterns criterion 4 names",
       len(write_capable_calls("import json\njson.dump({}, open('x','w'))\n")) == 2)
    # A sha256 that returned a constant would make every byte-identity check above
    # pass while proving nothing. Two different files must hash differently.
    ck("sha256 DISCRIMINATES: two different files do not hash alike",
       sha256(MASTERPLAN) != sha256(VERDICT_LEDGER))

    print("\n" + "=" * 78)
    print("E. GRADING IS UNTOUCHED (criterion 5)")
    print("=" * 78)
    vl_after = sha256(VERDICT_LEDGER)
    ck("sha256 of handoff/verdict_ledger.jsonl byte-identical before and after",
       vl_before == vl_after, f"{vl_before[:16]} -> {vl_after[:16]}")
    ck("no criterion text can be produced by this module (it only reads and labels)",
       not writes)

    print("\n" + "=" * 78)
    print("F. THE INPUT SURFACE (criterion 7 -- 'never READS' is the binding verb)")
    print("=" * 78)
    leaked = classifier_consequence_refs(src)
    ck("no classification function references a verdict history, a WIP record, a "
       "round index or a remaining attempt budget -- neither handed in nor "
       "SELF-read (AST, scoped to the classification path)",
       not leaked, "; ".join(leaked) or "none")
    # The probe must be shown to FIRE, or "none" is indistinguishable from a
    # broken walk. This is the control the first version of this check lacked.
    planted = classifier_consequence_refs(
        "def classify(criterion):\n"
        "    import subprocess\n"
        "    h = subprocess.run(['python3','scripts/qa/qa_wip.py',criterion])\n"
        "    return h\n", ["classify"])
    ck("...and that scan is NOT vacuous: a planted qa_wip self-read IS detected",
       len(planted) >= 1, "; ".join(planted[:2]) or "NOTHING DETECTED")
    planted2 = classifier_consequence_refs(
        "def census(plan):\n    return plan['retry_count']\n", ["census"])
    ck("...and it catches a consequence field read straight off the plan object",
       len(planted2) >= 1, "; ".join(planted2[:2]) or "NOTHING DETECTED")
    io_calls = classifier_io_calls(src)
    ck("the classification path's ENTIRE I/O surface is the plan of record -- "
       "named, not asserted",
       all("load_plan" in c for c in io_calls), "; ".join(io_calls) or "none")
    print("      classification-path I/O calls, in full: "
          + ("; ".join(io_calls) or "none"))
    print("      DISCLOSED: the SELF-TEST reads handoff/verdict_ledger.jsonl to hash "
          "it twice,\n      which criterion 5 requires. That read is outside the "
          "classification path and\n      is what the scope above enforces.")
    ck("classify() is pure over its argument: same text in, same label out, with no "
       "step identity available to it",
       classify("mutation-test every guard") == classify("mutation-test every guard"))
    import inspect
    ck("...and its signature admits no step id, verdict or attempt count",
       set(inspect.signature(classify).parameters) == {"criterion", "variant"},
       str(list(inspect.signature(classify).parameters)))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  failed: {len(fails)}")
    for f in fails:
        print(f"    FAIL {f}")
    return 1 if fails else 0


def print_census_quiet() -> None:
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_census()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--step-file", default=None)
    ns = ap.parse_args(argv)
    if ns.verify:
        return verify()
    if ns.step_file:
        return gate_step(json.loads(Path(ns.step_file).read_text(encoding="utf-8")))
    if ns.census or ns.rev:
        print_census()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
