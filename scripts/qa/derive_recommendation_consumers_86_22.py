#!/usr/bin/env python3
"""phase-86.22 criteria 2 + 4 -- DERIVE the recommendation-vocabulary population.

WHY AST, AND NOT A REGEX
------------------------
The first version of this check was a regex over source lines. It reported EIGHT
offenders, and SEVEN were wrong -- because a regex cannot tell these apart:

    if rec in ("Strong Buy", "Buy"):            # analyst RECOMMENDATION
    if order.action not in ("BUY", "SELL"):     # order SIDE
    if sig not in ("BUY", "SELL", "HOLD"):      # signal ACTION

They share tokens and share nothing else. The order/signal vocabularies are
separate closed scales that are already constrained upstream and never carry the
spaced spelling; folding them into this step would be a false positive that
trains a reader to ignore the check. It also matched a COMMENT -- including the
one I had just written describing the defect.

So membership is decided on the AST, by two rules, and a site is in-scope if
EITHER holds:

  R1  the literal set contains a STRONG_* / "Strong ..." token. Strong-conviction
      tokens occur ONLY in the recommendation vocabulary -- an order side is
      never "Strong Buy". This is the rule that catches a site whose variable is
      named something unhelpful.
  R2  the tested expression mentions "recommend" (e.g. `recommendation`,
      `rec_label`, `report["recommendation"]`). This catches a recommendation
      site whose literals happen to be plain BUY/SELL.
  R3  the INVERTED shape -- `"STRONG_BUY" in rec_label` -- where a canonical
      token is the LEFT operand and the container is a variable. This is a
      SUBSTRING test, not a membership test, and it is a strictly worse defect
      than R1/R2: `"STRONG_SELL"` contains `"SELL"`, so a substring chain grades
      a sell as a sell-or-a-buy depending only on clause order.

R3 was not in the first version of this script, and its absence was the bug.
Scanning the pre-fix tree with R1+R2 alone found 10 sites across 4 files and
silently MISSED `conflict_detector.py` entirely -- one of the five consumers --
because a substring test is a different AST shape, not a different spelling. A
derivation that covers one shape of a defect is a guard against an instance, not
against the class.

KNOWN LIMIT, stated rather than papered over: R3 flags a substring test against
ANY canonical token, so a hypothetical `"SELL" in order.side` would be reported
here even though order side is a separate vocabulary. No such site exists in the
tree today (measured: the R3 hits are conflict_detector's three, all genuine).
If one appears, it needs an ALLOWED entry with a reason -- not a weakened rule.

RECALL AND PRECISION ARE BOTH VALIDATED, AGAINST REAL PRE-FIX CODE
------------------------------------------------------------------
A derivation whose recall is asserted rather than measured is the defect this
project keeps re-learning. So `--validate` runs the method against:

  * KNOWN POSITIVES -- the verbatim pre-fix expressions from the five consumers,
    which MUST all be flagged;
  * KNOWN NEGATIVES -- the order-side and signal-action expressions, which MUST
    NOT be flagged.

Both directions gate the exit code. A method that finds every positive by
flagging everything fails here, and so does one that is silent.

`--against-git-rev REV` re-runs the scan over a git revision, so the population
can be measured against the DEFECTIVE tree rather than the fixed one -- the
"before" half of criterion 3.

    source .venv/bin/activate
    python scripts/qa/derive_recommendation_consumers_86_22.py --validate
    python scripts/qa/derive_recommendation_consumers_86_22.py --against-git-rev 4e9d1a7d
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"

#: Sites that legitimately test recommendation literals and must NOT be
#: rewritten. Keyed `basename.py::enclosing_function` -- PER FUNCTION, because
#: `formatters.py` proved a per-FILE key cannot express the truth: one function
#: in it is a real second vocabulary and the next one along is a different
#: vocabulary entirely. Each entry needs a reason; an unexplained entry is how a
#: real offender gets parked here forever.
ALLOWED: dict[str, str] = {
    "recommendation_vocab.py::*": "IS the vocabulary -- the sets have to live somewhere",
    # skill_optimizer.py was HERE, on the grounds that it reads a
    # schema-enforced Literal. The cycle-1 Q/A disproved the reason: it reads
    # `debate_consensus` from financial_reports.analysis_results, the same
    # persisted table as `recommendation`. Rather than reword the excuse, the
    # site was migrated -- an allow-list entry is only as good as its argument.
    "formatters.py::_signal_emoji": (
        "R3's documented false positive. Its `action` is `signal['signal']`, "
        "documented in the enclosing docstring as BUY/SELL/HOLD -- the SIGNAL "
        "vocabulary, which is a separate closed scale that never carries a "
        "spaced or STRONG_ spelling. Canonicalising here would be a "
        "wrong-vocabulary change, not a fix."
    ),
}

_STRONG = ("STRONG_BUY", "STRONG_SELL")


def _literal_strings(node: ast.AST) -> list[str] | None:
    """Return the string members if `node` is a literal tuple/list/set of str."""
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    out: list[str] = []
    for elt in node.elts:
        if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
            return None
        out.append(elt.value)
    return out or None


def _canon(value: str) -> str | None:
    # Import late so the module under audit is never a hard dependency of the
    # auditor -- this script must run even if the vocabulary module is broken.
    sys.path.insert(0, str(REPO_ROOT))
    from backend.services.recommendation_vocab import canonical_recommendation

    return canonical_recommendation(value)


def _is_recommendation_vocabulary(members: list[str], tested_src: str) -> str | None:
    """Return which rule puts this site in scope, or None."""
    canon = [_canon(m) for m in members]
    if any(c in _STRONG for c in canon):
        return "R1 strong-conviction token"
    if "recommend" in tested_src.lower():
        # ...but only if the literals are recommendation-shaped at all.
        if any(c is not None for c in canon):
            return "R2 tested expression names a recommendation"
    return None


def scan_source(src: str, label: str) -> list[dict]:
    """Scan one source string. Returns in-scope comparison sites."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [{"file": label, "line": 0, "rule": f"SyntaxError: {exc}",
                 "func": "<module>", "tested": "", "members": []}]
    # Map every node to its NEAREST enclosing function so ALLOWED can be
    # per-function. Recursive descent, not ast.walk: walk is breadth-first, so
    # a setdefault over it would attribute a nested function's body to the
    # OUTER function -- right answer for module-level defs, wrong in general.
    enclosing: dict[int, str] = {}

    def _descend(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = (child.name
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else current)
            enclosing[id(child)] = name
            _descend(child, name)

    _descend(tree, "<module>")

    rows: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        func = enclosing.get(id(node), "<module>")
        if not any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            continue
        for comparator in node.comparators:
            members = _literal_strings(comparator)
            if members:
                tested = ast.unparse(node.left)
                rule = _is_recommendation_vocabulary(members, tested)
                if rule:
                    rows.append({"file": label, "line": node.lineno, "rule": rule,
                                 "func": func, "tested": tested,
                                 "members": members})
                continue

            # R3 -- the INVERTED, substring shape: `"STRONG_BUY" in rec_label`.
            # Different AST shape, same defect class, and strictly worse.
            left = node.left
            if isinstance(left, ast.Constant) and isinstance(left.value, str):
                if _canon(left.value) is not None:
                    rows.append({
                        "file": label, "line": node.lineno, "func": func,
                        "rule": "R3 SUBSTRING test against a canonical token",
                        "tested": ast.unparse(comparator),
                        "members": [left.value],
                    })
    return rows


def scan_tree(read_file) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if "tests" in path.parts:
            continue
        src = read_file(rel)
        if src is None:
            continue
        for row in scan_source(src, str(rel)):
            fn = row.get("func", "<module>")
            key, wild = f"{path.name}::{fn}", f"{path.name}::*"
            row["allowed"] = key in ALLOWED or wild in ALLOWED
            rows.append(row)
    return rows


# ── validation of the METHOD itself ────────────────────────────────────────

#: Verbatim pre-fix expressions. Every one MUST be flagged.
KNOWN_POSITIVES = [
    ('outcome_tracker (title-case)',
     'x = recommendation in ("Strong Buy", "Buy")'),
    ('outcome_tracker sell leg',
     'x = recommendation in ("Strong Sell", "Sell")'),
    ('memory.py',
     'x = rec in ("Strong Buy", "Buy")'),
    ('bias_detector (upper-snake)',
     'x = (recommendation or "").upper() in ("STRONG_BUY", "BUY")'),
    ('api/portfolio.py',
     'x = (r.get("recommendation") or "").upper() in ("STRONG_BUY", "BUY")'),
    ('a plain-token recommendation site (R2 only)',
     'x = recommendation in ("BUY", "SELL")'),
    # R3 -- the shape R1+R2 missed entirely. conflict_detector, verbatim.
    ('conflict_detector strong-buy (R3 substring)',
     'x = "STRONG_BUY" in rec_label and score < 7.0'),
    ('conflict_detector buy (R3 substring)',
     'x = "BUY" in rec_label and score < 5.5'),
    ('conflict_detector sell (R3 substring)',
     'x = "SELL" in rec_label and score > 6.0'),
]

#: Separate closed vocabularies. NONE may be flagged.
KNOWN_NEGATIVES = [
    ('order side', 'x = order.action not in ("BUY", "SELL")'),
    ('signal action', 'x = sig not in ("BUY", "SELL", "HOLD")'),
    ('lite-analyzer action', 'x = analysis.get("action") not in ("BUY", "SELL", "HOLD")'),
    ('attribution side', 'x = s in ("BUY", "SELL")'),
    ('unrelated literals', 'x = mode in ("fast", "slow")'),
    ('already migrated', 'x = is_buy_intent(rec)'),
    # R3 must not fire on substring tests over NON-canonical tokens -- the
    # signal-sentiment chain right next to it in bias_detector uses exactly
    # this shape and is a separate, legitimate vocabulary.
    ('signal-sentiment substring', 'x = "BULL" in s or "RISING" in s'),
    ('bearish substring', 'x = "BEAR" in s or "DECLIN" in s'),
    # These two exist because mutation cell D4 SURVIVED without them. R2 only
    # fires when the tested expression mentions "recommend", so every negative
    # above leaves R2's second condition -- "...and the literals are actually
    # recommendation-shaped" -- completely uncovered. A name containing
    # "recommend" whose literals are a different vocabulary is the shape that
    # exercises it, and it is not hypothetical: a `recommendation_status` or a
    # `recommender` field is an ordinary thing to find in this codebase.
    ('recommend-named field, unrelated literals',
     'x = recommendation_status in ("pending", "done")'),
    ('recommend-named field, workflow literals',
     'x = report["recommendation_state"] in ("draft", "final")'),
]


def validate() -> int:
    print("METHOD VALIDATION -- recall AND precision, both gate the exit code\n")
    bad = 0

    print("KNOWN POSITIVES (must be flagged)")
    for name, src in KNOWN_POSITIVES:
        hits = scan_source(src, "<fixture>")
        ok = bool(hits)
        rule = hits[0]["rule"] if hits else "-"
        print(f"  {'OK  ' if ok else 'MISS'}  {name:<42} {rule}")
        if not ok:
            bad += 1

    print("\nKNOWN NEGATIVES (must NOT be flagged)")
    for name, src in KNOWN_NEGATIVES:
        hits = scan_source(src, "<fixture>")
        ok = not hits
        rule = hits[0]["rule"] if hits else "-"
        print(f"  {'OK  ' if ok else 'FALSE POSITIVE'}  {name:<42} {rule}")
        if not ok:
            bad += 1

    print(f"\nrecall {len(KNOWN_POSITIVES) - sum(1 for n, s in KNOWN_POSITIVES if not scan_source(s, '<f>'))}"
          f"/{len(KNOWN_POSITIVES)}   "
          f"precision {len(KNOWN_NEGATIVES) - sum(1 for n, s in KNOWN_NEGATIVES if scan_source(s, '<f>'))}"
          f"/{len(KNOWN_NEGATIVES)}")
    if bad:
        print(f"\nMETHOD REJECTED -- {bad} validation failure(s).")
        return 1
    print("\nMethod validated in BOTH directions.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="validate the method against known positives/negatives")
    ap.add_argument("--against-git-rev", metavar="REV",
                    help="scan REV's tree instead of the working tree")
    args = ap.parse_args()

    if args.validate:
        rc = validate()
        if rc:
            return rc
        print()

    if args.against_git_rev:
        rev = args.against_git_rev
        def read_file(rel: Path):
            r = subprocess.run(["git", "show", f"{rev}:{rel}"],
                               cwd=REPO_ROOT, capture_output=True, text=True)
            return r.stdout if r.returncode == 0 else None
        header = f"population at git rev {rev}"
    else:
        def read_file(rel: Path):
            try:
                return (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return None
        header = "population in the WORKING TREE"

    rows = scan_tree(read_file)
    print(f"{header}: {len(rows)} in-scope site(s)\n")
    hdr = (f"{'file':<40}{'line':>6}  {'function':<24}{'rule':<44}"
           f"{'tested expression':<28}members")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["file"], r["line"])):
        mark = "  (allowed)" if r.get("allowed") else ""
        print(f"{r['file']:<40}{r['line']:>6}  {r.get('func','')[:22]:<24}"
              f"{r['rule']:<44}{r['tested'][:26]:<28}{r['members']}{mark}")

    offenders = [r for r in rows if not r.get("allowed")]
    print(f"\nNOT on the allow-list: {len(offenders)}")
    for r in offenders:
        print(f"  !! {r['file']}:{r['line']} ({r.get('func','')})  "
              f"{r['tested']}  {r['members']}")
    if offenders and not args.against_git_rev:
        print("\nA second recommendation vocabulary survives (criterion 4).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
