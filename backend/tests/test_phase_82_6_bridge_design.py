"""phase-82.6 -- the registry-to-live selection bridge is DESIGNED, not built.

This step ships a document and a guard. The guard's job is to prove the live
selection path is UNCHANGED, because the live paper book is working and a
selection-path edit is the highest-regression-risk change in this system.

WHAT THE DESIGN FOUND, and why the guard is shaped the way it is:

  * The step's own premise was wrong in both halves. The registry's exit params
    do NOT cross into live behaviour either -- `summary["strategy_params"]` has
    zero readers repo-wide. Today nothing the optimizer produces changes live
    trading.
  * A strategy NAME already gates a live risk control: `paper_trader.py`
    skips the HWM trailing stop for `mean_reversion` / `pairs`. That branch is
    deliberate and cited (Kaminski-Lo Prop. 2) and its default is fail-safe --
    but it is currently UNREACHABLE, so the bridge's first natural wire would
    activate it as a side effect.

TRAPS THIS FILE DELIBERATELY AVOIDS, each measured by the research gate:

  * `len(set(values)) == len(keys)` FAILS on correct code -- STRATEGY_REGISTRY
    has 6 keys but 5 distinct method values (`meta_label` shares
    `triple_barrier`'s).
  * A blanket "autonomous_loop must not import backend.backtest" assertion is a
    FALSE POSITIVE: it legitimately imports `backend.backtest.universe_lists`
    and `backend.backtest.markets`.
  * Sweeping registry KEYS is unsafe -- `triple_barrier` legitimately appears in
    the live path as a label string. Sweep the METHOD VALUES.
  * A text scan trips on `perf_metrics.py:151`, which names `backtest_engine` in
    a comment.
"""

from __future__ import annotations

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
DESIGN = REPO / "docs/design/registry-to-live-selection-bridge.md"
LIVE_CYCLE = REPO / "backend/services/autonomous_loop.py"


# ---------------------------------------------------------------------------
# criterion 1 -- the design exists and names its three required parts
# ---------------------------------------------------------------------------


def test_the_design_document_exists():
    assert DESIGN.exists(), f"no bridge design at {DESIGN.relative_to(REPO)}"
    assert DESIGN.stat().st_size > 2000, "the design is a stub"


def test_the_design_names_insertion_point_gate_and_rollback():
    """CRITERION 1. Each part is checked by the CONCRETE symbol it must name,
    not by a section heading -- a heading can be written without content."""
    # Strip HTML comments first: the 82.6 Q/A showed a 2100-byte stub whose
    # only content was these tokens inside `<!-- -->` passed every criterion-1
    # test. Tokens must appear in the document's PROSE, not its comments.
    text = re.sub(r"<!--.*?-->", "", DESIGN.read_text(encoding="utf-8"), flags=re.S)

    required = {
        "insertion point (params seam)": "autonomous_loop.py:431",
        "insertion point (position seam)": "paper_positions.entry_strategy",
        "the function with no params argument": "decide_trades",
        "the live promotion gate": "PromotionGate",
        "the gate's real thresholds": "min_dsr 0.95",
        "the staged capital ladder": "evaluate_stage",
        "rollback -- kill switch": "kill switch",
        "rollback -- fail-open demotion": "promoted_strategies",
        "rollback -- drawdown demotion": "auto_demote_on_dd_breach",
    }
    # Identifiers are matched exactly; prose tokens case-insensitively -- a
    # design that writes "Kill switch" has named it.
    lowered = text.lower()
    missing = [
        k for k, token in required.items()
        if (token not in text) and (token.lower() not in lowered)
    ]
    assert not missing, (
        "the design does not name: " + ", ".join(missing)
    )


def test_the_design_records_the_build_prerequisites():
    """82.23 and 82.26 are BUILD-TIME blockers -- optimizer_best.json has no
    `pbo` key and PromotionGate is fail-closed on a missing one. A design that
    omitted this would send someone to build an ungateable bridge."""
    text = DESIGN.read_text(encoding="utf-8")
    assert "82.23" in text and "82.26" in text, (
        "the design does not name its hard prerequisites"
    )


def test_the_design_carries_the_trailing_stop_hazard():
    """The single most important thing the design says: populating
    `entry_strategy` activates a dormant live risk-behaviour change."""
    text = DESIGN.read_text(encoding="utf-8")
    # `trailing[- ]stop` as a regex, not a literal: the concept is prose and the
    # document hyphenates it. The literal form only ever matched because an
    # earlier draft carried a FABRICATED trailing comment inside a block
    # presented as a verbatim source quote; regenerating that block from source
    # (correctly) removed it.
    assert re.search(r"trailing[- ]stop", text, re.I), (
        "the design omits the trailing-stop hazard"
    )
    for token in ("mean_reversion", "Kaminski"):
        assert token in text, f"the design omits {token!r} from the §2 hazard"
    assert "stop_advanced_at_R" in text, (
        "the design does not state the precondition that bounds the hazard's "
        "blast radius -- without it the prose overstates what stops being trailed"
    )


# ---------------------------------------------------------------------------
# criterion 2 -- every file:line reference in the design RESOLVES
# ---------------------------------------------------------------------------

_REF = re.compile(r"`?([A-Za-z0-9_./]+\.(?:py|json|md))[`]?:(\d+)(?:-(\d+))?")


def _design_refs() -> list[tuple[str, int, int]]:
    """Every `path:line` or `path:line-line` in the design, with the repo-root
    path resolved. Paths are recorded relative to a few known roots because the
    prose cites some files by bare name."""
    out = []
    for m in _REF.finditer(DESIGN.read_text(encoding="utf-8")):
        path, lo, hi = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        out.append((path, lo, hi))
    return out


def _resolve(path: str) -> pathlib.Path | None:
    p = REPO / path
    if p.exists():
        return p
    # bare filenames cited in prose -- resolve by unique basename
    matches = [
        q for q in REPO.rglob(pathlib.PurePath(path).name)
        if ".venv" not in q.parts and "node_modules" not in q.parts
    ]
    return matches[0] if len(matches) == 1 else None


def test_the_design_cites_enough_to_be_checkable():
    """Recall test: a design with no file:line refs would pass the resolution
    test below vacuously."""
    refs = _design_refs()
    assert len(refs) >= 10, (
        f"only {len(refs)} file:line references found in the design -- criterion 2 "
        "asks for references that resolve, which presupposes there are some"
    )


def test_every_file_line_reference_in_the_design_resolves():
    """CRITERION 2. A design whose anchors have rotted is worse than none: it
    sends the next reader to the wrong place with false confidence."""
    unresolved = []
    for path, lo, hi in _design_refs():
        target = _resolve(path)
        if target is None:
            unresolved.append(f"{path}:{lo} -- file not found (or ambiguous)")
            continue
        n_lines = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        if hi > n_lines:
            unresolved.append(
                f"{path}:{lo}-{hi} -- file has only {n_lines} lines"
            )
    assert not unresolved, (
        "file:line references in the design that do not resolve:\n  "
        + "\n  ".join(unresolved)
    )


# ---------------------------------------------------------------------------
# criterion 3 -- the live selection path is UNCHANGED
# ---------------------------------------------------------------------------


def _registry() -> dict:
    from backend.backtest.backtest_engine import STRATEGY_REGISTRY

    return dict(STRATEGY_REGISTRY)


def test_the_registry_is_non_empty_and_its_shape_is_what_we_think():
    """Recall test for the sweep below, and a deliberate NON-assertion.

    The registry has MORE keys than distinct method values -- `meta_label`
    shares `triple_barrier`'s labeller. Asserting `len(set(values)) == len(keys)`
    would fail on correct code, which is exactly the kind of guard that gets
    deleted rather than understood.
    """
    reg = _registry()
    assert reg, "STRATEGY_REGISTRY is EMPTY -- the sweep below would be vacuous"
    assert len(reg) >= 5, f"only {len(reg)} strategies registered"
    assert len(set(reg.values())) < len(reg), (
        "every strategy now has a distinct label method -- the registry shape "
        "changed; re-derive this test's assumptions rather than loosening it"
    )


def test_no_registry_label_method_is_referenced_from_the_live_cycle():
    """CRITERION 3. The label methods are TRAINING-label functions
    ((ticker, entry_date) -> int|None), dispatched by getattr inside the backtest
    engine. None may appear in the live cycle as an attribute, a bare name, or a
    string constant. That is what THIS test proves -- not universal
    unreachability; the indirection shapes are covered by
    `test_the_live_cycle_does_not_reach_the_engine_by_indirection` above, and
    together they bound rather than eliminate the recall gap.

    Sweeps the method VALUES, not the registry KEYS: `triple_barrier` is a
    legitimate live label string, while `_label_triple_barrier` is not.
    """
    methods = set(_registry().values())
    assert methods, "no label methods derived -- sweep would be vacuous"

    src = LIVE_CYCLE.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Structural: any attribute access or bare name matching a label method.
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in methods:
            referenced.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in methods:
            referenced.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in methods:
                referenced.add(node.value)

    assert not referenced, (
        f"{LIVE_CYCLE.name} references STRATEGY_REGISTRY label method(s) "
        f"{sorted(referenced)} -- the live selection path has been wired, which "
        "phase-82.6 explicitly does not do"
    )


def test_the_live_cycle_does_not_reach_the_engine_by_indirection():
    """RECALL EXTENSION. The 82.6 Q/A demonstrated four wiring shapes the
    value-sweep and the direct-import guard both miss:

      * `from backend.backtest import backtest_engine` (submodule form -- the
        ImportFrom module is `backend.backtest`, not `...backtest_engine`)
      * `importlib.import_module("backend.backtest.backtest_engine")`
      * `STRATEGY_REGISTRY["mean_reversion"]` -- KEY dispatch, which the value
        sweep cannot see because keys legitimately appear live
      * `getattr(x, f"_compute_{sel}_label")` -- an f-string-built method name,
        which is exactly how the backtest engine itself dispatches, so it is
        the most likely shape a copy-paste wiring would take

    None is reachable today; this closes the recall gap rather than waiting for
    one of them to be the way selection gets wired.
    """
    src = LIVE_CYCLE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    offending = []

    for node in ast.walk(tree):
        # submodule import form
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "backend.backtest":
            for a in node.names:
                if a.name == "backtest_engine":
                    offending.append(f"line {node.lineno}: from backend.backtest import backtest_engine")
        # importlib by string
        if isinstance(node, ast.Call):
            fn = node.func
            if getattr(fn, "attr", getattr(fn, "id", None)) == "import_module":
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                       and "backtest_engine" in a.value:
                        offending.append(f"line {node.lineno}: importlib {a.value}")
            # f-string-built label method name
            if getattr(fn, "id", None) == "getattr":
                for a in node.args[1:]:
                    if isinstance(a, ast.JoinedStr) and "_compute_" in ast.unparse(a):
                        offending.append(f"line {node.lineno}: getattr with an f-string label name")
        # registry symbol referenced at all (covers KEY dispatch)
        if isinstance(node, ast.Name) and node.id == "STRATEGY_REGISTRY":
            offending.append(f"line {node.lineno}: STRATEGY_REGISTRY referenced")

    assert not offending, (
        "the live cycle reaches the strategy registry by indirection -- the "
        "value sweep alone would not have seen this:\n  " + "\n  ".join(offending)
    )


def test_the_live_cycle_does_not_import_the_backtest_engine():
    """Narrower than 'imports nothing from backend.backtest', which would be a
    FALSE POSITIVE -- the live cycle legitimately imports
    `backend.backtest.universe_lists` and `backend.backtest.markets`.

    Structural (ast.Import / ast.ImportFrom), not a text scan: a text scan trips
    on comments that merely name the module.
    """
    tree = ast.parse(LIVE_CYCLE.read_text(encoding="utf-8"))
    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "backend.backtest.backtest_engine"
        ):
            offending.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("backend.backtest.backtest_engine"):
                    offending.append(f"line {node.lineno}: import {a.name}")

    assert not offending, (
        "the live cycle imports the backtest engine -- selection may have been "
        "wired:\n  " + "\n  ".join(offending)
    )


def test_the_legitimate_backtest_imports_are_still_present():
    """Negative control for the test above. If `universe_lists` / `markets` ever
    stop being imported, the previous test would pass for the wrong reason and
    nobody would notice it had stopped discriminating."""
    src = LIVE_CYCLE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    modules = {
        (n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    }
    assert any("backend.backtest" in m for m in modules), (
        "the live cycle no longer imports ANY backend.backtest module, so the "
        "engine-import test above can no longer distinguish a real wiring from "
        "an unrelated refactor"
    )


def test_the_strategy_name_is_still_consumed_as_a_label_only():
    """Pins §1's measured finding so a future wiring is loud.

    `best_params` may be read for display and for the audit label, but the
    `strategy` key must not flow into anything that decides a trade.
    """
    src = LIVE_CYCLE.read_text(encoding="utf-8")
    tree = ast.parse(src)

    strategy_reads = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
        and n.args and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "strategy"
    ]
    # The 82.6 Q/A's fifth survivor: a subscript read `x["strategy"]` instead of
    # `.get("strategy")`. Counted alongside, so swapping the access form is not
    # a way to add a second read unnoticed.
    strategy_reads += [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Subscript)
        and isinstance(n.slice, ast.Constant) and n.slice.value == "strategy"
    ]

    assert strategy_reads, (
        "no `.get('strategy')` read found in the live cycle -- either the "
        "measured baseline moved or this instrument is broken"
    )
    assert len(strategy_reads) == 1, (
        f"`strategy` is now read at {len(strategy_reads)} sites {strategy_reads} "
        "in the live cycle; the design's measured baseline is ONE (the audit "
        "label). A second read is how selection gets wired by accident."
    )
