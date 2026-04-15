# Contract -- Phase 4.4.4.4 Risk limits hardcoded verification + evidence

**Depends on:** Phase 4.3 Risk Management (commit `be3accb`) and Phase 4.4 Go-Live Checklist scoping (commit `da4fe5d`).
**Research gate:** WAIVED. Pure verification cycle. No behavioral change. No new surface. The "risk limits hardcoded, not config-backed" invariant is documented in `PLAN.md` section 4.4.4.4 and mirrored in `docs/GO_LIVE_CHECKLIST.md` section 4.4.4.4. This cycle runs the deterministic check defined by that HOW recipe and records the evidence.

## Goal

Verify that the four Go-Live risk-limit keys in
`SignalsServer.get_risk_constraints` are hardcoded Python literals, not
sourced from an env var / YAML / TOML / config loader, and that
`SignalsServer.risk_check` uses that same single-source-of-truth
function. Mark `docs/GO_LIVE_CHECKLIST.md` item 4.4.4.4 as `[x]` with a
one-line evidence note pinning the commit + method locations.

## Files touched

1. `docs/GO_LIVE_CHECKLIST.md` (+1 line changed for the checkbox,
   +1 line added as the evidence note immediately under the bullet)

No other files modified. Zero `.py` files touched. Zero AST impact.
Zero test files. Zero migrations.

## Diff budget

- `<= 3` lines changed in `docs/GO_LIVE_CHECKLIST.md`
- `<= 0` lines added or deleted in any other file

## Success criteria

### A. Scope discipline (SC1-4)

- **SC1** Exactly 1 file modified this cycle (`docs/GO_LIVE_CHECKLIST.md`).
- **SC2** Zero `.py` files modified. Zero imports added. Zero AST impact in any backend module.
- **SC3** `git diff --stat origin/main -- backend/ scripts/ frontend/` is empty.
- **SC4** Diff budget honored: `<= 3` lines changed in `docs/GO_LIVE_CHECKLIST.md`.

### B. Checkbox + evidence (SC5-8)

- **SC5** Line for 4.4.4.4 flips from `- [ ]` to `- [x]`.
- **SC6** A new one-line evidence note is appended directly under the bullet, starting with `  - **Evidence**:` and citing (a) the current commit hash after push, (b) the method name `SignalsServer.get_risk_constraints`, (c) the 4 literal values.
- **SC7** The `## 4.4.4 Risk Management Validation` section header is unchanged.
- **SC8** No other checkbox in the file is modified. Items 4.4.1.*, 4.4.2.*, 4.4.3.*, 4.4.4.1, 4.4.4.2, 4.4.4.3, 4.4.5.*, 4.4.6.* all stay `[ ]`.

### C. Deterministic verification block (SC9-16)

The verification block below must run to completion with zero
assertion failures using `python3` + stdlib only:

```python
import ast
p = 'backend/agents/mcp_servers/signals_server.py'
src = open(p, 'r').read()
tree = ast.parse(src)
cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'SignalsServer')
grc = next(fn for fn in cls.body if isinstance(fn, ast.FunctionDef) and fn.name == 'get_risk_constraints')
rc = next(fn for fn in cls.body if isinstance(fn, ast.FunctionDef) and fn.name == 'risk_check')
ret = next(n for n in ast.walk(grc) if isinstance(n, ast.Return))
assert isinstance(ret.value, ast.Dict)
required = {'max_exposure_per_ticker_pct': 10.0, 'max_total_exposure_pct': 100.0,
            'max_drawdown_pct': -15.0, 'max_daily_trades': 5}
found = {}
for k, v in zip(ret.value.keys, ret.value.values):
    if isinstance(k, ast.Constant) and k.value in required:
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant):
            found[k.value] = -v.operand.value
        elif isinstance(v, ast.Constant):
            found[k.value] = v.value
assert found == required, found
# No indirection in file
for pat in ['os.environ', 'getenv', 'yaml.', 'toml.', 'ConfigParser', 'load_config', 'from_yaml', 'from_toml']:
    assert pat not in src
# get_risk_constraints does not touch self.settings
for n in ast.walk(grc):
    assert not (isinstance(n, ast.Attribute) and n.attr == 'settings')
# risk_check calls self.get_risk_constraints() at least once
calls = [n for n in ast.walk(rc)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == 'get_risk_constraints']
assert len(calls) >= 1
print('PASS 4.4.4.4')
```

- **SC9** `ast.parse(open(p).read())` is clean.
- **SC10** `get_risk_constraints` is a method of the `SignalsServer` class.
- **SC11** Its return value is a single `ast.Dict` literal (no computed keys, no `**kwargs` merge, no function call).
- **SC12** All 4 required keys are present with exact literal values: `max_exposure_per_ticker_pct=10.0`, `max_total_exposure_pct=100.0`, `max_drawdown_pct=-15.0`, `max_daily_trades=5`.
- **SC13** The literals for the negative value (`max_drawdown_pct=-15.0`) are recognized by unwrapping `ast.UnaryOp(USub, Constant(15.0))`.
- **SC14** No `os.environ`, `getenv`, `yaml.`, `toml.`, `ConfigParser`, `load_config`, `from_yaml`, or `from_toml` substring anywhere in the file.
- **SC15** `get_risk_constraints` does not reference `self.settings` (AST Attribute walk).
- **SC16** `risk_check` calls `self.get_risk_constraints()` at least once (AST Call walk). This confirms the method is the single source of truth for the hard-path limits.

## Adversarial probes (for self-evaluation)

- **ADV1** Mutate `max_drawdown_pct` default in `risk_check`'s `limits.get("max_drawdown_pct", -15.0)` -- does the block catch it? (No; SC14-16 only audit `get_risk_constraints`. The `.get()` defaults are fallbacks; the canonical source is the returned dict, which SC12 catches.)
- **ADV2** What if a new config loader is added later? SC14's substring audit is defense-in-depth and catches common patterns; any exotic loader (`from json import loads; loads(open(...).read())`) could slip through. Acceptable residual risk; future-Ford can tighten.
- **ADV3** What if `get_risk_constraints` is overridden by a subclass? Not verified -- only the current file is audited. Acceptable; SignalsServer is the canonical class.
- **ADV4** What if a caller bypasses `risk_check` entirely? SC16 only confirms `risk_check` itself uses the constraints; any other caller of the limits would need to be audited separately. Documented here as a known gap.

## Out of scope (explicitly)

- No code changes. No new methods. No new literals. Zero `.py` files touched.
- No new tests (4.4.4.1, 4.4.4.2, 4.4.4.3 remain `[ ]` and need their own cycles with standalone test modules).
- No run of the harness. `masterplan.json` is not edited (phase 4.4 stays `pending` until every one of its 27 items is `[x]`).
- No backfill on 4.4.3.5 (joint item, Ford-side can be checked but the Peder sign-off is required before the box flips).

## Self-evaluation justified

Cycle 7's documented precedent applies: pure-doc / zero-logic cycles
with deterministic verification and zero risk surface may skip the
`qa-evaluator` subagent and rely on lead-self verification. This cycle:

- Edits only `docs/GO_LIVE_CHECKLIST.md` (a markdown file, not loaded
  at runtime, not imported by any Python module, not a test fixture).
- Zero AST impact anywhere in `backend/`, `scripts/`, `frontend/`, or
  `.claude/`.
- The verification block is 100% deterministic and stdlib-only.
- The "hardcoded literals" invariant has been stable since Phase 4.3
  (`be3accb`, 2026-04-14); this cycle is verification, not protection.

Spawning an Opus qa-evaluator on a markdown checkbox flip with a
deterministic underlying AST invariant would burn turns for no signal.
