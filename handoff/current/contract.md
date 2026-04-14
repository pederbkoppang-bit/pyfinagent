# Contract -- Phase 4.2.3.3 / SN-audit: ASCII hardening of signals_server.py module docstring

**Step ID:** 4.2.3.3 (SN-audit micro-fix, follow-on to Phase 4.2.3.2 QA's overbroad `ascii_only` audit finding)
**Target file:** `backend/agents/mcp_servers/signals_server.py`
**Target scope:** Module-level docstring, lines 5-13 only
**Base commit:** `9a53cf6` (origin/main HEAD as of 2026-04-14T2300Z)

## Problem statement

Phase 4.2.3.2 QA evaluator's `ascii_only` audit flagged **7 U+2192
(RIGHTWARDS ARROW) glyphs** in `signals_server.py`. Prior QA
classified them as "pre-existing, not a regression" (correct --
they were not introduced by the SN4 fix). But they are still a
**defense-in-depth violation** per `.claude/rules/security.md`:

> **ASCII-only logger messages**: Never use Unicode characters
> (arrows `\u2192`, em dashes `\u2014`, etc.) [...] Use `--`, `->`,
> plain English instead.

Strict reading: the rule binds only to `logger.*()` calls. Wide
reading (and prior sessions' stated intent): any copy-paste risk
from docstrings or comments into a logger call propagates the
Windows cp1252 encoding crash. All 7 glyphs live in the module
header docstring (lines 5-13), describing the MCP tool and
resource surface.

## Concrete hit list (from `ord(ch) > 127` scan)

| Line | Col | Context |
|-----:|----:|---------|
| 5    |  32 | `- generate_signal(ticker, date) \u2192 BUY/SELL/HOLD with confidence` |
| 6    |  26 | `- validate_signal(signal) \u2192 Check constraints (market hours, liquidity, exposure)` |
| 7    |  25 | `- publish_signal(signal) \u2192 Post to Slack + portfolio` |
| 8    |  40 | `- risk_check(portfolio, proposed_trade) \u2192 Can we add this position?` |
| 11   |  22 | `- portfolio://current \u2192 Current holdings (tickers, shares, PnL)` |
| 12   |  21 | `- constraints://risk \u2192 Risk limits (max exposure, max drawdown, Sharpe floor)` |
| 13   |  20 | `- signals://history \u2192 All generated signals this month` |

All 7 are U+2192 (`0x2192`). All 7 are in the module docstring.
Zero logger calls currently reference these strings.

## Fix

Single `replace_all` Edit: `\u2192` (U+2192 arrow glyph) -> `->`
(two-char ASCII). That is the canonical ASCII substitution
specified by `security.md` itself.

## Success criteria (20)

### Group A -- Surgical scope (SC1-5)

- **SC1**: `ast.parse(open('backend/agents/mcp_servers/signals_server.py').read())` succeeds (clean syntax)
- **SC2**: `python3 -m py_compile backend/agents/mcp_servers/signals_server.py` exit 0
- **SC3**: Exactly 1 file touched: `backend/agents/mcp_servers/signals_server.py`
- **SC4**: Diff bound: **<= 7 added lines, <= 7 deleted lines** (conceptually 0 net: 7 lines changed in-place)
- **SC5**: Only lines 5, 6, 7, 8, 11, 12, 13 of the file (the docstring arrow lines) change

### Group B -- ASCII audit (SC6-10)

- **SC6**: Post-edit: 0 non-ASCII bytes in the entire file (`any(ord(c) > 127 for c in open(f).read())` is `False`)
- **SC7**: Post-edit: the exact substring `\u2192` (U+2192) appears 0 times in the file
- **SC8**: Post-edit: the substring ` -> ` (with spaces) appears at minimum 7 times (the 7 replacements)
- **SC9**: Post-edit: each of the 7 identified lines contains the substring `-> ` where it previously contained `\u2192 `
- **SC10**: `harness_memory.py` and `multi_agent_orchestrator.py` non-ASCII counts unchanged from pre-edit (not in scope)

### Group C -- Byte-identity preservation (SC11-16)

- **SC11**: All 21 `SignalsServer` methods are `ast.dump`-identical pre-vs-post: `_signal_id`, `_empty_response`, `_remember`, `_risk_response`, `generate_signal`, `validate_signal`, `risk_check`, `publish_signal`, `size_position`, `check_stop_loss`, `track_drawdown`, `get_portfolio`, `get_risk_constraints`, `get_signal_history`, `track_signal_accuracy`, `get_accuracy_report`, `_append_signal_history`, `_wilson_ci`, `_parse_iso_date`, plus `__init__`
- **SC12**: The new Phase 4.2.3.2 `_parse_iso_date` helper is byte-identical pre-vs-post (not retouched)
- **SC13**: The Phase 4.2.3.2 `get_signal_history` `since_date` compare block is byte-identical pre-vs-post
- **SC14**: `ast.parse(...).body` top-level structure count is unchanged (same number of imports, classes, functions)
- **SC15**: Module imports byte-identical: line 23 still `from datetime import datetime, timezone, date` (no touches)
- **SC16**: No new imports added; no imports removed

### Group D -- Docstring semantic preservation (SC17-20)

- **SC17**: Module docstring still exists as the first expression in `ast.parse(...).body[0]`
- **SC18**: Module docstring still contains the 4 tool names (`generate_signal`, `validate_signal`, `publish_signal`, `risk_check`) verbatim
- **SC19**: Module docstring still contains the 3 resource URIs (`portfolio://current`, `constraints://risk`, `signals://history`) verbatim
- **SC20**: Module docstring still contains the `Tools (FastMCP @mcp.tool):` and `Resources:` headers verbatim

## Adversarial probes (10) -- for qa-evaluator

- **ADV1**: Grep for `\\u2192` in the file post-edit -> 0 hits
- **ADV2**: Python `codecs.encode(src, 'ascii', 'strict')` on the post-edit file -> succeeds (no UnicodeEncodeError)
- **ADV3**: `open(f, encoding='ascii').read()` -> succeeds (no UnicodeDecodeError)
- **ADV4**: `hashlib.sha256(post_src.encode('ascii')).hexdigest()` -> deterministic (file is pure ASCII)
- **ADV5**: AST walker over every `logger.*()` Call -- 0 non-ASCII string constants (unchanged from pre)
- **ADV6**: Count `->` occurrences: pre-edit has N, post-edit has N+7 (the 7 new docstring arrows)
- **ADV7**: `SignalsServer().__class__.__doc__` after import -> unchanged (class docstring untouched -- module docstring != class docstring)
- **ADV8**: `inspect.getdoc(signals_server_module)` -> contains `-> BUY/SELL/HOLD` substring (the first replacement worked)
- **ADV9**: SN4 regression smoke: `_parse_iso_date("2026-4-1") == date(2026,4,1)` -- confirms Phase 4.2.3.2 scaffold intact
- **ADV10**: Phase 4.1 regression smoke: `SignalsServer().generate_signal("AAPL", "2026-01-01")` returns a valid response dict (no import-time breakage)

## Out of scope (explicitly deferred)

- Any touch to `_parse_iso_date` (stable Phase 4.2.3.2 scaffold)
- Any touch to the `get_signal_history` `since_date` compare block
- Any touch to the 19 other byte-identical `SignalsServer` methods
- Cleanup of non-ASCII in any other backend file (`multi_agent_orchestrator.py`, `harness_memory.py`, etc. -- separate cycles)
- Phase 4.2.4 BQ durable persistence
- Phase 4.4 Go-Live Checklist
- masterplan.json status sync

## Anti-leniency rules

- Diff <= 7/7 or contract FAILS
- Exactly 0 non-ASCII bytes in the file post-edit or contract FAILS
- Exactly 0 `\u2192` occurrences post-edit or contract FAILS
- 21 methods + helper byte-identical at `ast.dump` level or contract FAILS
- No new imports, no new methods, no new comments, no renames or contract FAILS

## Verification command (deterministic, stdlib only)

```bash
python3 - <<'PY'
import ast
p = 'backend/agents/mcp_servers/signals_server.py'
src = open(p).read()
# SC1+SC2 -- parses
ast.parse(src)
# SC6+SC7 -- no non-ASCII, no U+2192
assert all(ord(c) < 128 for c in src), "FAIL: non-ASCII present"
assert '\u2192' not in src, "FAIL: U+2192 present"
# SC8 -- arrow substring count
assert src.count(' -> ') >= 7, "FAIL: missing ASCII arrows"
# SC17+SC18+SC19+SC20 -- docstring intact
mod = ast.parse(src)
doc = ast.get_docstring(mod)
assert doc is not None
for s in ['generate_signal', 'validate_signal', 'publish_signal', 'risk_check',
          'portfolio://current', 'constraints://risk', 'signals://history',
          'Tools (FastMCP @mcp.tool):', 'Resources:']:
    assert s in doc, f"FAIL: docstring missing {s}"
print("SN-audit PASS")
PY
```

## Cycle metadata

- **Research gate**: satisfied in-session -- `.claude/rules/security.md` already documents the `->` / `--` substitution rule, and prior QA already classified the 7 hits. No new design needed.
- **Base commit**: `9a53cf6`
- **Branch**: `main` (per CLAUDE.md)
- **Dev env**: no `.venv`, stdlib-only verification
- **Byte-identity reference**: `git show 9a53cf6:backend/agents/mcp_servers/signals_server.py`
