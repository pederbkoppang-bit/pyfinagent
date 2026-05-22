# phase-38.5 Research Brief -- ASCII-only logger audit script

**Date:** 2026-05-22
**Tier:** simple
**Author:** researcher subagent

## Summary

Phase-38.5 will create `scripts/qa/ascii_logger_check.py`, an AST-based
static-check script that walks `backend/` and `scripts/` looking for
`logger.*()` calls whose string-literal arguments contain non-ASCII
characters. Today the rule is enforced by convention only
(`.claude/rules/security.md` line 37). A single non-ASCII char
(em-dash, arrow, smart-quote) in a `logger.*()` literal crashes the
cycle on Windows cp1252 uvicorn handlers. The script must exit 0
on a clean tree, exit 1 with line-precise violations otherwise, and
be runnable both standalone and from a CI workflow.

Initial grep (regex `logger\.\w+\([^)]*[^\x00-\x7F]`) reveals
**138 violations across 26 files** as of 2026-05-22 (verified
2026-05-22). Affected modules cluster in three areas:
`backend/autonomous_loop.py` (the autonomous loop core),
`backend/slack_bot/*` (12 files -- the Slack-integration layer is
the heaviest emoji-user), and `backend/services/*` (ticket queue,
SLA monitor, notifications). These should be SURFACED by the
script and remediated in a separate follow-up phase (phase-38.5
ships the SCRIPT, not the cleanup -- 138 line edits would
overwhelm a single ship).

## Section A -- Internal audit (file:line)

### A.1 Canonical rule

- `.claude/rules/security.md:37` -- "ASCII-only logger messages:
  Never use Unicode characters (arrows, em dashes, etc.) in
  `logger.*()` calls. Windows cp1252 encoding in uvicorn handlers
  crashes on non-ASCII. Use `--`, `->`, plain English instead."
- `.claude/rules/security.md:38` -- "`setup_logging()` in `main.py`
  clears uvicorn handlers and forces UTF-8 `TextIOWrapper`. Still
  use ASCII for defense-in-depth."

### A.2 UTF-8 wrapper definition

- `backend/main.py:84-110` (`setup_logging`):
  - line 93: `stream = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")`
  - lines 106-109: clears uvicorn / uvicorn.error / uvicorn.access
    handlers so they cannot reintroduce cp1252 on Windows
  - The wrapper is defense layer 1 (runtime); the audit script
    is defense layer 2 (CI).

### A.3 Existing violations (CRITICAL -- must surface)

Grep pattern `logger\.\w+\([^)]*[^\x00-\x7F]` finds **138
violations across 26 files** as of 2026-05-22.

#### Affected files (full list, alphabetical)

backend/agents/openclaw_client.py
backend/api/mas_events.py
backend/autonomous_loop.py
backend/db/tickets_db.py
backend/services/queue_notification.py
backend/services/response_delivery.py
backend/services/sla_monitor.py
backend/services/slack_ticket_webhook.py
backend/services/stuck_task_reaper.py
backend/services/ticket_ingestion.py
backend/services/ticket_queue_processor.py
backend/slack_bot/app_home.py
backend/slack_bot/app.py
backend/slack_bot/assistant_handler.py
backend/slack_bot/assistant_lifecycle.py
backend/slack_bot/commands.py
backend/slack_bot/context_management.py
backend/slack_bot/governance.py
backend/slack_bot/mcp_tools.py
backend/slack_bot/self_update.py
backend/slack_bot/streaming_handler.py
backend/slack_bot/streaming_integration.py
scripts/harness/run_autonomous_loop.py
scripts/harness/run_harness.py
scripts/migrations/add_phase27_columns.py
scripts/repair_phase_23_1_17.py

#### Cluster analysis

The 138 violations cluster in three groups:

1. **Slack bot (12 files)** -- heaviest emoji-user (status icons,
   reactions, button labels). Many `logger.info(f":white_check_mark: ...")`
   patterns alongside genuine non-ASCII (em-dashes, arrows).
2. **Backend services (6 files)** -- ticket queue, SLA monitor,
   notifications. Mixed em-dash narrative + emoji status.
3. **Autonomous loop + harness (4 files)** -- the heaviest
   cycle-text user. Lines 93, 113, 142, etc. of
   `backend/autonomous_loop.py` use emoji status icons in every
   PLAN/GENERATE/EVALUATE log line. This is the highest-risk
   cluster because the loop runs unattended.

#### Highest-risk samples (representative subset)

- `backend/autonomous_loop.py:113` -- `logger.info("\U0001F680 AUTONOMOUS LOOP: Starting...")` (ROCKET)
- `backend/autonomous_loop.py:206` -- `logger.error(f"\U0001F525 ERROR in cycle ...", exc_info=True)` (FIRE) -- error path: if this fires while non-UTF-8 stderr is in play, the error itself triggers a second exception
- `scripts/repair_phase_23_1_17.py:58` -- `logger.info("DRY RUN [U+2014] would call mark_to_market() ...")` (EM-DASH at codepoint U+2014) -- runs on operator-controlled scripts
- `scripts/migrations/add_phase27_columns.py:138` -- `logger.error("POST-FLIGHT FAILED [U+2014] still missing: %s", still_missing)` (EM-DASH U+2014) -- error-path em-dash in a migration log
- `scripts/harness/run_harness.py:629` -- `logger.info("  Position concentration: %d positions [U+2192] %.1f%% max (%s)", ...)` (ARROW U+2192) -- the harness driver itself

(The audit script will surface ALL 138; cleanup is OUT OF SCOPE
for phase-38.5.)

### A.4 Existing QA-script conventions (style reference)

Only one existing script in `scripts/qa/`:

- `scripts/qa/verify_qa_roster_live.sh` -- a Bash script using
  `set -u`, sectioned `[1/3]`, `[2/3]`, `[3/3]` output, exit 1
  on hard fail. The new script should match exit-code discipline
  (0 = clean, 1 = violation) and produce line-precise output.

Project convention: a CI/QA script lives in `scripts/qa/`, uses
absolute exit codes, and reports each violation with
`file:line: message` so editors can jump straight to the offending
line.

### A.5 Anchor: ASCII rule history

The ASCII-only rule was codified in `.claude/rules/security.md`
after a cp1252 crash on uvicorn observed during Windows-host
testing (anchor: `closure_roadmap.md` OPEN-14 -- "Today the
discipline is enforced by convention only; no static check").

## Section B -- External sources (>=5 read in full)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://peps.python.org/pep-0008/ | 2026-05-22 | Official PEP | WebFetch | PEP 8 "Source File Encoding" section: "Code in the core Python distribution should always use UTF-8". For library code targeting interop with cp1252 systems, ASCII-only string literals are the conservative-defense pattern. PEP 8 also says: "Use non-ASCII characters sparingly, preferably only to denote places and human names" and "All identifiers in the Python standard library MUST use ASCII-only identifiers". Reinforces the project's source-side ASCII discipline. |
| https://docs.python.org/3/library/ast.html | 2026-05-22 | Official doc | WebFetch | `ast.parse` -> `ast.walk` traversal lets a linter pattern-match `ast.Call` nodes where `func.attr` starts with a `logger.` method name. String-literal args appear as `ast.Constant` with `value: str`. `node.lineno` + `node.col_offset` provide line-precise reporting. Recommended over regex for any non-trivial source-code inspection. |
| https://docs.astral.sh/ruff/rules/ | 2026-05-22 | Official doc | WebFetch | Ruff implements G-series (`logging-format`, focused on %- vs f-string style) and Q-series (`flake8-quotes`, quote-style preferences) but, per the WebFetch read, **does NOT have a project-configurable rule for restricting non-ASCII string literals in named function calls**. The closest niches are PLE1205 / PLE1206 (logging-too-many-args) and G004 (logging-f-string) -- both orthogonal. Justifies a custom 80-line AST script over a Ruff config tweak. |
| https://docs.python.org/3/howto/logging.html | 2026-05-22 | Official doc | WebFetch | Python `logging.StreamHandler` writes via `stream.write(msg + self.terminator)`. On Windows the default `sys.stderr` is cp1252; passing a non-encodable str raises `UnicodeEncodeError` and tears down the handler. The standard mitigations are (a) replace the stream with a UTF-8 `TextIOWrapper` (defense at runtime, what `backend/main.py:93` does) and (b) restrict literals to ASCII (defense at source, what phase-38.5 enforces). Both layers should coexist. |
| https://12factor.net/logs | 2026-05-22 | Industry std | WebFetch | The 12-Factor "Logs" principle: "Treat logs as event streams ... write its event stream, unbuffered, to stdout". Log corruption / handler crashes break the stream invariant. The 12-Factor app philosophy implies that log integrity is an availability concern, not a cosmetic one -- justifies promoting the audit script from convention to CI gate. |
| https://docs.python.org/3/library/codecs.html#standard-encodings | 2026-05-22 | Official doc | WebFetch | cp1252 (Windows Western) covers 0x00-0xFF with several gaps. Characters like U+2014 EM DASH, U+2192 RIGHTWARDS ARROW, U+2705 WHITE HEAVY CHECK MARK are unmapped in cp1252 and raise `UnicodeEncodeError` on a non-wrapped stderr. The error message form is `UnicodeEncodeError: 'cp1252' codec can't encode character '<U+2014>' in position X: ordinal not in range(256)`. Confirms the failure mode the audit script defends against. |
| https://docs.python.org/3/library/logging.handlers.html | 2026-05-22 | Official doc | WebFetch | `StreamHandler.emit` writes via the stream; encoding errors are routed to `handleError()`. By default `logging.raiseExceptions=True`, so `handleError` prints a traceback to stderr -- which on Windows is the SAME cp1252 stream. The error-handler's `print(traceback)` triggers a SECOND UnicodeEncodeError, and depending on context can crash the worker. This is the deeper failure mode: a non-ASCII logger.error() call can convert a recoverable error into a hard cycle crash. Reinforces the case for the source-side audit. |

### B.1 Snippet-only (does NOT count toward gate)

| URL | Kind | Why not read in full |
|---|---|---|
| https://flake8.pycqa.org/en/latest/plugin-development/index.html | Doc | Plugin-author guide -- a custom AST script (single file, no plugin) is simpler for the scope here. Snippet confirms the API: register a `Plugin` class with `name`, `version`, and a generator returning `(line, col, msg, type)` tuples. |
| https://github.com/PyCQA/pylint | Code | Pylint has logging-format checks but no out-of-the-box "ASCII-only literal" rule. Snippet confirms why a custom script is simpler. |
| https://docs.python.org/3/library/unicodedata.html | Doc | `unicodedata.category(c)` lets the script emit human-friendly category names ("Sm" = math symbol, "Pd" = dash punctuation). Useful for the violation report but not core to the algorithm. |
| https://github.com/astral-sh/ruff/issues/8344 | Issue | Discusses whether Ruff should add a project-configurable "ASCII-only string-literal in specific function calls" rule. Confirms this is a known niche not covered by mainstream linters. |
| https://realpython.com/python-ast/ | Tutorial | AST primer -- snippet only; the official `ast` doc above is authoritative. |
| https://www.python.org/dev/peps/pep-0263/ | PEP | Source-file encoding declarations. Pre-PEP 8 era; superseded for current Python. |
| https://github.com/PyCQA/flake8-logging-format | Code | Plugin enforcing %-formatting in logger calls. Adjacent concern; doesn't cover non-ASCII literals. |

## Section C -- Recommended script shape

### C.1 Module structure

`scripts/qa/ascii_logger_check.py` -- single-file Python 3.10+
script, stdlib-only (no third-party deps), CPU-bound, < 5 sec on
the project tree.

### C.2 Public CLI

```
$ python scripts/qa/ascii_logger_check.py
# scans backend/ + scripts/ by default
$ python scripts/qa/ascii_logger_check.py backend/ scripts/ frontend-tools/
# explicit roots
$ python scripts/qa/ascii_logger_check.py --json
# machine-readable output for CI consumption
```

### C.3 Exit-code discipline

| Exit code | Meaning | Output |
|---|---|---|
| 0 | Clean tree | One-line summary `OK: N files scanned, M logger calls, 0 violations` |
| 1 | Violations found | Per-violation lines `<path>:<line>:<col>: <method> contains non-ASCII <hex codepoints> -- "<excerpt>"` followed by one-line summary |
| 2 | Script error (syntax error in target file, etc.) | Error to stderr |

### C.4 Algorithm

For each `*.py` under each root:
1. `ast.parse(source, filename=path)`. On `SyntaxError`, skip with
   a warning to stderr (do NOT fail the audit -- syntax errors are
   a separate concern).
2. `ast.walk(tree)` over all nodes.
3. For each `ast.Call` node where the callable is
   `Attribute(value=Name(id="logger"), attr="info"|"warning"|"error"|"debug"|"critical"|"exception"|"log")`,
   recurse into all string-literal arguments
   (`ast.Constant(value=str)`, `ast.JoinedStr` parts that are
   `ast.Constant(value=str)`, `ast.Str` for older AST nodes).
4. For each string literal, scan codepoints. Any `ord(c) > 0x7F`
   triggers a violation with `node.lineno`, `node.col_offset`,
   and an excerpt of the offending literal (truncated to 80
   chars + ellipsis if longer).
5. Emit `path:lineno:col_offset: method "excerpt" contains U+XXXX (category)`.

### C.5 Method-name configuration

The default method set should match the standard Python `logging`
module API:

```python
LOGGER_METHODS = {"debug", "info", "warning", "warn", "error",
                  "critical", "exception", "log"}
```

Allow `--methods` CLI flag to override (e.g., add `print` for
projects that route prints through a captured stream). Default
covers all violations observed today.

### C.6 Allowed ranges

ASCII range is `0x00 <= ord(c) <= 0x7F`. Everything outside is a
violation. Do NOT carve exceptions for "common" unicode chars
like U+2013 (en dash) or U+2014 (em dash) -- those are exactly
the chars the rule exists to prevent.

### C.7 What NOT to flag

- F-string interpolations (the `{variable_name}` parts) -- non-literal,
  cannot be statically inspected. The AST literal-parts of an
  f-string ARE inspected; the interpolated parts are skipped.
- Comments (`#`) -- not part of the logger call.
- Docstrings -- not part of a logger call.
- Module-level constants assigned to non-logger calls.
- `logger.<method>(some_var)` where `some_var` is not a literal --
  cannot be statically inspected; out of scope.
- Methods called on non-`logger` names (e.g.,
  `my_custom_logger.info("...")`). Phase-38.5 ships the default
  set; configurable name allowlist is a follow-up.

### C.8 Integration into CI

Add a make-style target or a pre-commit hook entry referencing the
script. Suggested patterns (the planner picks one):

- **Pre-commit hook** (`.git/hooks/pre-commit` or
  `.claude/hooks/pre-commit-ascii-check.sh`): runs the script on
  staged Python files, blocks commit on exit 1.
- **CI workflow step** (if/when project adds GH Actions): one
  `python scripts/qa/ascii_logger_check.py` step in the lint job.
- **Manual / harness step**: a row in `docs/runbooks/per-step-protocol.md`
  for the Q/A agent to run before EVALUATE returns PASS.

For phase-38.5 the planner should ship the script + one
integration point (the pre-commit hook is the minimum-friction
choice; see Section G).

### C.9 Self-test

Embed a doctest at the bottom of the script (`if __name__ ==
"__main__"`) or a small unit-test file `tests/test_ascii_logger_check.py`.
Test cases:
1. Empty file -- 0 violations.
2. File with one ASCII logger call -- 0 violations.
3. File with one em-dash logger call -- 1 violation, correct line + col.
4. File with an f-string containing a literal em-dash part -- 1 violation.
5. File with an f-string containing only interpolations -- 0 violations.
6. File with a non-logger method call containing em-dash -- 0 violations
   (out of scope).
7. File with a syntax error -- 0 violations, warning to stderr,
   exit code unchanged.

## Section D -- Recency scan (last 2 years, 2024-2026)

Searched for 2024-2026 literature and tooling on:
- Python logging encoding pitfalls on Windows
- AST-based linting for project-specific rules
- Ruff / Flake8 plugin patterns for non-ASCII literal detection

Findings:
- **No new findings in the 2024-2026 window that supersede
  the canonical sources.** The `ast` module API, the
  `logging.StreamHandler` encoding behavior, the cp1252 codec
  table, and the 12-Factor logs principle have been stable since
  Python 3.8 / 2011 respectively.
- Ruff (project began 2022) has matured significantly through 2025
  but still does not implement a project-configurable "ASCII-only
  literal in named-function calls" rule out of the box. The
  GitHub issue 8344 (snippet-only above) remains open in 2026.
  A custom 80-line script is still the right tool for this
  niche.
- The increased prevalence of LLM-generated code in 2024-2026 has
  amplified the risk of unicode slipping into logger calls --
  many LLMs produce em-dashes (U+2014) by default in
  human-friendly text. This makes the audit script MORE valuable
  in 2026 than it would have been in 2023.

No new tooling, no new mitigations -- the canonical
ast+stdlib pattern remains the right approach.

## Section E -- 3-variant queries (per .claude/rules/research-gate.md)

| Variant | Query | Purpose |
|---|---|---|
| Current-year frontier | "Python logging unicode encoding error Windows 2026" | Catches current published incidents + mitigations |
| Last-2-year window | "AST linter custom rule logger ASCII 2025" | Industry tooling for AST-based custom rules |
| Year-less canonical | "Python ast module Call node attribute Constant" | Founding-doc level API reference for the algorithm |
| Year-less canonical 2 | "12-Factor logs unbuffered stdout" | Founding principle for log-stream integrity |
| Year-less canonical 3 | "cp1252 UnicodeEncodeError Python logging Windows" | Classic incident report pattern |

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 7,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```

Gate logic:
- 7 sources read in full via WebFetch (PEP 8, ast doc, Ruff rules,
  logging howto, 12-Factor logs, codecs/standard-encodings,
  logging.handlers) -- exceeds the 5-source floor.
- 14 URLs collected total -- exceeds the 10-URL floor.
- Recency scan performed -- empty (no new findings) but performed
  and reported, per `.claude/rules/research-gate.md`.
- All hard-blocker checklist items satisfied.

## Section G -- Application notes for the planner

1. **Ship the script + the pre-commit hook in a single phase.** The
   script alone is defense-in-depth-on-paper; the hook is what
   actually catches a violation before it reaches main. The hook
   adds <20 lines and matches the project's existing hook idiom
   (`.claude/hooks/*.sh`).

2. **Surface the 28+ existing violations but do NOT clean them up
   in phase-38.5.** Cleanup is high-volume, high-merge-conflict,
   and orthogonal to the script ship. Open a follow-up phase
   (phase-38.5.1 suggested) for the cleanup. The script's first
   real-world run will be the violation list -- exit 1 -- and
   that's the correct behavior; CI gating it can come in 38.5.1
   after cleanup, NOT in 38.5.

3. **Make the script stdlib-only.** No `requirements.txt` changes,
   no virtualenv dependency. Importable by any Python 3.10+
   interpreter on the project tree. This matches the project's
   QA-script convention (the existing Bash script has no deps
   beyond standard Unix tools).

4. **Match the per-step protocol's exit-code semantics.** Exit 0
   on clean / exit 1 on violations / exit 2 on script error. The
   `--json` flag (Section C.2) lets the harness consume the
   output programmatically if a future phase wires it into the
   harness EVALUATE step.

5. **Test the script on a known-bad fixture before merging.** Add
   one small Python file under `tests/fixtures/ascii_logger/`
   containing one em-dash logger call; the test asserts the
   script returns exit 1 + the expected line number. This is
   the cheapest way to keep the script honest as it evolves.

## Hard-blocker checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (13)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] No emojis in the brief
- [x] No source code edited
- [x] ASCII-only in the brief itself
- [x] Cited URL + file:line per claim
