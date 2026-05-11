---
step_id: phase-23.6.0
step_name: "Ship dotenv-syntax validator + document operator-fix for backend/.env leading-space bug"
tier: moderate
generated: 2026-05-10
---

# Research Brief: phase-23.6.0

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://github.com/theskumar/python-dotenv/blob/main/README.md | 2026-05-10 | official doc | WebFetch | "Spaces before and after keys, equal signs, and values are ignored" — python-dotenv strips surrounding whitespace at parse time, so `KEY= value` returns `"value"` (trimmed). This is the critical divergence from bash `set -a` behavior. |
| https://github.com/dotenv-linter/dotenv-linter | 2026-05-10 | official code/doc | WebFetch | Rust-only binary (v4.0.0, Oct 2025). 14 checks including "Leading character" check (catches `KEY= value` pattern). Pre-commit YAML snippet included. No Python API — subprocess-only. Not recommended for this project. |
| https://github.com/wemake-services/dotenv-linter | 2026-05-10 | official code/doc | WebFetch | Python dotenv-linter v0.7.0 (Apr 28 2025). Catches "Leading/trailing spaces in variable assignments" and "Improper spacing around equal signs". Has a CLI. No library API documented. |
| https://stefaniemolin.com/articles/devx/pre-commit/hook-creation-guide/ | 2026-05-10 | authoritative blog | WebFetch | Full recipe for custom pre-commit hook: `.pre-commit-hooks.yaml` with `id`, `entry`, `language: python`, `types`, `pass_filenames`. Exit code 0 = pass, non-zero = block commit. Entry uses `console_scripts` from `pyproject.toml`. For `repo: local` hooks, `language: script` with a direct path to the script is simpler. |
| https://gist.github.com/judy2k/7656bfe3b322d669ef75364a46327836 | 2026-05-10 | authoritative code gist | WebFetch | Leading space before variable name creates a variable named ` VAR_NAME` instead of `VAR_NAME`; community sed fix to strip whitespace around `=`. Confirms the recommended `set -o allexport; source .env; set +o allexport` pattern and its failure with leading post-`=` spaces. |
| https://slhck.info/bash/2025/11/28/safe-env-file-loading-bash.html | 2026-05-10 | authoritative blog (2025) | WebFetch | Comprehensive 2025 treatment: `source .env` failure modes include leading whitespace in comments, values with special characters. Recommends `printf -v` for command-injection-safe assignment. Confirms `|| [[ -n "$line" ]]` pattern for files lacking final newline. |
| https://pre-commit.com/ | 2026-05-10 | official doc | WebFetch | `repo: local` hook format; `pass_filenames: false` for whole-file validators; exit-code semantics: non-zero exits block the commit; CI integration via `pre-commit run --all-files`. |
| https://pypi.org/project/python-dotenv/ | 2026-05-10 | official doc | WebFetch | python-dotenv v1.2.2 (Mar 1 2026, latest). No built-in validation CLI. `dotenv_values()` returns dict, strips whitespace around `=` and around values — meaning it silently accepts `KEY= value` as `KEY=value`. This makes it unsuitable as a bash-compatibility validator. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://dotenv-linter.github.io/ | official doc | Page returned "Please wait..." (JS-rendered, no static content) |
| https://pypi.org/project/dotenv-linter/ | PyPI | Page returned JS error "A required part of this site couldn't load" |
| https://github.com/pre-commit/pre-commit-hooks | official code | Snippet: YAML/JSON/TOML check hooks; no .env-specific hook built-in |
| https://spacelift.io/blog/exit-code-127 | blog | Previously read in phase-23.5.19 research (exit 127 = command not found); no new content for this step |
| https://hexdocs.pm/dotenvy/dotenv-file-format.html | spec doc | Fetched; Elixir dotenvy format spec — useful for cross-platform perspective but not directly applicable to Python |
| https://gist.github.com/mihow/9c7f559807069a03e302605691f85572 | code gist | Previously read in phase-23.5.19; shdotenv recommendation and robust sed approach confirmed |
| https://codegenes.net/blog/set-environment-variables-from-file-of-key-value-pairs/ | blog | Fetched; confirmed inline comment and spaces-in-value failure modes; no new validator patterns |
| https://evrone.com/blog/dotenv-linter-v300 | blog | Snippet only; confirms leading_character renamed from leading_space in v3.0.0 |
| https://dotenv-linter.readthedocs.io/ | doc | Snippet: v0.7.0 docs for Python dotenv-linter |
| https://check.town/blog/env-validator-guide | blog | Previously read in phase-23.5.19; recommends CI/CD validation; no new validator code |

## Recency scan (2024-2026)

Searched for:
1. "dotenv-linter leading space validation 2026" (current-year frontier)
2. "bash set -a source .env failure modes 2025" (last-2-year window)
3. "python-dotenv dotenv_values syntax validation edge cases" (year-less canonical)

Result: Two 2025-2026 findings of note:

1. **python-dotenv v1.2.2 released March 1, 2026** — latest stable. No new validation CLI added. The silent-whitespace-stripping behavior (`KEY= value` parsed as `KEY=value`) is unchanged and explicitly documented in this version. This is the definitive current behavior.

2. **slhck.info post (Nov 28, 2025)** — 2025 article with the most comprehensive current treatment of `set -a`/`source .env` failure modes, including the `printf -v` injection-safe pattern and handling of files without trailing newlines.

3. **dotenv-linter (Rust) v4.0.0 released Oct 18, 2025** — major release. Still Rust-only binary. The "leading_character" check (renamed from "leading_space" in v3.0.0) catches `KEY= value`. Not recommended as a dependency for this project.

4. **wemake-services/dotenv-linter (Python) v0.7.0 released April 28, 2025** — Python package, but CLI-only with no documented library API. Catches leading/trailing space violations.

No new findings supersede the core mechanism: `KEY= value` under bash `set -a` tokenizes as `KEY=""` + `value` command execution. This is unchanged POSIX shell behavior.

## Key findings

1. **python-dotenv silently accepts `KEY= value`** — strips surrounding whitespace at parse time, returning `"value"` without the leading space. This makes `dotenv_values()` unsuitable as a bash-compatibility validator: it will report no error on precisely the lines that break `set -a`. (Source: python-dotenv README, https://github.com/theskumar/python-dotenv/blob/main/README.md, v1.2.2 PyPI, accessed 2026-05-10)

2. **Pure Python regex is the right approach** — no external tool dependency; zero install friction; can be run ad-hoc by the operator OR imported in a pytest test. The regex pattern `^[A-Z_][A-Z0-9_]*=\s+\S` identifies the `KEY= value` bug class. (Derived from bash parsing mechanics; confirmed by spacelift.io exit-127 explanation, phase-23.5.19 brief)

3. **Bash `set -a` + `source` failure modes (comprehensive list):**
   - **Leading space after `=`** (`KEY= value`): bash tokenizes as `KEY=""` then executes `value` as a command → exit 127. This is the confirmed bug in pyfinagent `backend/.env` lines 24, 25, 56.
   - **Inline comment after unquoted value** (`KEY=val # comment`): bash includes `# comment` in the value (no stripping), which may cause downstream failures but does NOT cause exit 127.
   - **Special characters in values** (spaces, `$`, `!`, `&`, `*`): may cause word-splitting, glob expansion, or command substitution if value is unquoted.
   - **Missing trailing newline**: bash `read` misses the last line; the `set -a; source` idiom handles this via shell's built-in file sourcing (not `read` loop), so trailing-newline is generally safe with `source` but risky with `while IFS= read -r` loops.
   - **Lines with spaces around key** (` KEY=value`): bash treats as command, not assignment → exit 127.
   (Sources: slhck.info 2025, gist.github.com/judy2k and mihow, codegenes.net, spacelift.io)

4. **Cross-platform divergence — bash `set -a` vs python-dotenv:**
   | Syntax | bash `set -a` result | python-dotenv result |
   |--------|---------------------|---------------------|
   | `KEY=value` | KEY="value" (correct) | KEY="value" (correct) |
   | `KEY= value` | "" then executes "value" → exit 127 | KEY="value" (strips space, silent) |
   | `KEY=value # comment` | KEY="value # comment" (includes comment) | KEY="value" (strips comment) |
   | `KEY ="value"` | syntax error | KEY="value" (strips space around =) |
   | `KEY=` | KEY="" | KEY="" |
   | ` KEY=value` | command not found ("KEY=value" as cmd) | KEY="value" |
   (Sources: python-dotenv README; slhck.info 2025; gist.github.com/judy2k; dotenvy spec)

5. **Pre-commit hook pattern** — `repo: local` with `language: script` is the minimal approach for a single-file Python script without packaging. Alternatively, `language: python` with `additional_dependencies: []` and the script as a `console_scripts` entry. For ad-hoc use, direct `python scripts/validators/check_dotenv_syntax.py` is sufficient. Exit 0 = clean, exit 1 = violations found. (Source: pre-commit.com docs; stefaniemolin.com hook guide, accessed 2026-05-10)

6. **dotenv-linter (Rust) exit behavior** — non-zero on any violation; pre-commit integration via `rev: 3.3.1` (or 4.0.0) snippet exists. But: requires Rust binary globally installed, which is inappropriate for a Mac-only local deployment per `project_local_only_deployment.md`. (Source: github.com/dotenv-linter/dotenv-linter, accessed 2026-05-10)

7. **Idempotency** — a read-only validator that only reads the file and prints violations is inherently idempotent. Exit-code semantics: 0 = file is clean, 1 = one or more violations found. Running it N times on the same file always returns the same result. The operator-fix `sed` sequence is the only mutating operation, and it too is idempotent if restricted to lines where the pattern still matches. (Derived from standard Unix validator conventions)

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/autoresearch/run_nightly.sh` | 35 | Launchd wrapper: sources `backend/.env` via `set -a; . backend/.env; set +a` at lines 14-19, then runs `run_memo.py` | Read in full; `set -euo pipefail` means first exit-127 aborts the script |
| `handoff/archive/phase-23.3.5/phase-23.3.5-audit-findings.md` | 112 | Phase-23.3.5 operator-fix sequence for lines 24, 25, 56 | Read in full; operator-fix sequence quoted verbatim below |
| `handoff/archive/phase-23.5.19/research_brief.md` | ~80 | Phase-23.5.19 research on autoresearch exit code (127→1 transition) | Read partially; confirms exit-code mechanism and 6 prior sources |
| `.git/hooks/pre-commit` | 32 | Existing pre-commit hook (stale Claude snapshot IDs + bak-files check) | Read in full; NO dotenv check currently; safe to extend |
| `tests/services/` | dir | Existing service-layer test directory | `ls` only; contains test_log_path_allowlist.py as a structural model |
| `tests/api/` | dir | Existing API test directory | `ls` only; no dotenv-related test files found |
| `scripts/` | dir | Scripts directory | `ls` only; no `validators/` subdirectory exists yet |
| `backend/.env` | unknown | The target file | **Sandbox-blocked**: `ls`, `stat`, `python3 open()` all denied by session permissions |

### Operator-fix sequence (verbatim from phase-23.3.5 audit findings, file:line 67-88)

```bash
# Inspect the broken lines first:
awk 'NR==24 || NR==25 || NR==56 {print NR": "$0}' backend/.env

# Surgical fix: collapse the leading space after `=` on broken lines.
# The pattern is `KEY= value` -> `KEY=value`. sed handles all 3 in one go:
sed -i '' '24s/^\([A-Z_]*\)= /\1=/' backend/.env
sed -i '' '25s/^\([A-Z_]*\)= /\1=/' backend/.env
sed -i '' '56s/^\([A-Z_]*\)= /\1=/' backend/.env

# Verify:
awk 'NR==24 || NR==25 || NR==56 {print NR": "$0}' backend/.env

# Recovery (force the next nightly to pick up the fix immediately):
launchctl bootout gui/501/com.pyfinagent.autoresearch 2>/dev/null
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
launchctl bootout gui/501/com.pyfinagent.ablation 2>/dev/null
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.pyfinagent.ablation.plist
launchctl kickstart gui/501/com.pyfinagent.autoresearch
launchctl kickstart gui/501/com.pyfinagent.ablation
sleep 5
launchctl list | grep -E "(autoresearch|ablation)"   # both should be exit 0
```

### Internal .env access confirmation

All three access attempts (`ls backend/.env`, `stat backend/.env`, `python3 -c "open('backend/.env')"`) returned permission-denied at the Bash tool level. The file is sandbox-blocked for this Claude Code session, confirming the deliverable is the validator + operator-fix doc, not direct file editing.

## Consensus vs debate (external)

**Consensus:** Pure Python regex validator is the right tool for this project. python-dotenv's `dotenv_values()` is NOT appropriate for bash-compatibility validation because it silently strips the leading space that causes bash exit 127. The Rust dotenv-linter is inappropriate because it requires a globally installed binary on a local Mac deployment.

**Debate:** Whether the validator should live in `scripts/validators/` (ad-hoc operator use) or `tests/services/` (CI/pytest). The evidence supports BOTH: an operator-runnable `scripts/validators/check_dotenv_syntax.py` plus a `tests/services/test_dotenv_syntax.py` that imports from it. No conflict — different invocation contexts.

## Pitfalls (from literature)

1. **Using python-dotenv to validate** — it silently accepts `KEY= value` as valid, masking the exact bug class this validator must catch. Never use `dotenv_values()` as the validation mechanism.
2. **Line-number hardcoding in sed** — the phase-23.3.5 sed sequence uses line numbers (24, 25, 56). If the operator has already partially fixed some lines, running the sed again on an already-fixed line may corrupt the value if the pattern no longer matches. The validator must re-scan to identify current offending lines before the operator applies the sed.
3. **Not handling the final-newline edge case** — a `.env` file without a trailing newline may cause the last line to be missed by a `while read` loop. The validator must use `python open()` with `readlines()` or ensure the read loop handles it.
4. **Pre-commit hook blocking .env commits** — if the validator runs as a pre-commit hook and the operator commits a partially-fixed `.env`, the hook blocks the commit. The hook should only trigger on changes to `backend/.env` specifically (use `files: backend/\.env$` in the hook config).
5. **Dotenv-linter (Rust) v4.0.0 is a Rust binary** — downloading/installing it globally on the operator's Mac introduces a system-level dependency that is inconsistent with the local-only, no-infra deployment pattern documented in memory.

## Application to pyfinagent (mapping findings to file:line anchors)

| Finding | File:Line | Implication |
|---------|-----------|-------------|
| `set -a; . backend/.env; set +a` is the bash sourcing mechanism | `scripts/autoresearch/run_nightly.sh:14-19` | Any leading-space line after `=` will cause `set -euo pipefail` to abort at that point |
| Lines 24, 25, 56 identified as leading-space offenders | `handoff/archive/phase-23.3.5/phase-23.3.5-audit-findings.md:53-75` | Exit-1 transition (from 127) in phase-23.5.19 suggests lines 24/25 may now be fixed; line 56 status unknown |
| No `.pre-commit-config.yaml` exists | repo root | Easy to add; `.git/hooks/pre-commit` exists and is safe to extend or replace with a pre-commit managed hook |
| No `scripts/validators/` directory exists | repo root | Must be created; `tests/services/` exists and is the model for the pytest test |
| `tests/services/test_log_path_allowlist.py` pattern | `tests/services/test_log_path_allowlist.py:1-32` | Use same module-import + assertion pattern for `test_dotenv_syntax.py` |

---

## Four Concrete Recommendations for Main

### 1. Validator implementation language/approach

**Recommendation: pure Python regex, no external dependencies.**

Rationale:
- python-dotenv's `dotenv_values()` CANNOT be used: it silently accepts `KEY= value` as `KEY=value`, masking the exact bug class. (Source: python-dotenv README, v1.2.2, https://github.com/theskumar/python-dotenv/blob/main/README.md)
- Rust dotenv-linter (v4.0.0, Oct 2025) requires a globally installed Rust binary — inappropriate for the Mac-only local deployment. (Source: github.com/dotenv-linter/dotenv-linter)
- Python dotenv-linter (v0.7.0, Apr 2025, wemake-services) catches the right violations but is CLI-only with no library API, adding a pip dependency for something a 40-line stdlib script can do.
- Pure Python regex with stdlib only (`re`, `pathlib`, `sys`) is zero-dependency, trivially installable (it's already in the venv), importable from pytest, and idempotent. The operator can run it directly; pytest can import and assert on its output.

### 2. Validation rules to enforce

The validator MUST catch (causing bash exit 127 or silent value corruption):

| Rule | Pattern | Severity |
|------|---------|----------|
| **Leading space after `=`** | `^[A-Z_][A-Z0-9_]*=\s+\S` | CRITICAL — causes exit 127 |
| **Leading space before key** | `^\s+[A-Z_]` | CRITICAL — causes exit 127 (key treated as command) |
| **Trailing whitespace in value** (unquoted) | `^[A-Z_][A-Z0-9_]*=[^\n\r#"']+\s+$` | WARNING — value includes trailing space |
| **Inline comment after unquoted value** | `^[A-Z_][A-Z0-9_]*=[^"'\n]+\s+#` | WARNING — comment included in value |
| **Missing trailing newline** | file doesn't end with `\n` | INFO — last line may be missed by some parsers |

The CRITICAL rules are the minimum viable set for this phase. WARNING/INFO can be reported but should not cause non-zero exit in a first implementation if the operator wants a permissive mode.

### 3. Operator-fix sed sequence (refined)

The phase-23.3.5 sequence hardcodes line numbers. The safer general-purpose sequence is:

```bash
# Step 1: scan for ALL leading-space-after-= lines (don't hardcode line numbers):
grep -n '^\([A-Z_][A-Z0-9_]*\)= ' backend/.env

# Step 2: surgical in-place fix (macOS BSD sed syntax):
# Strips the single leading space after = only when the pattern KEY= value matches.
# Safe: only matches lines where the key is ALL_CAPS_WITH_UNDERSCORES and there is
# exactly one space after = followed by a non-space character.
sed -i '' 's/^\([A-Z_][A-Z0-9_]*\)= \([^ ]\)/\1=\2/' backend/.env

# For multiple leading spaces (KEY=  value):
sed -i '' 's/^\([A-Z_][A-Z0-9_]*\)=  *\([^ ]\)/\1=\2/' backend/.env

# Step 3: verify no leading-space lines remain:
grep -n '^\([A-Z_][A-Z0-9_]*\)= ' backend/.env && echo "STILL BROKEN" || echo "CLEAN"

# Step 4 (from phase-23.3.5): restart the launchd services:
launchctl bootout gui/501/com.pyfinagent.autoresearch 2>/dev/null
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist
launchctl kickstart gui/501/com.pyfinagent.autoresearch
sleep 5
launchctl list | grep autoresearch   # should show "last exit code = 0"
```

**Why the refinement over phase-23.3.5:** The original sequence used line-specific `24s/...` and `56s/...` addressing. Since lines 24/25 may already be fixed (exit code transitioned from 127 to 1 per phase-23.5.19), re-applying a line-specific sed to an already-clean line is safe only if the pattern no longer matches. The global pattern `s/^\([A-Z_][A-Z0-9_]*\)= \([^ ]\)/\1=\2/` is idempotent: it only modifies lines where the bug pattern still exists.

### 4. Where to put the validator

**Both locations, different roles:**

- **`scripts/validators/check_dotenv_syntax.py`** — operator-runnable ad-hoc script. Takes a `.env` file path as CLI argument (default: `backend/.env`). Prints violations with line numbers. Exits 0 = clean, 1 = violations found. The operator runs this BEFORE and AFTER applying the sed fix.

- **`tests/services/test_dotenv_syntax.py`** — pytest test that imports the validator function from `scripts/validators/check_dotenv_syntax.py` and asserts zero violations against a known-good synthetic `.env` string. Does NOT attempt to open the real `backend/.env` (sandbox-blocked in CI). Instead, tests the validator logic itself with embedded fixtures.

For the pre-commit hook, add to `.git/hooks/pre-commit` (simplest: no `.pre-commit-config.yaml` needed, extends the existing hook):

```bash
# In .git/hooks/pre-commit, append after existing checks:
# Check backend/.env for leading-space-after-= violations if .env is staged:
if git diff --cached --name-only | grep -q 'backend/\.env$'; then
    python scripts/validators/check_dotenv_syntax.py backend/.env || {
        echo "pre-commit: backend/.env has leading-space violations (see above)"
        exit 1
    }
fi
```

This approach avoids creating a `.pre-commit-config.yaml` (no new infrastructure), uses the existing hook file, and only triggers when `backend/.env` is staged for commit.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 fetched)
- [x] 10+ unique URLs total (incl. snippet-only) (18 unique URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted (python-dotenv vs bash divergence)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 10,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
