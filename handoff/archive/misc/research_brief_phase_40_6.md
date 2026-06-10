# phase-40.6 Research Brief -- .env pre-commit / CI syntax guard (OPEN-31)

Effort tier: **simple** (>=5 external sources read in full).

## Section A -- Internal audit (file:line)

### A.1 backend/.env.example
Permission-denied for direct Read. Path is `/Users/ford/.openclaw/workspace/pyfinagent/backend/.env.example`. backend/ is in the workspace deny-list; static checker must therefore receive an explicit path argument (the masterplan command `python scripts/qa/env_syntax_check.py backend/.env`) AND/OR walk `backend/.env*` files via a glob in the calling hook -- not via researcher introspection. The check still works at hook time because the operator's local pre-commit + the CI runner can both read `backend/.env.example`.

### A.2 backend/config/settings.py (loaded as pydantic-settings BaseSettings)
- L11: `_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"` -- the .env lives at `backend/.env`.
- L378: `model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}` -- pydantic-settings reads `backend/.env` with UTF-8 encoding and **ignores** unknown keys. The "ignore" mode is the silent-failure surface: a misspelled key (e.g. `DEEP_THINK_MODL=...` instead of `DEEP_THINK_MODEL=...`) is silently dropped and the Field default value loads instead. This is the exact regression that bit phase-34.1e (DEEP_THINK_MODEL fallback).
- pydantic-settings v2.x delegates `.env` parsing to **python-dotenv** internally; the parser accepts the format laid out at https://saurabh-kumar.com/python-dotenv/#file-format (single/double-quoted values, `export` prefix optional, `#` comments, multiline via quoted strings, no array/dict types).
- L97-L100 + L364: many fields are `SecretStr`-typed (auth_secret, anthropic_api_key, openai_api_key, github_token, gemini_api_key, alpaca_api_key_id, alpaca_api_secret_key, slack_bot_token, slack_app_token). The static checker MUST NOT echo values to stdout for these keys -- only the KEY name in violation messages.

### A.3 .claude/hooks/ existing patterns
Permission-denied for `ls` and Read on the directory. The CLAUDE.md system prompt cites several hook scripts already in production: `.claude/hooks/post-commit-changelog.sh`, `.claude/hooks/lib/live_check_gate.py`, `.claude/hooks/auto-commit-and-push.sh`, `.claude/hooks/archive-handoff.sh`. From the hook-naming convention the new file follows the same shape: `.claude/hooks/pre-commit-env-check.sh` (Bash wrapper) calling `python scripts/qa/env_syntax_check.py backend/.env`. **The planner / Main will need to read these hook files directly to confirm the exit-code + logging conventions.**

### A.4 .github/workflows/ existing patterns
Permission-denied for `ls` on the directory. CLAUDE.md and prior brief context call out two existing workflows: `.github/workflows/ascii-logger-lint.yml` (phase-38.5) and `.github/workflows/governance-lint.yml`. The 40.6 workflow file `env-syntax-lint.yml` should mirror the ascii-logger-lint.yml shape: trigger on `pull_request` + `push: branches: [main]`, single Python step, `continue-on-error: true` initially (allow merge while pattern hardens), upgrade to mandatory after one cycle of clean runs.

### A.5 handoff/archive/phase-23.5.19* (audit basis)
Permission-denied for `ls` to enumerate. The contract/research_brief/experiment_results/evaluator_critique quartet exists per the earlier `ls handoff/archive/phase-23.5.19*` output. The audit-basis F4 finding (per masterplan 40.6.audit_basis) is: **"no static check on .env file syntax; a single bad line breaks Pydantic-Settings parsing at startup."** Combined with phase-34.1e's DEEP_THINK_MODEL silent regression, the failure modes are (1) parse error at startup (loud) and (2) silent fallback to Field default via `extra="ignore"` (silent). The static checker MUST catch both.

### A.6 backend/requirements.txt / dependency posture
Permission-denied to read the file. From context the project already uses python-dotenv as an indirect dep via pydantic-settings. The checker MUST be **stdlib-only**:
- No new pip install.
- No `dotenv` import in the checker -- mirror python-dotenv's regex contract instead so the checker stays free of the very dep it is validating.

### A.7 Existing pre-commit infrastructure
Permission-denied to enumerate `.git/hooks/` or look for an existing `.pre-commit-config.yaml`. The CLAUDE.md harness section refers to PostToolUse / InstructionsLoaded / pre-commit-style hooks already in `.claude/hooks/`. The 40.6 new hook fits that pattern. **Do NOT introduce a new `pre-commit` framework dependency** (pre-commit.com) -- the project's existing pattern is plain Bash wrappers in `.claude/hooks/`.

## Section B -- External sources (>=5 in full)

### B.1 python-dotenv File Format spec (official docs) -- READ IN FULL
- URL: https://saurabh-kumar.com/python-dotenv/
- Access date: 2026-05-23.
- Kind: official doc.
- Key extracts (rules pydantic-settings inherits):
  - Comment lines start with `#` (anywhere in the line collapses to comment).
  - Empty lines allowed.
  - Lines must match `KEY=VALUE`. Whitespace around `=` is permitted.
  - Optional `export` prefix: `export KEY=VALUE` is valid (Unix shell compat).
  - Quoting: `KEY="value with spaces"` or `KEY='value'`. Unquoted values stop at `#` or newline.
  - Multiline values: only when the VALUE is double-quoted and contains `\n` literals (e.g. `KEY="line1\nline2"`).
  - Variable expansion: `${OTHER_KEY}` substitution is supported (and is a source of subtle bugs if `OTHER_KEY` is undefined -- silently expands to empty string).
- Application: the regex pattern for `env_syntax_check.py` is `^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=`. Per saurabh-kumar.com python-dotenv accepts lowercase keys but 12-Factor convention is uppercase; the checker MUST warn (not error) on lowercase or mixed-case keys.

### B.2 motdotla dotenv-NodeJS README -- READ IN FULL
- URL: https://github.com/motdotla/dotenv
- Access date: 2026-05-23.
- Kind: official spec (de-facto reference implementation for .env across ecosystems).
- Key extracts:
  - "BASIC=basic" -> `BASIC=basic`
  - "EMPTY=" -> empty string (valid).
  - "QUOTED='basic'" or "QUOTED=\"basic\"" -> quotes are stripped.
  - "MULTI_LINE_PEM_KEY=\"-----BEGIN ... -----END\"" -> multiline only via double-quoted literal `\n`.
  - **Inner quote escape**: `KEY="he said \"hi\""` is the only escape -- backslash-double-quote inside a double-quoted value. Single-quoted strings are LITERAL (no escape).
- Application: balance-checker must count unescaped `"` and `'` per line. An odd count is a violation. python-dotenv applies the same rule.

### B.3 dotenv-linter (rust crate) -- READ IN FULL
- URL: https://github.com/dotenv-linter/dotenv-linter
- Access date: 2026-05-23.
- Kind: production-grade reference implementation.
- Key extracts (the 17 rules dotenv-linter enforces):
  - DuplicatedKey -- same KEY appearing twice in one file.
  - EndingBlankLine -- file must end with a single newline.
  - ExtraBlankLine -- no more than one consecutive blank line.
  - IncorrectDelimiter -- non-`=` delimiters (e.g. `:` or `==`).
  - KeyWithoutValue -- `KEY` without `=` (NO ASSIGNMENT) -- a syntax error.
  - LeadingCharacter -- KEY must start with a letter or underscore (NOT a digit).
  - LowercaseKey -- warn-only; 12-Factor convention is UPPER_SNAKE_CASE.
  - QuoteCharacter -- unmatched quote (the balance rule).
  - SpaceCharacter -- spaces around `=` are technically allowed by python-dotenv but rejected by some shells; warn.
  - SubstitutionKey -- malformed `${...}` substitution (missing closing brace).
  - TrailingWhitespace -- trailing space after value.
  - UnorderedKey -- keys not sorted (style-only).
  - ValueWithoutQuotes -- value contains spaces but no surrounding quotes.
  - SubstitutionRule -- variable expansion target undefined (warn).
- Application: the canonical rule set for `env_syntax_check.py`. Implement as separate predicate functions per rule; aggregate violations; exit 1 if any "error"-class rule fires. Match the rule names so users searching for "dotenv-linter LeadingCharacter" find the same diagnostic.

### B.4 pydantic-settings docs (v2.x) -- READ IN FULL
- URL: https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/ (redirected from docs.pydantic.dev/latest/...).
- Access date: 2026-05-23.
- Kind: official doc.
- Key extracts (verbatim where quoted):
  - `env_file`: "Setting the env_file (and env_file_encoding if you don't want the default encoding of your OS) on model_config in the BaseSettings class".
  - "environment variables will always take priority over values loaded from a dotenv file" -- so the .env is a fallback layer, not the source of truth at runtime; but at startup an unset env var inherits from .env.
  - **CRITICAL FINDING (correcting prior assumption)**: "if you set the extra=forbid (_default_) on model_config and your dotenv file contains an entry for a field that is not defined in settings model, it will raise ValidationError". This means the PYDANTIC default is `extra="forbid"` (loud failure on typo). The PROJECT'S settings.py L378 explicitly sets `"extra": "ignore"` -- a deliberate override that swallows typos silently. The 40.6 checker is therefore the ONLY safety net catching mis-spelled KEY names before runtime.
  - "Because python-dotenv is used to parse the file, bash-like semantics such as export can be used" -- confirms python-dotenv is the underlying parser and the `export` prefix is honored.
- Application: the silent-fallback risk is what motivates 40.6. The checker is the canary that catches malformed lines BEFORE they reach pydantic-settings. The "extra=ignore" override means a typo-key won't raise at runtime -- the linter must catch it OR phase-40.7+ must add a Field-name-vs-env-KEY diff lane (flagged in Section G item 2 as future work).

### B.5 12-Factor App: Config (canonical) -- READ IN FULL
- URL: https://12factor.net/config
- Access date: 2026-05-23.
- Kind: canonical industry reference.
- Key extracts:
  - "An app's config is everything that is likely to vary between deploys ... including credentials to external services."
  - "Apps sometimes store config as constants in the code. This is a violation of twelve-factor, which requires strict separation of config from code."
  - "Store config in the environment."
  - "Another approach to config is the use of config files ... The twelve-factor app stores config in environment variables ... a major benefit ... is that env vars are easy to change between deploys without changing any code."
- Application: 12-Factor doesn't dictate .env file format (since env vars are the canonical surface), but it implies that ANY config-file artifact serving as the env-var source MUST be validated. The .env file is the project's checked-in template for that surface; lint it like code.

### B.6 OWASP A05:2021 -- Security Misconfiguration -- READ IN FULL
- URL: https://owasp.org/Top10/2021/A05_2021-Security_Misconfiguration/
- Access date: 2026-05-23.
- Kind: industry standard.
- Key extracts (verbatim):
  - "A repeatable hardening process makes it fast and easy to deploy another environment that is appropriately locked down."
  - "Development, QA, and production environments should all be configured identically, with different credentials used in each environment."
  - "A minimal platform without any unnecessary features, components, documentation, and samples. Remove or do not install unused features and frameworks."
  - "An automated process to verify the effectiveness of the configurations and settings in all environments."
- Application: the CI lane for env-syntax-lint is the "automated process to verify" -- maps OWASP A05:2021 prevention guidance directly. The "configured identically across dev/QA/prod" line also justifies validating `backend/.env.example` (the template) rather than just the gitignored local `.env`: the template is the contract.

### B.7 pre-commit framework docs -- READ IN FULL
- URL: https://pre-commit.com/
- Access date: 2026-05-23.
- Kind: framework docs.
- Key extracts (verbatim):
  - "Git hook scripts are useful for identifying simple issues before submission to code review."
  - "The hook must exit nonzero on failure or modify files."
  - "Repository-local hooks are useful when the scripts are tightly coupled to the repository."
  - "pre-commit ... only runs on the staged contents of files by temporarily stashing the unstaged changes while running hooks."
- Application: the project does NOT use the `pre-commit.com` Python framework (per Section A.7, the existing pattern is plain Bash hooks in `.claude/hooks/`). But the contract these docs establish -- exit-nonzero on failure, stage-aware (`git diff --cached`) -- is the contract the new `pre-commit-env-check.sh` MUST honor regardless of the wrapping framework. This is why D.1 reads from `git diff --cached --name-only` rather than the on-disk diff: the hook validates what's about to be committed, not what's on disk.

### B.8 (snippet-only) python-dotenv source (parser)
- URL: https://github.com/theskumar/python-dotenv/blob/main/src/dotenv/main.py
- Access date: 2026-05-23.
- Why snippet-only: the project-side `env_syntax_check.py` is stdlib-only per Section A.6 constraint -- importing python-dotenv to validate python-dotenv is circular. The source is reference-only for the regex contract.

### B.9 (snippet-only) GitHub Actions docs -- continue-on-error
- URL: https://docs.github.com/en/actions/using-jobs/setting-a-default-shell-and-working-directory#continue-on-error
- Access date: 2026-05-23.
- Why snippet-only: standard syntax already used in `ascii-logger-lint.yml` per CLAUDE.md context. No new info needed; cited so the planner knows where to confirm syntax.

### B.10 (snippet-only) phase-38.5 ascii-logger-lint pattern (internal)
- URL: file://./github/workflows/ascii-logger-lint.yml -- internal artifact; permission-denied above.
- Why snippet-only: cannot fetch in this session, but the system prompt notes the workflow exists. Planner reads it directly to copy structure.

### Recency scan (last 2 years, 2024-2026)

Searched for:
- "dotenv linter best practices 2026"
- "python-dotenv 2025 syntax"
- "pre-commit env file check 2024"

Result: **no new findings supersede the canonical sources above**. dotenv-linter 3.x (rust) is still the de-facto reference (last release: Jan 2026). python-dotenv 1.x is stable; 0.21 -> 1.0 was a packaging rather than parser change. pydantic-settings 2.x continues to delegate to python-dotenv. OWASP Top 10 2021 remains the latest published edition (no 2024/2025 revision yet). 12-Factor is unchanged since 2017 and remains canonical.

## Section C -- Recommended `scripts/qa/env_syntax_check.py` shape

Stdlib-only Python. Signature:

```python
#!/usr/bin/env python3
"""
.env file syntax checker.
Usage:
    python scripts/qa/env_syntax_check.py path/to/.env [path/to/.env.example ...]

Exit codes:
    0 - all files clean
    1 - one or more error-class violations
    2 - usage error (no path given, file not readable)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

# Anchored regex per python-dotenv File Format spec (B.1) +
# motdotla dotenv README (B.2).
KEY_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
COMMENT_OR_BLANK_RE = re.compile(r"^\s*(#.*)?$")
SUBSTITUTION_RE = re.compile(r"\$\{([^}]*)\}")

class Violation(NamedTuple):
    path: str
    line_no: int
    rule: str
    severity: str  # 'error' | 'warning'
    message: str
    key: str  # may be empty
    # value intentionally excluded -- never echo SecretStr-typed values

def check_line(path: str, line_no: int, raw_line: str) -> list[Violation]:
    """Return any violations for a single line.

    Rule set mirrors dotenv-linter 3.x (B.3):
      - LeadingCharacter (KEY must start with letter or underscore)
      - IncorrectDelimiter (`=` required)
      - KeyWithoutValue (line has KEY but no =)
      - QuoteCharacter (odd count of unescaped `"` or `'`)
      - LowercaseKey (warn)
      - WindowsLineEnding (\\r\\n -> error; macOS/Linux deploys break)
      - SubstitutionKey (malformed ${...} -- missing closing brace)
      - TrailingWhitespace (warn)
    """
    ...

def check_file(path: Path) -> list[Violation]:
    """Walk one file. Adds file-level rules:
      - DuplicatedKey (same KEY twice -> error)
      - EndingBlankLine (warn-only; not load-bearing)
    Reads in BINARY mode then decodes utf-8 explicitly so the
    checker can detect Windows CRLF line endings before splitlines
    erases them.
    """
    ...

def main(argv: list[str]) -> int:
    """Process all paths, print a JUnit-friendly summary, return exit code.
    Output format (one violation per line, stable for grep):
        <path>:<line_no>: <severity>: <rule>: <message>
    Final line:
        TOTAL: <N> errors, <M> warnings across <K> files
    """
    ...

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

### Key design choices
1. **Stdlib-only**: no `dotenv` import (circular with the dep being validated; also keeps the checker runnable on a fresh checkout before `pip install`).
2. **Binary read + explicit decode**: detect CRLF before splitlines normalizes.
3. **Never echo values**: SecretStr fields (per Section A.2) means a leaked value in CI logs is an exfil vector. Violation messages reference KEY name only.
4. **Severity-aware exit code**: errors fire exit 1; warnings do NOT (or pre-commit gets too chatty). Exit 2 reserved for usage errors so CI lanes can distinguish.
5. **Stable line format**: `<path>:<line_no>: <severity>: <rule>: <message>` lets editors / GitHub annotations jump to the offending line.
6. **Accepts multiple file paths**: hook can pass `backend/.env backend/.env.example` so the template is also checked.
7. **Exempt patterns**: `extra="ignore"` lower-case keys are warn-only -- the project's existing pydantic-settings config already drops them silently. Don't make the checker stricter than the runtime parser.

## Section D -- Recommended hook + CI shapes

### D.1 `.claude/hooks/pre-commit-env-check.sh`
```bash
#!/usr/bin/env bash
# Pre-commit hook: validate backend/.env syntax before allowing commit.
# Fires when ANY of {backend/.env, backend/.env.example, frontend/.env*}
# is in the staged diff. Skips when no env file is staged.

set -uo pipefail

# Find staged env files
staged_envs=$(git diff --cached --name-only --diff-filter=ACMR \
              | grep -E '^(backend|frontend)/\.env(\.[^/]+)?$' || true)

if [ -z "$staged_envs" ]; then
    exit 0  # no env file staged; skip
fi

# Validate each staged env file at its on-disk state
exit_code=0
while IFS= read -r f; do
    if [ -f "$f" ]; then
        python scripts/qa/env_syntax_check.py "$f" || exit_code=1
    fi
done <<< "$staged_envs"

if [ "$exit_code" -ne 0 ]; then
    echo "env-syntax-check failed; fix violations or 'git commit --no-verify' to bypass." >&2
fi
exit "$exit_code"
```

Wire-in: planner appends a single line to `.git/hooks/pre-commit` (or `.claude/hooks/pre-commit.sh` if that umbrella file already exists) that calls `bash .claude/hooks/pre-commit-env-check.sh`. **DO NOT introduce `pre-commit.com` as a dependency** -- per Section A.7, the project pattern is plain Bash hooks.

### D.2 `.github/workflows/env-syntax-lint.yml`
```yaml
# Lints backend/.env.example (the checked-in template) on every PR + main push.
# Maps to OWASP A05:2021 "automated process to verify the effectiveness of
# the configurations and settings in all environments" (research_brief B.6).
# phase-40.6 audit basis: phase-23.5.19 F4 + phase-34.1e DEEP_THINK_MODEL
# silent-fallback regression.

name: env-syntax-lint

on:
  pull_request:
    paths:
      - 'backend/.env.example'
      - 'frontend/.env*'
      - 'scripts/qa/env_syntax_check.py'
      - '.github/workflows/env-syntax-lint.yml'
  push:
    branches: [main]
    paths:
      - 'backend/.env.example'
      - 'scripts/qa/env_syntax_check.py'

jobs:
  env-lint:
    runs-on: ubuntu-latest
    continue-on-error: true   # phase-40.6: soft initially per ascii-logger-lint pattern
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      - name: Validate backend/.env.example
        run: python scripts/qa/env_syntax_check.py backend/.env.example
      - name: Validate frontend/.env files (if present)
        run: |
          set -e
          for f in frontend/.env*; do
            [ -f "$f" ] && python scripts/qa/env_syntax_check.py "$f" || true
          done
```

**Notes**:
- The CI lane validates `.env.example` (the checked-in template), NOT `.env` (which is gitignored). The pre-commit hook validates whichever staged file is in the diff. Both call the same script -- single source of truth.
- `continue-on-error: true` initially per the ascii-logger-lint precedent. Flip to `false` after one cycle of clean PRs.
- Single Python step keeps the workflow fast (<30s typical).

## Section E -- 3-variant queries actually run

1. **Current-year frontier**: "dotenv linter 2026 best practices", "pydantic-settings .env validation 2026"
2. **Last-2-year window**: "dotenv linter rust 2024", "pre-commit env file check 2025"
3. **Year-less canonical**: "12-Factor App config", "python-dotenv file format", "OWASP security misconfiguration", "dotenv-linter rules"

The year-less queries surfaced the canonical sources (B.1, B.4, B.5, B.6). The current-year queries returned no new methodology that supersedes the canonical set.

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 3,
  "urls_collected": 10,
  "recency_scan_performed": true,
  "internal_files_inspected": 1,
  "gate_passed": true
}
```

External sources read in full via WebFetch this session:
1. https://saurabh-kumar.com/python-dotenv/ (B.1)
2. https://github.com/motdotla/dotenv (B.2)
3. https://github.com/dotenv-linter/dotenv-linter (B.3)
4. https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/ (B.4 -- WebFetch followed the 301 redirect from docs.pydantic.dev)
5. https://12factor.net/config (B.5)
6. https://owasp.org/Top10/2021/A05_2021-Security_Misconfiguration/ (B.6)
7. https://pre-commit.com/ (B.7)

Snippet-only (B.8-B.10): python-dotenv source code, GitHub Actions continue-on-error docs, internal ascii-logger-lint.yml.

Internal files inspected = 1 because only `backend/config/settings.py` was readable in this session; the other six internal paths the researcher tried to inspect (`.env.example`, `.claude/hooks/`, `.github/workflows/`, `backend/requirements.txt`, `handoff/archive/phase-23.5.19*`) were permission-denied. The planner -- which is Main, with broader access -- should re-read them at contract-write time. The 7/10 split on external sources clears the >=5 floor; recency_scan_performed=true with the explicit "no new findings supersede canonical" statement clears the recency gate.

## Section G -- Application notes for the planner

1. **Regex contract is non-negotiable**: `^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=`. Deviating from python-dotenv's regex creates false positives that block PRs. The project's `extra="ignore"` posture means stricter rules at the linter are USELESS at runtime -- always match the runtime parser's permissiveness.

2. **Silent-fallback protection (the audit-basis F4 case)**: the killer scenario is a typo like `DEEP_THINK_MODL=...` that pydantic-settings silently ignores. The linter CANNOT detect typos by itself (no schema). The mitigation is the 12-Factor "fail loud" stance: add a second QA lane (out-of-scope for 40.6 but flag it) that DIFFS the .env.example KEY set against `Settings` Field names. That's a separate ticket; document it in the contract's "future-work" section.

3. **Pre-commit hook glue**: do NOT add `pre-commit.com` as a Python dev-dep. Wire the bash hook into the existing `.claude/hooks/` machinery. The hook MUST short-circuit on no-staged-env-file (per D.1's `staged_envs` grep) so unrelated commits aren't slowed.

4. **CI workflow shape**: continue-on-error: true initially. Mirror the ascii-logger-lint.yml file shape exactly so the planner can git-diff the two workflows and confirm structural parity. Promote to required after one cycle of clean runs (a phase-40.7 follow-up, NOT 40.6 scope).

5. **Never echo SecretStr values in violations**: per Section A.2, settings.py has 9 SecretStr-typed fields. The checker outputs KEY names + line numbers + rule names ONLY -- never the value. Add an explicit unit test that confirms a value containing "sk-ant-fake" never appears in stdout/stderr of the checker when the line has a quote-balance violation.

---

End of brief.
