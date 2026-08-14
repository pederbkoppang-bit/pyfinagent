---
name: away-ops-session-gitignore-86-67
description: 86.67 away_ops session_*.json -- sole consumer reads from DISK so gitignore is safe; `sk-[A-Za-z0-9]{20,}` CANNOT match sk-ant-oat01- tokens (hyphens); a sound brief still fails the gate without a sources_read_in_full array; .git/hooks/pre-commit already exists and is fail-CLOSED; redaction-at-write beats scanning (73.5% of agent leaks are stdout capture)
metadata:
  type: project
---

Step 86.67 (`handoff/away_ops/session_*.json` not gitignored, `git add -A` can
publish a credential). Researched 2026-08-14.

**The consumer question resolves cleanly: DISK, not repo.**
`scripts/away_ops/auth_state.py:67` is the ONLY consumer --
`glob.glob(os.path.join(ops, "session_*.json"))` sorted by `os.path.getmtime`,
newest only, looking for `"api_error_status": 401` in a 36h window
(`DEFAULT_WINDOW_S=129600` at `:51`). `ops` is a REQUIRED CLI arg (`:113`),
passed by `healthcheck.sh:106` as `$REPO/handoff/away_ops`. A repo-path fallback
is *forbidden by test* (`test_phase_85_3_auth_latch_freshness.py:270`). Writer is
`run_away_session.sh:135`. The Slack away-digest (`scheduler.py:519/533/539/548`)
reads `pending_tokens.json` / `health.jsonl` / `session.log` / `defect_register.md`
-- **never** a session JSON.

**Why:** criterion 1 turned entirely on disk-vs-repo, and a consumer keyed on
`getmtime` can't be reading a git-materialised artifact (git doesn't store mtime).

**How to apply:** for any "can we gitignore X" question, find the consumer's PATH
SOURCE (hardcoded repo path vs injected param) and its FRESHNESS KEY (mtime vs
content) -- those two settle it faster than reading the whole rail.

## The live precedent nobody had cited
`handoff/away_ops/session.log` is ALREADY gitignored by `.gitignore:28 *.log`,
is untracked, and `scheduler.py:539` reads it from disk every digest run. Also
`run_away_session.sh:99` already excludes `handoff/away_ops/` from its own
dirty-tree gate ("perpetually dirty by design"). Look for an existing
ignored-and-still-consumed sibling before arguing from first principles.

## Credential-regex class trap (the highest-value lesson)
A caller handed me "ZERO credential-shaped values, positive-controlled" as GIVEN.
It was FALSE: 5 TRACKED files, all on `origin/main`, contain
`sk-ant-oat01-sk-ant-oat01-...` (92 chars) inside `.result`, from an API error
echoing `Authorization: Bearer ...`.

Root cause, proven with a discriminating probe:
`sk-[A-Za-z0-9]{20,}` returns **0** on `sk-ant-oat01-OvM72Xwg...` because the
class **cannot cross the hyphens** in `ant-oat01-`; `sk-ant-[A-Za-z0-9_-]{20,}`
returns 1. Their positive control used a hyphen-free `sk-` token, so it passed on
a shape that does not resemble a real Anthropic token.

**Why:** this is the "control and fail-safe answer coincide" probe failure --
the control proved the harness ran, not that the branch that mattered could fire.
See [[feedback_mutation_probe_must_discriminate]] and [[feedback_suspect_the_clean_check]].

**How to apply:** (1) always include `_` and `-` in credential character classes;
(2) build the positive control from a REAL token's literal prefix
(`sk-ant-oat01-`, `xoxb-`, `AIza`), never a stylised stand-in; (3) note these
files are single-line compact JSON with **zero** `": "` occurrences -- any regex
requiring a literal space after the colon can never fire on them.

## git check-ignore mechanics worth reusing
- Test a candidate rule WITHOUT editing `.gitignore`:
  `git -c core.excludesFile=<tmpfile> check-ignore -v --no-index <path>`.
  Faithful because `core.excludesFile` is the *lowest* precedence and is anchored
  at the worktree root -- matching there implies matching in `.gitignore`.
- **`--no-index` is REQUIRED for TRACKED paths**: without it a tracked file
  returns rc=1 even when the rule matches (tracked files are exempt). Measured
  both ways. Same trap as
  `.claude/agent-memory/qa/project_committed_criterion_gitignore_check.md:21-23`.
- Rule shape is load-bearing: `handoff/away_ops/session_*.json` is correct;
  `handoff/away_ops/session*` also swallows the TRACKED, rail-critical
  `session_notes.md` (written by `prompt_am.md:67`, `prompt_pm.md:22,41,44,48`).

## Doctrine short of history rewriting (criterion 4 / ask 06-2)
GitHub Docs: rotate/revoke is the FIRST step and rewriting *"may not be
warranted"* once rotated; rewriting risks recontamination from stale clones --
acute here given concurrent sessions + hook auto-push. OWASP orders
Revocation -> Rotation and says secrets must *"never be logged"*. GitGuardian
2026: **>64% of credentials leaked in 2022 were still valid in Jan 2026** -- never
assume a published token is dead, even a malformed-looking one.

---

# Re-run additions (2026-08-14, second researcher pass)

## A sound brief can FAIL the gate on pure artifact accounting
The first 86.67 brief was 40,839 bytes, 6 genuine full reads, recency scan done,
38 real URLs -- and the gate FAILED because its closing envelope carried **no
`sources_read_in_full` array**. `enforceGate` cross-checks claimed URLs against
the brief text; with no array there is nothing to corroborate, so the count is
unverifiable regardless of how good the prose is.

**Why:** the script never trusts `gate_passed`; it RECOMPUTES from the array +
the file. Research quality and gate-passing are separate properties.

**How to apply:** the envelope's URL array is a deliverable, not decoration.
Before returning, run the cross-check yourself:
`while read u; do grep -cF "$u" $BRIEF; done` over every URL you intend to
claim. Also: sources read by a PRIOR run are not yours -- re-fetch them, or the
claim is inherited rather than made.

## `.git/hooks/pre-commit` ALREADY EXISTS in this repo (changes any hook proposal)
Not in the first brief. Measured:
- `auto-commit-and-push.sh` has **no `--no-verify`**, so `.git/hooks/pre-commit`
  DOES fire on the auto-commit path -- a guard there would have blocked the leak.
- The hook already blocks 3 things (stray `.claude/*.bak-*` `:10-17`; retired
  Claude snapshot IDs in staged `*.py` `:19-29`; dotenv syntax `:35-47`), all via
  `git diff --cached --name-only --diff-filter=ACM` -> filter -> `exit 1`.
- **`set -e` at `:5` = fail-CLOSED.** Every guard wraps grep in `|| true` because
  grep exits 1 on no-match and `set -e` would abort -> git reads that as "commit
  rejected". Forget `|| true` and you block every CLEAN commit.
- **A rejected commit is SILENT**: `auto-commit-and-push.sh:380` is
  `if ! git commit ...; then log ...; exit 0; fi` -- no commit, no push, exit 0,
  visible only in `handoff/logs/auto-push.log`.
- `.git/hooks/` is **not version-controlled**; no `.pre-commit-config.yaml`
  exists; zero secret-scanning refs anywhere in `.claude/hooks/`.

## The publisher's own safety comment is FALSIFIED by this incident
`.claude/hooks/auto-commit-and-push.sh:348-349`: *"Broad capture; the pre-commit
pre-tool-use-danger guard + gitignore for .env files cover safety."* Neither
covers `handoff/away_ops/session_*.json`. A written safety model that the
evidence refutes is worth more than a missing one -- grep for the comment that
states WHY an unsafe operation is believed safe, then test that claim.

## Redaction-at-write outranks scanning and gitignore, and it is MEASURED
arXiv 2604.03070v1 (n=17,022 agent skills, kappa=0.88): **73.5% of credential
leaks are stdout/log capture** -- *"agent frameworks capture stdout into the LLM
context window"* -- and the top recommendation is to strip credential patterns
from stdout **before** it is persisted; 89.6% exploitable in normal execution.
That is `run_away_session.sh:170` verbatim (`> "$OUT_JSON"`, raw, unfiltered).
Same paper kills the scrub-first instinct mechanistically: *"credentials removed
from 107 upstream repositories remain live across 50+ independent forks"*.

## GitHub push protection is weaker than it sounds
Fires at **push** time not commit time; the free on-by-default variant is
user-level and only blocks pushes to **public** repos; repo-level needs GitHub
Secret Protection; and any writer can bypass by picking a reason. gitleaks is
MIT and `--baseline-path` is what makes adoption possible on a repo with
pre-existing findings -- but it is **"feature complete... security patches
only"**, so a new `sk-ant-oat01-` detector must be YOUR `.gitleaks.toml` rule.
