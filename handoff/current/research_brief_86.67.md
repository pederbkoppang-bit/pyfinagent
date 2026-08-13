# Research Brief -- step 86.67

**Topic:** `handoff/away_ops/session_*.json` are not gitignored, so the auto-commit
hook's `git add -A` can publish the next one that contains a credential.

**Tier:** simple (caller-stated). **Audit-class:** NO (`coverage.dry` not required).
**Role:** Layer-3 Researcher (external literature + internal code exploration).
**Started:** 2026-08-14.

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 32,
  "urls_collected": 38,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The single consumer of handoff/away_ops/session_*.json is scripts/away_ops/auth_state.py:67, which globs them from a --ops DISK path keyed on mtime; the Slack away-digest reads four OTHER away_ops files and none of these. Gitignoring is therefore strictly safe. session.log is already gitignored (.gitignore:28) and still read by scheduler.py:539 -- a live in-repo precedent. Rule shape is load-bearing: handoff/away_ops/session_*.json matches real paths (rc=0) and NOT session_notes.md/pending_tokens.json/health.jsonl (rc=1), while a sloppy session* glob swallows the tracked, rail-critical session_notes.md. --no-index is required to demonstrate a match on already-tracked paths. CRITERION 5 CONTRADICTED: five TRACKED files, all on origin/main, carry an Anthropic OAuth token echoed from an Authorization header into .result; the caller's sk-<20+> regex missed it because the class cannot cross hyphens in sk-ant-oat01-. Literature: .gitignore never affects tracked files (git-scm); rotate first, rewriting history is often unwarranted (GitHub, OWASP); >64% of 2022-leaked credentials were still valid Jan 2026.",
  "brief_path": "handoff/current/research_brief_86.67.md",
  "gate_passed": true
}
```

> Envelope was born inert (`INCOMPLETE`, zeroed counts) in the first tool call and
> updated in place as sources landed. Flipped to `COMPLETE` as the final act.
> Gate arithmetic: `external_sources_read_in_full` 6 >= 5 AND
> `recency_scan_performed` == true AND all hard blockers checked AND
> `coverage.audit_class` == false => `gate_passed: true`.

---

## Status log

- [x] Read `.claude/agents/researcher.md` in full (operating instructions).
- [x] Read `.claude/rules/research-gate.md` in full (authoritative floors).
- [x] Internal: enumerate the CONSUMER of `handoff/away_ops/session_*.json`.
- [x] Internal: ignore-rule shape + `git check-ignore -v` both-direction proof.
- [ ] External: >=5 sources read in full via WebFetch; >=10 URLs; recency scan.

---

# PART A -- INTERNAL EXPLORATION (the decisive half)

## A0. Tooling trap, discharged first

The caller warned that the shell's `grep` is a **ugrep wrapper**. Confirmed
verbatim: `which grep` prints a shell *function* that re-execs
`${CLAUDE_CODE_EXECPATH}` with `ARGV0=ugrep` and `--ignore-files`. That
`--ignore-files` flag makes ugrep **honour .gitignore**, which on this exact
question would have silently hidden ignored files from every search -- a
self-referential trap. **Every grep below is pinned to `/usr/bin/grep`**
(`-rwxr-xr-x root wheel 153760`, verified present).

## A1. Baseline re-measurement (criterion 5 says re-measure, do not trust the basis)

| Quantity | Basis (masterplan) | Caller's re-measure | **My measurement (2026-08-14)** |
|---|---|---|---|
| Untracked `session_*.json` | 4 | 5 | **5** (confirmed) |
| Tracked `session_*.json` | -- | 72 | **71** |
| Total `session_*` on disk | -- | -- | **77** |

**The 71-vs-72 discrepancy is explained, not a disagreement.** `git ls-files
'handoff/away_ops/session_*'` returns **72**; `git ls-files
'handoff/away_ops/session_*.json'` returns **71**. The 72nd entry is
**`handoff/away_ops/session_notes.md`** -- a *tracked Markdown* file, not a
session JSON. Reconciliation closes exactly: 77 on disk = 76 `.json` +
1 `session_notes.md`; 76 = 71 tracked + 5 untracked.

That one-file difference is **not cosmetic** -- see A5, where `session_notes.md`
is the negative control that separates a safe ignore rule from a
rail-breaking one.

The 5 untracked files (verbatim `git status --porcelain`):

```
?? handoff/away_ops/session_am_20260812T053032Z.json
?? handoff/away_ops/session_am_20260813T053028Z.json
?? handoff/away_ops/session_pm_20260811T200033Z.json
?? handoff/away_ops/session_pm_20260812T200030Z.json
?? handoff/away_ops/session_pm_20260813T200009Z.json
```

Note the set **grew by one during this session** (`session_pm_20260813T200009Z`
was not in the git-status snapshot taken at session start). The channel is
live and produces a new file per away session -- roughly 2/day.

Current ignore state, measured:
`git check-ignore -v handoff/away_ops/session_am_20260813T053028Z.json` -> **rc=1,
no output** = matched by no rule. Also confirmed: no `core.excludesFile` is
configured (`git config --get core.excludesFile` -> rc=1), and
`.git/info/exclude` contains only `**/.claude/*` entries -- none of which touch
`handoff/`. So there is genuinely **no** ignore rule on these paths from any of
the three ignore sources.

## A2. THE CONSUMER -- criterion 2, the decisive question

**Answer: exactly ONE consumer reads `session_*.json`, and it reads them
FROM DISK, never as committed artifacts. Gitignoring is therefore strictly
safe for the away-ops rail.**

### A2.1 The single real consumer

`scripts/away_ops/auth_state.py:67-68`:

```python
sessions = sorted(glob.glob(os.path.join(ops, "session_*.json")),
                  key=os.path.getmtime)
```

It then takes `sessions[-1]` (the newest), reads its body
(`auth_state.py:70-74`), searches for `'"api_error_status": 401'`
(`:75`), and gates the result on the file's **mtime** against a 36h window
(`:76-81`; `DEFAULT_WINDOW_S = 129600` at `:51`).

Three independent properties make this a **filesystem** consumer, not a
repository consumer:

1. **`ops` is a required CLI parameter, not a repo path.**
   `auth_state.py:113` -- `ap.add_argument("--ops", required=True)`. The
   caller supplies it: `scripts/away_ops/healthcheck.sh:106` invokes
   `python3 "$REPO/scripts/away_ops/auth_state.py" --ops "$OPS" ...` where
   `healthcheck.sh:14` sets `OPS="$REPO/handoff/away_ops"`.
2. **A repo-path fallback is FORBIDDEN BY TEST.**
   `backend/tests/test_phase_85_3_auth_latch_freshness.py:270` asserts
   `assert "handoff/away_ops" not in src, "seam carries a repo-path fallback"`.
   The seam is *designed* to be path-injectable, and the test suite drives it
   over synthetic files under `tmp_path` (`:66`, `:110`).
3. **It keys on `os.path.getmtime`.** Git neither stores nor restores mtime.
   A consumer whose freshness logic is mtime-based cannot be reading a
   git-materialised artifact in any meaningful sense.

### A2.2 The writer (same directory, same disk path)

`scripts/away_ops/run_away_session.sh:135`:

```sh
OUT_JSON="$OPS/session_${SESSION}_$(date -u +%Y%m%dT%H%M%SZ).json"
```

with `OPS="$REPO/handoff/away_ops"` at `:23`. The Claude session's
`--output-format json` stdout is redirected into it at `:170`. The wrapper
then re-reads **its own just-written file** by absolute path at `:186`
(the 401 credential-death detector) and `:209-219` (cost + limit surfacing).
All disk reads, all immediate, all independent of git.

**This is why the file is credential-shaped in the first place**: it is the
raw JSON envelope of an unattended `claude -p` run, dumped verbatim into the
repo working tree.

### A2.3 What the away DIGEST actually reads -- and it is NOT these files

`backend/slack_bot/scheduler.py` reads exactly four away_ops artifacts:

| Line | File read |
|---|---|
| `scheduler.py:519` | `handoff/away_ops/pending_tokens.json` |
| `scheduler.py:533` | `handoff/away_ops/health.jsonl` |
| `scheduler.py:539` | `handoff/away_ops/session.log` |
| `scheduler.py:548` | `handoff/away_ops/defect_register.md` |

**None is `session_*.json`.** `scripts/away_ops/send_away_digest.py` likewise
contains zero `session_*.json` references (its only "session" hits are prose
in the module docstring at `:5`, `:9-10`). The digest's session-outcome line
comes from `session.log` (`scheduler.py:540-542`, filtering for
`"END session"`), which is the human-readable log written by the `slog()`
helper at `run_away_session.sh:38` -- **not** the JSON envelope.

### A2.4 Positive control for the negative claim (a zero must be proven)

A bare "nothing else reads them" is worth nothing without evidence the search
would have found a reader. Controls run with the **same pinned
`/usr/bin/grep` invocation**:

- **Control string `pending_tokens`** (known to exist at `scheduler.py:519`):
  returned **6 matches**. Method alive.
- The same filename-pattern sweep
  (`-E 'session_am|session_pm|session_\*|session_\{|session_%s|session_" *\+|session_\$'`)
  **did positively find** `auth_state.py:67` and `run_away_session.sh:135`
  plus 3 test-fixture hits. So the sweep demonstrably *can* surface a
  `session_*.json` consumer -- it found the only one there is.
- A separate `glob|iterdir|listdir|scandir` sweep across
  `backend/ scripts/ .claude/hooks .claude/workflows` returned exactly one
  away_ops-relevant hit: `auth_state.py:67`.

**Conclusion for criterion 1 + 2:** the away-ops rail consumes these files
from **disk**. Nothing consumes them as committed artifacts. Ignoring them
does not break the rail; it removes a publication channel while leaving every
consumer untouched.

## A3. The live precedent already in the repo (strongest single argument)

`handoff/away_ops/session.log` is **already gitignored** -- measured:

```
git check-ignore -v --no-index handoff/away_ops/session.log
  -> .gitignore:28:*.log   handoff/away_ops/session.log
```

...it is **untracked** (`git ls-files --error-unmatch` -> no), and
`scheduler.py:539` **reads it from disk for the digest every run**. So
"an away_ops file that is gitignored *and* consumed by the away-ops rail"
is not a hypothetical -- it is the status quo, working today.

Corroborating design intent, `run_away_session.sh:97-99` already treats the
whole directory as untracked-by-nature evidence:

```sh
dirty=$(git status --porcelain 2>/dev/null | grep -vE '^.. (handoff/audit/|handoff/away_ops/|handoff/logs/)')
```

with the comment at `:94-97`: *"handoff/away_ops/ holds the wrapper's own logs
-- both are perpetually dirty by design and would route EVERY session into
recovery."* The rail already excludes `handoff/away_ops/` from its own
dirty-tree gate. Gitignoring the session JSONs *aligns* the ignore state with
an intent the code already encodes.

## A4. The publication mechanism

`.claude/hooks/auto-commit-and-push.sh:360` -- `add_stderr=$(git add -A 2>&1)`
(with a 1s retry at `:365`). It is tree-wide; the hook's own comment at `:351`
concedes *"`git add -A` is tree-wide, so it will also stage a PEER session's
work."* It fires on a masterplan status flip. There is **no pathspec and no
exclusion** for `handoff/away_ops/`.

This is exactly the mechanism the masterplan basis names: *"a credential-bearing
away_ops session file reached origin/main under an unrelated step's commit
message on 2026-08-11 at 06:42Z"* (masterplan `86.67.audit_basis`). The
channel has **already leaked once**.

Also relevant: `git add -A` **silently skips** gitignored paths -- no error, no
warning (prior Q/A memory, `.claude/agent-memory/qa/project_committed_criterion_gitignore_check.md:10-11`).
That silence is precisely what makes an ignore rule an effective *staging*
block, and precisely why it must be proven to match (A5).

## A5. Criterion 3 -- the ignore rule shape, proven in BOTH directions

Criterion 3 exists because *"a rule that matches nothing is the failure mode
here"* -- a no-op rule and a working rule are indistinguishable unless you
also show a non-match.

**These were run WITHOUT editing `.gitignore`** (hard constraint), by
injecting a candidate rule through a throwaway excludes file:
`git -c core.excludesFile=<tmpfile> check-ignore -v --no-index <path>`.

### Candidate A (recommended): `handoff/away_ops/session_*.json`

| Path | rc | Matched by |
|---|---|---|
| `handoff/away_ops/session_am_20260813T053028Z.json` | **0** | `handoff/away_ops/session_*.json` |
| `handoff/away_ops/session_pm_20260813T200009Z.json` | **0** | `handoff/away_ops/session_*.json` |
| `handoff/away_ops/session_notes.md` | **1** | *(NO MATCH -- required)* |
| `handoff/away_ops/pending_tokens.json` | **1** | *(NO MATCH -- required)* |
| `handoff/away_ops/health.jsonl` | **1** | *(NO MATCH -- required)* |

Both directions demonstrated: it matches the real target paths **and** fails
to match four deliberately non-matching neighbours, three of which are live
digest inputs (`scheduler.py:519/533/548`).

### Candidate B (REJECT): `handoff/away_ops/session*`

| Path | rc | Matched by |
|---|---|---|
| `handoff/away_ops/session_am_20260813T053028Z.json` | 0 | `handoff/away_ops/session*` |
| `handoff/away_ops/session_notes.md` | **0** | `handoff/away_ops/session*` -- **BREAKAGE** |

`session_notes.md` is **tracked** and is the away-ops rail's own running
notebook: written by `prompt_am.md:67`, `prompt_pm.md:22,41,44,48`,
`prompt_digest_only.md:17,23`, and read back by `prompt_recovery.md:15`
(the crash-recovery path). A `session*` glob would sweep it up. **The
suffix-anchored `.json` form is load-bearing, not stylistic.**

### Two disclosure notes so the evidence is not misread

1. **`session.log` matched via `.gitignore:28:*.log` in BOTH candidate runs.**
   That is the **pre-existing** global rule, not the candidate. Do not
   attribute that match to the proposed rule.
2. **`--no-index` is REQUIRED to demonstrate a match on an already-TRACKED
   path.** Measured on the same tracked file
   (`session_am_20260612T092025Z.json`) with the identical candidate loaded:

   ```
   without --no-index : rc=1, no output     <-- looks like "rule doesn't match"
   with    --no-index : rc=0, rule printed  <-- the rule DOES match
   ```

   Tracked files are exempt from ignore rules, so plain `check-ignore` returns
   1 for them regardless. Prior Q/A memory flags exactly this trap:
   *"once the path is tracked, `git check-ignore` exits 1 ... do not misread
   exit-1 as 'was never ignored'"*
   (`.claude/agent-memory/qa/project_committed_criterion_gitignore_check.md:21-23`).
   **An execution-time live_check that runs `check-ignore` over the 71 tracked
   files without `--no-index` will produce 71 misleading rc=1s.**

### Suggested exact invocations for `live_check_86.67.md`

```sh
# BEFORE (must be rc=1 / no output today)
git check-ignore -v handoff/away_ops/session_am_20260813T053028Z.json ; echo "rc=$?"

# AFTER -- positive direction (must be rc=0 and print the rule + line number)
git check-ignore -v handoff/away_ops/session_am_20260813T053028Z.json ; echo "rc=$?"

# AFTER -- negative direction (must be rc=1: the rule must NOT swallow these)
for p in handoff/away_ops/session_notes.md \
         handoff/away_ops/pending_tokens.json \
         handoff/away_ops/health.jsonl ; do
  git check-ignore -v "$p" ; echo "rc=$? $p"
done

# TRACKED-PATH nuance (needs --no-index or every line lies with rc=1)
git check-ignore -v --no-index handoff/away_ops/session_am_20260612T092025Z.json ; echo "rc=$?"
```

## A6. What an ignore rule does and does NOT accomplish here

- **DOES** stop `git add -A` from staging any *future* session file. Every
  session file is a **new, uniquely timestamped path**
  (`run_away_session.sh:135` embeds `date -u +%Y%m%dT%H%M%SZ`), so the forward
  channel is fully covered -- no path can slip in under an existing tracked name.
- **DOES NOT** untrack the 71 already-tracked files. `.gitignore` has no effect
  on files already in the index. Mitigating fact: session JSONs are
  **write-once** -- the wrapper never rewrites an old one -- so a tracked
  historical file will not be re-staged with new content.
- **DOES NOT** remove anything from published history. That is criterion 4's
  explicit boundary and operator ask 06-2's territory (rotation), and this
  brief proposes nothing there.

## A7. Internal file inventory

| File | Anchor | Role | Status |
|---|---|---|---|
| `scripts/away_ops/auth_state.py` | `:67-68`, `:51`, `:113` | **The only consumer.** Globs `session_*.json` from a `--ops` path, newest-by-mtime, 401 + 36h freshness | LIVE |
| `scripts/away_ops/run_away_session.sh` | `:135` (write), `:23`, `:170`, `:186`, `:209-219` (self-read), `:99` (away_ops excluded from dirty gate) | Writer + self-reader | LIVE |
| `scripts/away_ops/healthcheck.sh` | `:14`, `:106` | Passes `--ops "$REPO/handoff/away_ops"` to auth_state.py | LIVE |
| `backend/slack_bot/scheduler.py` | `:519`, `:533`, `:539`, `:548` | Away digest -- reads 4 away_ops files, **none a session JSON** | LIVE |
| `scripts/away_ops/send_away_digest.py` | `:5`, `:9-10` (prose only) | Digest sender -- **zero** session JSON reads | LIVE |
| `.claude/hooks/auto-commit-and-push.sh` | `:360` (`git add -A`), `:365` retry, `:351` comment | The publication mechanism | LIVE |
| `.gitignore` | `:28` (`*.log`), `:75-96` (handoff rules) | No rule matches `session_*.json`; `:28` already covers `session.log` | LIVE |
| `backend/tests/test_phase_85_3_auth_latch_freshness.py` | `:66`, `:110`, `:270` | Drives auth_state.py over synthetic tmp_path files; forbids repo-path fallback | LIVE |
| `scripts/away_ops/prompt_am.md` / `prompt_pm.md` / `prompt_digest_only.md` / `prompt_recovery.md` | `:67` / `:22,41,44,48` / `:17,23` / `:15` | Write+read `session_notes.md` -- the negative control | LIVE |
| `.claude/agent-memory/qa/project_committed_criterion_gitignore_check.md` | `:10-11`, `:21-23` | Prior Q/A memory: `git add -A` silently skips ignored paths; tracked-path exit-1 trap | REFERENCE |
| `.claude/masterplan.json` | step `86.67`, step `86.2` | Step definition + the 2026-08-11T06:42Z leak record | REFERENCE |
| `.git/info/exclude`, `git config core.excludesFile` | -- | Neither carries any `handoff/` rule (rules out a hidden ignore source) | CHECKED |

---

# PART A8 -- CRITERION 5 RE-MEASURED: **THE GIVEN IS WRONG**

The caller supplied criterion 5 as "given, but re-verify before restating".
**Re-verification contradicts it, and the contradiction is material.**

> Caller's claim: *"ZERO credential-shaped values in ANY of them, tracked or
> untracked."*

**Measured 2026-08-14: FIVE TRACKED files contain an Anthropic OAuth token,
and all five are on `origin/main`.**

```
session_am_20260809T053008Z.json   TRACKED   ON-ORIGIN/MAIN
session_am_20260810T053009Z.json   TRACKED   ON-ORIGIN/MAIN
session_pm_20260808T200008Z.json   TRACKED   ON-ORIGIN/MAIN
session_pm_20260809T200008Z.json   TRACKED   ON-ORIGIN/MAIN
session_pm_20260810T200010Z.json   TRACKED   ON-ORIGIN/MAIN
```

Location (value redacted -- JSON key path via `json.load` + recursive walk):

```
KEY PATH : .result
context  : "API Error: Header 'Authorization' has invalid value: 'Bearer "
token    : sk-ant-oat01-sk-...<REDACTED, 92 chars total>
```

The token is **doubled** (`sk-ant-oat01-sk-ant-oat01-...`) -- which matches the
project's long-standing "malformed token" symptom (20/20 rail calls failing with
`duration_api_ms=0`). It reached the repo because an API **error message**
echoed the `Authorization` header verbatim into `.result`, and
`run_away_session.sh:170` redirects that JSON straight into the working tree.

**The caller's five UNTRACKED files ARE clean** -- re-confirmed under the wider
regex (0 matches each). So the caller's finding is right about the files they
looked at and wrong in its universal quantifier.

### Root cause of the false negative -- proven, not guessed

The caller's `sk-<20+>` branch is **character-class-blind to hyphens**.
Discriminating probe on the literal string `sk-ant-oat01-OvM72XwgABCDEFGHIJKLMNOP`:

| Regex | Matches |
|---|---|
| `sk-[A-Za-z0-9]{20,}` (caller's shape) | **0** |
| `sk-ant-[A-Za-z0-9_-]{20,}` (mine) | **1** |

After `sk-`, the class `[A-Za-z0-9]` cannot cross the hyphens in `ant-oat01-`,
so it never reaches 20 characters. The caller's **positive control passed on a
token shape that does not resemble a real Anthropic OAuth token** -- the control
and the real subject differ exactly where the regex is brittle. This is the
"control answer and fail-safe answer coincide" probe failure: the control proved
the *harness* ran, not that the *branch that mattered* could fire.

A second latent trap, measured: these files are **single-line compact JSON**
(`lines=1`, and `": "` occurs **0** times). Any regex requiring a literal
`": "` after a key can **never** fire on this file family. Both the caller's
`": "` variant and my whitespace-tolerant variant return 0 here, so that branch
is genuinely dry -- but it is dry for a reason that would hide a future
pretty-printed leak.

### What this changes for the step

- Criterion 5 ("re-measure, do not trust the basis") is **doing its job** -- it
  caught a real, still-published credential.
- It **does not change** criteria 1-3: the forward-looking ignore decision is
  unaffected, and it makes the case for it stronger -- the mechanism is not
  hypothetical, it has published a credential that is still on `origin/main`.
- It lands squarely in **criterion 4 / operator ask 06-2** territory
  (rotation of an already-published credential, operator-gated). **This brief
  proposes no action on it and I have touched nothing.** Rotation is the
  operator's call; per GitHub's own guidance (below) rotation -- not history
  rewriting -- is the step that actually removes the risk.

---

# PART B -- EXTERNAL RESEARCH

## B0. Search-query composition (three-variant discipline)

| Variant | Query run |
|---|---|
| Current-year frontier | `secret leakage source code repositories agent CI telemetry artifacts 2026` |
| Last-2-year window | `credentials already committed to git rotate revoke instead of rewriting history 2025` |
| Year-less canonical | `gitignore is not a security control for secrets already tracked files` |
| Recency/academic | `arXiv 2025 2024 empirical study secret leakage git repositories detection remediation lifetime` |

## B1. Read in full (>=5 required; 6 achieved) -- all via WebFetch, HTML, no PDFs

| # | URL | Accessed | Kind | Tier | Key finding |
|---|---|---|---|---|---|
| 1 | https://git-scm.com/docs/gitignore | 2026-08-14 | Official docs | 2 | *"Files already tracked by Git are not affected"* -- the primary-source basis for A6 |
| 2 | https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository | 2026-08-14 | Official docs | 2 | Rotate FIRST; history rewrite *"may not be warranted"* |
| 3 | https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html | 2026-08-14 | Standards body | 1/2 | Revocation -> rotation sequence; secrets must *"never be logged"* |
| 4 | https://blog.gitguardian.com/the-state-of-secrets-sprawl-2026/ | 2026-08-14 | Industry report | 4 | 28.65M secrets in 2025; internal repos ~6x public; **>64% of 2022 leaks still valid in Jan 2026** |
| 5 | https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-coding-agent-cicd-secrets-20260808-csa/ | 2026-08-14 | Industry research note (2026-08-08) | 4 | Claude Code / Gemini CLI / Codex CI secret exposure; agent-loaded repo files are untrusted input |
| 6 | https://microsoft.github.io/code-with-engineering-playbook/source-control/secrets-management/ | 2026-08-14 | Vendor engineering playbook | 3 | `.gitignore` is the baseline, credential scanning is the *"extra security measure"* |

## B2. Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://thehackernews.com/2026/03/the-state-of-secrets-sprawl-2026-9.html | News | Secondary coverage of source #4 |
| https://passwork.pro/blog/the-state-of-secrets-sprawl-in-2026/ | Vendor blog | Secondary coverage of source #4 |
| https://snyk.io/articles/state-of-secrets/ | Vendor blog | Secondary coverage of the same dataset |
| https://www.cloudsek.com/blog/ai-supply-chain-breach-2500-companies-434000-cicd-pipelines | Vendor blog | Supply-chain incident, not the ignore/rotation question |
| https://medium.com/@umangmishra2327/what-the-claude-code-leak-teaches-us-about-ai-supply-chain-security-cd687f2ee5ce | Community | Tier-5, superseded by source #5 |
| https://www.akeyless.io/blog/the-hidden-risks-of-secrets-management-in-ci-cd-pipelines/ | Vendor blog | Marketing-tier |
| https://www.invisirisk.com/blog/what-are-secrets-leaks-in-ci-cd-pipelines/ | Vendor blog | Marketing-tier |
| https://cyberupdates365.com/mercedes-benz-data-breach-source-code-leak/ | News | Unrelated incident |
| https://techcommunity.microsoft.com/blog/azureinfrastructureblog/how-to-safely-remove-secrets-from-your-git-history-the-right-way/4464722 | Vendor blog | History-rewriting -- **forbidden by criterion 4** |
| https://azurefeeds.com/2025/10/29/how-to-safely-remove-secrets-from-your-git-history-the-right-way/ | Syndication | Duplicate of the above |
| https://blog.gitguardian.com/rewriting-git-history-cheatsheet/ | Vendor blog | History-rewriting -- out of scope per criterion 4 |
| https://coreui.io/answers/how-to-remove-sensitive-data-from-git-history/ | Community | Tier-5, history rewriting |
| https://gist.github.com/R1verRat/989911c6566235ec076ceac69f4ed04d | Community gist | Tier-5 |
| https://aquilax.ai/blog/secrets-git-history-rotation | Vendor blog | Useful framing ("git history as a credential store"); tier-4 |
| https://github.com/orgs/community/discussions/161907 | Forum | Tier-5 |
| https://github.com/orgs/community/discussions/165862 | Forum | Tier-5 |
| https://github.com/orgs/community/discussions/158668 | Forum | Tier-5 |
| https://www.twalker.dev/avoid-committing-secrets-by-ignore-tracked-files-in-git/ | Personal blog | Tier-5; covers `--skip-worktree`, not needed here |
| https://gitcheatsheet.dev/docs/getting-started/ignoring-files/ | Reference | Superseded by primary source #1 |
| https://www.envzero.com/blog/gitignore-command-guide-practical-examples-and-terraform-tips | Vendor blog | Superseded by source #1 |
| https://coreui.io/answers/how-to-ignore-files-in-git-with-gitignore/ | Community | Superseded by source #1 |
| https://dev.to/yusbuntu/chapter-4-gitignore-1ojc | Community | Tier-5 |
| https://dev.to/just_ritik/why-a-gitignore-file-is-essential-for-your-projects-4odm | Community | Tier-5 |
| https://educative.io/courses/getting-started-with-git-version-control/ignore-files-with-gitignore | Course | Tier-5 |
| https://about.gitlab.com/blog/whats-new-in-git-2-52-0/ | Vendor blog | Git release notes, not on-topic |
| https://arxiv.org/html/2604.03070v1 | Preprint (2026) | **Highly relevant** (credential leakage in LLM agent skills) -- flagged for a deeper tier; floor already met |
| https://arxiv.org/html/2605.31520 | Preprint (2026) | Detection-model architecture; not decision-relevant here |
| https://arxiv.org/html/2504.18784 | Preprint (2025) | LLM secret-breach detection; tooling, not policy |
| https://arxiv.org/html/2410.23657 | Preprint (2024) | Secret leaks in issue reports; adjacent channel |
| https://arxiv.org/pdf/2410.23657 | Preprint (PDF form) | PDF form of the above -- not fetched (PDF policy) |
| https://arxiv.org/pdf/2307.00714 | Preprint (2023) | Secret-detection tool comparison |
| https://dl.acm.org/doi/10.1145/3793302.3793348 | Peer-reviewed (MSR) | Paywalled landing page |

**Total unique URLs collected: 38** (6 read in full + 32 snippet-only).

## B3. Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

**Performed.** Result: **found 4 new findings that COMPLEMENT (do not supersede)
the canonical guidance.** The canonical rule -- ".gitignore does not affect
tracked files; rotate an exposed secret" -- is unchanged and is still stated in
current primary docs. What the 2024-2026 window adds:

1. **Scale and trajectory.** 28.65M new hardcoded secrets on public GitHub in
   2025, +34% YoY; AI-service secrets 1,275,105, **+81% YoY**
   (GitGuardian 2026, https://blog.gitguardian.com/the-state-of-secrets-sprawl-2026/).
2. **Remediation basically does not happen.** *"Nearly 70% of credentials
   confirmed as valid in 2022 were still valid in January 2025"*, and by January
   2026 *"the validity rate was still above 64%"* (same source). This is the
   single most decision-relevant new number: an exposed credential should be
   assumed live until explicitly rotated.
3. **Private repos are the worse offender.** *"Internal repos are roughly 6x
   more likely than public ones to contain hardcoded secrets."* A private repo
   is **not** a mitigating control.
4. **Agent-specific channel is now documented (2026-08-08).** CSA Labs showed a
   GitHub issue from an unprivileged account reaching CI-runner secrets across
   Claude Code, Gemini CLI and Codex, and stresses treating *"any repository
   file an agent loads automatically ... as untrusted input"*
   (https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-coding-agent-cicd-secrets-20260808-csa/).

Nothing found in the window contradicts the ignore-then-rotate approach or
recommends history rewriting as a first move.

## B4. Key findings, cited per claim

1. **`.gitignore` has NO effect on already-tracked files -- primary source.**
   *"Files already tracked by Git are not affected; see the NOTES below for
   details."* To untrack, *"use `git rm --cached` to remove the file from the
   index."*
   (git-scm gitignore docs, https://git-scm.com/docs/gitignore, accessed 2026-08-14)
   -> Confirms A6: an ignore rule protects the **forward** channel only.

2. **Ignore-pattern anchoring works exactly as candidate A assumes.**
   *"If there is a separator at the beginning or middle (or both) of the
   pattern, then the pattern is relative to the directory level of the
   particular `.gitignore` file itself."*
   (https://git-scm.com/docs/gitignore) -> `handoff/away_ops/session_*.json` in
   the ROOT `.gitignore` anchors at the repo root, which is what the A5
   measurement showed empirically.

3. **Ignore-source precedence justifies the A5 test method.**
   `.gitignore` > `$GIT_COMMON_DIR/info/exclude` > `core.excludesFile`, and
   *"Patterns read from exclude sources that are outside the working tree ...
   are treated as if they are specified at the root of the working tree."*
   (https://git-scm.com/docs/gitignore) -> Injecting the candidate through
   `core.excludesFile` is a **faithful, lower-precedence** simulation of the
   same pattern in `.gitignore`: if it matches at the weakest precedence it will
   match at the strongest. It is also non-destructive, satisfying the
   "do not edit .gitignore" constraint.

4. **Rotation is the FIRST step, and history rewriting is explicitly optional.**
   *"if the sensitive data you need to remove is a secret ... then as a first
   step you need to revoke and/or rotate that secret"*; *"Once the secret is
   revoked or rotated, it can no longer be used for access, and that may be
   sufficient to solve your problem. Going through the extra steps to rewrite
   the history and remove the secret may not be warranted."*
   (GitHub Docs, https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository,
   accessed 2026-08-14) -> **This is the direct answer to the caller's third
   external question.** Criterion 4's ban on history rewriting is not a
   compromise; it is the vendor-recommended default.

5. **Rewriting carries a recontamination failure mode.** *"If a fellow developer
   has a clone from before your rewrite, and after your rewrite simply runs
   `git pull` followed by `git push`, the sensitive data will return."*
   (same source) -> Materially relevant: this repo is **known to run two
   concurrent Claude sessions** plus scheduled away sessions, so stale clones
   and hook-driven auto-pushes are a live recontamination vector. Independent
   corroboration for keeping criterion 4's boundary.

6. **What to do INSTEAD of rewriting: push protection + scanning.**
   *"Enable push protection for your repository to detect and prevent pushes
   which contain hardcoded secrets from being committed to your codebase."*
   (same source) -> The recommended short-of-rewriting control set is
   (a) rotate, (b) block future pushes, (c) ignore/exclude the producing path.

7. **OWASP orders the response Revocation -> Rotation, and warns rewriting
   causes collateral damage.** *"Revocation: Keys that were exposed should
   undergo immediate revocation."* ... *"Rotation: A new secret must be quickly
   created and implemented."* And on squashing history: *"this may introduce
   other problems as it rewrites git history and will break any other links to
   a given commit."*
   (OWASP Secrets Management Cheat Sheet §9.2 Remediation,
   https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html,
   accessed 2026-08-14)

8. **Secrets must never be written to logs -- the root-cause control for A8.**
   *"Never be logged (must implement either an encryption or masking approach in
   place to avoid logging plaintext secrets)."*
   (OWASP §8.3 Detection lifecycle, same URL) -> This is the deepest fix for
   the A8 finding: an upstream error string echoed a `Bearer` header into
   `.result`. Ignoring the path stops **publication**; it does not stop
   **production**. Worth queueing as a separate step (redact/mask
   `Authorization` in captured session output), not folded into 86.67.

9. **Prevention should be layered at the developer/pre-commit level.**
   *"Consider enabling secrets detection at the developer level to avoid
   checking secrets into code before commit/PR either in the IDE, as part of
   test-driven development, or via pre-commit hook."*
   (OWASP §8.1, same URL) -> pyfinagent already has the hook surface
   (`.claude/hooks/auto-commit-and-push.sh`) where such a check would sit.

10. **`.gitignore` alone is explicitly framed as insufficient.**
    *"The best way to avoid leaking secrets is to store them in local/private
    files and exclude these from git tracking with a `.gitignore` file"* --
    immediately followed by *"As an extra security measure, apply credential
    scanning in your CI/CD pipeline."*
    (Microsoft Code-with Engineering Playbook,
    https://microsoft.github.io/code-with-engineering-playbook/source-control/secrets-management/,
    accessed 2026-08-14)

11. **Assume an exposed credential is still live.** *"Nearly 70% of credentials
    confirmed as valid in 2022 were still valid in January 2025"*; by January
    2026 *"the validity rate was still above 64%"*.
    (GitGuardian 2026, https://blog.gitguardian.com/the-state-of-secrets-sprawl-2026/,
    accessed 2026-08-14) -> Do not assume the A8 token is dead because it looks
    malformed.

12. **Private/internal repos are the higher-risk population.** *"Internal repos
    are roughly 6x more likely than public ones to contain hardcoded secrets"*;
    32.2% of internal repos vs 5.6% of public.
    (same source) -> Removes "it's a private repo" as a mitigating argument.

13. **Agent-written repo files are an established attack surface (2026).**
    *"treat GitHub issue content, pull request descriptions, comments, and any
    repository file an agent loads automatically ... as untrusted input"*;
    Codex's *"first pass can write to `AGENTS.md`, an instruction file Codex
    loads from disk and treats as authoritative context."*
    (CSA Labs, 2026-08-08,
    https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-coding-agent-cicd-secrets-20260808-csa/,
    accessed 2026-08-14) -> Direct analogue: `run_away_session.sh:170` writes raw
    agent output into the working tree, and `auth_state.py:67` reads it back.
    The literature now treats exactly this write-then-read-from-tree loop as
    security-relevant.

## B5. Consensus vs debate

**Consensus (unanimous across all 6 sources):**
- `.gitignore` prevents *tracking of untracked paths*; it is not a control over
  already-tracked content (git-scm, primary).
- An exposed secret must be **revoked/rotated**, and that is the *first* step
  (GitHub, OWASP).
- `.gitignore` alone is insufficient; layer scanning/push-protection on top
  (Microsoft, OWASP, GitHub).

**Debate / divergence:**
- **How hard to push history rewriting.** GitHub is the most permissive --
  rewriting *"may not be warranted"* once rotated. OWASP notes squashing
  *"may introduce other problems."* Vendor blogs (GitGuardian's cheatsheet,
  Microsoft TechCommunity) lean toward "clean the history properly." For 86.67
  the question is moot: **criterion 4 forbids it and the operator owns ask 06-2**
  -- and the most authoritative source is also the one most comfortable with not
  rewriting.
- **Whether `.gitignore` is a "security control" at all.** Microsoft calls it
  *"the best way to avoid leaking secrets"* for not-yet-tracked files; the
  academic/industry framing treats it as hygiene. Both are reconcilable: it is a
  strong **staging-prevention** control and a **zero** exposure-remediation
  control. That is precisely the framing 86.67 should adopt.

## B6. Pitfalls from the literature, mapped to this step

| Pitfall | Source | Mitigation for 86.67 |
|---|---|---|
| Assuming `.gitignore` retroactively protects tracked files | git-scm primary | Scope the claim to the forward channel only (A6) |
| Assuming a leaked credential is dead | GitGuardian 2026 (>64% still valid) | Do not infer the A8 token is harmless from its malformed look |
| History rewriting causing recontamination via stale clones | GitHub Docs | Criterion 4 already forbids it; concurrent-session repo makes this acute |
| Ignoring the path but not the *producer* | OWASP §8.3 "never be logged" | Queue a separate redaction step; do not fold into 86.67 |
| A rule that matches nothing looking identical to one that works | criterion 3 itself | A5 both-direction proof |
| Misreading `check-ignore` exit-1 on tracked paths | Internal Q/A memory | Use `--no-index` for tracked paths (A5 note 2) |

## B7. Application to pyfinagent

1. **Criterion 1 -- decide explicitly.** The consumer evidence (A2) says the
   away-ops rail reads these from **disk**, never from the repo. Combined with
   git-scm's primary statement and the live `session.log` precedent (A3),
   **gitignoring is strictly better than scrubbing** and breaks nothing.
   `scripts/away_ops/auth_state.py:67` + `run_away_session.sh:135` are the only
   two touchpoints and both use absolute disk paths.
2. **Criterion 2 -- proven against the consumer.** Enumerated in A2; zero
   consumers read them as committed artifacts; positive control reported (A2.4).
3. **Criterion 3 -- rule shape.** Use `handoff/away_ops/session_*.json`. Do
   **not** use `handoff/away_ops/session*` (matches the tracked, rail-critical
   `session_notes.md` -- A5). Prove both directions with the A5 invocations;
   remember `--no-index` for tracked paths.
4. **Criterion 4 -- history untouched.** Nothing in this brief touches history.
   The authoritative source agrees rewriting is often unwarranted once the
   secret is rotated (B4 #4).
5. **Criterion 5 -- re-measured, and it FAILED the given.** See A8: 5 tracked,
   published files carry an Anthropic OAuth token. Escalate to the operator
   under ask 06-2; the untracked 5 are clean.
6. **Out of scope but worth queueing** (per the standing "queue discovered
   defects" rule): (a) redact `Authorization` headers before session JSON is
   written (`run_away_session.sh:170`) -- OWASP §8.3; (b) consider GitHub push
   protection (B4 #6); (c) fix the credential-scan regex class used in
   away-ops tooling so `sk-ant-*` hyphenated tokens cannot slip past (A8).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6**
- [x] 10+ unique URLs total -- **38**
- [x] Recency scan (last 2 years) performed + reported -- §B3
- [x] Full pages read (not abstracts) for the read-in-full set -- all 6 HTML, no PDFs
- [x] file:line anchors for every internal claim -- §A2, §A5, §A7

Soft checks:
- [x] Internal exploration covered every relevant module (§A7, 18 files)
- [x] Contradictions / consensus noted (§B5) -- incl. a contradiction of the caller's given (§A8)
- [x] All claims cited per-claim (§B4)

Known gap (disclosed, not gating): `arXiv:2604.03070v1` ("Credential Leakage in
LLM Agent Skills") is topically the closest academic work and was left
snippet-only -- the `simple` tier's floor was already exceeded and the decision
does not turn on it. Recommend it as the first read if this is ever re-run at a
higher tier.
