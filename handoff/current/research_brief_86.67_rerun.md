# Research Brief -- step 86.67 (RE-RUN)

**Topic:** away_ops session telemetry files are not gitignored; the auto-commit
hook publishes them to a PUBLIC repo, and five of them carried a live Anthropic
OAuth token. What SHOULD be done -- gitignore vs a commit-boundary secret
scanner vs redaction-at-write in the producer -- and in what order.

**Tier:** moderate. **Audit-class:** NO (`coverage` reported for information only).
**Role:** Layer-3 Researcher (external literature + internal code exploration).
**Date:** 2026-08-14.

**Nature of this run.** REMEDIATION RE-RUN, not new research. A sound brief
exists at `handoff/current/research_brief_86.67.md` (40,839 bytes, 6 sources,
recency scan done). Its gate failed for an **artifact-accounting** reason only:
its closing envelope carried no `sources_read_in_full` array, so `enforceGate`
could not corroborate a single claimed URL. This run carries those findings
forward, **re-fetches every inherited source so the read-in-full claim is this
run's own**, adds four new sources aimed at the caller's scope note, and emits
an envelope corroborated by this file's own text.

---

## ENVELOPE

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 22,
  "urls_collected": 31,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "gate_passed": true
}
```

> Envelope was born inert (`INCOMPLETE`, zeroed counts) in the first tool call
> and updated in place as sources landed; flipped to `COMPLETE` as the final act.
> Gate arithmetic: 9 >= 5 AND `recency_scan_performed` == true AND all hard
> blockers checked AND `coverage.audit_class` == false => `gate_passed: true`.

---

## Sources read in full (>=5 required; counts toward the gate)

Every row was fetched **by this run** via `WebFetch` (all HTML; no PDF fetches,
per the arXiv html-first policy). Rows 1-3, 6, 7 were also read by the prior
run and were re-fetched here rather than inherited. Rows 4, 5, 8, 9 are NEW.

| # | URL | Accessed | Kind | Tier | Key finding |
|---|---|---|---|---|---|
| 1 | https://git-scm.com/docs/gitignore | 2026-08-14 | Official docs (primary) | 2 | *"Files already tracked by Git are not affected"* |
| 2 | https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository | 2026-08-14 | Official docs | 2 | Revoke/rotate FIRST; rewrite *"may not be warranted"*; recommends gitleaks + push protection |
| 3 | https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html | 2026-08-14 | Standards body | 1/2 | Revocation -> Rotation -> Deletion; secrets must be masked/encrypted in logs; pre-commit detection |
| 4 | https://docs.github.com/en/code-security/secret-scanning/introduction/about-push-protection | 2026-08-14 | Official docs | 2 | **NEW.** Fires at **push** time, not commit time; user-level default covers **public repos only**; bypassable with a reason |
| 5 | https://github.com/gitleaks/gitleaks | 2026-08-14 | Official tool docs | 2 | **NEW.** MIT; `dir` scans working dir/staged, `git` scans history; `.gitleaks.toml` custom rules + allowlists + baseline; exit 1 on leak; **"feature complete... security patches only"** |
| 6 | https://blog.gitguardian.com/the-state-of-secrets-sprawl-2026/ | 2026-08-14 | Industry report | 4 | 28.65M secrets in 2025 (+34% YoY); AI-service secrets **+81% YoY**; **>64% of 2022-valid credentials still valid Jan 2026**; internal repos ~6x public |
| 7 | https://microsoft.github.io/code-with-engineering-playbook/source-control/secrets-management/ | 2026-08-14 | Vendor playbook | 3 | `.gitignore` is the baseline; credential scanning is the *"extra security measure"* |
| 8 | https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-coding-agent-cicd-secrets-20260808-csa/ | 2026-08-14 | Industry research note (2026-08-08) | 4 | **NEW.** Claude Code CVE-2026-54316 exfiltrated an API key; treat *"any repository file an agent loads automatically"* as untrusted input |
| 9 | https://arxiv.org/html/2604.03070v1 | 2026-08-14 | Peer-review-track preprint (2026) | **1** | **NEW + decisive.** *Credential Leakage in LLM Agent Skills*, n=17,022 skills: **73.5% of leaks are stdout/log capture**; fix is to strip credentials from the stdout stream **before** it is persisted |

Source 9 was the prior brief's explicitly disclosed **known gap** ("recommend it
as the first read if this is ever re-run at a higher tier"). It is now read, it
is the only Tier-1 peer-review-track source in the set, and it changes the
recommended ordering (see F1).

## Sources identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://csrc.nist.gov/pubs/sp/800/218/final | Standards body | **Fetch ATTEMPTED and FAILED to yield content** -- the landing page carries only metadata + abstract; the practice text is PDF-only. Recorded as an attempt, NOT counted. OWASP (source 3) covers the same ordering question. |
| https://appsecsanta.com/secret-scanning-tools/gitleaks-vs-trufflehog | Vendor comparison | Tier-4 benchmark marketing; source 5 is primary |
| https://appsecsanta.com/sast-tools/gitleaks-vs-trufflehog | Vendor comparison | Duplicate of the above |
| https://rafter.so/blog/secrets/gitleaks-vs-trufflehog | Vendor blog | Tier-4, superseded by source 5 |
| https://rafter.so/blog/secrets/secret-scanning-tools-comparison | Vendor blog | Tier-4 |
| https://devsecops.ae/secrets-scanners-comparison-2026/ | Vendor blog | Tier-4 |
| https://www.elegantsoftwaresolutions.com/blog/gitleaks-pre-commit-hooks-stop-leaks-before-push | Vendor blog | Tier-4; source 5 covers the hook integration |
| https://learn.microsoft.com/en-us/dotnet/core/extensions/data-redaction | Official docs | .NET-specific redaction API; wrong runtime |
| https://www.dash0.com/guides/opentelemetry-redaction-processor | Vendor guide | OTel Collector redaction; pyfinagent has no OTel pipeline |
| https://www.dash0.com/guides/scrubbing-sensitive-data-with-opentelemetry | Vendor guide | Duplicate topic |
| https://oneuptime.com/blog/post/2026-02-06-redaction-processor-opentelemetry-collector/view | Vendor blog | Duplicate topic |
| https://oneuptime.com/blog/post/2026-02-06-redact-sensitive-data-pii-opentelemetry-pipeline/view | Vendor blog | Duplicate topic |
| https://allan.reyes.sh/posts/keeping-secrets-out-of-logs/ | Personal blog | Tier-5; OWASP (3) + arXiv (9) cover it authoritatively |
| https://qaskills.sh/blog/ci-mask-secrets-in-test-logs | Community | Tier-5 |
| https://hoop.dev/blog/how-automatic-sensitive-data-redaction-and-telemetry-rich-audit-logging-allow-for-faster-safer-infrastructure-access | Vendor blog | Marketing-tier |
| https://chuniversiteit.nl/operations/stop-committing-secrets | Personal blog | Tier-5; agrees with source 2 |
| https://dev.to/safvantsy/accidentally-committed-secrets-a-simple-git-fix-is-not-enough-3gem | Community | Tier-5 |
| https://techcommunity.microsoft.com/blog/azureinfrastructureblog/how-to-safely-remove-secrets-from-your-git-history-the-right-way/4464722 | Vendor blog | History rewriting -- **forbidden by this step's criterion** |
| https://coreui.io/answers/how-to-remove-sensitive-data-from-git-history/ | Community | History rewriting -- out of scope |
| https://khadirullah.com/blog/git-filter-repo-scrub-secrets/ | Personal blog | History rewriting -- out of scope |
| https://gist.github.com/R1verRat/989911c6566235ec076ceac69f4ed04d | Community gist | Tier-5, history rewriting |
| https://github.com/orgs/community/discussions/161907 | Forum | Tier-5 |

**Total unique URLs: 31** (9 read in full + 22 snippet-only).

## Search-query composition (three-variant discipline)

| Variant | Query run |
|---|---|
| Current-year frontier (2026) | `gitleaks trufflehog pre-commit hook secret scanning commit boundary 2026` |
| Year-less canonical | `redaction at write log masking credentials telemetry sink never log secrets` |
| Year-less canonical | `secret accidentally committed to git repository rotate credential instead of rewriting history` |

The read-in-full table mixes 2026 hits (4, 5, 6, 8, 9) with year-less canonical
primary docs (1, 2, 3, 7), satisfying the mix requirement.

---

# PART A -- EXTERNAL FINDINGS (cited per claim)

## F1. The three options are not alternatives; they sit at three different points, and the literature ranks them

The caller framed gitignore / pre-commit scanner / redaction-at-write as a
choice. The literature is unambiguous that they are **layers**, and it ranks
them by where the credential actually enters the artifact.

**Redaction-at-write is the highest-impact point, and this is measured, not
asserted.** The largest empirical study of the exact failure class -- 17,022
LLM agent skills, 37,409 source files, three independent expert labellers
(Cohen's kappa = 0.88) -- finds that **"print/console.log statements account for
73.5% of vulnerabilities"**, and explains the mechanism: *"the largest single
vector because agent frameworks capture stdout into the LLM context window"*
(Chen et al. 2026, https://arxiv.org/html/2604.03070v1, accessed 2026-08-14).
Its primary recommendation for framework designers is to *"extract recognized
credential patterns from the stdout stream before it enters the LLM's
conversational memory"*, and it names the dominant vector as *"the highest-impact
remediation point"*. **This describes pyfinagent's mechanism exactly** -- see B2.

OWASP reaches the same conclusion from the standards side: logging must
implement *"either an encryption or masking approach in place to avoid logging
plaintext secrets"*
(https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html,
accessed 2026-08-14). Two independent sources -- one Tier-1 empirical, one
Tier-1/2 standards -- converge.

**Commit-boundary scanning is the defence-in-depth layer, explicitly
recommended.** OWASP: *"Consider enabling secrets detection at the developer
level to avoid checking secrets into code before commit/PR either in the IDE, as
part of test-driven development, or via pre-commit hook"* (same URL). GitHub's
own remediation page recommends *"pre-commit hooks like git-secrets or
gitleaks"* (https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository,
accessed 2026-08-14). Microsoft frames it as strictly additive: `.gitignore` is
*"the best way to avoid leaking secrets"* for untracked files, *"As an extra
security measure, apply credential scanning in your CI/CD pipeline"*
(https://microsoft.github.io/code-with-engineering-playbook/source-control/secrets-management/,
accessed 2026-08-14).

**`.gitignore` is the weakest of the three for this incident, and the caller's
scope note is confirmed by the primary source.** *"A gitignore file specifies
intentionally untracked files that Git should ignore. **Files already tracked by
Git are not affected**; see the NOTES below for details."* ... *"To stop tracking
a file that is currently tracked, use git rm --cached to remove the file from the
index."* (https://git-scm.com/docs/gitignore, accessed 2026-08-14). The five
leaking files are **tracked** (measured, B1) -- **an ignore rule would not have
prevented this leak.** It is still correct and necessary for the *forward*
channel, because every session file is a new uniquely-timestamped path.

## F2. Response ordering: revoke first, scrub second -- and there is a mechanistic reason, not just a policy

GitHub, primary and unambiguous: *"as a first step you need to revoke and/or
rotate that secret"*; and then *"Once the secret is revoked or rotated, it can no
longer be used for access, and that may be sufficient to solve your problem.
Going through the extra steps to rewrite the history and remove the secret may
not be warranted."*
(https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)

OWASP gives the same ordering as an explicit sequence: *"Revocation: Keys that
were exposed should undergo immediate revocation...Rotation: A new secret must be
quickly created and implemented...Deletion: Secrets revoked/rotated must be
removed from the exposed system immediately"*, and warns that squashing history
*"may introduce other problems as it rewrites git history and will break any
other links to a given commit."*
(https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)

**The mechanistic reason scrubbing is secondary** -- this is the highest-value
new finding of the re-run, because it is a *causal* argument rather than a
recommendation. Chen et al. measured what happens when you scrub without
rotating: *"credentials removed from 107 upstream repositories remain live across
50+ independent forks"* (https://arxiv.org/html/2604.03070v1). Removing a
credential from the origin does not remove it from the world. GitHub's
recontamination warning is the same phenomenon at smaller scale: *"If a fellow
developer has a clone from before your rewrite, and after your rewrite simply
runs git pull followed by git push, the sensitive data will return."*
(https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
**This repo is a live instance of that risk** -- it runs two concurrent Claude
sessions plus scheduled away sessions with an auto-push hook.

**Do not assume the exposed token is dead.** *"Nearly 70% of credentials
confirmed as valid in 2022 were still valid in January 2025"*, and retested in
January 2026 *"the validity rate was still above 64%"*
(https://blog.gitguardian.com/the-state-of-secrets-sprawl-2026/, accessed
2026-08-14). The prior brief noted the leaked token *looks* malformed (doubled
`sk-ant-oat01-sk-ant-oat01-`); that appearance is **not** evidence it is
unusable, and this statistic is the reason to treat rotation as operator-gated
work that still needs doing rather than as moot.

## F3. GitHub push protection: real, but partial, and it fires too late to be the primary control

Push protection *"prevent[s] hardcoded credentials, such as secrets or tokens,
from ever being pushed to your repository"* and engages **at push time**
(https://docs.github.com/en/code-security/secret-scanning/introduction/about-push-protection,
accessed 2026-08-14). Three limits matter for 86.67:

1. **It is not a commit-time control.** It would not stop the local commit, only
   the publication. Given this repo's auto-push hook the two are ~simultaneous,
   so it is still a meaningful last line -- but it cannot be the *only* line.
2. **Coverage is split.** Repository-level protection *"requires GitHub Secret
   Protection enabled"*; the free, on-by-default variant is user-level and
   *"stop[s] users from pushing secrets to public repositories on GitHub"*.
3. **It is bypassable** by any contributor with write access by choosing a
   reason (`It's used in tests` / `It's a false positive` / `I'll fix it later`).

**An open question this brief flags but does NOT resolve:** the repo is public
and five token-bearing files reached `origin/main` anyway. That is consistent
with either (a) push protection not being active on this repo, or (b) the
doubled/malformed token shape not matching GitHub's Anthropic pattern. Both are
cheap to check and neither is established here. Note (b) would rhyme exactly
with the prior brief's regex finding, where `sk-[A-Za-z0-9]{20,}` could not
cross the hyphens in `sk-ant-oat01-`.

## F4. Tool choice, if a commit-boundary scanner is adopted

gitleaks is MIT-licensed and its `dir` command *"enable[s] scanning of
directories and individual files"* -- i.e. the working tree / staged content --
while `git` scans history; `detect`/`protect` are deprecated since v8.19.0 in
favour of these (https://github.com/gitleaks/gitleaks, accessed 2026-08-14). It
exits **1** on any leak and **0** when clean, supports `.gitleaks.toml` custom
rules (regex + entropy + keyword prefilter), `[[allowlists]]`, `#gitleaks:allow`
inline annotations, a `.gitleaksignore` fingerprint file, and a **baseline**
(`--baseline-path`) so *"only new issues appear in the output report"*.

Two consequences specific to this repo:

- **The baseline is the mechanism that makes adoption possible here.** 71 tracked
  session files already exist and 5 contain a token. Without a baseline a
  history-scanning guard would be red on every commit forever. With
  `--baseline-path`, pre-existing findings are suppressed and only new ones fire.
- **Do not expect upstream to add an `sk-ant-oat01-` detector for you.**
  *"Gitleaks is feature complete. I'm not merging new features into Gitleaks.
  Future releases will be security patches only."* (same URL). A custom rule in
  `.gitleaks.toml` is the supported path and is a first-class feature, so this is
  a maintenance note, not a blocker. TruffleHog's differentiator is live
  credential *verification* (snippet-tier sources), which is orthogonal to
  blocking at the boundary.

## F5. Agent-written files in the working tree are now a documented attack surface

CSA Labs (2026-08-08) recommends treating *"GitHub issue content, pull request
descriptions, comments, and any repository file an agent loads automatically,
including convention files like AGENTS.md or CLAUDE.md and configuration files
like .env, as untrusted input regardless of the agent vendor's default trust
settings."*
(https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-coding-agent-cicd-secrets-20260808-csa/,
accessed 2026-08-14). It documents CVE-2026-54316 against Claude Code, which
*"abused Claude Code's pre-approved access to Hugging Face to exfiltrate an API
key one character at a time"*. The direct pyfinagent analogue is the
write-then-read-from-tree loop in B2: raw agent output is written into the
working tree and read back by the rail.

## F6. Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

**Performed.** Result: **3 new findings that COMPLEMENT, and 1 that
MATERIALLY RE-RANKS, the canonical guidance.** The canonical rule -- ".gitignore
does not affect tracked files; rotate an exposed secret first" -- is unchanged
and still stated in current primary docs (sources 1, 2). What the window adds:

1. *(complements)* **Scale and trajectory.** 28.65M new hardcoded secrets on
   public GitHub in 2025, +34% YoY; AI-service secrets 1,275,105, **+81% YoY**
   (GitGuardian 2026, https://blog.gitguardian.com/the-state-of-secrets-sprawl-2026/).
2. *(complements)* **Remediation largely does not happen** -- >64% of 2022-valid
   credentials still valid Jan 2026 (same source). Assume exposed == live.
3. *(complements)* **The agent-specific channel is now documented** (CSA Labs,
   2026-08-08, https://labs.cloudsecurityalliance.org/research/csa-research-note-ai-coding-agent-cicd-secrets-20260808-csa/).
4. *(RE-RANKS)* **Chen et al. 2026** (https://arxiv.org/html/2604.03070v1)
   supplies the first large-N measurement of *where* agent credential leaks
   originate: **73.5% from stdout/log capture**, 89.6% exploitable in normal
   execution, and fork-persistence defeating post-hoc scrubbing. This moves
   redaction-at-write from "one of three options" to "the option the evidence
   points at first", and supplies the causal argument for revoke-before-scrub.

Nothing found in the window contradicts the revoke-first ordering or recommends
history rewriting as a first move.

## F7. Consensus vs debate

**Consensus (all 9 sources where the topic arises):** `.gitignore` governs
untracked paths only and is not an exposure remedy (1, 7); an exposed secret is
revoked/rotated **first** (2, 3); scanning/push-protection is layered on top, not
substituted (2, 3, 7); secrets must not be written to logs/telemetry in the first
place (3, 9).

**Debate:** *how hard to push history rewriting.* GitHub is the most permissive
(*"may not be warranted"*); OWASP warns of collateral damage; the snippet-tier
vendor blogs lean "clean the history properly". For 86.67 this is moot -- the
step's criterion forbids rewriting and rotation is operator-gated -- and notably
**the most authoritative source is also the one most comfortable with not
rewriting**, and the Tier-1 paper supplies the mechanism for why (F2).

**Genuine tension worth recording:** Microsoft calls `.gitignore` *"the best way
to avoid leaking secrets"* while the empirical literature treats it as hygiene.
Both are true under a distinction this step should adopt explicitly:
`.gitignore` is a strong **staging-prevention** control and a **zero**
exposure-remediation control.

---

# PART B -- INTERNAL CODE INVENTORY (file:line anchored)

All greps pinned to `/usr/bin/grep`. The shell's `grep` is a ugrep wrapper
carrying `--ignore-files`, which honours `.gitignore` -- on this question that
would silently hide the very files under investigation.

## B1. Measured state (2026-08-14)

- An untracked session file matches **no** ignore rule:
  `git check-ignore -v handoff/away_ops/session_am_20260813T053028Z.json` -> **rc=1**, no output.
- The two spot-checked leakers are **tracked**:
  `session_am_20260809T053008Z.json` -> tracked=YES;
  `session_pm_20260810T200010Z.json` -> tracked=YES.
  This is the measurement that makes F1's `.gitignore`-cannot-have-helped
  conclusion concrete rather than theoretical.
- `.gitignore` handoff rules today: `:28 *.log`, `:76 handoff/logs/`,
  `:77 handoff/*.log`, `:80 handoff/archive/_quarantine_*/`,
  `:88 handoff/.away-session.lock`, `:93 handoff/.autonomous_loop.lock`.
  **Nothing matches `handoff/away_ops/session_*.json`.**

## B2. The producer -- where the credential enters the artifact (F1's target)

`scripts/away_ops/run_away_session.sh`:

| Anchor | Behaviour |
|---|---|
| `:23` | `OPS="$REPO/handoff/away_ops"` -- the sink is **inside the working tree** |
| `:135` | `OUT_JSON="$OPS/session_${SESSION}_$(date -u +%Y%m%dT%H%M%SZ).json"` -- new unique path per session |
| `:170` | `< "$PROMPT_FILE" > "$OUT_JSON" 2>> "$SLOG"` -- **raw agent stdout redirected verbatim into the tree, unfiltered** |
| `:186` | re-reads its own file: `grep -q '"api_error_status": *401' "$OUT_JSON"` |
| `:209`, `:215-217` | re-reads for cost + usage-limit surfacing |

`:170` **is** the 73.5% vector from Chen et al. verbatim: agent stdout captured
and persisted with no credential filter. An upstream API error echoed the
`Authorization: Bearer` header into `.result` and it landed on disk. This is the
single line a redaction-at-write fix would target, and it is the only place the
credential can be stopped *before* it exists as a file.

## B3. The consumer -- confirms an ignore rule is safe

`scripts/away_ops/auth_state.py:67-68`:

```python
sessions = sorted(glob.glob(os.path.join(ops, "session_*.json")),
                  key=os.path.getmtime)
```

`:113` `ap.add_argument("--ops", required=True)` -- the directory is a **CLI
parameter**, not a repo constant; `:75-76` gate on `"api_error_status": 401` and
`os.path.getmtime`. A consumer keyed on mtime cannot meaningfully be reading a
git-materialised artifact (git neither stores nor restores mtime). Carried
forward from the prior brief and re-confirmed: this is a **disk** consumer, so
gitignoring the path breaks nothing.

## B4. The publication mechanism, and its own falsified safety comment

`.claude/hooks/auto-commit-and-push.sh:360` -- `add_stderr=$(git add -A 2>&1)`,
retried at `:365`. Tree-wide, no pathspec, no exclusion for `handoff/away_ops/`.

The load-bearing find is the comment immediately above it, `:348-349`:

> `# Broad capture; the pre-commit pre-tool-use-danger guard + gitignore for`
> `# .env files cover safety.`

**That is a written safety model, and this incident falsifies it.** It claims
`git add -A` is safe because of (a) the pre-commit guard and (b) a `.gitignore`
rule for `.env`. Neither covers `handoff/away_ops/session_*.json`. Any fix
should also correct this comment, or the next reader inherits the same false
assurance.

## B5. The integration point already exists -- `.git/hooks/pre-commit` is live

This materially changes the cost of option 2 and was not in the prior brief.

- `auto-commit-and-push.sh` contains **no `--no-verify`** (measured) -- so
  `.git/hooks/pre-commit` **does fire** on the auto-commit path. A guard added
  there would have blocked the leaking commit.
- `.git/hooks/pre-commit` **already exists and already blocks**: three active
  guards -- stray `.claude/*.bak-*` files (`:10-17`), retired Claude snapshot IDs
  in staged `*.py` (`:19-29`), and dotenv syntax on staged `.env` (`:35-47`) --
  each using the exact idiom a secret guard needs:
  `git diff --cached --name-only --diff-filter=ACM` -> filter -> `exit 1`.
- There is **no secret scanning anywhere in `.claude/hooks/`** (measured: zero
  matches for `gitleaks|trufflehog|secret[_-]?scan|detect[_-]secret|sk-ant`).
- There is **no `.pre-commit-config.yaml`** -- the `pre-commit` *framework* is not
  adopted; the hook is hand-written.

**Three pitfalls for whoever implements this, all derived from the file itself:**

1. **`set -e` at `:5` makes this hook fail-CLOSED.** Every existing guard wraps
   its grep in `|| true` precisely because `grep` exits 1 on no-match and `set -e`
   would abort the hook -- which git reads as "commit rejected". A new guard that
   forgets `|| true` blocks **every clean commit**. This is the repo's
   "operations that cannot fail loudly" class inverted: here it fails loudly on
   the wrong input.
2. **A rejected commit is SILENT on the auto-commit path.**
   `auto-commit-and-push.sh:380` is
   `if ! git commit -m "$SUBJECT" ...; then log "git commit failed"; exit 0; fi`.
   A pre-commit rejection therefore produces no commit, no push, and **exit 0** --
   visible only in `handoff/logs/auto-push.log`. Blocking is correct; being
   invisible is not. Any adoption needs a `systemMessage` or equivalent surfacing.
3. **`.git/hooks/` is not version-controlled.** The existing guard is unreviewed,
   unpropagated, and lost on a fresh clone. This is the trade-off the
   `pre-commit` framework exists to solve, at the cost of a new dependency.

## B6. Internal file inventory

| File | Anchors | Role | Status |
|---|---|---|---|
| `scripts/away_ops/run_away_session.sh` | `:23`, `:135`, `:170`, `:186`, `:209-217` | **Producer.** Redirects raw agent stdout into the tree -- the leak's origin | LIVE |
| `scripts/away_ops/auth_state.py` | `:47`, `:67-68`, `:75-76`, `:113` | **Sole consumer.** Globs from a `--ops` disk path, keyed on mtime | LIVE |
| `.claude/hooks/auto-commit-and-push.sh` | `:348-349` (falsified safety comment), `:360` `git add -A`, `:380` silent-commit-failure | **Publisher** | LIVE |
| `.git/hooks/pre-commit` | `:5` `set -e`, `:10-17`, `:19-29`, `:35-47` | **Existing blocking guard** -- the ready integration point; NOT version-controlled | LIVE |
| `.gitignore` | `:28`, `:76-77`, `:80`, `:88`, `:93` | No rule matches `session_*.json` | LIVE |
| `handoff/away_ops/` | -- | Sink dir; 5 untracked + 71 tracked session JSONs | LIVE |
| `.claude/hooks/` (tree) | -- | Zero secret-scanning references (measured) | GAP |
| `.pre-commit-config.yaml` | -- | **Absent** -- framework not adopted | ABSENT |

---

# PART C -- APPLICATION TO pyfinagent

**C1. The three options, ranked by the evidence, as layers not alternatives.**

| Layer | Where it acts | What it fixes | What it cannot fix | Evidence |
|---|---|---|---|---|
| **Redaction-at-write** (`run_away_session.sh:170`) | Before the file exists | The root cause; the credential never reaches disk, so it cannot leak via git, backups, or the digest | Token shapes the redactor does not know | Chen et al. 73.5% + *"highest-impact remediation point"* (https://arxiv.org/html/2604.03070v1); OWASP *"never be logged"* (https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html) |
| **Commit-boundary scanner** (`.git/hooks/pre-commit`) | Before the commit | Catches what redaction misses, across the WHOLE tree not just away_ops | Anything after the commit; needs a baseline for 71 pre-existing files | OWASP pre-commit guidance; GitHub names gitleaks (https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository); tool mechanics (https://github.com/gitleaks/gitleaks) |
| **`.gitignore`** | Before staging | The FORWARD channel only -- every future session file is a new unique path | **The 5 tracked files. It would not have prevented this incident.** | *"Files already tracked by Git are not affected"* (https://git-scm.com/docs/gitignore) |

**Recommendation for PLAN (Main owns the decision):** all three, in that
priority order, with `.gitignore` as the cheapest and most immediate and
redaction-at-write as the one that actually closes the class. The scanner is the
only layer that generalises beyond `handoff/away_ops/`, which matters because
`git add -A` at `:360` is tree-wide.

**C2. Ignore-rule shape (carried forward, unchanged, still correct).** Use
`handoff/away_ops/session_*.json`. Do **not** use `handoff/away_ops/session*`:
that also matches the **tracked, rail-critical** `handoff/away_ops/session_notes.md`.
Pattern anchoring is confirmed by the primary source -- *"a leading slash is not
relevant if there is already a middle slash in the pattern"*, and the pattern is
relative to the `.gitignore`'s own directory (https://git-scm.com/docs/gitignore).
Precedence from the same source (`.gitignore` > `info/exclude` >
`core.excludesFile`, with out-of-tree patterns *"treated as if they are specified
at the root of the working tree"*) is what makes the prior brief's
non-destructive `core.excludesFile` test method faithful: a match at the weakest
precedence implies a match at the strongest.

**C3. Verification trap that will bite the `live_check`.** `git check-ignore`
returns **1 for tracked paths regardless of whether a rule matches**, because
tracked files are exempt. Running it over the 71 tracked session files without
`--no-index` yields 71 misleading rc=1s. Use `--no-index` for tracked paths and
plain `check-ignore` for untracked ones, and state which you used.

**C4. Ordering for the operator-gated half.** Revoke/rotate the exposed token
**first**; scrubbing published history is second and this step forbids it anyway.
That ordering is GitHub's explicit first step, OWASP's explicit sequence, and is
mechanistically justified by fork/clone persistence (F2). Do **not** infer the
token is harmless from its malformed appearance -- >64% of 2022-exposed
credentials were still valid in Jan 2026
(https://blog.gitguardian.com/the-state-of-secrets-sprawl-2026/).

**C5. Defects to queue as their own steps (not to be folded into 86.67).**

1. **Redaction-at-write** at `run_away_session.sh:170` -- strip
   `Authorization`/`Bearer` and `sk-ant-*` shapes before the JSON is persisted.
   This is the root-cause fix and the highest-impact one per F1.
2. **Commit-boundary secret guard** in the existing `.git/hooks/pre-commit`,
   with a gitleaks baseline for the 71 pre-existing files, `|| true` on the
   grep (B5 pitfall 1), and a surfaced rejection (B5 pitfall 2).
3. **Correct the falsified safety comment** at
   `.claude/hooks/auto-commit-and-push.sh:348-349`.
4. **Check whether GitHub push protection is active** on this repo, and whether
   the doubled-token shape evades its Anthropic pattern (F3, open question).
5. **Fix the credential-scan regex class** used by away-ops tooling so
   hyphenated `sk-ant-oat01-` tokens cannot slip past a `sk-[A-Za-z0-9]{20,}`
   character class (carried forward from the prior brief's §A8).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**, all fetched by this run
- [x] 10+ unique URLs total -- **31**
- [x] Recency scan (last 2 years) performed + reported -- §F6
- [x] Full pages read (not abstracts) for the read-in-full set -- 9/9 HTML; arXiv via `/html/` per policy, no PDF fetches
- [x] file:line anchors for every internal claim -- §B1-B6

Soft checks:
- [x] Internal exploration covered every module in the caller's stated scope (8 files/dirs, §B6)
- [x] Contradictions / consensus noted -- §F7, incl. a genuine Microsoft-vs-empirical tension and one open question flagged as unresolved (§F3)
- [x] All claims cited per-claim with URL + access date
- [x] Three-variant search discipline visible -- query table above

**Disclosed limitation (non-gating):** the NIST leg of the caller's
"OWASP/NIST" ask was **attempted and failed** --
https://csrc.nist.gov/pubs/sp/800/218/final serves only metadata and an
abstract; SP 800-218's practice text is PDF-only. It is recorded as
snippet-only, not counted toward the floor. OWASP (source 3) answers the same
response-ordering question authoritatively, so the gap does not affect any
conclusion. A future run wanting NIST specifically should extract the PDF with
`pypdf`/`pdfplumber` per the arXiv/PDF chain rather than fetching the landing
page.
