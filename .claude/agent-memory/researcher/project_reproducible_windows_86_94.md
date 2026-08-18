---
name: project-reproducible-windows-86-94
description: "Pinning a git --since timestamp is NOT enough: a naive timestamp is TZ-local (707 vs 787 commits, both ends pinned). Also --since=today is NOT midnight though both man pages say it is, and @{date} fails open."
metadata:
  type: project
---

Phase-86.94 research. Measured 2026-08-16, `git version 2.50.1 (Apple Git-155)`.
Probe technique worth reusing: **`git rev-parse --since=<X>` prints the resolved
`--max-age=<epoch>`** — a direct read of approxidate's output, no repo needed.

**The finding that survives the obvious fix.** phase-86.91 pinned
`CORPUS_SINCE = "2026-08-11T00:00:00"` to stop a bare date sliding with the clock. That
is still not reproducible: **a naive timestamp is TZ-LOCAL.** The same string spans
**13 hours** between Seoul and New York, and with **both ends already pinned** the corpus
is **707 commits under Europe/Oslo and UTC but 787 under Asia/Seoul**. TZ-invariant forms,
measured identical: `...T00:00:00Z`, `...+0000`, `@<epoch>`.

**Why: the docs are wrong.** git-log(1) and git-rev-list(1) both say *"As a special case,
today means the last midnight."* Measured, `--since=today` -> **0 commits**,
`--since=midnight` -> **64**. `today` is absent from `date.c`'s `special[]` table (which
holds `midnight`, `noon`, `now`, `yesterday`), so it falls through to the generic parser
and keeps the wall-clock time of day. **Never cite the man page for this; cite a probe.**

**A window has more than two ends.** Same `(since, until)` gives 707 under the tip but
**766 under `--all`**. `A..` silently defaults its upper end to `HEAD`. `HEAD@{date}` is
worse than the disease: it reads the **local, per-clone, 90-day-expiring reflog**, warns
to stderr, and **still exits 0 with a sha**. The reproducible unit is
`(since, until, ref-scope, traversal flags, non-shallow)`.

**Why it matters:** a detector keyed only on `--since`/`--until` misses `A..`, `--all`,
`--since=30.days`, and every `datetime.now() - timedelta(...)` window.

**How to apply:** when any count will be quoted in a criterion, an `audit_basis`, or a
handoff artifact — pin with an explicit **UTC** instant, pin the upper end to a **commit
sha**, and print the resolved endpoint next to the number.
`scripts/qa/replay_changelog_rule_86_68.py:113` and `:124-129` already do the last two
correctly and are the template. **Do NOT propose banning now-relative windows**:
`backend/slack_bot/scheduler.py:503` (`--since-as-filter=midnight`, a daily digest) is a
correct relative use, and every mature system studied — SOURCE_DATE_EPOCH,
`java.time.Clock`, PromQL `@`, dbt `--event-time-start/--event-time-end`, AWS durable
steps — makes the instant **overridable, not forbidden**.

`--since` stops traversal at the first older commit (`--since-as-filter`, git >= 2.37,
does not). Measured **inactive on this repo**: identical counts at three cutoffs,
committer dates monotonic across 2,999 adjacent pairs. Guard it; don't claim it as a bug.

No off-the-shelf detector exists: Ruff DTZ005 bans **tz-naive** `now()`, not `now()`;
there is no `pyproject.toml` here anyway; Semgrep has no registry rule; and two separate
2026 reproducibility papers (arXiv 2602.08561v3, 2604.25944) omit time windows from their
failure taxonomies entirely. Template to copy: `scripts/governance/lint_limits_usage.py`
(AST walk + allowlist + `0/1/2` exits) plus a `.github/workflows/*-lint.yml` sibling.

Full brief with all 17 sources: `handoff/current/research_brief_86.94.md`.
See also [[reference-webfetch-pdf-summaries-fabricate-quotes]] — the `date.c` fetch
returned a paraphrase, so the mechanism was re-established by local measurement.
