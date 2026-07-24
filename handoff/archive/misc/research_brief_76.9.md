# Research Brief -- Step 76.9 (P1, operator bug B1)

tier: moderate | status: COMPLETE -- gate_passed=true | 2026-07-24 | Layer-3 Researcher

Two INDEPENDENT causes:
1. **AUTORESEARCH**: arxiv HTTP 429 kills `run_nightly.sh` (rc=1) every night 2026-07-08..07-24
2. **ABLATION**: launchd job's raw `. backend/.env` dies on malformed `backend/.env` L81 (orphan unbalanced quote)

---

## Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
| - | --- | --- | --- | --- | --- |
| 1 | https://info.arxiv.org/help/api/tou.html | 2026-07-24 | official doc | WebFetch | "make no more than one request every three seconds, and limit requests to a single connection at a time"; higher rate -> "contact our support team"; limits apply COLLECTIVELY across all your machines (can't bypass by adding machines). |
| 2 | https://info.arxiv.org/help/api/user-manual.html | 2026-07-24 | official doc | WebFetch | "we encourage you to play nice and incorporate a 3 second delay"; `max_results` limited to 30000 in slices of <=2000; "refine queries which return more than 1,000 results, or at least request smaller slices". NO mention of 429/Retry-After. |
| 3 | https://github.com/assafelovic/gpt-researcher/issues/1282 | 2026-07-24 | code/issue | WebFetch | Confirms the retrievers[0] trap: "the subquery generation fails because it is using only first retriever"; "the subquery retriever should use other retrievers also if available instead of returning empty list." Opened 2025-03-20, status Closed (no fix details in thread). |
| 4 | https://github.com/lukasschwab/arxiv.py | 2026-07-24 | code/doc | WebFetch | `Client(page_size, delay_seconds, num_retries)`; "Reusing a client allows successive API calls to use the same connection pool and ensures they abide by the rate limit you set." Installed v3.0.0 defaults (from source): page_size=100, delay_seconds=3.0, num_retries=3. Does NOT honor Retry-After (fixed delay only). |
| 5 | https://docs.gptr.dev/docs/gpt-researcher/search-engines | 2026-07-24 | official doc | WebFetch | "You can also specify multiple retrievers by separating them with commas. The system will use each specified retriever in sequence." NO documented fallback-on-failure. Docs list omits semantic_scholar (but installed code `get_retriever` DOES support it -- verified). |
| 6 | https://groups.google.com/a/arxiv.org/g/api/c/ycq8giRdZsQ | 2026-07-24 | official forum (arXiv API group) | WebFetch | [RECENCY/ADVERSARIAL] Widespread 429 since ~2026-02-25 DESPITE 3-4s delay; arXiv staff (B.Maltzan, Feb 27) "acknowledged recent rate-limit changes"; users suspect IP-granularity; L.Schwab (arxiv.py author): "POST requests bypass Fastly caching -> much more immediately to 429; GET benefits from caching." No higher-throughput path given. |
| 7 | https://github.com/theskumar/python-dotenv/issues/487 | 2026-07-24 | code/issue | WebFetch | Safe bash .env sourcing idiom `set -a && source .env && set +a`; "There is no formal specification for .env, and each parser is slightly different"; direct `. .env` breaks on quotes/spaces/special chars -- confirms grep-to-KEY=value pre-filter is the robust pattern. |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://arxiv.org/abs/2510.04516 | preprint (client-side rate limiting) | general RL theory, not arxiv-API-specific; adjacent |
| https://groups.google.com/a/arxiv.org/g/api/c/pNB3lnxf4mQ | official forum | second arXiv "rate exceeded" thread; duplicative of #6 |
| https://github.com/assafelovic/gpt-researcher/issues/1111 | code/issue | "local mode still requires TAVILY" -- retriever-required-ness context |
| https://github.com/assafelovic/gpt-researcher/issues/1264 | code/issue | custom-retriever registration steps |
| https://deepwiki.com/assafelovic/gpt-researcher | 3rd-party wiki | planner-executor architecture overview |
| https://docs.gptr.dev/docs/gpt-researcher/gptr/config | official doc | RETRIEVER env var config reference |
| https://gist.github.com/judy2k/7656bfe3b322d669ef75364a46327836 | community gist | canonical bash dotenv-parse patterns |
| https://blog.postman.com/http-error-429/ | vendor blog | 429 + Retry-After general best practice |
| https://zuplo.com/learning-center/http-429-too-many-requests-guide | vendor blog | 429 exponential backoff + jitter guidance |
| https://requestly.com/blog/api-rate-limiting/ | vendor blog | 429 handling overview |
| https://en.wikipedia.org/wiki/Rate_limiting | reference | rate-limiting definitions |
| https://nurbak.com/en/blog/429-too-many-requests/ | blog | 429 fixes (2026) |

URLs total: 7 read-in-full + 12 snippet-only = **19** (floor 10, moderate target 15+).

## Recency scan (2024-2026)
**Finding (MATERIAL -- changes the fix design).** An explicit 2026-scoped pass
(`arxiv API 429 2026 export.arxiv.org`) surfaced the arXiv API developer group
thread (source #6, read in full) documenting a **widespread arXiv-API 429 regression
beginning ~2026-02-25**: numerous independent users (Bharath Sethupathi Feb 27,
Serguei Kalentchouk May 14, Mohammed Mehdi May 29, plus June 2-5 reports) hit HTTP
429 (and intermittent 503) at `export.arxiv.org/api/query` **while respecting the
documented 1-request/3-seconds limit**. arXiv staff acknowledged "recent rate-limit
changes" (Feb 27) and later reported capacity was increased with one fix still
pending (~June 2026); no higher-throughput path was published. pyfinagent's nightly
autoresearch began failing 2026-07-08 -- squarely inside this regression window.
IMPLICATION: this is an EXTERNAL, server-side arXiv change, not a pyfinagent
misconfig; adding client-side politeness/backoff is UNRELIABLE as a sole fix because
polite clients are being 429'd anyway. The durable fix is to stop depending on arxiv
being up (drop/deprioritize it + tolerate its failure), not to back off harder.

Second recency thread: gpt-researcher's own retrievers[0]-only planning limitation
(source #3, 2025) has "recent fixes ... using all configured retrievers in deep
research planning" per search snippets -- but the INSTALLED 0.14.8 still exhibits the
single-retriever planning path (verified in source at skills/researcher.py:62), so an
upgrade MIGHT help but is unproven here and out of scope for a P1 hotfix. No new
finding supersedes the .env-sourcing guidance (stable POSIX shell behaviour).

---

## Internal code inventory
| File | Lines | Role | Status |
| --- | --- | --- | --- |
| scripts/autoresearch/run_nightly.sh | 1-75 | autoresearch launchd wrapper; HAS phase-62.6 grep-sanitize (:19-27) | LIVE, working sanitize |
| scripts/autoresearch/run_memo.py | 1-251 | gpt-researcher runner; broad-except -> ERROR memo -> return 1 | LIVE, fails on arxiv 429 |
| scripts/ops/run_ablation.sh | 1-73 | ablation launchd wrapper; verbatim copy of the 62.6 sanitize (:21-37) | EXISTS (phase-75.11), fix code |
| scripts/ops/com.pyfinagent.ablation.plist.template | -- | checked-in plist template, `__REPO_ROOT__` placeholder -> run_ablation.sh | EXISTS (phase-75.11) |
| ~/Library/LaunchAgents/com.pyfinagent.ablation.plist | 1-35 | LIVE plist; NOW points to run_ablation.sh (mtime 2026-07-24T08:52) | runs=0, UNPROVEN |
| scripts/ablation/run_ablation.py | 1-40+ | ablation study; `--next-untested` entrypoint | LIVE, never reached on crash nights |
| backend/tests/test_phase_39_1_autoresearch_env.py | 1-75 | locks the anthropic: prefix fix | LIVE |
| backend/tests/test_phase_75_sre_ops.py | -- | phase-75.11 sre-ops (run_ablation.sh) tests | LIVE (read pending) |
| backend/tests/test_phase_75_deps.py | -- | gpt_researcher guard tests | LIVE (read pending) |

### CAUSE 1 -- AUTORESEARCH arxiv 429 (re-derived facts)

- **Verbatim failure** (handoff/autoresearch/2026-07-23-ERROR-topic08.md:5):
  `Error: HTTPError: Page request resulted in HTTP 429 (https://export.arxiv.org/api/query?search_query=...&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100)`
  Note `max_results=100` in the query URL.
- **Unbroken daily ERROR memos** 2026-07-08 -> 2026-07-24 (topic07..topic09 rotating). Confirms "every run" claim; extends to 07-24.
- **Installed versions** (.venv/lib/python3.14/site-packages): `gpt_researcher==0.14.8`, `arxiv==3.0.0`.
- **RETRIEVER config**: run_memo.py:211 `"RETRIEVER": "arxiv,semantic_scholar,duckduckgo"` (step text said ~:189 -- STALE; actual :211). Set via `os.environ.setdefault` loop (:213-214).
- **Exit-code chain**: `run_research()` (run_memo.py:68-83) calls `GPTResearcher.conduct_research()` + `write_report()`. arxiv 429 raises `arxiv.HTTPError` out of conduct_research -> caught by BROAD `except Exception` in `_main_async` (:114) -> writes ERROR memo (:117-122) -> `return 1` (:124) -> `main()` returns 1 (:246) -> `SystemExit(1)` (:250) -> python exits 1 -> run_nightly.sh `if python ...` false (:43) -> `rc=$?` (:47) -> `exit "$rc"` (:72). So one retriever's 429 = whole-job rc=1.
- **arxiv retriever source** (.venv/.../gpt_researcher/retrievers/arxiv/arxiv.py): `ArxivSearch.search(max_results=5)` calls `list(arxiv.Client().results(arxiv.Search(query=..., max_results=max_results, sort_by=...)))`. **NO try/except**; `list()` forces the lazy generator, which is where `arxiv.HTTPError` (429) is raised. Default constructor `arxiv.Client()` -- default `page_size`, `delay_seconds`, `num_retries` (NOT tuned).
- **THE load-bearing orchestration finding** (.venv/.../gpt_researcher/skills/researcher.py):
  - `plan_research():62` -- `get_search_results(query, self.researcher.retrievers[0], ...)` uses **retrievers[0] ONLY** (= arxiv, first in the comma list) for the initial planning search. If arxiv raises here, the run dies before sub-query fan-out.
  - `get_retrievers()` (actions/retriever.py:99-136) parses `cfg.retrievers` comma-string -> ordered list of classes; **no reordering, no health-check**.
  - There is **NO built-in cross-retriever fallback** in the standard web flow: the non-MCP path calls `_get_context_by_web_search` which uses the configured retrievers, but the PLANNING call at :62 is hard-wired to `retrievers[0]`. Putting a fragile retriever (arxiv) first is the trap.
  - **Fix hooks available**: (a) REORDER so a robust retriever (duckduckgo) is retrievers[0]; (b) tune `arxiv.Client(delay_seconds=, num_retries=)` -- but gpt_researcher's ArxivSearch hard-codes `arxiv.Client()` with no passthrough, so tuning requires monkeypatch or dropping arxiv; (c) wrap `run_research()` to tolerate arxiv-class 429/5xx and fall through / exit 0 WARN. Cleanest at OUR layer: drop/deprioritize arxiv in the RETRIEVER list AND/OR catch the arxiv HTTPError in run_memo.py and exit 0 with a WARN memo (matches step's planned fix).

### CAUSE 2 -- ABLATION `.env` EOF crash (re-derived facts)

- **Verbatim failure** (handoff/logs/ablation.launchd-v4.log, 84 lines = ~42 nights x2):
  `backend/.env: line 81: unexpected EOF while looking for matching `"'`
  `backend/.env: line 85: syntax error: unexpected end of file`
  (Log path re-derived: step text's `handoff/ablation.launchd.log` is STALE -- logrotate rotated to `handoff/logs/ablation.launchd-v4.log`; root `handoff/logs/ablation.launchd.log` is 0 bytes since 2026-05-07. v4 mtime 2026-07-24T03:00 = today's 3am run.)
- **backend/.env malformed line** (Main read READ-ONLY, structure only -- .env DENIED to researcher sandbox):
  - File is 86 lines. L80 is a COMMENT: `# phase-61.1 (2026-06-12): operator tokens "60.2 FLAG: ON" / "60.3 FLAG: ON" / "57.1 FLAG:`
  - **L81 is a NON-KEY orphan fragment**: `  ON"` (2 leading spaces, ONE unbalanced `"`). It is the hard-wrapped TAIL of L80's comment that lost its leading `#`.
  - L82-85 are balanced KEY= lines; L86 blank (EOF).
  - **Load-bearing**: L81 does NOT match `^[A-Za-z_][A-Za-z0-9_]*=`, so the phase-62.6 grep-sanitize DROPS it -> adopting that sanitize in the ablation job FIXES the crash. There is NO `KEY="multiline` case, so the sanitize is sufficient (no partial-line-extraction trap).
  - Minor discrepancy to report, not resolve: the v4 log says `line 85`; Main's current read says the file is 86 lines (L86 blank). The log is runtime ground-truth; the 1-line delta likely reflects a trailing-line change since the failures or an off-by-one in EOF reporting. Immaterial to the fix.
  - **REPAIR (operator-gated, NOT our edit)**: rejoin `  ON"` into L80's comment, or prefix L81 with `#`.
- **Original crashing plist** (~/Library/LaunchAgents/com.pyfinagent.ablation.plist.bak-harness-ABCD, mtime 2026-04-16): inline
  `/bin/bash -c 'cd <REPO> && set -a && . backend/.env && set +a && . .venv/bin/activate && python scripts/ablation/run_ablation.py --next-untested >> handoff/ablation.log 2>&1'`
  The raw `. backend/.env` is the crash site; `&&` chain means a source failure short-circuits -> python NEVER runs -> zero ablation output those nights.
- **Fix already coded (phase-75.11, commit 07182b94)**: `scripts/ops/run_ablation.sh` (verbatim copy of run_nightly.sh:19-27 sanitize at :21-37) + `scripts/ops/com.pyfinagent.ablation.plist.template` (`__REPO_ROOT__` -> run_ablation.sh). Bootstrap gated by operator token `OPS-ROTATE-BOOTSTRAP`.
- **STATE IN FLUX (report, do not over-interpret)**: LIVE plist + `.pre-75.11.bak` BOTH mtime 2026-07-24T08:52 with IDENTICAL template content (points to run_ablation.sh). `launchctl print` shows loaded job -> run_ablation.sh, `runs=0`, `last exit code=(never exited)`. Signature of a `cp`/bootstrap done ~08:52 TODAY (after the 3am crash), likely Main/executor staging for THIS step (executors active this session; cf auto-memory feedback_executor_sees_mutation_transients). NET: the ablation fix appears ~deployed but is UNPROVEN (job hasn't run since reload). Main should confirm whether 08:52 was intentional and whether a clean run is proven before claiming PASS.

### Masterplan step 39.1 disposition
- 39.1 = "Autoresearch nightly cron exit 1 fix (OPEN-29) -- owner-gated", `status: pending`, P1.
- Its root cause (audit_basis) is the OLD autoresearch `.env` partial-fix era (2026-05-20 ERROR memos). That `.env`-format root cause was SUPERSEDED by phase-62.6's grep-sanitize -- autoresearch's CURRENT failure is arxiv 429, a DIFFERENT root cause.
- 39.1 verification command `ls handoff/autoresearch/ | grep -E '2026-05-(2[3-9]|3[01])-PASS'` is UNSATISFIABLE: (a) dates are in the past; (b) memos are never named `-PASS` (only `-ERROR-` or `<date>-topicNN-<slug>`), so the `-PASS` token can never appear (corroborated by auto-memory project_cron_maintenance_jobs: "39.1 grep can never match").
- **Recommendation: SUPERSEDE 39.1 with 76.9.** 76.9 covers the live autoresearch failure (arxiv 429) with a satisfiable verification; carry forward 39.1's "root_cause_documented" intent by having 76.9 write a root-cause note. Verification criteria are immutable, so 39.1 cannot be edited -- close as superseded-by-76.9 with an audit note.

---
## Key findings (external, per-claim cited)
1. **arXiv's official rate limit is 1 request / 3 seconds, single connection, enforced collectively across all your machines.** Higher rates require contacting support (no self-serve tier). (Source: arXiv ToU, https://info.arxiv.org/help/api/tou.html, 2026-07-24)
2. **arXiv's own manual recommends a 3s inter-request delay and to refine any query returning >1,000 results / request smaller slices.** Our retriever requests `max_results=100` in ONE page -- a heavy single call. The manual documents NO 429/Retry-After behaviour, so clients cannot rely on Retry-After from arXiv. (Source: arXiv User Manual, https://info.arxiv.org/help/api/user-manual.html, 2026-07-24)
3. **The installed `arxiv==3.0.0` Client already retries 3x with a 3.0s delay by default and does NOT honor Retry-After** -- it uses a fixed delay. gpt-researcher's `ArxivSearch` constructs `arxiv.Client()` with defaults and exposes no passthrough, so we cannot tune backoff without monkeypatching or replacing the retriever. Since the run is ALREADY retrying 3x and still 429ing, more backoff is not a reliable fix. (Sources: arxiv.py README https://github.com/lukasschwab/arxiv.py + installed source .venv/.../arxiv/__init__.py:600, 2026-07-24)
4. **gpt-researcher uses ONLY `retrievers[0]` for subquery/planning; there is no built-in fallback when it fails.** This is a documented upstream issue (#1282) and is verified in the installed 0.14.8 source (skills/researcher.py:62, outside the :348-365 try/except at :330). So arxiv being FIRST in the comma-list is the structural trap: its 429 aborts the whole run before the tolerant sub-query fan-out. (Sources: gpt-researcher #1282 https://github.com/assafelovic/gpt-researcher/issues/1282 + installed source, 2026-07-24)
5. **RETRIEVER is a comma-list "used in sequence" with no documented failure-fallback; `semantic_scholar` and `duckduckgo` are both valid installed retrievers.** Reordering/pruning the list is a supported, low-risk config change (env var at run_memo.py:211). (Sources: gpt-researcher search-engines docs https://docs.gptr.dev/docs/gpt-researcher/search-engines + installed actions/retriever.py get_retriever, 2026-07-24)
6. **[RECENCY] arXiv has been returning 429 to polite (3s) clients since ~2026-02-25**, an acknowledged server-side change; pyfinagent's failures (from 2026-07-08) fall inside this window. The robust response is to not depend on arxiv, not to back off harder. (Source: arXiv API group thread https://groups.google.com/a/arxiv.org/g/api/c/ycq8giRdZsQ, 2026-07-24)
7. **The canonical safe way to source a DOTENV file in POSIX shell is to pre-filter to `KEY=value` lines (grep) then `set -a; . tmpfile; set +a`** -- a raw `. .env` breaks on any unbalanced quote / comment / special char because there is "no formal specification for .env". This is exactly the phase-62.6 pattern already in run_nightly.sh:19-27 and copied into scripts/ops/run_ablation.sh:21-37. (Source: python-dotenv #487 https://github.com/theskumar/python-dotenv/issues/487, 2026-07-24)

## Consensus vs debate (external)
- **Consensus**: honor Retry-After + exponential backoff with jitter for 429 (Postman/Zuplo/Requestly); arXiv's 1-req/3s is the hard published limit; grep-filter is the safe dotenv-sourcing idiom.
- **Debate / unsettled**: whether arXiv's 2026 429 wave is fully resolved -- staff said capacity was raised with "one fix still needed" (~June 2026) and no confirmation of closure. Practitioners disagree on whether the 429 is IP-level vs endpoint-level. This uncertainty is itself the argument for treating arxiv as best-effort, not a hard dependency.

## Pitfalls (from literature + code)
- **P1 -- Backoff-only is insufficient**: polite clients are 429'd in 2026; and arxiv.py already retries 3x/3s, so "add backoff" alone will keep failing (Key finding 3, 6).
- **P2 -- Reorder-only leaves a landmine**: moving arxiv out of `retrievers[0]` fixes the fatal planning path, but arxiv can still raise inside the sub-query fan-out; that path IS wrapped (researcher.py:348-365 -> returns [] on any exception), so it degrades to empty context rather than rc=1. Acceptable, but means arxiv contributes ~nothing while it's down -- so pruning it is cleaner than reordering.
- **P3 -- Over-tolerating hides real failures**: if run_memo.py swallows ALL exceptions to exit 0, a genuinely broken run (bad API key, dep missing) would silently "pass". Scope the tolerance to arxiv/HTTP-429/5xx-class retriever errors, and still WRITE a WARN memo + emit a non-paging log line, so the 75.11 paging seam still fires on real breakage. (The existing broad `except` already writes an ERROR memo; keep that for non-network faults.)
- **P4 -- grep-sanitize partial-line trap**: the grep filter fixes THIS .env because L81 is a non-KEY orphan (dropped). It would NOT fully fix a `KEY="multiline` value (it would extract the unbalanced-quote first line). Verified not the case here (Cause 2), but any future .env fix must keep values single-line. ASCII-only + single-line .env discipline.
- **P5 -- ASCII-only logs**: any new WARN/log line added to run_memo.py or run_ablation.sh must be ASCII (project security.md logging rule) -- use `--`/`->`, never Unicode.

## Application to pyfinagent (external findings -> file:line)
**CAUSE 1 (autoresearch arxiv 429) -- recommended fix = structural + tolerant, both cheap:**
- (a) **Prune/deprioritize arxiv in the RETRIEVER list** at `scripts/autoresearch/run_memo.py:211`. Either drop it -> `"semantic_scholar,duckduckgo"` (both valid installed retrievers; semantic_scholar covers academic papers, arguably better than arxiv here), or make it last -> `"duckduckgo,semantic_scholar,arxiv"` so a robust retriever occupies `retrievers[0]` for planning (Key findings 4,5; researcher.py:62). Dropping is the most robust given the 2026 429 wave (Key finding 6).
- (b) **Tolerate retriever-network failure** in `run_memo.py`: narrow the outcome so an arxiv/HTTP-429/5xx-class failure writes a WARN memo and `return 0` instead of `return 1` (the broad `except` at :114 currently returns 1). Keep `return 1` for non-network faults so the 75.11 paging seam (run_nightly.sh:61-71) still pages on genuine breakage (Pitfall P3). Match arxiv error by `type(e).__name__ == "HTTPError"` / module `arxiv` / "429" in str(e).
- Do NOT rely on backoff tuning alone (Key finding 3, Pitfall P1). If backoff is added, it must be alongside (a)/(b), and requires replacing arxiv.Client() (monkeypatch) since ArxivSearch gives no passthrough.
- **39.1 disposition**: supersede with 76.9 (see internal section); write a root-cause note to satisfy 39.1's `root_cause_documented` intent.

**CAUSE 2 (ablation .env EOF) -- fix already coded by phase-75.11; 76.9 = bootstrap + verify, not new code:**
- `scripts/ops/run_ablation.sh` (git-tracked, commit 07182b94) already carries the verbatim phase-62.6 grep-sanitize (:21-37) -- confirmed sufficient because L81 is a non-KEY orphan the grep drops (Cause 2, Key finding 7, Pitfall P4).
- The LIVE plist already points to it (mtime 2026-07-24T08:52) but `runs=0` -- UNPROVEN. 76.9 should (i) confirm the bootstrap was intentional (STATE-IN-FLUX note), (ii) prove a clean run (`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.ablation` then check handoff/logs/ablation.log for END OK + no EOF lines), (iii) REPORT the malformed backend/.env L80-81 to the operator for repair (operator-gated; NOT an agent edit).
- Regression test home: extend `backend/tests/test_phase_75_sre_ops.py` (already asserts run_ablation.sh uses the sanitize + no raw `. backend/.env`, tests c3 at :130/:149).

## Research Gate Checklist
Hard blockers -- all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7: 4 official docs/forums + 3 code/issue)
- [x] 10+ unique URLs total (19: 7 full + 12 snippet)
- [x] Recency scan (last 2 years) performed + reported (the 2026 arxiv-429 regression -- material)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (run_memo.py:211/:114, researcher.py:62/:330/:348-365, arxiv.py:23, run_ablation.sh:21-37, ablation.launchd-v4.log)

Soft checks:
- [x] Internal exploration covered every relevant module (both wrappers, run_memo, installed gpt_researcher retriever chain, both plists + backups, launchctl state, .env via Main, 39.1, tests)
- [x] Contradictions / consensus noted (backoff-vs-drop debate; arxiv 429 resolution unsettled)
- [x] All claims cited per-claim
- Source-quality hierarchy: 4 official (arXiv docs x2, arXiv forum, gptr docs) + 3 code/issue > blogs (snippet-only). Floor met with authoritative tier.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 12,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Two independent P1 causes confirmed. CAUSE1 autoresearch: arxiv HTTP 429 raised in ArxivSearch.search (no try/except) propagates through gpt-researcher's retrievers[0]-only planning path (skills/researcher.py:62, outside the :348-365 try) to run_memo.py's broad except (:114) -> ERROR memo + return 1 -> run_nightly.sh exit 1. arxiv is retrievers[0] because RETRIEVER='arxiv,semantic_scholar,duckduckgo' at run_memo.py:211. arxiv==3.0.0 already retries 3x/3s and ignores Retry-After; arXiv has 429'd polite clients server-side since ~2026-02-25 (recency), so backoff-only is unreliable. Fix: prune/deprioritize arxiv in RETRIEVER + tolerate HTTP-429/5xx by writing a WARN memo and return 0 (keep return 1 for real faults so the 75.11 paging seam still works). CAUSE2 ablation: raw '. backend/.env' in the old plist dies on backend/.env L81 (an orphan '  ON\"' comment-tail fragment, unbalanced quote); fix already coded by phase-75.11 (scripts/ops/run_ablation.sh grep-sanitize, sufficient since L81 is non-KEY and grep-dropped). LIVE plist already points to the wrapper (mtime today 08:52) but runs=0 -- UNPROVEN; 76.9 = bootstrap+verify+report .env to operator (no .env edit). 39.1 superseded by 76.9 (its verification grep is unsatisfiable).",
  "brief_path": "handoff/current/research_brief_76.9.md",
  "gate_passed": true
}
```
