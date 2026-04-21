# Experiment Results — Cycle 4.15.6

Step: phase-4.15.6 Batches + Files + Citations + Search results

## What was built

Researcher produced `docs/audits/compliance-batches-files-
citations.md` (~2000 words, 25 patterns).

## Zero-usage claim CONFIRMED via live greps

All 5 live grep checks returned empty output:
- `messages.batches` — 0 matches
- `beta.files` / `files-api-2025-04-14` — 0 matches
- `citations.*enabled` / `cited_text` — 0 matches
- `type.*search_result` / `char_location` / `page_location` — 0 matches
- `base64.*application/pdf` / `media_type.*pdf` — 0 matches
- `anthropic-beta` — 0 matches (confirms structural blocker —
  ClaudeClient has no `betas=` kwarg plumbing, so even if we add
  one feature we'd need the kwarg first)

## 25 patterns, by feature

| Feature | Status summary |
|---------|----------------|
| Batches API (6 patterns) | ALL ❌ missing. `autonomous_loop._run_claude_analysis()` line 436 is sequential `asyncio.to_thread` — archetypal batch candidate. 50% discount uncaptured. |
| Files API (7 patterns) | ALL ❌ missing. SEC Form 4 XML is parsed to dict + f-string'd. Earnings transcripts as raw text. 10-Ks only reach Claude via Vertex Gemini grounding, never as document blocks. |
| Citations (6 patterns) | ALL ❌ missing. No document blocks exist — precondition for citations. `cited_text` is free (0 output tokens); current prompt-quote extraction is token-wasteful. ZDR-eligible. |
| Search results (4 patterns) | ALL ❌ missing. BQ rows reach Claude as `json.dumps()`-stringified text. `autonomous_loop.py:438` still uses `claude-sonnet-4-20250514` (deprecated). ZDR-eligible. |
| PDF native (1 pattern) | ❌ missing. fpdf2 present for generation only; no base64 ingestion path. |
| Cross-cutting (1 pattern) | `ClaudeClient.generate_content` has no `betas` kwarg — structural blocker for all 4 features. |

## ZDR impact matrix

| MF task | ZDR-safe? |
|---------|-----------|
| MF-31 Citations on SEC filings | **YES** |
| MF-32 search_result blocks for BQ RAG | **YES** |
| MF-33 Files API for filing reuse | **NO** (Files API not ZDR-eligible) |
| MF-34 Batches for overnight + PDF native | **NO** (Batches not ZDR-eligible; PDF via `document` block is fine but pairing with Batches isn't) |

If compliance ever requires ZDR, MF-33 + MF-34 become unavailable.
MF-31 + MF-32 remain available and are the safer first wave.

## Success criteria (from contract)

1. every_doc_pattern_status_evidenced — PASS (25 patterns)
2. qa_runs_live_code_checks_not_review — PARTIAL (researcher ran
   live checks; Q/A verifies next)
3. deviations_cite_doc_page — PASS

## Artifact

- `docs/audits/compliance-batches-files-citations.md`

## Novel findings not in prior audits

- **ClaudeClient has no `betas=` kwarg plumbing** — confirmed
  structural blocker for cache-TTL-1h, interleaved-thinking,
  files-api-2025-04-14. Matches phase-4.15.5 cache audit finding.
  (Already captured as MF-37 in phase-4.15 running list.)
- **`autonomous_loop.py:438` uses deprecated `claude-sonnet-4-
  20250514`** — already in MF-7 but this audit verifies it's the
  exact line for the nightly ticker sweep. Same line should be
  bumped AND wrapped in Batches API to claim 50% discount.
- **`autonomous_loop._run_claude_analysis` is the ARCHETYPAL batch
  candidate** — per-ticker sequential Claude calls, non-urgent,
  high-volume. MF-34 fix has a specific line target.

## Fixes landed this turn (MF-40 through MF-44)

In response to user pushback, fixed small MAS config items from
phase-4.15.3 findings:
- MF-40: added `permissionMode: plan` to both qa.md and
  researcher.md
- MF-41: rewrote qa.md NEVER constraint to allow Bash for
  verification commands only (`python -c`, `pytest`, `grep`, `jq`,
  `test -f`, `ls`, `git log --oneline`) — no `rm`, `mv`, `sed -i`,
  `git commit`, `git push`, no `>`/`>>` redirects
- MF-42: added `SubagentStop` hook to `.claude/settings.json`
  (loop-prevention gate)
- MF-44: added session-restart note to CLAUDE.md (agent file
  changes require restart; can't dispatch new agents this session)
- MF-43: added separation-of-duties note to CLAUDE.md (same
  session authoring + self-evaluating agent edits should leave a
  Peder-review note in harness_log)

All 5 verified via live grep + `jq` + Python JSON validity check.
