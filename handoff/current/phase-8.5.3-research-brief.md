---
step: phase-8.5.3
topic: LLM proposer with narrow-surface diff -- STRIP vs reject-whole-diff safety
tier: simple
date: 2026-04-19
---

## Research: phase-8.5.3 LLM Proposer -- Narrow-Surface Diff Discipline

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://arxiv.org/html/2510.03217v1 | 2026-04-19 | paper | WebFetch | Dual-LLM validation: aggressive filtering improves precision but reduces recall; strip-bad vs reject-whole tradeoff documented; no adversarial/smuggling analysis |
| https://arxiv.org/html/2603.01257v1 | 2026-04-19 | paper | WebFetch | LLM patch architectures; patch validity = compile + PoV elimination + test suite; Claude Code self-reported success ≠ actual test outcomes (mismatch risk); no whitelist/strip analysis |
| https://www.knostic.ai/blog/ai-coding-agent-security | 2026-04-19 | blog (industry) | WebFetch | Deny-by-default allowlists; full human review of entire diff favored over surgical filtering; no strip-partial discussion |
| https://best.openssf.org/Security-Focused-Guide-for-AI-Code-Assistant-Instructions.html | 2026-04-19 | official doc | WebFetch | Human-in-the-loop oversight; iterative correction (RCI) pattern; constraint via instructions not allowlist gating; no reject-whole guidance |
| https://www.anthropic.com/research/measuring-agent-autonomy | 2026-04-19 | official doc | WebFetch | 80% of agent deployments have scoped permissions or human approval; adaptive oversight; no technical allowlist / strip semantics |
| https://arxiv.org/html/2602.17753v1 | 2026-04-19 | paper | WebFetch | 2025 AI Agent Index; allowlisting frameworks noted as a governance path; partial-filter vs full-reject not addressed; prompt injection in 2/5 browser agents documented |

### Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://github.com/bytedance/PatchEval | code/bench | GitHub repo, CLI scan only |
| https://deepmind.google/blog/introducing-codemender-an-ai-agent-for-code-security/ | blog | Less relevant than fetched sources |
| https://www.prnewswire.com/news-releases/hiddenlayer-releases-the-2026-ai-threat-landscape-report-302716687.html | press | Lower authority; not a technical paper |
| https://checkmarx.com/learn/ai-security/top-12-ai-developer-tools-in-2026-for-security-coding-and-quality/ | blog | General survey; no patch-semantics depth |
| https://github.com/aiming-lab/AutoResearchClaw | code | Not directly relevant to diff semantics |

### Recency scan (2024-2026)

Searched for 2026-window sources on "autonomous research agent narrow-surface diff whitelist 2026" and 2025-window sources on "AI coding agent safety diff validation 2025". Result: found useful 2025 empirical data (AI Agent Index, CodeMender, dual-LLM abstain-validate paper). No 2026 paper directly addresses STRIP-vs-reject-whole semantics for LLM-generated diffs. The adversarial partial-compromise question is not addressed in any 2024-2026 publication found; it is a live gap.

### Query variants run

1. Current-year frontier: `autonomous research agent narrow-surface diff whitelist 2026`
2. Last-2-year window: `AI coding agent safety diff validation 2025`
3. Year-less canonical: `LLM patch validation whitelist pattern`

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/autoresearch/proposer.py` | 110 | Proposer class + WHITELIST + Diff TypedDict + _default_llm_call stub | Active |
| `scripts/harness/autoresearch_proposer_test.py` | 97 | 3-case verification script | Active |

#### Key file:line anchors

- `proposer.py:24-27` -- WHITELIST = {optimizer_best.json, candidate_space.yaml}. Two-file surface only.
- `proposer.py:38-58` -- `_default_llm_call`: deterministic stub, no network, no API key. Offline-safe confirmed.
- `proposer.py:61-70` -- `validate_diff`: returns `(ok, bad_paths)`. Validation logic is a set membership check, not a format check.
- `proposer.py:100-106` -- STRIP semantics: `good = {p: c for p, c in raw["files"].items() if p in self.whitelist}`. Non-whitelisted paths silently dropped; whitelisted paths pass through unchanged. Warning is logged (`logger.warning`), rationale string appended with `stripped_paths=`.
- `proposer.py:98-99` -- `read_results_tsv` / `read_git_log` flags set via `setdefault` from the presence of non-empty inputs. Confirmed both flags tracked.
- `proposer.py:93-96` -- Exception guard: if `llm_call_fn` raises, returns empty diff with `llm_call_failed` rationale. Fail-open semantics on exception.
- `autoresearch_proposer_test.py:34-57` -- `case_diff_touches_only_whitelisted_files`: malicious LLM proposes `kill_switch.py` (not whitelisted) + `optimizer_best.json` (whitelisted). After STRIP, `kill_switch.py` absent, `optimizer_best.json` present. Test verifies only the strip outcome, not the content of the surviving whitelisted file.

---

### Key findings

1. **Offline-safe default confirmed** -- `_default_llm_call` (`proposer.py:38-58`) requires no real API key and returns a deterministic stub. Test scaffolding is offline-safe. (Internal audit)

2. **STRIP semantics, not reject-whole** -- `proposer.py:100-106` implements per-path filtering: keep whitelisted, drop non-whitelisted, append warning to rationale. The surviving whitelisted paths are passed through to the caller without re-validation of their content. (Internal audit)

3. **`read_results_tsv` + `read_git_log` flags tracked** -- `proposer.py:98-99` sets both via `setdefault` against the truthiness of inputs. The test at `autoresearch_proposer_test.py:60-72` checks both flags and checks the rationale string references input sizes. (Internal audit)

4. **Industry norm is deny-by-default allowlist, not strip** -- Knostic (2025) and the AI Agent Index (2025) recommend explicit allowlisting with full-diff human approval, not surgical stripping. No authoritative source endorses STRIP semantics as the primary safety mechanism. (Source: Knostic, URL above; AI Agent Index, URL above)

5. **Partial-compromise vector exists and is unaddressed in literature** -- If a malicious LLM proposes {whitelisted_path: malicious_content, non-whitelisted_path: benign_or_malicious_content}, STRIP keeps the whitelisted path's content without inspecting it. The content of `optimizer_best.json` is JSON; a malicious but syntactically valid JSON value (e.g., extreme learning_rate, injected eval string) passes through undetected. The test at `autoresearch_proposer_test.py:34-57` only checks path membership, not content safety. No 2024-2026 paper found addresses this specific vector. (Gap identified from internal audit + literature review)

6. **Dual-LLM abstain-validate (2025) shows filtering tradeoffs** -- Stricter percentile thresholds for patch validation improve precision but drop recall substantially (53% precision on 12 bugs at P90 vs 174 bugs at baseline). STRIP is an equivalent tradeoff applied to paths: high-recall (keeps whitelisted work) at cost of content integrity. (Source: arXiv:2510.03217v1)

7. **LLM self-reported success != actual outcomes** -- arXiv:2603.01257 documents Claude Code claiming success on every challenge while 3 patches failed actual tests. This is directly relevant: a real LLM caller could claim `read_results_tsv: True` + `read_git_log: True` without actually using the inputs. The `setdefault` approach at `proposer.py:98-99` would be overridden only if the LLM explicitly sets these flags to False. (Source: arXiv:2603.01257v1)

---

### Consensus vs debate

- **Consensus**: Allowlists are better than denylists for agent output scoping (Knostic, AI Agent Index, Anthropic autonomy research).
- **Consensus**: Narrow surface (two files in WHITELIST) is the right direction; minimizing writable surface minimizes blast radius.
- **Debate**: STRIP-partial vs reject-whole is contested. No published standard mandates either. Industry security tooling (Knostic) implicitly favors reject-whole via human-in-the-loop. Academic patch repair papers (arXiv:2510.03217v1) use per-patch filtering, which is analogous to STRIP.

### Pitfalls (from literature + internal audit)

- **Content injection through whitelisted paths**: STRIP validates path, not content. A malicious LLM can put harmful parameter values in `optimizer_best.json`. This is the primary adversarial risk.
- **Flag spoofing**: A real LLM can set `read_results_tsv: False` or simply omit the key; `setdefault` only fills missing keys, not overrides.
- **Exception fail-open**: `proposer.py:93-96` returns empty diff on exception, not a hard error. Downstream code must check `diff["files"]` before acting.
- **No content schema validation**: `validate_diff` checks path membership only. There is no JSON schema validation of the content written to `optimizer_best.json`.

---

### Application to pyfinagent

The STRIP semantics at `proposer.py:100-106` are **defensible for the current scaffold phase** (phase-8.5.3) because:
1. The WHITELIST is two files with a narrow, well-understood schema (JSON params + YAML search space).
2. This is a scaffold/proposer layer, not a committing layer. The contract notes "a committing layer (phase-8.5.6 or later)" will translate the diff into an actual patch or `git apply`.
3. The adversarial scenario (malicious LLM proposing bad content for a whitelisted file) is a real risk, but the committing layer is the correct place to add content-schema validation (e.g., JSON Schema check on `optimizer_best.json`, bounds check on numeric parameters).

However, the contract for phase-8.5.3 should note the open gap and either:
- Add content-schema validation to `validate_diff` now (preferred), or
- Document it as a phase-8.5.6 acceptance criterion.

Reject-whole-diff would be safer against the partial-compromise vector but would reduce proposer utility: any non-whitelisted path from a real LLM (e.g., a path hallucination) would nuke the entire proposal including valid whitelisted changes. For this two-file WHITELIST with no human in the loop, STRIP is the pragmatic choice -- **provided content validation is added at the committing layer**.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources fetched)
- [x] 10+ unique URLs total (11 collected: 6 read in full + 5 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (proposer.py + test file)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

### Verdict: is STRIP defensible?

**Yes, conditionally.** STRIP is appropriate for the proposer scaffold because the WHITELIST is narrow (2 files) and a committing layer follows. The critical gap is content validation: path-level STRIP does not prevent a malicious or hallucinating LLM from writing dangerous parameter values to a whitelisted file. The committing layer (phase-8.5.6+) must add JSON Schema bounds-checking on `optimizer_best.json` content before STRIP can be called fully safe. Without that, reject-whole-diff would be the safer choice.

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 5,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/phase-8.5.3-research-brief.md",
  "gate_passed": true
}
```
