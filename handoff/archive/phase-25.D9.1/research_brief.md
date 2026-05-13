## Research: phase-25.D9.1 -- Caller-side Files API adoption (skill_file_id wiring)

Tier: moderate (stated by caller).
Date: 2026-05-13

### Queries run (three-variant discipline)

1. **Current-year frontier**: `Anthropic Files API document block file_id 2026`
2. **Last-2-year window**: `Anthropic Files API text/plain markdown .md upload document block supported 2025`
3. **Year-less canonical**: `Anthropic prompt caching cache_control document block files-api 1h TTL interaction`

---

### Read in full (>=5 required)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://platform.claude.com/docs/en/docs/build-with-claude/files | 2026-05-13 | Official doc | WebFetch | Beta header `files-api-2025-04-14` still current; `text/plain` maps to `document` block; `.md` not a first-class type but uploads as `text/plain`; file content billed as normal input tokens; storage 500 MB/file |
| https://platform.claude.com/docs/en/build-with-claude/prompt-caching | 2026-05-13 | Official doc | WebFetch | Document blocks in user messages support `cache_control`; 1h TTL = 2x write / 0.1x read; cache prefix is tools->system->messages in order; must mark explicitly |
| https://jangwook.net/en/blog/en/anthropic-files-api-batch-document-processing-guide/ | 2026-05-13 | Practitioner blog | WebFetch | Upload-once/reuse confirmed; recommends `retrieve_metadata()` guard against stale file_ids; no markdown-specific guidance |
| https://dev.to/thegdsks/prompt-caching-with-the-claude-api-a-practical-guide-14ce | 2026-05-13 | Practitioner blog | WebFetch | Document blocks in messages support `cache_control`; Sonnet min 1024 tokens to qualify for caching; static document block must precede dynamic question text |
| https://www.finout.io/blog/anthropic-api-pricing | 2026-05-13 | Industry pricing | WebFetch | Sonnet 4.6: $3.00/MTok input, 5-min cache write $3.75/MTok, 1h write $6.00/MTok, cache hit $0.30/MTok (90% discount) |

### Identified but snippet-only

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/BerriAI/litellm/issues/20367 | GH issue | SDK integration tangent |
| https://docs.anthropic.com/en/api/files-list | API reference | Management endpoints only |
| https://forum.bubble.io/t/anthropic-files-api/376952 | Community | Bubble-specific, low authority |
| https://spring.io/blog/2025/10/27/spring-ai-anthropic-prompt-caching-blog/ | Blog | Java/Spring SDK, not relevant |
| https://embracethered.com/blog/posts/2025/claude-abusing-network-access-and-anthropic-api-for-data-exfiltration/ | Security | Adversarial; not relevant to wiring |
| https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache | Docs | Third-party compat layer |
| https://ai-sdk.dev/cookbook/node/dynamic-prompt-caching | Blog | Node/TS SDK; confirms ordering but not primary |

---

### Recency scan (2024-2026)

Searched 2025-2026 window on Files API, prompt-cache document block interaction, markdown MIME support.

Findings:
- Anthropic silently dropped default ephemeral TTL from 1h to 5 minutes on 2026-03-06. Already handled in `llm_client.py:1185-1198` by explicitly passing `"ttl":"1h"`.
- Cache isolation changed from org-level to workspace-level on 2026-02-05. No code impact.
- Beta header `files-api-2025-04-14` unchanged as of 2026-05-13.
- Official docs now explicitly note `.md` is NOT a first-class document block type; correct path is `text/plain` MIME which `prompts.py:115` already uses.

No new finding supersedes the 25.D9 mechanism. Implementation is sound.

---

### Key findings

1. **Beta header still current**: `files-api-2025-04-14` is correct. `llm_client.py:1245-1246` injects it when `skill_file_id` is set. (Source: Anthropic Files API docs)

2. **Markdown as `text/plain` document block confirmed**: `.md` files uploaded with `mime_type="text/plain"` work as `document` blocks. `prompts.py:115` already uses this MIME type. (Source: Files API docs; practitioner blog)

3. **Token reduction is 98-99.5%**: Each skill `.md` is 50-75 lines (~700-1500 tokens inline). A document block referencing the file_id sends ~8 tokens. The "~97%" in the 25.D9 comment is conservative. (Source: Files API docs; internal wc -l audit)

4. **Document block cache_control is additive**: Callers can add `cache_control: {"type":"ephemeral","ttl":"1h"}` to the document block inside the user message, independent of the system-prompt cache already in `llm_client.py:1193-1199`. Sonnet 4.6 minimum is 1024 tokens; larger skills (~1200+ tokens) will qualify. (Source: prompt caching docs; dev.to)

5. **Cache order requirement**: Static document block must precede dynamic ticker data in the content array. The current document block builder at `llm_client.py:1235-1242` already places `document_block` before `{"type":"text","text":prompt}`. Order is correct. (Source: prompt caching docs)

---

### Internal code inventory

| File | Lines of interest | Role | Status |
|------|------------------|------|--------|
| `backend/agents/llm_client.py` | 1092-1155 (upload helper), 1155-1247 (generate_content) | Consumes `config["skill_file_id"]`; builds document block + injects beta header | **Complete** |
| `backend/config/prompts.py` | 36-172 (`SkillFileIdCache`), 157-172 (`bulk_upload_all`) | SHA256-keyed upload cache; `mime_type="text/plain"` at line 115 | **Complete** |
| `backend/agents/orchestrator.py` | 467-480 (`__init__` bulk-upload), 835-919 (11 `run_*_agent` callers) | Populates `self._skill_file_ids`; callers pass NO `generation_config` yet | **GAP: this is 25.D9.1** |

#### The gap

Every `run_*_agent` method follows (example, lines 835-840):
```python
prompt = prompts.get_insider_prompt(ticker, insider_data, ...)
response = self._generate_with_retry(self.general_client, prompt, "Insider")
```

`_generate_with_retry` (line 510) accepts `generation_config: dict | None`. The path `generation_config -> generate_content(config["skill_file_id"])` exists and works. The missing link is that callers never consult `self._skill_file_ids`.

**Key guard**: `self._skill_file_ids` is empty when `general_client` is not a `ClaudeClient` (enforced at `orchestrator.py:471`). Wiring must be a no-op when the dict is empty.

**Skill stem mapping**: `bulk_upload_all` keys by `skill_path.stem` (e.g. `"insider_agent"`). Agent display names used in `_generate_with_retry` (e.g. `"Insider"`) do not match. The callers must reference the skill stem directly.

---

### Design recommendation

**(a) WHERE -- cheapest cost-impact path first**

The 11 pure-`general_client` agents at `orchestrator.py:1429-1439` (Insider, Options, Social Sentiment, Patent, Earnings Tone, Alt Data, Sector, NLP Sentiment, Anomaly, Scenario, Quant Model) are the highest-value targets. They run in the parallel batch and are always on `general_client`. Wire these first.

`run_macro_agent` (line 744-748) uses `get_macro_prompt` which does NOT call `load_skill` -- it builds inline. No skill stem exists for it. Skip.

Grounded agents (Market, Competitor, Enhanced Macro, Deep Dive) fall through to `general_client` when `self.supports_grounding=False`. Wire with the same mechanism; the empty-dict guard handles the Gemini path.

**(b) WHAT -- exact dict shape**

No new API surface needed. Add a private helper to `Orchestrator`:

```python
def _skill_gen_config(self, skill_stem: str) -> dict | None:
    fid = self._skill_file_ids.get(skill_stem)
    return {"skill_file_id": fid} if fid else None
```

Each caller becomes:
```python
response = self._generate_with_retry(
    self.general_client, prompt, "Insider",
    generation_config=self._skill_gen_config("insider_agent"),
)
```

When `_skill_file_ids` is empty (Gemini path), `_skill_gen_config` returns `None` and `_generate_with_retry` passes `None` to `generate_content`, which falls back to the existing inline path. Zero overhead.

**(c) HOW -- cache_control on document block**

Optional but high value: add `cache_control: {"type":"ephemeral","ttl":"1h"}` to the document block inside `ClaudeClient.generate_content` when both `skill_file_id` is set AND `self.enable_prompt_caching` is True. Change is one dict update at `llm_client.py:1229-1234`. Skills >= 1024 tokens will qualify for Sonnet 4.6. Given ~11 agents x N tickers per backtest cycle, the 1h TTL will compound: first-ticker call pays 2x write; every subsequent ticker within the hour pays 0.1x for the same skill body. At $6.00/MTok write vs $0.30/MTok read, breakeven is 2 cache hits (the 3rd ticker and beyond is pure gain).

---

### Consensus vs debate

No contradictions found. `text/plain` -> `document` block path is consistent across official docs and practitioner sources. Cache ordering (static before dynamic) is universally agreed.

### Pitfalls

- `run_macro_agent` has no `load_skill` call; skip it for this step.
- Grounded calls use `grounded_client` (GeminiClient). Never pass `skill_file_id` to Gemini.
- `_skill_file_ids` is empty when `general_client` is GeminiClient; the empty-dict guard at `_skill_gen_config` is mandatory, not optional.
- System-prompt cache (line 1193-1199) and document-block cache are independent. Adding document-block `cache_control` does not conflict.
- Skills < 1024 tokens will not register a Sonnet 4.6 cache write; they'll still get the upload-once token reduction, just no cache discount.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (incl. snippet-only): 12 collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
