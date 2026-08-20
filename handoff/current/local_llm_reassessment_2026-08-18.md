# Local-LLM reassessment — 2026-08-18 (REPORT ONLY)

**Status: no changes made.** No install, no masterplan edit, no process killed, no
config touched. Operator elected "write the findings up only". Phase-74.0–74.3 remain
`pending`.

Supersedes the measurements in `handoff/archive/misc/local_llm_assessment_2026-07-18.md`
on three points; leaves that document's role-partition verdict intact.

---

## 0. Methodological correction (the reason this document exists)

The first pass of this reassessment measured the app stack with **RSS**
(`ps -o rss`, `psutil.memory_info().rss`) and concluded the app stack was **0.386 GB**,
i.e. that the July assessment's "13.7 GB already resident" was wrong by ~35x. **That
conclusion was wrong.**

On macOS, RSS counts only pages currently resident in physical RAM. It **excludes
compressed and swapped pages**. This box currently holds **6.80 GB compressed** and
**2.88 GB swapped**, so RSS omits the majority of most processes' real memory claim.

The correct metric is **`phys_footprint`** — what Activity Monitor's "Minne" column
shows, and what macOS uses for memory-pressure accounting. Read it with:

```bash
vmmap -summary <pid> | grep "Physical footprint:"
footprint -p <pid>                      # also reports phys_footprint_peak
```

Recorded as auto-memory `reference_rss_understates_macos_memory`.

**Net effect on the July document:** its 13.7 GB figure was *correct as a box total*
(Activity Monitor "Minne i bruk" reads 13.78 GB today, unchanged). Its only real
imprecision was attributing the whole box total to "the app stack". The conclusion it
drew from that number — a constrained inference budget — was sound.

---

## 1. Measured state (2026-08-18, operator session active)

Box: base M4 Mac mini, 16 GB unified, 120 GB/s memory bandwidth, 118 Gi disk free.

### Box totals (Activity Monitor, operator screenshot + `vm_stat`)

| Field | Value |
|---|---|
| Fysisk minne | 16.00 GB |
| Minne i bruk | 13.78 GB |
| Appminne | 4.44 GB |
| Fast minne (wired) | 2.01 GB |
| Komprimert | 6.80 GB |
| Bufrede filer | 2.16 GB |
| Flyttet (swap) | 2.88 GB / 4.00 GB |

`psutil.virtual_memory().available` = **2.96 GB** at time of measurement.

### App stack (`phys_footprint`)

| Process | pid | Footprint | Peak |
|---|---|---|---|
| next-server (v15.5.12) | 13308 | **3.9 GB** | 4.44 GB |
| uvicorn backend.main :8000 | 25117 | 735.2 MB | 758 MB |
| backend.slack_bot.app | 66031 | 526.0 MB | 575 MB |
| **Total** | | **~5.16 GB** | |

Caveat: taken at ~6h41m uptime without an in-flight backtest. Peak under
`cache.preload_macro()` + a live backtest is **not** measured here and should be
before anything is sized against the remainder.

---

## 2. Finding A — the production frontend is a development server (~3.4 GB)

`~/Library/LaunchAgents/com.pyfinagent.frontend.plist` runs:

```xml
<key>ProgramArguments</key>
<array>
    <string>.../frontend/node_modules/.bin/next</string>
    <string>dev</string>
    <string>--port</string>
    <string>3000</string>
</array>
```

Confirmed live: pid 13281 `next dev --port 3000` → child pid 13308 `next-server`.

`next dev` holds the compiler, HMR machinery, source maps and per-route compilation
caches resident for the process lifetime. That is why a single Node process carries
**3.9 GB** (peak 4.44 GB) — roughly **28% of the entire machine**. An equivalent
`next build` + `next start` server typically sits at 300–500 MB.

**Estimated reclaim: ~3.4 GB.** This is the largest single lever on the box, and it is
larger than the entire footprint of the model this phase wants to run.

Trade-off: loses hot-reload. Sane shape is `next start` as the standing launchd
service, with `next dev` run by hand only during frontend work.

Blocking constraints (both from CLAUDE.md, neither resolved here):
- Frontend edits break an open operator UI session (ChunkLoadError + Auth.js cookie error).
- Backend/service restarts are batched to session end; `launchctl kickstart -k` does
  **not** re-read a plist's `EnvironmentVariables` — only `bootout` + `bootstrap` does,
  and away-ops rail 9 reserves that verb for the operator.

## 3. Finding B — leaked Playwright processes (~1.3 GB)

Live right now: **6** `playwright-mcp` server processes and **14** Chrome for Testing
processes, totalling **~1.32 GB** `phys_footprint`.

| Started | pids | Note |
|---|---|---|
| 2026-08-13 09:52 | 48979, 49842 | 5 days old |
| 2026-08-13 20:55 | 8072, 8107 | 5 days old |
| 2026-08-17 10:39 | 77938, 77951 | |
| 2026-08-17 16:00 | 45047 + 8 Chrome children | Chrome renderer alone 331.5 MB |

Only one session should hold a Playwright MCP at a time. Matches the existing
auto-memory `reference_leaked_playwright_browser_spins_cpu` (they outlive the step, not
just the session).

**Not killed** — some may belong to the concurrent Claude session
(`project_concurrent_claude_sessions`). Killing another session's MCP server breaks it
mid-flight. Any cleanup should target only pids older than the current session.

## 4. Combined headroom arithmetic

| | Reclaim | Free after |
|---|---|---|
| Today | — | 2.22 GB |
| + frontend to `next start` | ~3.4 GB | ~5.6 GB |
| + Playwright leak cleanup | ~1.3 GB | ~6.9 GB |

Both are hygiene fixes with no loss of app function. Neither depends on phase-74.

---

## 5. Model landscape — the July pin is a generation stale

**Qwen3.5 small series** (0.8B / 2B / 4B / 9B) released **2026-03-01/02** — i.e. it
already existed when the July assessment was written, and was missed there. Apache 2.0,
262K native context, native vision, hybrid thinking/non-thinking.

Architecture (from the HF card):
`8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))` — only one block in
four is full attention; the rest are linear-attention with a constant-size recurrent
state.

### Artificial Analysis Intelligence Index (reasoning variants)

| Model | Score | Ollama Q4 size |
|---|---|---|
| Qwen3.5-9B | **32** — highest under 10B | 6.6 GB |
| Qwen3.5-4B | **27** — highest under 5B | 3.4 GB |
| Qwen3.5-2B | 16 | 2.7 GB |
| Qwen3-4B-2507 *(the July pin)* | **18** | ~3 GB |
| Qwen3-VL-8B | 17 | — |

The 4B gains **+9 points** over the July pin at the same footprint — a free upgrade.
The 9B gains **+14**, and reportedly beats the previous-generation Qwen3-30B on
reasoning at a third the size.

Nothing newer exists in this size class: **Qwen3.6** starts at 27B (17 GB) and
**Qwen3.8** is 27B plus a single `qwen3.8-9b-coder` (coder-specialised, wrong fit for
chat/extraction roles).

### Known weaknesses (both sizes)

- **AA-Omniscience hallucination rate 80–82% — MIS-FRAMED HERE; CORRECTED 2026-08-18
  (brief F16, the single most decision-relevant finding of the research gate).** The
  number is real but measures **closed-book adversarial factual recall**, with no
  penalty for refusing. On **grounded** tasks — answering from a supplied passage —
  independent Vectara HHEM puts a Qwen 4B-class model at **5.7% hallucination / 94.3%
  factual consistency**, within **1.6 pts of Llama-3.3-70B**. A ~14x spread on the same
  size class, explained entirely by whether the answer is in the prompt.
  **Every role phase-74 proposes (news_screen extraction, Slack replies over supplied
  context, degraded-mode fallback) is GROUNDED; every role on the NEVER-LOCAL list is
  UNGROUNDED.** So this figure does not argue against the pilot — it argues for exactly
  the role partition already drawn. Two riders: it does **not** rescue numeric work
  (FAITH: Qwen-3-8B 30.6%), and model choice inside a size class dominates size
  (Phi-4 3.7% vs Phi-4-mini 23.5%).
- **Verbose thinking mode** — 230–390M output tokens across the AA benchmark suite,
  far above larger siblings. This is the latency risk, not quality.

### Latency ceiling (bandwidth-bound, 120 GB/s)

Dense-model token generation reads all weights per token, so `tok/s ≈ bandwidth / size`:

| Model | Q4 size | Ceiling | Realistic |
|---|---|---|---|
| qwen3.5:4b | 3.4 GB | ~35 tok/s | ~21–28 |
| qwen3.5:9b (Q4_K_M) | 6.6 GB | ~18 tok/s | ~11–15 |
| qwen3.5:9b (IQ4_XS) | ~5.0 GB | ~24 tok/s | ~15–19 |

A 2,000-token think on the 9B ≈ **150 s**, which breaches the 120 s cc_rail timeout
precedent. Thinking mode must stay **off** on any path with a clock on it.

### Revised recommendation

Conditional on Findings A and B being actioned — without them, neither model fits
alongside an active operator session:

| Role | Model | Mode |
|---|---|---|
| Slack bot, news_screen, macro_regime | `qwen3.5:4b` | thinking **off** |
| Terminal last-resort rail | `qwen3.5:9b` (IQ4_XS) | thinking **on** |

Never-local, unchanged and reinforced by the 80–82% hallucination rate: enrichment,
debate, synthesis, risk-judge, meta-scorer-primary, autoresearch, MAS harness.

### Additional RAM levers, model-side

- `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` — **CORRECTED 2026-08-18
  (brief F17, settled from llama.cpp source): this does NOT halve the KV cache on
  Qwen3.5.** llama.cpp stores recurrent state in `llama_memory_recurrent`, separate
  from the KV cache, with its own `type_r`/`type_s` ggml types that the KV-cache-type
  flag does not reach. Only **8 of 32** layers are full Gated Attention, so the flag
  halves the cache for a **quarter of the layers only** and cannot compress the
  linear-attention state at all — roughly a quarter of the benefit the same flag buys
  on a dense transformer. The silent-f16-fallback worry recorded above is **moot for
  the recurrent path**, which was never covered by the flag to begin with.
- Cap `num_ctx` to what each role needs (8–16K), not the 262K native ceiling.
- `IQ4_XS` over `Q4_K_M` on the 9B: ~5.0 GB vs 6.6 GB, better quality-per-GB than Q4.
- `keep_alive=0` (already in 74.0) — model resident only during the call.

**Two things to verify, not assume:**
1. Quantized KV cache **silently falls back to f16** on unsupported architectures.
   Qwen3.5's Gated DeltaNet hybrid is exactly the kind of non-standard attention where
   that fallback is plausible. Check actual memory use; do not trust the env var.
2. The hybrid's constant-size linear-attention state *should* make long-context KV cost
   far below a standard transformer, which would suit this box well. **That is an
   inference from the published architecture, not a measurement.**

---

## 6. What is stale in phase-74 as queued

### 6a. Model pin
74.0 says `ollama pull qwen3-4b-instruct-2507`. Superseded — see §5.

### 6b. Every file:line anchor has drifted

| Step text claims | Measured 2026-08-18 |
|---|---|
| `llm_client.py:1200-1202` = the base_url schema skip | **`:1249`** — `if (mime == "application/json" or schema) and not self._base_url:`. `:1200-1211` is the *prompt-hint* branch, a different thing |
| `cost_tracker.py:20-83`, default pricing at `:83` | `_DEFAULT_PRICING = (0.10, 0.40)` at **`:95`**, applied at **`:177`** and **`:266-267`** |
| `model_tiers.py:57` = mas_communication | **`:98`** (`claude-sonnet-4-6`); effort `low` at **`:326`** |

The *substance* of all three holds: the `base_url` path really does get only a soft
prompt hint with no enforced JSON, and an unknown model id really does book phantom
$0.10/$0.40. Only the anchors moved.

### 6c. The memory guard formula is the wrong shape

74.0 specifies "refuse if free RAM < 2 GB". That does not gate on model size. Correct
shape is `available >= model_size + ~1.5 GB headroom`:

- `qwen3.5:4b` → needs ~5 GB available
- `qwen3.5:9b` IQ4_XS → needs ~6.5 GB available

At today's 2.96 GB available, **both would be correctly refused**. Expect local
inference to run mostly while the operator is away — that should be a stated
expectation of the pilot, not a surprise discovered in the live_check.

### 6d. Upstream tool-calling bug affects the 74.2 pilot

[ollama#14745](https://github.com/ollama/ollama/issues/14745) — `qwen3.5:9b` prints
tool calls as text instead of executing them, "fairly often". Opened 2026-03-09,
reported on Ollama 0.17.7.

**CORRECTED 2026-08-18 by the phase-74.0 research gate (brief L1) — the pin advice
below was BACKWARDS and must not be followed.** This paragraph originally read
"workaround is to pin 0.17.5". PR #15022 was **merged 2026-03-27 and shipped in
v0.19.0** ("Fixed tool call parsing issue with Qwen3.5 where tool calls would be output
in thinking"). **Pinning 0.17.5 pins to BEFORE the fix.** Correct advice: **pin Ollama
>= 0.19.0.**

Worse, the bug class is not closed: [ollama#16686](https://github.com/ollama/ollama/issues/16686)
was opened 2026-06-12 on Ollama **0.30.7**, is **still open**, and shows the same
observable symptom three months after #15022 shipped — the model emits a valid
`<function=...>` block but omits the opening `<tool_call>` tag, so the parser skips it
and returns the payload as plain text. Treat Qwen-family tool-call parsing on Ollama as
a **recurring failure class**, not a fixed one-off.

74.2 (Slack bot) depends on MCP tool calling, so: pin a known-good Ollama version and
keep the graceful tool-miss reply the step already calls for.

### 6e. Structured output is better-supported than the step assumes

Ollama accepts a JSON schema in the `format` parameter and converts it to a GBNF
grammar internally (since v0.5). The 74.1 fix is therefore straightforward — but note
the OpenAI-compat `/v1` surface and the native `/api/chat` surface expose this
differently, so the plumbing must target whichever one `make_client()` uses.

---

## 7. The "free" motive, tested

Free is true but it is the weakest leg. The reliability leg is much stronger than in
July. From `pyfinagent_data.llm_call_log`, trailing 14 days:

| provider | model | calls | ok% |
|---|---|---|---|
| anthropic | claude-haiku-4-5 | 41 | **48.8** |
| anthropic | claude-opus-5 | 69 | **63.8** |
| anthropic | claude-opus-4-8 | 237 | **66.2** |
| anthropic | claude-sonnet-4-6 | 1088 | 88.6 |
| gemini | gemini-2.5-flash | 248 | **100.0** |

The Anthropic direct legs are failing between 11% and 51% of calls. The one clean leg
retires **2026-10-16**. That — not the money — is the argument for a rail that cannot
be credit-killed. July's "~$0.2/day, nothing meaningful to save" framing stands.

---

## 8. Open items (none actioned)

1. Measure backend peak `phys_footprint` under a real backtest with `preload_macro()`.
2. Decide on `next dev` → `next start` (needs build + plist rewrite + session-end restart).
3. Decide on Playwright leak cleanup (must not kill the concurrent session's server).
4. If phase-74 proceeds: re-pin model, fix guard formula, re-derive anchors, pin Ollama version.
5. 74.0 still requires explicit operator approval for the system-level install; no
   artifact records that approval.

## Sources

- [Qwen3.5 small models — Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- [Qwen/Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Ollama qwen3.5 library](https://ollama.com/library/qwen3.5)
- [Alibaba releases Qwen 3.5 Small — MarkTechPost, 2026-03-02](https://www.marktechpost.com/2026/03/02/alibaba-just-released-qwen-3-5-small-models-a-family-of-0-8b-to-9b-parameters-built-for-on-device-applications/)
- [ollama#14745 — qwen3.5:9b prints tool call instead of executing](https://github.com/ollama/ollama/issues/14745)
- [Mac mini technical specifications — Apple](https://www.apple.com/mac-mini/specs/)
- [OLLAMA_KV_CACHE_TYPE — ModelPiper](https://modelpiper.com/blog/ollama-kv-cache-quantization)
- [Bringing K/V context quantisation to Ollama — smcleod.net](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- Prior assessment: `handoff/archive/misc/local_llm_assessment_2026-07-18.md`
