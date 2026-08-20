# Phase-74.0 — peer-supplied candidate URL pool (UNVERIFIED, NOT EVIDENCE)

**Source:** peer Claude session `lit-fetcher`, delivered 2026-08-18 ~19:05Z.
**Status: NONE of these were fetched by the peer.** Everything below is
search-snippet-derived. Titles and arXiv IDs are probably right; **every number
in a note is unverified.** No URL here counts toward the research-gate source
floor unless the researcher independently fetched and read it, and recorded it
in `handoff/current/research_brief_74.0.md`.

This file is a **provenance-audit reference**, not an input to the brief and not
a finding. Do not cite it. Do not merge it into the brief.

---

## CRITICAL: the coverage cross-check this pool was requested for is COMPROMISED

I requested this pool to test whether the audit-class loop-until-dry on run
`wf_98c646a4-8b1` actually went dry, or merely stopped. That test is **void**:

- The peer disclosed it had already sent "a broader-but-shallower version of
  this pool" to agent `ab73bb623cb3dd432`.
- `ListAgents` at 19:05:47Z resolves `ab73bb623cb3dd432` as
  `researcher · running · started 3m ago` — i.e. **the subagent spawned by my
  own gate run**, not a separate effort.
- Therefore the "independent" pool and the researcher's own URL set share a
  source. Comparing them measures nothing about coverage — it is a control
  built from the thing it is meant to test.
  See `feedback_a_control_built_from_your_own_pattern_tests_nothing` and
  `feedback_check_collinearity_before_crediting_an_attribution`.

The peer flagged this risk itself and asked me to confirm before scoring
coverage. Confirmed: compromised. **Do not report loop-until-dry as
independently corroborated.**

Timeline note: the peer said it sent the earlier pool "about an hour ago", but
the receiving agent started ~19:02Z, ~3 minutes before the message. The peer's
elapsed-time claim does not reconcile with the agent's start time; the
*recipient identity* is what matters here and that is confirmed from
`ListAgents`, not from the peer's narration.

## Second integrity risk this creates (open)

The gate enforces that every URL claimed in `sources_read_in_full` **appears in
the brief**. It does NOT and cannot verify the URL was actually *fetched*. A
researcher handed ~100 plausible URLs with descriptions can therefore produce a
brief that passes the cross-check while being substantially snippet-derived.

The doctrine already forbids this — `.claude/rules/research-gate.md`: "Search
snippets do NOT count"; "Never count a paper as read in full if only the
abstract page was fetched." But doctrine plus a large tempting list is exactly
the drift this project has repeatedly measured. Reinforcement sent to the
running researcher at 19:06Z.

## Peer's own stated limits (verbatim in substance)

- Its session hit the WebSearch cap (200/200) on the last query. **Axis 6
  (tool-calling) got no dedicated search** — it rests on an earlier BFCL/tau-bench
  batch. The peer's pool is **not dry**; absence here is not evidence of absence.
- Axis 5 has a **genuine gap**: no source found addressing the OpenAI-compat
  `/v1` vs native `/api/chat` structured-output difference. The peer flagged
  this as a gap, not a dry round.

## Peer's side effects, and a strength correction on the two escalated items

Peer's own accounting (2026-08-18, follow-up message): it **wrote nothing to
disk this session** — no brief, no masterplan, no notes — and ran **no further
searches** after hitting its cap. Its only side effects were three SendMessage
calls: two to me, one to `ab73bb623cb3dd432`. So the only contamination channel
into this cycle was that single message to my researcher; no artifact on disk is
peer-authored except this file, which I wrote.

The peer also **retracted its "about an hour ago" timeline claim as fabricated**
— it had no timestamp for its own send and no start time for the recipient, and
invented the interval. Noted here because that number appeared inside the very
disclosure that surfaced the contamination: had I reasoned from it instead of
from `ListAgents`, it would have pointed *away* from the contamination, since an
hour-old send cannot reach a 3-minute-old agent. The conclusion survived only
because identity was resolved from the listing. Instance of
`feedback_never_narrate_a_clock_you_did_not_read`.

**Strength correction on the two items I escalated to the researcher** — both are
weaker than my first framing implied, and neither is a finding:

- `llama_memory_recurrent` / DeltaNet-has-no-standard-KV-cache came from a
  **DeepWiki mirror, not llama.cpp source**. The repo itself is where this gets
  settled. Treat as a LEAD.
- The **">=32GB for the Ollama MLX backend"** claim came from a **third-party
  blog, NOT from `ollama.com/blog/mlx`**. That official page is listed in axis 3
  but the peer never fetched it. Do not attribute the claim to Ollama. Treat as
  a LEAD.

---

## The single most valuable thing in this pool: AA-Omniscience provenance

**AA-Omniscience is Artificial Analysis's own benchmark.** Their leaderboard,
their article and their X threads are **one publisher, not three sources**.

This directly weakens a claim I made to the operator with more confidence than
it deserved: the "80-82% hallucination rate for Qwen3.5 4B/9B" came from
Artificial Analysis's article about Artificial Analysis's own index. That figure
is load-bearing in the "never-local for judgment roles" argument, so corroboration
must come from a **different benchmark by different authors**.

Same-publisher (pins provenance; does NOT corroborate):
- https://artificialanalysis.ai/articles/qwen3-5-small-models — origin of the figure
- https://artificialanalysis.ai/evaluations/omniscience — live leaderboard
- https://arxiv.org/abs/2511.13029 — AA-Omniscience methodology paper
- https://benchgen.com/benchmarks/artificial-analysis/aa-omniscience — third-party mirror
- https://x.com/ArtificialAnlys/status/1990455484844003821 — launch thread
- https://x.com/ArtificialAnlys/status/2008570655047118914 — index construction

Independent instruments (different benchmarks, different authors):
- https://arxiv.org/abs/2602.14778 — geometric analysis of SMALL-model hallucination
- https://arxiv.org/abs/2504.17550 — HalluLens
- https://cdn.openai.com/papers/simpleqa.pdf — SimpleQA (canonical prior art, 2024)
- https://arxiv.org/abs/2509.07968 — SimpleQA Verified
- https://arxiv.org/abs/2602.19643 — KGHaluBench
- https://arxiv.org/abs/2605.17007 — HalluScore
- https://arxiv.org/abs/2605.02504 — MultiWikiQHalluA
- https://arxiv.org/abs/2509.21104 — PerHalluEval (method comparison only)
- https://arxiv.org/abs/2505.18658 — robustness survey
- https://llm-stats.com/benchmarks/hallusion-bench — HallusionBench leaderboard

## Axis 2 — KV-cache quant on hybrid / Gated DeltaNet

Peer rated this its best-targeted set; the f16-fallback question may be directly
addressed. Note the DeepWiki item: `llama_memory_recurrent` is reportedly the
DeltaNet/recurrent path, i.e. **not a standard KV cache at all** — which if true
reshapes the whole memory argument I gave the operator.

- https://github.com/ollama/ollama/issues/5091 — KV cache quantization thread
- https://github.com/ollama/ollama/issues/10794 — differentiated KV quant (type_k/type_v)
- https://github.com/ollama/ollama/pull/15090 — TurboQuant rotation KV compression
- https://github.com/ggml-org/llama.cpp/issues/21385 — per-head adaptive KV quant on hybrids; names Qwen3.5
- https://github.com/ggml-org/llama.cpp/discussions/20969 — TurboQuant extreme KV quant
- https://deepwiki.com/ggml-org/llama.cpp/3.6-memory-management-and-kv-cache — llama_memory_recurrent path
- https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/ — by the implementer
- https://mitjamartini.com/posts/ollama-kv-cache-quantization/
- https://modelpiper.com/blog/ollama-kv-cache-quantization — where the silent-f16-fallback claim surfaced
- https://modelpiper.com/blog/ollama-environment-variables
- https://arxiv.org/abs/2412.06464 — Gated Delta Networks (founding architecture paper)
- https://docs.lmcache.ai/mp/hybrid_models.html
- https://wal.sh/research/qwen3.6-local-first-inference/
- https://techdocs.broadcom.com/us/en/vmware-tanzu/platform/ai-services/10-0/ai/explanation-understanding-ollama-configuration.html

## Axis 3 — measured tok/s, 3-7 GB models on base M4

Peer flags one claim worth checking hard: a snippet suggested the **Ollama MLX
backend requires >=32GB unified memory**, which if true excludes this 16GB box.

- https://www.mayhemcode.com/2026/07/best-local-llm-setup-for-mac-mini-m4.html — M4 mini 16GB, incl. swap cliff
- https://thoughts.jock.pl/p/local-llm-35b-mac-mini-gemma-swap-production-2026
- https://llmcheck.net/benchmarks — Apple Silicon tok/s M1-M5
- https://modelpiper.com/blog/local-llm-benchmarks-apple-silicon
- https://pub.towardsai.net/apples-mlx-runs-local-llms-3x-faster-than-llama-cpp-until-your-context-hits-40k-715ec441afbb
- https://ollama.com/blog/mlx — OFFICIAL MLX backend announcement (check the 32GB claim here)
- https://dev.to/alanwest/ollama-just-got-93-faster-on-mac-heres-how-to-enable-it-3gce
- https://localaimaster.com/tools/apple-silicon-ai-calculator
- https://yang3kc.substack.com/p/running-llms-on-your-desktop-with

## Axis 4 — small-model cliffs on FINANCIAL reasoning

- https://arxiv.org/abs/2508.05201 — FAITH
- https://arxiv.org/abs/2402.12659 — FinBen (preprint)
- https://proceedings.neurips.cc/paper_files/paper/2024/file/adb1d9fa8be4576d28703b396b82ba1b-Paper-Datasets_and_Benchmarks_Track.pdf — FinBen NeurIPS camera-ready; cite over preprint
- https://arxiv.org/abs/2608.08634 — open-weight vs proprietary on financial text (Aug 2026)
- https://arxiv.org/abs/2506.02515 — FinChain
- https://arxiv.org/abs/2506.21591 — FinEval-KR (knowledge vs reasoning failure)
- https://arxiv.org/abs/2507.06057 — FEVO
- https://arxiv.org/abs/2604.10015 — FinTrace (tool calling on financial tasks; axes 4+6)
- https://arxiv.org/abs/2602.07294 — Fin-RATE
- https://arxiv.org/abs/2603.19254 — FinReasoning
- https://arxiv.org/abs/2605.29586 — FinVerBench (validity + calibration)
- https://arxiv.org/abs/2603.20252 — FinReflectKG-HalluBench (axes 1+4)
- https://arxiv.org/abs/2602.19073 — financial LLM/agent eval suite
- https://arxiv.org/abs/2506.04574 — reasoning vs overthinking, financial sentiment

## Axis 5 — Ollama structured output, JSON-schema -> GBNF

- https://blog.danielclayton.co.uk/posts/ollama-structured-outputs/ — schema constrains sampling, NOT injected into prompt
- https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md — GBNF reference
- https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output
- https://arxiv.org/abs/2403.01632 — SynCode (canonical mechanism)
- https://markaicode.com/ollama-structured-output-pipeline/
- https://llmconfigurator.com/en/guides/llm-json-structured-output
- https://heidloff.net/article/llm-structured-output/

Constrained-decoding vs reasoning quality — bears on whether schema-forcing costs judgment:
- https://aclanthology.org/2024.emnlp-industry.91/ — "Let Me Speak Freely?" (peer-reviewed)
- https://arxiv.org/abs/2408.02442 — same, preprint
- https://openreview.net/forum?id=JSyo3dpEs6 — reviewer pushback on it
- https://arxiv.org/abs/2502.09061 — CRANE (counter-position)
- https://proceedings.neurips.cc/paper_files/paper/2024/file/2bdc2267c3d7d01523e2e17ac0a754f3-Paper-Conference.pdf — Grammar-Aligned Decoding (NeurIPS 2024)
- https://arxiv.org/abs/2605.02363 — structured-output reliability in SMALL models

## Axis 6 — tool calling at 4B-9B (NO dedicated search ran; not dry)

- https://proceedings.mlr.press/v267/patil25a.html — BFCL (PMLR, peer-reviewed)
- https://openreview.net/forum?id=2GmDdhBdDk — BFCL OpenReview
- https://gorilla.cs.berkeley.edu/leaderboard.html — live BFCL V4, filterable by size
- https://arxiv.org/abs/2511.22138 — TinyLLM: SLM agents on edge devices (closest to our scale)
- https://arxiv.org/abs/2410.04587 — Hammer: on-device function calling
- https://arxiv.org/abs/2407.00121 — Granite function-calling
- https://github.com/sierra-research/tau2-bench
- https://github.com/LLM360/tau2-bench
- https://www.spheron.network/blog/tool-calling-benchmarks-bfcl-tau-bench-latency-optimization/

## Axis 7 — arguments AGAINST local deployment

**Peer's structural caveat, which I endorse and which matters more than the
list:** nearly all of this is written by parties selling an API or hosting. The
maintenance-burden arguments transfer; the cost/break-even numbers largely do
not, because this literature is about **GPU-fleet self-hosting, not a single Mac
mini**. Least-conflicted: KDnuggets and deepsense.

- https://www.kdnuggets.com/self-hosted-llms-in-the-real-world-limits-workarounds-and-hard-lessons — least conflicted
- https://deepsense.ai/blog/llm-inference-as-a-service-vs-self-hosted-which-is-right-for-your-business/ — least conflicted
- https://telnyx.com/resources/why-self-hosting-llms-fails — vendor sells an API; discount
- https://bentoml.com/llm/getting-started/serverless-vs-self-hosted-llm-inference — vendor
- https://alpacked.io/blog/self-hosted-llm-guide/
- https://www.sitepoint.com/local-llms-vs-cloud-api-cost-analysis-2026/
- https://www.sitepoint.com/self-hosted-llm-costs-2026/
- https://www.braincuber.com/blog/self-hosted-llms-vs-api-based-llms-cost-performance-analysis
- https://devtk.ai/en/blog/self-hosting-llm-vs-api-cost-2026/
- https://aisuperior.com/llm-hosting-cost/
- https://www.premai.io/blog/self-hosted-llm-guide-setup-tools-cost-comparison-2026/
- https://costlens.dev/blog/self-hosting-llms-vs-cloud-apis-the-2026-showdown
- https://www.aicloudit.com/blog/ai/open-source-vs-proprietary-llms-tco-calculator/
- https://datarootlabs.com/blog/llm-hosting-strategy
