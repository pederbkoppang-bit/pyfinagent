# Research Brief — phase-76.9.2: durable Max-rail routing for nightly autoresearch

Status: COMPLETE. Tier: moderate (caller-set). Date: 2026-07-24. Researcher: Layer-3 (merged researcher+Explore).

## Read in full (6; counts toward gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/README.md | 2026-07-24 | official code/doc | WebFetch full | README now thin — only `ANTHROPIC_API_KEY` shown; full docs moved to platform.claude.com. Installed-source read (below) is authoritative for behavior. |
| https://code.claude.com/docs/en/cli-reference | 2026-07-24 | official doc | WebFetch full | `--model`: "an alias for the latest model (`sonnet`, `opus`, `haiku`, or `fable`) or a model's full name"; `-p` print mode; `--output-format text|json|stream-json`; `claude update` / `claude install [version]` manage the native binary. |
| https://docs.python.org/3/library/ssl.html | 2026-07-24 | official doc | WebFetch full | "The default settings for this context include `VERIFY_X509_PARTIAL_CHAIN` and `VERIFY_X509_STRICT`... behave more like a conforming implementation of RFC 5280" — **Changed in 3.13** (project runs 3.14). `hostname_checks_common_name` default true (CN fallback exists, but CN=api.anthropic.com ≠ localhost anyway). `load_verify_locations`/`cadata` is the self-signed trust path. |
| https://docs.openssl.org/master/man1/openssl-verification-options/ | 2026-07-24 | official doc | WebFetch full | Trust anchor = element of trust store, "apparently self-signed" OK at defaults; **strict mode**: "CA certificates must explicitly include the keyUsage extension... The subjectKeyIdentifier must be given for all X.509v3 CA certs"; SAN if given must be non-empty. |
| https://keith.github.io/xcode-man-pages/launchd.plist.5.html | 2026-07-24 | official man page | WebFetch full | `EnvironmentVariables`: "additional environmental variables to be set before running the job" — per-job dict of strings; PATH settable. |
| https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic/anthropic_api_url | 2026-07-24 | official doc | WebFetch full | `anthropic_api_url` (alias `base_url`) reads `ANTHROPIC_API_URL` first, then `ANTHROPIC_BASE_URL` (latest docs add a LANGSMITH_GATEWAY fallback not present in installed 1.4.8). |

## Identified but snippet-only (12; context, not gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/Wei-Shaw/sub2api/issues/867 | community issue | Title is the finding: "/v1/messages ignores stream=false — always returns SSE, breaking non-streaming Anthropic SDK clients" — external corroboration of our exact proxy defect class; primary proof is installed SDK source |
| https://github.com/vercel/ai/issues/15542 | community issue | ANTHROPIC_BASE_URL convention drift across SDKs (2026) — supports exporting BOTH env vars |
| https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python | official | Search-corroborated: SDK honors `base_url` param + `ANTHROPIC_BASE_URL` env |
| https://code.claude.com/docs/en/model-config | official | Aliases "point to the recommended version... update over time"; env vars can remap aliases |
| https://support.claude.com/en/articles/11940350-claude-code-model-configuration | official | Alias/model switching help |
| https://www.scriptbyai.com/anthropic-claude-timeline/ | blog | Sonnet 5 GA 2026-06-30 (matches probe) |
| https://claudefa.st/blog/models | blog | Model lineup incl. Fable 5 |
| https://github.com/anthropics/anthropic-sdk-python/blob/main/helpers.md | official code | Streaming helpers |
| https://docs.anthropic.com/en/api/messages-streaming | official | SSE event grammar (message_start/delta/stop) the bridge aggregates |
| https://github.com/zed-industries/zed/discussions/35333 | community | base-URL override demand pattern |
| https://fazm.ai/blog/route-claude-api-through-custom-endpoint-anthropic-base-url | blog | ANTHROPIC_BASE_URL routing pattern |
| https://python.langchain.com/api_reference/anthropic/chat_models/... | official | 308-redirect to reference.langchain.com (fetched there); index page itself had no param docs |

## Queries run (3-variant discipline)
1. Current-year frontier: "anthropic sdk python ANTHROPIC_BASE_URL base_url proxy support 2026"
2. Last-2-year: "claude code CLI --model alias fable sonnet opus 2025"
3. Year-less canonical: "anthropic sdk python non-streaming response text/event-stream content-type error"

## Recency scan (2024-2026)
Findings: (1) **Python 3.13 (Oct 2024) turned on `VERIFY_X509_STRICT` + `VERIFY_X509_PARTIAL_CHAIN` by default** in `create_default_context()` — inherited by our 3.14 venv; this supersedes the older "pass the self-signed cert as cafile and it just works" folklore and is the mechanism behind the measured cert rejection. (2) Claude Code aliases: `fable` alias since v2.1.170 (2026); Sonnet 5 GA 2026-06-30 and `sonnet` now resolves to it (probe-verified on 2.1.218). (3) anthropic-sdk-python docs moved off the README to platform.claude.com in 2026; `ANTHROPIC_BASE_URL` remains honored; vercel/ai #15542 (2026) shows cross-SDK env-var drift → pin both vars. (4) sub2api #867 documents always-SSE-on-stream=false breaking SDK clients — current, active failure class.

## Key findings
1. **Non-streaming create() cannot tolerate SSE — aggregation is REQUIRED regardless of transport fix.** Installed anthropic 0.96.0 `_response.py:266-278` (+`_legacy_response.py:332-356`): content-type must `.endswith("json")`; on SSE it tries `response.json()`, swallows the parse failure (`log.debug`), and — since `_strict_response_validation` defaults False (`_client.py:84`) — **returns raw `response.text` instead of a `Message`** (silent type corruption, worse than an exception). Corroborated externally (sub2api #867). This alone eliminates options (a) and (b) as sufficient.
2. **Cert-regen bar is high on Python 3.14:** default-strict RFC 5280 (ssl docs, changed 3.13) + OpenSSL strict requirements (explicit keyUsage, subjectKeyIdentifier, CA:TRUE, non-empty SAN). Measured cert (re-verified today via `openssl x509 -text`): Subject=Issuer=CN=api.anthropic.com, **no extensions at all**.
3. **No live consumer anchors the proxy cert** (re-verified): live `~/.openclaw/openclaw.json` has no `baseUrl`; only stale `openclaw.json.bak.1:17` pointed at :18796; `combined-certs.pem` referenced by nothing; gateway plist runs `NODE_TLS_REJECT_UNAUTHORIZED=0` (ai.openclaw.gateway.plist:59-60). Cert/listener changes are low-risk — but also low-value given finding 1.
4. **Env-var pin:** installed langchain_anthropic 1.4.8 `chat_models.py:947-951` reads `["ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"]` (API_URL wins); `_client_utils.py:55/:74` reads only `ANTHROPIC_BASE_URL`; legacy `llms.py:57` only `ANTHROPIC_API_URL`. Export BOTH.
5. **Alias probe (2.1.218, one live call, Max rail):** `claude -p --model sonnet --output-format json` → `modelUsage` keys `['claude-haiku-4-5-20251001', 'claude-sonnet-5']` → `sonnet` = **claude-sonnet-5**. `--help` documents aliases `fable`/`opus`/`sonnet` + full names (`claude-fable-5`).
6. **`~/.local/bin/claude` is the installer-maintained stable path:** symlink → `versions/2.1.218`, rewritten by auto-update this morning (versions 2.1.215/217/218 on disk); `/opt/homebrew/bin/claude` is today's manual, unmanaged symlink.

## Internal code inventory
| File | Lines | Role | Status |
|---|---|---|---|
| ~/.openclaw/openclaw.json (live + .bak*) | grepped | OpenClaw config | RE-VERIFIED: no baseUrl/18796 in live config; only `.bak.1:17` (stale, and it was plain `http://`); combined-certs.pem orphaned |
| ~/.openclaw/claude-code-proxy.js | 1-192 | Max-rail proxy | MODEL_MAP :15-20 (opus-4-6/sonnet-4-6/haiku-4-5 only; unknown→`sonnet` :22 — silent-downgrade trap); ALWAYS SSE on /v1/messages :89; `CLAUDE_PATH` env or bare `claude` :10; serializes one `claude -p` :109-120 w/ 180s timeout :120; binds 127.0.0.1 :189; /health :47-50 |
| ~/Library/LaunchAgents/com.pyfinagent.claude-code-proxy.plist | env block | proxy launchd | PATH=`/opt/homebrew/bin:...`, NO CLAUDE_PATH → the ENOENT root cause until today's symlink |
| ~/Library/LaunchAgents/ai.openclaw.gateway.plist | :59-60 | gateway | `NODE_TLS_REJECT_UNAUTHORIZED=0` confirmed; gateway does not anchor proxy-cert.pem |
| ~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist | env block | nightly job | Calls run_nightly.sh; PATH has .venv/bin + /opt/homebrew/bin |
| ~/.openclaw/proxy-cert.pem | openssl -text | proxy TLS cert | RE-VERIFIED: self-signed leaf CN=api.anthropic.com, no basicConstraints, no SAN |
| .venv .../langchain_anthropic/{chat_models,\_client_utils,llms}.py | 947-956 / 53-75 / 55-64 | base-URL resolution | v1.4.8; see key finding 4 |
| .venv .../anthropic/{_response,_legacy_response,_client}.py | 266-278 / 332-356 / 84 | SDK 0.96.0 non-streaming parse | See key finding 1 |
| scripts/autoresearch/run_nightly.sh | 6,19-27,31,43,61-72 | nightly wrapper | `set -euo pipefail`; sanitized-grep .env sourcing :19-27 (`set -a` → every KEY=value incl. a new flag lands in env); `python run_memo.py` :43; fail-state + page-after-3 :54-71; flag insertion point = between :31 and :43 |
| scripts/autoresearch/run_memo.py | 121, 288 | memo runner | Uses **gpt_researcher** :121 (→ langchain_anthropic under the hood); :288 FAILs early if `ANTHROPIC_API_KEY` unset → dummy key export needed when flag ON |
| scratchpad/anthropic_bridge.py (session tmp, uncommitted) | 1-118 | proven bridge | HTTP 127.0.0.1:18797 → https://localhost:18796 `verify=False`; SSE→JSON aggregation :26-58; SSE passthrough when client requests stream :86-95; 600s upstream timeout :84 (> proxy 180s); GET relays /health through the chain |
| scripts/ops/run_ablation.sh | 1-60 | 75.11 pattern | `REPO="${SRE_OPS_REPO:-...}"`; verbatim sanitize copy; fail-state + paging seam — template for the bridge wrapper/plist |
| backend/tests/test_phase_76_9_launchd_fixes.py | 159-202 | test pattern | Throwaway repo-tree fixture (poison .env line, .venv/bin python shim, stub script) + SRE_OPS_REPO redirect — reusable for flag-wiring tests |
| ~/.local/bin/claude, /opt/homebrew/bin/claude, ~/.local/share/claude/versions/ | fs | binary chain | See key finding 6; `claude --version` = 2.1.218 |

## Design answers
**Q1 — transport: OPTION (c), productionize the SSE-aggregating bridge as a repo-owned service.** Deciding facts: (i) the aggregation layer is required NO MATTER WHAT — anthropic 0.96.0 non-streaming create() silently returns raw SSE text as `response.text` on non-JSON content-type (key finding 1), so (a)/(b) alone leave gpt_researcher broken; (ii) (a) additionally demands a fully RFC5280-strict cert (Python 3.14 strict-by-default) and (a)/(b) both edit un-versioned `~/.openclaw` operator infra; (iii) (c) is the ONLY path proven end-to-end (2026-07-24 run_memo topic09 rc=0, 4m40s, $0 metered via dummy key) and is versioned in-repo. Shape: `scripts/ops/anthropic_max_bridge.py` (harden the scratchpad bridge: keep 127.0.0.1 bind, /health relay, stream passthrough, 600s timeout) + `com.pyfinagent.anthropic-bridge.plist` template + wrapper per 75.11 `run_ablation.sh` pattern. `verify=False` to :18796 is acceptable at loopback (nothing else anchors the cert; gateway itself disables TLS verification). Optional low-risk residue (documented runbook edits, not transport): add MODEL_MAP entries and fix the unknown→sonnet silent fallback in proxy.js.
**Q2 — env var:** export **BOTH** `ANTHROPIC_API_URL` and `ANTHROPIC_BASE_URL` = `http://127.0.0.1:18797`. Load-bearing for ChatAnthropic 1.4.8 is `ANTHROPIC_API_URL` (read first, chat_models.py:949); `ANTHROPIC_BASE_URL` covers `_client_utils` default-client path + raw-SDK constructions inside gpt_researcher. Plus `ANTHROPIC_API_KEY=<dummy>` (run_memo.py:288 requires non-empty; dummy guarantees the metered API can never be silently billed).
**Q3 — flag:** `AUTORESEARCH_USE_MAX_RAIL` (values `1`/absent; default OFF) in `backend/.env` — operator flips it; it reaches the wrapper automatically via the existing sanitized `set -a` sourcing (run_nightly.sh:19-27). Wiring between :31 and :43: when ON → preflight `curl -sf -m 10 http://127.0.0.1:18797/health`; on failure, run the SAME fail-state + paging block (factor :54-71 into a function) and `exit` non-zero — LOUD FAIL, never unset-and-fall-back to metered; when preflight passes → export the three vars above (after .env sourcing, so dummy key overrides the real one). When OFF: the guard is a single false `if` — behavior byte-identical to today.
**Q4 — MODEL_MAP additions:** `'claude-opus-4-8': 'opus'`, `'claude-sonnet-5': 'sonnet'` (probe-verified resolves to claude-sonnet-5 on 2.1.218), `'claude-fable-5': 'fable'` (alias in 2.1.218 `--help`; exists since 2.1.170). CLI also accepts full model names, so a safer fallback than the current unknown→`sonnet` is passing `claude-*` ids through verbatim.
**Q5 — CLAUDE_PATH:** set `CLAUDE_PATH=/Users/ford/.local/bin/claude` in com.pyfinagent.claude-code-proxy.plist `EnvironmentVariables` (proxy.js:10 reads it; launchd man page confirms per-job env). That symlink is installer-maintained across version bumps (rewritten this morning 2.1.217→2.1.218); the `/opt/homebrew/bin/claude` symlink is manual and unmanaged — keep as belt-and-suspenders or retire, never load-bearing.
**Q6 — reuse:** test = `test_phase_76_9_launchd_fixes.py` fixture pattern (SRE_OPS_REPO redirect + throwaway tree + stub bridge answering /health) asserting ON-exports / OFF-byte-identity / loud-fail-when-down; wrapper+plist = 75.11 `run_ablation.sh` pattern; a `scripts/qa/verify_max_rail.sh` (curl /health + one tiny POST) mirrors existing scripts/qa verify style.

## Pitfalls (from literature + measurement)
- Always-SSE upstreams break non-streaming SDK clients *silently* here (raw text returned, not an exception) — do not "fix" by cert alone (sub2api #867 class).
- Alias drift: `sonnet` re-points as models GA (docs: aliases "update over time") — MODEL_MAP values as aliases auto-track latest; full-name passthrough is exact. Note either in the runbook.
- Proxy 180s per-call timeout (:120) vs 4m40s total run: fine per-call today, but the bridge must keep its upstream timeout > 180s (600s in the proven bridge).
- Exporting the dummy key BEFORE .env sourcing would be overwritten — insertion point after :31 is mandatory.

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (18)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered every relevant module
- [x] Contradictions/consensus noted (latest langchain docs add LANGSMITH_GATEWAY fallback absent from installed 1.4.8 — installed source wins)
- [x] Per-claim citations

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 17,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0, "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "summary": "Option (c) — productionize the SSE-aggregating bridge as a repo-owned service — is forced: installed anthropic 0.96.0 silently returns raw SSE text (not a Message) on non-streaming create() (_response.py:266-278), so cert-regen (a) or plain-HTTP (b) alone cannot work; (c) is the only live-proven path (topic09 rc=0, $0 metered) and the only versioned one. Env pin: ChatAnthropic 1.4.8 reads ANTHROPIC_API_URL then ANTHROPIC_BASE_URL (chat_models.py:949) — export both + dummy ANTHROPIC_API_KEY (run_memo.py:288 requires it). Flag AUTORESEARCH_USE_MAX_RAIL in backend/.env via existing sanitize; ON = preflight bridge /health, loud non-zero fail through the existing paging seam; OFF = byte-identical. MODEL_MAP: opus-4-8→opus, sonnet-5→sonnet (probe: sonnet alias ran claude-sonnet-5 on 2.1.218), fable-5→fable. CLAUDE_PATH: plist EnvironmentVariables → ~/.local/bin/claude (installer-maintained symlink, rewritten on this morning's auto-update); homebrew symlink is manual/unmanaged. Reuse 76.9 fixture + 75.11 wrapper patterns.",
  "brief_path": "handoff/current/research_brief_76.9.2.md",
  "gate_passed": true
}
```
