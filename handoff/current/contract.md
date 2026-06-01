# Contract — phase-54.2 (Reliable daily Slack digests for the away week)

**Date:** 2026-06-01. **Tier:** moderate. **Step:** phase-54.2 (P0). Operator REMOTE
2026-06-01 → 2026-06-08, Slack-only. THE lifeline step.

## N* delta (N* = Profit − Risk − Burn)

**Risk↓** (operational visibility): guarantees the operator's only window (Slack)
actually delivers a daily status digest with NAV/P&L, kill-switch/gate, and cron
health — so a problem during the unattended week is SEEN, not silent. No P delta.
$0 (digest is template/data-only, confirmed). No money-path change.

## Research-gate summary

`researcher` ran first (gate **PASSED**: 7 sources read in full, 23 URLs, recency
scan, 12 internal files). Brief: `handoff/current/research_brief.md`. Decisive:
1. **The bot IS supervised** (corrects 54.1): `scripts/slack_bot_monitor.sh` runs
   every 5 min from the user crontab (since 2026-04-01) — grep-guarded `nohup`
   restart + iMessage alert to the operator's phone. The Mac will NOT sleep
   (`caffeinate -i -s` under the backend launchd job). The lifeline is already
   resilient.
2. **A launchd KeepAlive plist is the WRONG move** — it spawns a SECOND instance →
   the monitor's grep masks failures + TWO APScheduler schedulers → double-fired
   digests AND double-fired heavy crons (nightly_mda_retrain etc.). Do NOT add it.
3. **Confirmation digest = standalone one-shot `AsyncWebClient`** (no Socket Mode
   connection; never touches PID 42151). `slack_channel_id`/`slack_bot_token` are
   real + set. `chat.postMessage` → `ok:true` + `ts`; include a `text` fallback
   (a11y); no idempotency (do NOT blind-retry).
4. **Digest is $0/template** (`formatters.py` imports only math+datetime; no LLM) →
   NOT operator-gated. Safe to send + to add an internal `/api/jobs/all` GET.

## Hypothesis

Sending one labelled confirmation digest via the one-shot Web-API path proves the
operator's Slack window receives messages end-to-end; folding a fail-open
`cron_health` kwarg into `format_morning_digest` (byte-identical when `None`) gives
the operator daily cron visibility; a controlled single-restart (monitor greps-first
→ one instance) deploys it for the week without risking the lifeline.

## Immutable success criteria (verbatim from masterplan phase-54.2)

1. the morning + evening Slack digests are confirmed scheduled (slack_bot/scheduler.py)
   AND the Slack bot process is confirmed running; if either is down it is fixed or
   escalated to the operator.
2. at least ONE live digest is delivered to the operator's Slack channel during this
   step and receipt is confirmed (message ts / channel id recorded), proving the
   away-week pipeline works end-to-end.
3. the digest content covers the remote-supervision essentials: NAV / total P&L / open
   positions, kill-switch + go-live-gate state, the 54.1 cron-health summary, and the
   best-in-class-elevation autonomous-cycle progress.
4. any LLM-summarized digest body that would incur API spend is flagged operator-gated
   (not silently spent); live_check_54.2.md records the delivered digest + channel + ts
   + the daily cadence for the 2026-06-01 → 2026-06-08 window.

## Plan steps (researcher §12 — every item $0 + zero-risk to PID 42151)

1. **Verify the supervisor** (read-only): the `*/5` cron monitor present + greps the
   live process name + sources the venv + `imsg` on PATH; Mac won't sleep
   (caffeinate). Confirm morning/evening digests scheduled + bot up. Document
   (corrects the 54.1 "unsupervised" framing). (criterion 1)
2. **Fold the cron-health line** into `format_morning_digest` (Option a): new
   `cron_health: str | None = None` kwarg → a `section` block before the `divider`
   (`formatters.py:382`); the scheduler computes the line from `/api/jobs/all`
   wrapped in try/except (fail-open → `None`). Byte-identical when `None`. Test:
   byte-identity with kwarg absent + render with a synthetic failed job. (criterion 3)
3. **Send ONE live confirmation digest** (standalone one-shot script
   `scripts/ops/send_confirmation_digest.py`): `AsyncWebClient(bot_token)` +
   `format_morning_digest(..., cron_health=<computed>)` + a short away-week note
   (sync done, 54.1 done, 54.2 in progress, planned 50.6→43.0→53.x) + `text`
   fallback → `chat_postMessage(channel=slack_channel_id)`. Record `ok`+`ts`+channel.
   (criteria 2, 3)
4. **Controlled single-restart to deploy** (researcher §12.3 option ii):
   `pkill -f backend.slack_bot.app`; the 5-min monitor respawns ONE instance
   (greps-first → no double-instance), OR run the monitor script directly to respawn
   immediately. Verify exactly one bot process + scheduler registered; if the restart
   misbehaves, fall back to ensuring a working bot (the one-shot path already proved
   outbound delivery as a fallback). (deploys criterion 3 for daily digests)
5. **Write `live_check_54.2.md`**: supervisor proof, the delivered confirmation digest
   (channel+ts+content), the cron-health line, the daily cadence, and the
   elevation-progress delivery plan (milestone Slack updates from this session +
   GitHub commits). (criterion 4)
6. **Fresh qa → log → flip → commit.**

## Scope / guardrails

- DO NOT add a launchd plist (double-instance). DO NOT blind-retry chat.postMessage.
- Sending the digest is authorized: the operator explicitly asked for Slack updates
  while away; the destination is the configured operator channel (settings, not
  observed content). $0/template → not operator-gated.
- `cron_health=None` default ⇒ byte-identical existing digests (DO-NO-HARM).
- The controlled restart is the ONLY process action; the monitor guarantees a single
  instance. No `.env`/secret edit (read token via the existing SecretStr accessor).
- Elevation-cycle progress (criterion 3): delivered in the confirmation digest's
  away-week note + as milestone Slack posts from this session as 50.6/43.0/53.x land,
  since the bot's daily digest can't see the Claude-Code harness state.

## References

- `handoff/current/research_brief.md` (Slack Socket Mode/multi-connection, launchd
  KeepAlive, chat.postMessage, on-call digest practices).
- `scripts/slack_bot_monitor.sh` (the supervisor); `backend/slack_bot/scheduler.py`
  (`_send_morning_digest` :330/:351, base URL :95); `backend/slack_bot/formatters.py`
  (`format_morning_digest` :323, divider :382); `backend/config/settings.py`
  (`slack_channel_id`/`slack_bot_token`/digest hours).
