# live_check 66.0 -- Recovery re-baseline (2026-07-06, return-day session)

Required shape (masterplan): "live_check_66.0.md with push output, pending_tokens
before/after summary, and the prompt-selection evidence."

## 1. Push output (backlog + install + step artifacts all on origin/main)

```
$ git push origin main            # commits A (sweep) + B (install)
   899d4a90..68909af1  main -> main
$ git push origin main            # commit C (brief/contract/tokens) + changelog trailers
   68909af1..3292baed  main -> main
```

Backlog content pushed in commit A: 37 untracked away-window files (34 session JSONs
2026-06-19..07-06 incl. the 401 series, wrapper session.log rotations excluded by
gitignore) + 6 modified evidence/audit files + goal_phase66_reactivation_DRAFT.md.

## 2. pending_tokens.json before/after

BEFORE (unchanged since 2026-06-19T05:55:00+00:00): 8 asks, none dispositioned;
METERED-BREACH-RECURRING open P1 with root cause "NOT diagnosed".

AFTER (updated 2026-07-06T21:30:00+00:00): 8/8 asks carry disposition/
disposition_note/disposition_at --
| ask | disposition |
|---|---|
| METERED-BREACH-RECURRING | root_caused_pending_fix (phantom accounting; BQ: 06-17 137/137 failed cc_rail $16.30, 06-18 207/207 $42.20, 0 tokens, flat $0.50 rows; real window metered ~$8.24 Gemini; fix = 66.3) |
| MAS-PLIST-ZOMBIE | resolved (mv executed 2026-07-06, `MOVED`, ls-verified) |
| FABLE-HEADLESS | resolved (KEEP OPUS OVERRIDE; operator approved Part-1 table in-session) |
| SDK-CREDIT | deferred (re-decide before next away window) |
| TEST-TOKEN-62.2 | open_operator_gated (re-asked return-day) |
| WEBHOOK | open_operator_gated (re-asked return-day) |
| AUTORESEARCH-SPEND | open_operator_gated (re-asked return-day) |
| ENV-LINE-81 | open_operator_gated (.env = operator keystroke) |

## 3. Prompt-selection evidence (criterion 3)

Wrapper condition, quoted verbatim from scripts/away_ops/run_away_session.sh:96-102:

```sh
    if [ "$PROMPT_KIND" = "am" ] || [ "$PROMPT_KIND" = "pm" ]; then
        dirty=$(git status --porcelain 2>/dev/null | grep -vE '^.. (handoff/audit/|handoff/away_ops/|handoff/logs/)')
        if [ -n "$dirty" ]; then
            slog "dirty tree detected (non-evidence paths) -- recovery prompt selected"
            PROMPT_KIND="recovery"
        fi
    fi
```

Wrapper-visible dirty set at probe time:

```
$ git status --porcelain | grep -vE '^.. (handoff/audit/|handoff/away_ops/|handoff/logs/)'
(clean by wrapper definition)
```

REAL preflight chain (HALT-DEV -> sentinel -> dirty-tree -> prompt selection; no
git/claude side effects per run_away_session.sh:122-126; NOT the DRY_RUN=1 mode, which
skips the dirty check at :80):

```
$ AWAY_SESSION_TEST_PREFLIGHT=1 AWAY_SESSION_KIND=am bash scripts/away_ops/run_away_session.sh am
PREFLIGHT_PROMPT=am
```

=> the next scheduled session (2026-07-07 05:30 UTC) selects the NORMAL AM prompt, not
recovery. Sentinel same run: `metered_llm_usd_today: 0.85 ... ok: true, gates_failed: []`
(session.log 2026-07-06T20:00 entry shows the identical healthy gate output).

## 4. Bonus live evidence (not criterion-bearing, recorded for 66.1/66.2 baselines)

- `claude_code_health_probe(timeout_s=30)` -> `probe ok: True / msg: ok` -- the cc_rail
  credential works again post-/login; next scheduled cycle 2026-07-07 18:00 UTC is the
  first chance of ok=true cc_rail rows.
- Kill-switch MCP: `is_paused: false`, NAV 23997.71, peak 24124.77 (100% cash baseline
  for 66.2).
