# Alpaca MCP / ExecutionRouter — Rollback Runbook

## When to use

- Paper orders landing on Alpaca look wrong (prices off, unexpected size, sudden volume spike).
- Drift between `bq_sim` reference and `alpaca_paper` fills exceeds 2%.
- Kill switch armed or tripped unexpectedly.
- Any suspicion the agent is placing orders it shouldn't.

## Immediate rollback (30 seconds)

```bash
# 1. Flip backend to bq_sim (stops real submissions to Alpaca paper immediately).
export EXECUTION_BACKEND=bq_sim

# 2. Restart backend so the new env is loaded. (launchd respects plist env,
#    so edit ~/Library/LaunchAgents/com.pyfinagent.backend.plist to set
#    EXECUTION_BACKEND=bq_sim OR unset it, then reload.)
launchctl unload ~/Library/LaunchAgents/com.pyfinagent.backend.plist
launchctl load   ~/Library/LaunchAgents/com.pyfinagent.backend.plist

# 3. Cancel any open Alpaca paper orders (DOES NOT affect bq_sim history).
source .venv/bin/activate
python3 -c "
import requests
from dotenv import dotenv_values
e = dotenv_values('backend/.env')
h = {
    'APCA-API-KEY-ID': e['ALPACA_API_KEY_ID'],
    'APCA-API-SECRET-KEY': e['ALPACA_API_SECRET_KEY'],
}
r = requests.delete('https://paper-api.alpaca.markets/v2/orders', headers=h, timeout=15)
print('cancel_all HTTP', r.status_code)
"

# 4. Optional: flatten all Alpaca paper positions (CLOSE-ALL).
python3 -c "
import requests
from dotenv import dotenv_values
e = dotenv_values('backend/.env')
h = {
    'APCA-API-KEY-ID': e['ALPACA_API_KEY_ID'],
    'APCA-API-SECRET-KEY': e['ALPACA_API_SECRET_KEY'],
}
r = requests.delete('https://paper-api.alpaca.markets/v2/positions', headers=h, timeout=15)
print('close_all HTTP', r.status_code)
"
```

## State invariants after rollback

- `EXECUTION_BACKEND=bq_sim` (or unset → same default).
- Open Alpaca paper orders = 0.
- Alpaca paper positions = 0 (if step 4 ran).
- `financial_reports.paper_trades` BQ history is unchanged — the
  `source` field tracks which backend was active for each trade.
- `financial_reports.paper_portfolio` NAV is whatever the last bq_sim
  mark-to-market wrote; Alpaca-side P&L is irrelevant once canceled.

## Permanent disable (require code review to re-enable)

```bash
# In backend/.env, set EXECUTION_BACKEND=bq_sim explicitly and remove
# Alpaca keys:
python3 - <<'PY'
from pathlib import Path
p = Path('backend/.env')
out = []
for ln in p.read_text().splitlines():
    if ln.startswith('ALPACA_API_KEY_ID='):
        out.append('# ALPACA_API_KEY_ID=<rolled-back>')
    elif ln.startswith('ALPACA_API_SECRET_KEY='):
        out.append('# ALPACA_API_SECRET_KEY=<rolled-back>')
    elif ln.startswith('EXECUTION_BACKEND='):
        out.append('EXECUTION_BACKEND=bq_sim')
    else:
        out.append(ln)
if not any(l.startswith('EXECUTION_BACKEND=') for l in out):
    out.append('EXECUTION_BACKEND=bq_sim')
p.write_text('\n'.join(out) + '\n')
print('disabled')
PY
```

## max_notional_usd clamp

Default $10,000 notional per order. `ALPACA_MAX_NOTIONAL_USD=25000` to
raise, or keep lower for stricter safety. Clamp raises `RuntimeError`
before any `TradingClient.submit_order` call. Evidence:
`backend/services/execution_router.py::_max_notional_usd`.

## Related safety layers (all 4 active)

1. `_refuse_live_keys()` — raises on `PKLIVE*` / `ALPACA_PAPER_TRADE=false`.
2. `.mcp.json` — pins `ALPACA_PAPER_TRADE=true` for the MCP server subprocess.
3. `ExecutionRouter` default mode is `bq_sim`; flipping to alpaca_paper is
   a deliberate env-var action.
4. `max_notional_usd` — pre-submit clamp on order size.

## Escalation

If rollback does not stop Alpaca-side activity within 5 minutes, kill
the backend process hard and investigate:

```bash
pkill -f "uvicorn backend.main:app"
launchctl unload ~/Library/LaunchAgents/com.pyfinagent.backend.plist
# investigate handoff/ logs before reload
```

BLOCKER-4 (task #46) Paper→Live transition MUST NOT proceed until this
runbook has been rehearsed end-to-end.
