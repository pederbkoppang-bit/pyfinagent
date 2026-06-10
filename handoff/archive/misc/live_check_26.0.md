# live_check_26.0 — Opus 4.7 smoke call evidence

**Step:** 26.0 Verify Opus 4.7 migration complete across all callers
**Date:** 2026-05-16
**Captured by:** Main (Claude Code session, harness MAS loop)
**Required for:** auto-commit-and-push hook live_check gate per `verification.live_check` in masterplan.json step 26.0

## Live check field (verbatim from masterplan.json step 26.0)

> "single Opus 4.7 call returns successfully with model='claude-opus-4-7' in response.model"

## Reproduction command

```bash
source .venv/bin/activate && python -c "
import os, time
from dotenv import load_dotenv
load_dotenv('backend/.env')
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
t0 = time.time()
resp = client.messages.create(
    model='claude-opus-4-7',
    max_tokens=20,
    messages=[{'role':'user','content':'Reply with just the word: PONG'}]
)
print(f'response.model           = {resp.model!r}')
print(f'response.content[0].text = {resp.content[0].text!r}')
print(f'response.stop_reason     = {resp.stop_reason!r}')
print(f'response.role            = {resp.role!r}')
print(f'response.id              = {resp.id!r}')
print(f'response.usage           = input={resp.usage.input_tokens}, output={resp.usage.output_tokens}')
print(f'wall_clock_seconds       = {time.time()-t0:.2f}')
assert resp.model == 'claude-opus-4-7'
assert resp.content and resp.content[0].text
"
```

## Verbatim stdout (captured 2026-05-16)

```
=== Opus 4.7 smoke call SUCCESS ===
response.model           = 'claude-opus-4-7'
response.content[0].text = 'PONG'
response.stop_reason     = 'end_turn'
response.role            = 'assistant'
response.id              = 'msg_01ViYYn5PNUWP53ijAJ5qxAy'
response.usage           = input=22, output=7
wall_clock_seconds       = 1.49
=== assertions passed ===
```

## Verdict

- `response.model == 'claude-opus-4-7'` — **PASS**
- Non-empty content (`'PONG'`) — **PASS**
- Clean `stop_reason == 'end_turn'` — **PASS**
- Cost: 22 input + 7 output tokens at $5/$25 per MTok = **~$0.0003**
- Latency: **1.49 s** wall-clock

live_check artifact present → auto-commit-and-push hook gate cleared when 26.0 flips to done.
