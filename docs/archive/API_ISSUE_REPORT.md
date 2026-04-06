# ANTHROPIC API ISSUE — DETAILED ANALYSIS REPORT

**Date:** 2026-04-05 10:06 GMT+2  
**Severity:** CRITICAL (blocks agent execution)  
**Status:** UNRESOLVED  

---

## Executive Summary

Anthropic Claude API (both Opus and Sonnet models) is returning **HTTP 429 (Rate Limit)** errors on ALL requests since approximately **2026-04-03 17:00 GMT+2**.

This is an **account-level rate limit**, not a key issue or model-specific problem. All agents (Planner, Evaluator, queue processor agents) **fail silently** when hitting the API, preventing any LLM-based functionality from executing.

---

## Timeline of Issue

| Time | Event | Status |
|------|-------|--------|
| 2026-04-03 14:00 | Ticket queue system working | ✅ |
| 2026-04-03 15:00 | First 429 errors observed | 🚨 |
| 2026-04-03 17:00 | Rate limiting becomes persistent | 🚨 |
| 2026-04-03 20:00 | All agents failing (Opus + Sonnet) | 🚨 |
| 2026-04-04 07:00 | Still throttled | 🚨 |
| 2026-04-05 10:00 | STILL ACTIVE (32+ hours) | 🚨 |

---

## Error Details

### HTTP 429 Response Structure
```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "Error code: 429"
  },
  "request_id": "req_011CZh..."
}
```

### Request Pattern
- **Model Attempts:** claude-opus-4-6, claude-3-5-haiku-20241022, claude-sonnet-4-6
- **All Failing:** Yes (not model-specific)
- **Retry Behavior:** Anthropic SDK retries automatically, but eventually gives up
- **Timeout:** After ~3-5 retries (60+ seconds total)

### Error Logs
```
2026-04-03 20:47:24,189 INFO httpx: HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 429 Too Many Requests"
2026-04-03 20:47:24,981 INFO httpx: HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 429 Too Many Requests"
2026-04-03 20:47:25,189 ERROR backend.services.ticket_queue_processor: Error invoking agent q-and-a: Error code: 429
```

---

## Root Cause Analysis

### What We've Ruled Out ✅
- ✅ **API Key Issue:** Key is valid format (`sk-ant-oat01-...`), loads correctly
- ✅ **Model Name Issue:** Switched between Opus/Sonnet/Haiku — all return 429
- ✅ **Token Size Issue:** Reduced prompts to <2000 tokens — still 429
- ✅ **Rate Limit Mitigation Issue:** Implemented 60s, 120s, 240s delays — still 429

### Likely Causes (Account-Level)
1. **Quota Exceeded** — Account monthly/daily limit hit
2. **Rate Limit Exceeded** — Tokens per minute (TPM) or requests per minute (RPM) limit breached
3. **Account Suspended** — Account flagged or temporarily restricted
4. **Billing Issue** — Card declined or subscription lapsed

---

## Impact Assessment

### Services Affected
| Service | Status | Impact |
|---------|--------|--------|
| Planner Agent | ❌ BLOCKED | Cannot propose features |
| Evaluator Agent | ❌ BLOCKED | Cannot review proposals |
| Queue Processor Agents | ❌ BLOCKED | Tickets stuck in ASSIGNED state |
| Ticket System | ⚠️ DEGRADED | Can create tickets, cannot execute |
| MCP Servers | ✅ OK | Data/backtest/signals working |
| Paper Trading | ✅ OK | Pre-computed, not LLM-dependent |

### Business Impact
- **Phase 3.1 LLM-as-Planner:** Cannot go live
- **Feature Generation:** Frozen (no new features can be proposed)
- **Manual Backtest:** Still possible (Python-only)
- **May 2026 Go-Live:** At risk if not resolved

---

## Mitigation Strategies (Implemented)

### 1. Exponential Backoff (✅ Implemented)
```python
# Retry 1: 60s
# Retry 2: 120s  
# Retry 3: 240s max
```
**Result:** Does NOT help. API still returns 429 immediately.

### 2. Token Reduction (✅ Implemented)
- Reduced prompts from 3000 tokens → <2000 tokens
- Capped proposal generation at 1500 tokens max
- **Result:** Does NOT help. Still 429.

### 3. Model Failover (✅ Implemented)
- Opus 429 → switch to Sonnet
- Sonnet 429 → switch to Research agent
- **Result:** All models throttled. Failover doesn't help.

### 4. Request Spacing (✅ Implemented)
- 10s initial delay
- 60s between retries
- **Result:** API still returns 429 immediately (not a rate limit we can space out).

---

## Recommended Resolution

### Immediate (Today)
1. **Check Anthropic Dashboard:**
   - Log into https://console.anthropic.com/
   - Check "Usage" section for TPM/RPM limits
   - Check "Account Settings" for billing/suspension status
   - Review "Rate Limits" settings

2. **Contact Anthropic Support:**
   - Email: support@anthropic.com
   - Reference Request IDs from error logs
   - Ask: "Why are all our requests being rate-limited?"

3. **Check Account Quota:**
   - Are we on a free tier (limited quota)?
   - Do we need to upgrade to paid account?
   - Do we need to increase rate limit manually?

### Workaround (Short-term)
- Use Gemini API as fallback (VertexAI already integrated)
- Delegate non-critical LLM work to Gemini
- Keep Anthropic for critical agent reasoning only

### Long-term (Post-Resolution)
- Implement multi-provider fallback (Anthropic → Gemini → OpenAI)
- Add rate limit monitoring dashboard
- Set up alerts for 429 errors
- Pre-agree on rate limits before scaling

---

## Current Workarounds in Place

1. **Mock Agent Testing:** Phase 3.1 evaluation uses simulated backtest data
2. **Gemini Fallback:** Some non-critical tasks can use Gemini
3. **Queue Processor:** Can be paused; doesn't crash the system
4. **Fallback to Manual:** Features can be added manually via code until API restored

---

## Dependencies & Blockers

**Phase 3.1 (LLM-as-Planner)** requires resolved API:
- Planner agent needs live API access
- Evaluator agent needs live API access
- Phase 3.2 (LLM-as-Evaluator) also blocked
- Phase 4 (Autonomous harness) also blocked

**Only unblocked by:** Anthropic account quota restored

---

## Questions for Peder

1. Is the Anthropic account on a free tier or paid plan?
2. Have you received any billing warnings or suspension notices?
3. Do we have pre-agreed rate limits with Anthropic?
4. Should we pivot to Gemini API for the critical path?
5. Can you check the Anthropic console for current status?

---

**Report Prepared By:** Ford  
**Confidence Level:** 9/10 (account-level diagnosis is high-confidence)  
**Next Steps:** Await Peder's API account investigation
