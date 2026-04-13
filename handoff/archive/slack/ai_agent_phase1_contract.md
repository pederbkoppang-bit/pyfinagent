# Slack AI Agent Upgrade — Phase 1 Implementation Contract

**Phase:** 1 — App Configuration & Feature Enablement  
**Date:** 2026-04-05 17:48 GMT+2  
**Authority:** Peder B. Koppang (APPROVED)  
**Timeline:** 1 hour  
**Cost:** $0 (configuration only)  
**Status:** PLAN PHASE — Ready to implement

---

## Problem Statement

Current Slack app configuration uses Socket Mode with legacy patterns. Missing:
- Agent container entry point (top-bar launch)
- Agents & AI Apps feature flag
- Suggested prompts support
- Official assistant API methods
- MCP server integration capability

**Solution:** Enable `Agents & AI Apps` feature in Slack app settings and update manifest.

---

## Implementation Plan

### Step 1: Enable Feature in Slack App Settings
**Action:** Navigate to https://api.slack.com/apps → pyfinAgent app

**In left sidebar:**
1. Find **Agents & AI Apps** (under Features)
2. Click to open
3. Toggle **ON** — "Enable Agents & AI Apps"
4. Verify `assistant:write` scope added (appears in OAuth scopes)
5. Save

**Expected result:**
- Feature enabled
- `assistant:write` scope auto-added
- Ready for Assistant API methods

### Step 2: Update App Manifest

**Add to `manifest.json` in `features` section:**

```json
"features": {
  "app_home": { ... },
  "bot_user": { ... },
  "assistant_view": {
    "assistant_description": "PyFinAgent AI Analyst — research-backed stock analysis with multi-agent reasoning",
    "suggested_prompts": [
      {
        "title": "Analyze AAPL",
        "message": "Analyze AAPL. What's the investment thesis?"
      },
      {
        "title": "Backtest a hypothesis",
        "message": "Backtest mean reversion in large-cap tech. What's the Sharpe ratio?"
      },
      {
        "title": "Research a topic",
        "message": "Research latest trends in AI-driven trading. What's the evidence?"
      },
      {
        "title": "Review portfolio",
        "message": "What's our current portfolio allocation and risk profile?"
      }
    ]
  }
}
```

### Step 3: Verify Event Subscriptions

**In app settings → Event Subscriptions:**

Ensure these bot events are subscribed:
- [ ] `assistant_thread_started` — User opens container
- [ ] `assistant_thread_context_changed` — User switches channels
- [ ] `message.im` — User sends message in DM/thread

**URL must be:**
- For Socket Mode: Uses `AsyncSocketModeHandler` (no request URL needed)
- Verify in app settings that Socket Mode is enabled

### Step 4: Verify OAuth Scopes

**In app settings → OAuth & Permissions → Scopes:**

**Bot token scopes (should include):**
- `assistant:write` ← Auto-added by feature toggle
- `chat:write` ← Existing
- `chat:write.public` ← Existing
- Any others currently configured

**User token scopes (add if using Slack MCP):**
- `chat:write`
- `search:read.public`
- `search:read.private`
- `search:read.files`
- `channels:history`
- `groups:history`
- `im:history`

### Step 5: Code Integration (Minimal)

**In `app.py`, ensure:**
1. `AsyncApp` initialized with bot token
2. `AsyncSocketModeHandler` set up
3. Ready for `Assistant` class in Phase 2

**Current `app.py` already has:**
- ✅ AsyncApp + Socket Mode handler
- ✅ Event registration
- ✅ Background task startup

**No changes needed yet** — Phase 2 will add Assistant lifecycle.

---

## Verification Checklist

After completing steps above:

- [ ] Feature toggle ON in Slack app settings
- [ ] `assistant:write` scope present in manifest
- [ ] `assistant_view` block added to manifest
- [ ] `suggested_prompts` defined (4 contextual prompts)
- [ ] Event subscriptions verified (3 events)
- [ ] OAuth scopes reviewed + updated if needed
- [ ] `app.py` confirmed ready for Phase 2
- [ ] Manifest deployed (reinstall app if needed)

---

## Success Criteria

✅ **Feature visible in Slack client:**
- Top bar shows pyfinAgent app launcher
- Clicking opens agent container (split pane)
- Welcome message appears
- Suggested prompts render

✅ **Event handling ready:**
- App can receive `assistant_thread_started` events
- App can receive `message.im` events
- Socket Mode actively listening

✅ **Scopes sufficient:**
- All required scopes present
- No permission errors on API calls

---

## Deliverables

1. ✅ Updated manifest.json with `assistant_view` block
2. ✅ Feature toggle enabled in Slack app settings
3. ✅ Event subscriptions verified
4. ✅ OAuth scopes checked
5. ✅ Ready for Phase 2 (Assistant lifecycle)

---

## Next Step (Phase 2)

Implement assistant lifecycle handlers:
- `handle_thread_started()` → send welcome + prompts
- `handle_context_changed()` → track channel context
- `handle_user_message()` → core response loop

**Timeline:** 2-3 hours after Phase 1 complete

---

**Prepared by:** Ford  
**Status:** PLAN PHASE — Ready to implement  
**Authorization:** ✅ APPROVED by Peder B. Koppang
