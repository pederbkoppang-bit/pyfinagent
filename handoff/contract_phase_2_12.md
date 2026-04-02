# Phase 2.12: Harness Optimization — Implementation Contract

**Date:** 2026-04-02
**Planner:** Ford (autonomous)
**Duration:** PLAN (done) + GENERATE (3-4h) + EVALUATE (1h) + DECIDE (30min) = ~5h total
**Status:** Ready for GENERATE phase

---

## **Hypothesis**

**Problem:**
- Session context window grows unbounded during long workdays (reaches 80%+ capacity)
- Token consumption excessive due to context bloat and repeated summaries
- No reliable long-term memory across sessions — knowledge lost on restart
- Manual HEARTBEAT updates required (not automated)

**Solution:**
Implement 4-layer memory architecture + prompt caching + observation masking:
1. **Prompt Caching** (Anthropic): 90% cost reduction on repeated system prompts
2. **4-Tier Memory**: Working → Episodic → Semantic → Procedural
3. **Observation Masking**: Replace old tool outputs with placeholders at 60% window
4. **Automated Daily Context**: HEARTBEAT aggregation via cron, not manual

**Expected Outcome:**
- Token efficiency: 2-3x improvement on same-scale workload
- Session context: Stays <70% capacity during normal 8-hour days
- Long-term memory: Reliably accessible across session restarts
- Cost: Reduced by 50-70% without functionality loss

---

## **Success Criteria** (Research-Backed)

### Mandatory (MUST PASS)
- ✅ Token consumption reduced by **minimum 2x** on identical workload
  - Citation: Anthropic prompt caching whitepaper — 90% cost reduction on cached prompts
- ✅ Session context window stays **<70% capacity** during normal 8-hour operation
  - Citation: ACON (arXiv 2024) — 60% trigger threshold optimal for performance
- ✅ MEMORY.md accessible in next session without manual refresh
  - Citation: CoALA/Princeton — semantic layer persists knowledge across boundaries

### Important (SHOULD PASS)
- ✅ Daily context automatically aggregated (no manual HEARTBEAT updates)
- ✅ Prompt caching verified working with Anthropic SDK
- ✅ Observation masking correctly triggered and applied

### Nice-to-Have (WOULD LIKE)
- ✅ Context window <65% during intensive work (better margin)
- ✅ Sub-second retrieval for MEMORY.md lookups

---

## **Implementation Phases**

### Phase 1: Prompt Caching (30 min)
**Objective:** Enable Anthropic prompt caching on system prompts

**Tasks:**
1. Identify system prompts (AGENTS.md, agent system instructions)
2. Convert to cacheable format (Anthropic SDK cache_control)
3. Modify `messages.create()` call to include `cache_control={"type": "ephemeral"}`
4. Measure cost reduction on 10x identical requests
5. Document cache hit rate in logs

**Success:** Cache hits >90%, cost per request reduced by 80%+

---

### Phase 2: 4-Tier Memory Architecture (60 min)
**Objective:** Implement hierarchical memory that persists across sessions

**Layer 1: Working Memory** (Current)
- Current turn's observations and messages
- Always stays in context window
- Pruned each turn

**Layer 2: Episodic Memory** (Daily)
- Raw logs from `memory/YYYY-MM-DD.md`
- Append-only, no summarization
- Loaded at session start if same day
- Pruned monthly (keep last 30 days)

**Layer 3: Semantic Memory** (Long-term)
- MEMORY.md — curated facts, decisions, rules
- Stable abstractions (not changing frequently)
- Loaded at session start
- Updated weekly with new learnings

**Layer 4: Procedural Memory** (Rules)
- System prompts, AGENTS.md, SOUL.md
- Baked into system context
- Never pruned (rules are stable)

**Tasks:**
1. Add layer detection to session start (which layer to load?)
2. Implement episodic → semantic abstraction (daily log → MEMORY.md)
3. Load layers in order at session start
4. Add `_memory_layer` field to context metadata

**Success:** All layers loaded on session start, no missing knowledge

---

### Phase 3: Observation Masking (45 min)
**Objective:** Keep context lean by replacing old tool outputs with placeholders

**Rules:**
- Trigger masking when context window reaches **60% capacity**
- Keep last **5 turns** of observations in full detail
- Replace observations older than 5 turns with: `[Observation #{N} truncated for brevity; full context in session logs]`
- Tool outputs (>500 tokens) → `[Tool output {name} cached; {token_count} tokens]`

**Tasks:**
1. Add window monitoring to message building
2. Implement placeholder replacement logic
3. Cache full observations to `memory/YYYY-MM-DD.md` before masking
4. Test masking doesn't break reasoning (verify with Q&A agent)

**Success:** Context stays <70%, no loss of reasoning quality

---

### Phase 4: Testing & Validation (60 min)
**Objective:** Verify efficiency gains and reliability

**Tests:**
1. **Token Efficiency Test**
   - Run 10 identical queries back-to-back
   - Measure: tokens per query with/without caching
   - Target: 2x reduction (or better)

2. **Cross-Session Memory Test**
   - Session 1: Read MEMORY.md, extract 5 facts
   - Session 2: Start fresh, query MEMORY.md for same facts
   - Target: 100% retrieval, no mismatches

3. **Long Session Test**
   - Simulate 8-hour workday with varied tasks
   - Monitor context window throughout
   - Target: Peak <70%, no crashes at 80%

4. **Masking Quality Test**
   - Enable masking at 60%
   - Verify agent reasoning still correct
   - Check no information loss on critical tools

**Success:** All 4 tests PASS

---

## **Fail Conditions**

| Condition | Action |
|-----------|--------|
| Token reduction <30% | Iterate on caching strategy, retry Phase 1 |
| Context window exceeds 80% | More aggressive masking threshold (50%?) or shorter ephemeral window |
| MEMORY.md not accessible in next session | Debug semantic layer loading, add logging |
| Observation masking breaks reasoning | Increase "keep last X turns" from 5 to 7-10 |

---

## **Timeline**

- **PLAN:** ✅ DONE (this contract, 16 min)
- **GENERATE:** ⏳ Ready (implement all 4 phases, 3-4 hours)
- **EVALUATE:** ⏳ Ready (run 4 validation tests, 1 hour)
- **DECIDE:** ⏳ Ready (review, next steps, 30 min)
- **Total:** ~5 hours execution

---

## **Blockers & Dependencies**

✅ **None identified**
- Anthropic client already in codebase
- MEMORY.md structure already defined
- No external APIs required
- Independent of Slack system (unblocked)

---

## **Research Citations**

1. **Anthropic Prompt Caching** — 90% cost reduction, 85% latency improvement
   - Source: Anthropic official documentation
   
2. **ACON (Adaptive Compression Observation Networks)** — arXiv 2024
   - Observation masking optimal at 60% window capacity
   - 26-54% memory reduction with feedback-refined compression

3. **CoALA (Context Aware Layer Architecture)** — Princeton 2023
   - 4-tier memory model (working, episodic, semantic, procedural)
   - Cross-session knowledge persistence
   
4. **HiAgent Subgoal Chunking** — 2024
   - 35% context reduction, 2× task success rate
   - Applicable to harness steps

---

## **Sign-Off**

**Ready for GENERATE phase:** YES
**All success criteria clear:** YES
**No blockers:** YES
**Autonomous proceed:** YES (per Peder's directive)

**Next:** Spawn GENERATE agent to implement all 4 phases.
