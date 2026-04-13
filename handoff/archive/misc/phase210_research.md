# Phase 2.10: Karpathy Autoresearch Integration — RESEARCH (Pre-Execution)

**Status:** Research phase (awaiting Phase 2.8 PASS, can proceed in parallel with Phase 3 budget approval)  
**Start:** 2026-03-29 10:57 UTC  
**Goal:** Understand Karpathy autoresearch framework, identify integration points with our harness

---

## Research Gate Status (Peder's Minimum: ≥3 URLs for simple, ≥10 for complex)

This is a **complex integration task** (wrapping external framework). Target: ≥10 sources.

### Sources to Fetch (TODO)

**Must read (before code):**
- [ ] Karpathy autoresearch GitHub repo: https://github.com/karpathy/autoresearch
- [ ] README + architecture docs in repo
- [ ] Example: "nanoGPT" autoresearch use case (if documented)
- [ ] Karpathy blog/essays on AI research methodology
- [ ] Three-agent harness paper (Anthropic reference)

**Academic/practitioner context:**
- [ ] Papers: "AutoML" + "Hyperparameter Optimization" (Google Scholar)
- [ ] arXiv: recent work on meta-learning / automated research
- [ ] Quant context: AQR's approach to systematic parameter search
- [ ] Case study: How other teams wrap external ML frameworks with evaluation layers

---

## Preliminary Understanding (before deep research)

### What is Karpathy Autoresearch?

**Definition:** Automated research framework for iteratively improving ML models. Core loop:
1. **Start** with baseline model/params
2. **Generate** candidates (modify one hyperparameter at a time)
3. **Evaluate** each candidate (run, measure performance)
4. **Keep/Discard** based on improvement
5. **Repeat** until plateau

**Karpathy's insight:** This loop is generalizable. Same pattern applies to:
- Neural architecture search (NAS)
- Hyperparameter tuning (HPO)
- Feature engineering
- Research direction selection

### How Our Harness Relates

Our **three-agent harness** (Phase 2.0):
- **Planner:** Decides what to try next (currently heuristic rules)
- **Generator:** Runs experiments (runs backtest engine)
- **Evaluator:** Judges if improvements are real (statistical tests)

**Phase 2.10 proposal:** Treat autoresearch framework as the Generator, wrap it with our Planner + Evaluator.

**Benefits:**
- Autoresearch handles parameter mutation + experiment generation (we just write the evaluator)
- Our evaluator adds domain-specific statistical rigor (DSR, Lo-adjusted Sharpe, etc.)
- Our planner adds strategic direction (what parameter family to focus on)

---

## Key Questions to Answer (via research)

1. **Architecture:** How does autoresearch framework iterate? Where can we inject Planner/Evaluator logic?
2. **Integration points:** Can we replace just the "decide keep/discard" logic with our evaluator?
3. **Parameter generation:** Does autoresearch support parameter namespacing (e.g., "only mutate barrier params this round")?
4. **Cost:** Is autoresearch framework zero-cost (no paid services)? What are computational requirements?
5. **Extensibility:** How do we add new parameter types or evaluation criteria?
6. **Applicability:** Does autoresearch work well for finance/quant models, or is it NAS-focused?

---

## Expected Findings (Hypothesis)

**What we'll probably learn:**
- Autoresearch is a generic optimization loop (applies to any parameter space)
- We can wrap it by implementing a custom `evaluate()` callback
- Planner integration means implementing `propose_next_params()` callback
- Our DSR/Ljung-Box/Lo-adjusted Sharpe go into the evaluator
- The loop becomes: autoresearch proposes → our evaluator judges → planner sets search space

**Unknowns to resolve via research:**
- Does autoresearch have a published API/SDK, or is it just reference code?
- Is it actively maintained (Karpathy still uses it)?
- Are there known limitations in financial use cases?

---

## Execution Plan (Post-Research)

Once research is complete:

1. **Document findings in RESEARCH.md** (URLsm key insights, integration points)
2. **Write handoff/contract.md** for Phase 2.10
   - Success criteria: Autoresearch + our evaluator runs without errors
   - Test case: Wrap a single parameter (e.g., `min_samples_leaf`) in autoresearch loop with our evaluator
3. **Proof of concept** (~4 hours):
   - Clone autoresearch repo
   - Understand parameter mutation logic
   - Implement evaluator callback with our DSR logic
   - Run 5 iterations (tiny test)
4. **Integrate into harness** (Phase 2.10 proper):
   - Replace heuristic planner with autoresearch framework
   - Keep our evaluator logic unchanged
5. **Evaluate:** Does autoresearch + our evaluator find improvements faster than current heuristic planner?

---

## Why This Matters for May Launch

**Current state:** Our harness (Phase 2) uses heuristic rules to decide what to try next.  
**Limitation:** Rules are ad-hoc. E.g., "if 5 consecutive experiments fail on same param, disable that param" — works but not principled.

**With Phase 2.10:** Autoresearch provides a more principled search strategy (local search, mutation + evaluation).  
**Benefit:** Potentially faster convergence to better parameters before May launch.  
**Risk:** Low (Phase 2.10 is optional; can skip if research shows no clear benefit).

---

## Next Steps

1. **Fetch Karpathy autoresearch GitHub** + read architecture
2. **Search Google Scholar** for recent work on automated hyperparameter search in finance
3. **Find ≥10 sources** to meet research gate
4. **Document findings** in RESEARCH.md
5. **Write contract.md** success criteria
6. **Proceed with proof of concept** (if research shows clear integration path)

---

**Status:** Awaiting Phase 2.8 completion. Can start research now if time available, or defer until Phase 3 approval pending.
