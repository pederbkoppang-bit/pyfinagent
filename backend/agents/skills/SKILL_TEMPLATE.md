# {Agent Name}

## Goal
{Money-making oriented goal — tied to maximizing risk-adjusted return (return_pct × beat_benchmark_rate). State explicitly how this agent's output contributes to identifying winning trades and avoiding losing ones.}

## Identity
{Role in the 15-step pipeline, what step number, what upstream data it receives, what downstream agents consume its output. E.g., "Step 7 enrichment agent — analyzes insider trading data from sec_insider.py, feeds signal into debate framework and synthesis."}

## What You CAN Modify (Fair Game)
- Analytical techniques and reasoning strategies
- Signal thresholds and classification boundaries
- Emphasis weighting between data points
- Anti-pattern detection rules
- Output narrative and evidence presentation
- Confidence calibration approach

## What You CANNOT Modify (Fixed Harness)
- Output JSON schema (keys and value types)
- Input data format (what the data tools provide)
- Function signature in prompts.py
- Data tool implementations (backend/tools/*)
- Orchestrator pipeline order
- BigQuery schema columns

## Data Inputs
{Structured description of input data with {{template_variables}} using double-brace syntax. E.g.:
- `{{ticker}}` — stock symbol
- `{{insider_data}}` — JSON from sec_insider.py containing buy/sell counts, trade details
}

## Skills & Techniques
{Numbered list of analytical techniques this agent should apply. These are the "model architecture" — the reasoning strategies that drive signal quality.}

## Anti-Patterns
{Biases and errors to avoid. Drawn from bias_detector findings, conflict_detector patterns, and past outcome failures. E.g.:
- Do NOT default to NEUTRAL/HOLD as a safe fallback
- Do NOT ignore contradictory signals
- Do NOT anchor on a single data point
}
- Do NOT invent, compute, or round financial numbers — cite ONLY values from FACT_LEDGER verbatim
- Do NOT use approximate language ("around", "roughly", "about") for FACT_LEDGER values — use exact figures
- Do NOT reference metrics not present in FACT_LEDGER — say "data unavailable" instead
- Do NOT contradict FACT_LEDGER values even if your training data suggests different numbers — flag the discrepancy explicitly
- Do NOT hallucinate company names, ticker symbols, sector classifications, or industry labels — use ONLY what FACT_LEDGER provides

## Research Foundations
{Academic/industry research backing this agent's approach, with citations from AGENTS.md research table.}

## Evaluation Criteria
{How to measure if this agent's output improved returns. Tied to outcome_tracker metrics:
- Primary: Does modifying this agent's prompt increase avg return_pct of subsequent recommendations?
- Secondary: Does it increase beat_benchmark_rate?
- Proxy: Does it improve signal accuracy (signal direction vs actual price movement)?
}

## Output Format
{JSON schema — FIXED, do not modify. This is part of the evaluation harness.}

## Prompt Template
{{fact_ledger_section}}
{The actual LLM instructions sent to the model. THIS section is the "train.py" — the ONLY part the optimizer modifies.

Uses {{variable}} double-brace syntax for runtime injection. Variables are replaced by the skill loader at call time.}

## Experiment Log
| Date | Commit | Metric Before | Metric After | Status | Description |
|------|--------|--------------|-------------|--------|-------------|
| — | — | — | — | baseline | Initial prompt from prompts.py |
