#!/usr/bin/env python
"""phase-4.17.5 smoke test -- CoALA memory layers.

Per Sumers et al. 2024, CoALA agents have four memory modules:
  - working (conversation / orchestrator scratch space)
  - episodic (events / cycles / trials)
  - semantic (long-term knowledge; BM25-retrievable in pyfinagent)
  - procedural (skills / prompts)

pyfinagent maps them to:
  working    -> backend.agents.orchestrator import + runtime cache
  episodic   -> backend.backtest.learning_logger (BQ iteration log)
                + `pyfinagent_data.harness_learning_log` (slot events)
  semantic   -> backend.agents.memory::FinancialSituationMemory (BM25)
  procedural -> backend/agents/skills/*.md

Drill probes each layer INDEPENDENTLY and reports per-layer pass/fail.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


def test_coala_all_four_layers_operational():
    errors: list[str] = []

    # 1. Working memory -- orchestrator importable
    try:
        import backend.agents.orchestrator  # noqa: F401
        print("PASS working_memory_orchestrator_importable")
    except Exception as e:
        errors.append(f"working_memory: {e!r}")

    # 2. Episodic -- learning logger importable + IterationLog defined
    try:
        from backend.backtest.learning_logger import IterationLog, log_iteration_to_bq  # noqa: F401
        print("PASS episodic_memory_learning_logger_importable")
    except Exception as e:
        errors.append(f"episodic_memory: {e!r}")

    # 3. Semantic -- FinancialSituationMemory importable (BM25 corpus)
    try:
        from backend.agents.memory import FinancialSituationMemory  # noqa: F401
        print("PASS semantic_memory_agent_memories_importable")
    except Exception as e:
        errors.append(f"semantic_memory: {e!r}")

    # 4. Procedural -- every backend/agents/skills/*.md non-empty
    skills = glob.glob("backend/agents/skills/*.md")
    assert skills, "procedural_memory FAIL: no skills/*.md files"
    empty = [s for s in skills if os.path.getsize(s) == 0]
    if empty:
        errors.append(f"procedural_memory: empty files {empty}")
    else:
        print(f"PASS procedural_memory_all_skills_readable_and_nonempty ({len(skills)} files)")

    if errors:
        raise AssertionError("; ".join(errors))
    print(f"PASS 4.17.5 CoALA memory layers: working + episodic + semantic + procedural all operational")


if __name__ == "__main__":
    try:
        test_coala_all_four_layers_operational()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
