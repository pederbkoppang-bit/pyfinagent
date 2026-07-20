"""verify_phase_25_D9_1 -- Caller-side Files API adoption (skill_file_id wiring).

Verifies:
  1. AnalysisOrchestrator has `_skill_gen_config(skill_stem)` helper.
  2. At least 11 `run_*_agent` call sites pass
     `generation_config=self._skill_gen_config(...)`.
  3. Helper returns the enrichment cap alone when `_skill_file_ids` is empty
     (Gemini fallback).
  4. Helper returns `{"max_output_tokens": 1024, "skill_file_id": "<file_id>"}`
     when stem is mapped.
  5. Helper returns the enrichment cap alone when stem is missing (no KeyError).

phase-75.4 (gap5-06) AMENDED claims 3/4/5: the helper used to return `None` on the
two fallback paths and a file-id-only dict on the mapped path -- carrying the
documented 1024 enrichment cap on NEITHER, so the Claude rail silently used its own
2048 default. It now ALWAYS carries `max_output_tokens`. Gemini behavior is
unchanged (its bundle base_config already merged the same 1024 via setdefault).

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("RAG_DATA_STORE_ID", "test-store")

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: helper exists ────────────────────────────────────────────
src = (REPO / "backend/agents/orchestrator.py").read_text(encoding="utf-8")
tree = ast.parse(src)
helper_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "AnalysisOrchestrator":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "_skill_gen_config":
                helper_node = item
                break
        break

claim(
    "1. orchestrator_has_skill_gen_config_helper",
    helper_node is not None,
    "_skill_gen_config method present" if helper_node else "missing",
)


# ── Claim 2: 11+ call sites pass generation_config=self._skill_gen_config ─
matches = re.findall(r"generation_config=self\._skill_gen_config\(", src)
claim(
    "2. enrichment_agents_pass_generation_config_with_skill_file_id",
    len(matches) >= 11,
    f"call_sites={len(matches)} (expected >=11)",
)


# Build a stub orchestrator bypassing __init__ to exercise helper logic.
from backend.agents.orchestrator import AnalysisOrchestrator  # noqa: E402

stub = AnalysisOrchestrator.__new__(AnalysisOrchestrator)


# ── Claim 3: empty dict (Gemini path) returns None ────────────────────
stub._skill_file_ids = {}
out_empty = stub._skill_gen_config("insider_agent")
claim(
    "3. helper_returns_enrichment_cap_only_when_skill_file_ids_empty_gemini_fallback",
    out_empty == {"max_output_tokens": 1024},
    f"got {out_empty}",
)


# ── Claim 4: mapped stem returns the expected dict ────────────────────
stub._skill_file_ids = {"insider_agent": "file_xyz_123"}
out_mapped = stub._skill_gen_config("insider_agent")
claim(
    "4. helper_returns_skill_file_id_dict_for_mapped_stem",
    out_mapped == {"max_output_tokens": 1024, "skill_file_id": "file_xyz_123"},
    f"got {out_mapped}",
)


# ── Claim 5: missing stem returns None (no KeyError) ──────────────────
stub._skill_file_ids = {"insider_agent": "file_xyz_123"}
out_missing = stub._skill_gen_config("nonexistent_agent")
claim(
    "5. helper_returns_enrichment_cap_only_for_unmapped_stem_no_keyerror",
    out_missing == {"max_output_tokens": 1024},
    f"got {out_missing}",
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.D9.1 verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
