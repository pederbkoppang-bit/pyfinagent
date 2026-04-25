"""phase-10.7.4 unit tests for the Cron Budget Allocator.

Allocator tests (cron_allocator.py):
  1. proportional_basic: 3 slots high/medium/low -> 6:3:1 ratio
  2. disabled_excluded: enabled=false slot absent; weights renormalise
  3. min_floor_enforced
  4. max_ceiling_enforced
  5. single_slot_full_budget
  6. allocate_uses_yaml_default_budget (when total_budget=None)
  7. invalid_priority_raises
  8. min_gt_max_raises

Validator tests (subprocess + scripts/meta/validate_cron_budget.py):
  9. validator_real_yaml_exits_0 (real .claude/cron_budget.yaml)
 10. validator_duplicate_job_name_exits_1
 11. validator_bad_priority_exits_1
 12. validator_missing_file_exits_2
 13. validator_min_gt_max_exits_1

No external deps beyond pyyaml + pytest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.meta_evolution.cron_allocator import (  # noqa: E402
    DEFAULT_MAX_TOKENS_PER_FIRE,
    DEFAULT_MIN_TOKENS_PER_FIRE,
    PRIORITY_WEIGHTS,
    Allocation,
    allocate,
    compute_allocations,
)


VALIDATOR = REPO_ROOT / "scripts" / "meta" / "validate_cron_budget.py"
REAL_YAML = REPO_ROOT / ".claude" / "cron_budget.yaml"


def _write_yaml(tmp_path: Path, slots: list[dict], **top_level) -> Path:
    cfg = {
        "version": top_level.get("version", 3),
        "total_slots": top_level.get("total_slots", len(slots)),
        "slots": slots,
    }
    if "total_daily_token_budget" in top_level:
        cfg["total_daily_token_budget"] = top_level["total_daily_token_budget"]
    p = tmp_path / "budget.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _slot(name: str, priority: str = "medium", **kw) -> dict:
    s = {
        "slot_id": kw.get("slot_id", 1),
        "job_name": name,
        "priority": priority,
        "cadence": kw.get("cadence", "daily"),
        "surface": kw.get("surface", "routine"),
    }
    for k in ("enabled", "min_tokens_per_fire", "max_tokens_per_fire", "category"):
        if k in kw:
            s[k] = kw[k]
    return s


# -----------------------
# Allocator tests
# -----------------------

def test_proportional_basic(tmp_path):
    slots = [
        _slot("a", "high", slot_id=1),
        _slot("b", "medium", slot_id=2),
        _slot("c", "low", slot_id=3),
    ]
    p = _write_yaml(tmp_path, slots)
    out = allocate(p, total_budget=10000)
    # weights 6+3+1 = 10; raw shares 6000/3000/1000 (all > default min 1000, < max 50000)
    assert out["a"] == 6000
    assert out["b"] == 3000
    assert out["c"] == 1000
    assert sum(out.values()) == 10000


def test_disabled_excluded(tmp_path):
    slots = [
        _slot("a", "high", slot_id=1),
        _slot("b", "medium", slot_id=2, enabled=False),
        _slot("c", "low", slot_id=3),
    ]
    p = _write_yaml(tmp_path, slots)
    out = allocate(p, total_budget=7000)
    assert "b" not in out
    # weights 6+1=7; raw 6000/1000
    assert out["a"] == 6000
    assert out["c"] == 1000


def test_min_floor_enforced(tmp_path):
    # low-priority slot should get raw share = 1000, but min_floor lifts it
    slots = [
        _slot("big", "high", slot_id=1, max_tokens_per_fire=50000),
        _slot("tiny", "low", slot_id=2, min_tokens_per_fire=5000),
    ]
    p = _write_yaml(tmp_path, slots)
    out = allocate(p, total_budget=7000)
    # raw: big=6000, tiny=1000; floor lifts tiny to 5000
    assert out["tiny"] == 5000
    # big keeps its raw share; sum != total (clamp drift -- expected)
    assert out["big"] == 6000


def test_max_ceiling_enforced(tmp_path):
    slots = [
        _slot("capped", "reserved", slot_id=1, max_tokens_per_fire=2000),
        _slot("other", "low", slot_id=2),
    ]
    p = _write_yaml(tmp_path, slots)
    # weights 10+1=11; raw capped = (10/11)*5500 = 5000; ceiling clamps to 2000
    out = allocate(p, total_budget=5500)
    assert out["capped"] == 2000


def test_single_slot_gets_full_budget(tmp_path):
    slots = [_slot("solo", "high", slot_id=1)]
    p = _write_yaml(tmp_path, slots)
    out = allocate(p, total_budget=12345)
    # raw = 12345; default max 50000 doesn't clip
    assert out["solo"] == 12345


def test_allocate_uses_yaml_default_budget(tmp_path):
    slots = [_slot("a", "high", slot_id=1), _slot("b", "low", slot_id=2)]
    p = _write_yaml(tmp_path, slots, total_daily_token_budget=14000)
    out = allocate(p)  # no total_budget arg -> reads from yaml
    # weights 6+1=7; raw 12000/2000
    assert out["a"] == 12000
    assert out["b"] == 2000


def test_invalid_priority_raises(tmp_path):
    slots = [_slot("a", "bogus_priority", slot_id=1)]
    p = _write_yaml(tmp_path, slots)
    with pytest.raises(KeyError, match="invalid priority"):
        allocate(p, total_budget=1000)


def test_min_gt_max_raises(tmp_path):
    slots = [_slot("a", "high", slot_id=1, min_tokens_per_fire=5000, max_tokens_per_fire=1000)]
    p = _write_yaml(tmp_path, slots)
    with pytest.raises(ValueError, match="min_tokens_per_fire"):
        allocate(p, total_budget=1000)


def test_compute_allocations_returns_richer_data(tmp_path):
    slots = [_slot("a", "high", slot_id=1)]
    p = _write_yaml(tmp_path, slots)
    out = compute_allocations(p, total_budget=10000)
    assert len(out) == 1
    a = out[0]
    assert isinstance(a, Allocation)
    assert a.job_name == "a"
    assert a.priority == "high"
    assert a.weight == PRIORITY_WEIGHTS["high"]
    assert a.raw_budget == 10000
    assert a.clamped_budget == 10000
    assert a.was_clamped is False


def test_priority_weights_constants():
    """Pin the priority weight scheme so a sneaky edit gets caught."""
    assert PRIORITY_WEIGHTS == {"reserved": 10, "high": 6, "medium": 3, "low": 1}
    assert DEFAULT_MIN_TOKENS_PER_FIRE == 1000
    assert DEFAULT_MAX_TOKENS_PER_FIRE == 50000


# -----------------------
# Validator tests (subprocess)
# -----------------------

def _run_validator(yaml_path, *extra_args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(yaml_path), *extra_args],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_validator_real_yaml_exits_0():
    """The actual .claude/cron_budget.yaml MUST validate."""
    assert REAL_YAML.exists(), f"missing: {REAL_YAML}"
    r = _run_validator(REAL_YAML)
    assert r.returncode == 0, (
        f"real yaml failed validation:\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}"
    )


def test_validator_duplicate_job_name_exits_1(tmp_path):
    slots = [
        _slot("dup", "high", slot_id=1),
        _slot("dup", "low", slot_id=2),
    ]
    p = _write_yaml(tmp_path, slots)
    r = _run_validator(p)
    assert r.returncode == 1
    assert "duplicate" in r.stderr.lower()


def test_validator_bad_priority_exits_1(tmp_path):
    slots = [_slot("a", "bogus", slot_id=1)]
    p = _write_yaml(tmp_path, slots)
    r = _run_validator(p)
    assert r.returncode == 1
    assert "priorit" in r.stderr.lower()


def test_validator_missing_file_exits_2(tmp_path):
    r = _run_validator(tmp_path / "does_not_exist.yaml")
    assert r.returncode == 2
    assert "not found" in r.stderr.lower()


def test_validator_min_gt_max_exits_1(tmp_path):
    slots = [_slot("a", "high", slot_id=1, min_tokens_per_fire=5000, max_tokens_per_fire=1000)]
    p = _write_yaml(tmp_path, slots)
    r = _run_validator(p)
    assert r.returncode == 1
    assert "min_tokens_per_fire" in r.stderr or "min" in r.stderr.lower()


def test_validator_quiet_flag(tmp_path):
    slots = [_slot("a", "high", slot_id=1)]
    p = _write_yaml(tmp_path, slots)
    r = _run_validator(p, "--quiet")
    assert r.returncode == 0
    assert r.stdout == "" or r.stdout.strip() == ""


def test_validator_total_slots_mismatch_exits_1(tmp_path):
    slots = [_slot("a", "high", slot_id=1)]
    p = _write_yaml(tmp_path, slots, total_slots=999)
    r = _run_validator(p)
    assert r.returncode == 1
