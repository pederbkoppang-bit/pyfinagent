"""phase-62.4: sentinel gates -- tamper, infra, preflight wiring.

Tamper tests use env overrides BY RESEARCHED NECESSITY: a synthetic prod row
in llm_call_log would sit in the BQ streaming buffer (~30 min, un-DML-able)
and inflate the real metered figure all day (self-DoS). The overrides
exercise the identical gate logic. The healthy path needs live BQ + ADC and
is marked requires_live.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SENTINEL = REPO / "scripts" / "away_ops" / "sentinel.sh"
WRAPPER = REPO / "scripts" / "away_ops" / "run_away_session.sh"


def run_sentinel(extra_env: dict | None = None) -> tuple[int, dict]:
    env = dict(os.environ)
    env.update(extra_env or {})
    r = subprocess.run(["bash", str(SENTINEL)], env=env, capture_output=True,
                       text=True, timeout=120, cwd=REPO)
    line = [l for l in r.stdout.strip().splitlines() if l.startswith("{")][-1]
    return r.returncode, json.loads(line)


def test_tamper_metered_budget_named_gate():
    rc, rep = run_sentinel({"SENTINEL_TEST_METERED_USD": "99"})
    assert rc == 1
    assert "metered_budget" in rep["gates_failed"]
    assert rep["metered_llm_usd_today"] == 99.0
    assert rep["ok"] is False


def test_tamper_unauthorized_flag_named_gate(tmp_path):
    envf = tmp_path / "test.env"
    envf.write_text("PAPER_FAKE_UNAUTHORIZED_FLAG=true\n")
    rc, rep = run_sentinel({"SENTINEL_ENV_FILE": str(envf),
                            # isolate from BQ so only the flag gate trips
                            "SENTINEL_TEST_METERED_USD": "0"})
    assert rc == 1
    assert "flags_match_tokens" in rep["gates_failed"]
    assert rep["flags_match_tokens"] is False


def test_grandfathered_flags_pass(tmp_path):
    envf = tmp_path / "test.env"
    envf.write_text("PAPER_SWAP_CHURN_FIX_ENABLED=true\nPAPER_TRADING_ENABLED=true\n")
    rc, rep = run_sentinel({"SENTINEL_ENV_FILE": str(envf),
                            "SENTINEL_TEST_METERED_USD": "0"})
    assert rep["flags_match_tokens"] is True
    assert rc == 0


def test_ops_flag_exempt(tmp_path):
    envf = tmp_path / "test.env"
    envf.write_text("AWAY_MODE_ENABLED=true\n")  # ops flag, not PAPER_*; also exempt-listed
    rc, rep = run_sentinel({"SENTINEL_ENV_FILE": str(envf),
                            "SENTINEL_TEST_METERED_USD": "0"})
    assert rep["flags_match_tokens"] is True


def test_infra_path_distinct_exit():
    rc, rep = run_sentinel({"SENTINEL_TEST_BQ_FAIL": "1"})
    assert rc == 2
    assert "metered_source_unavailable" in rep["gates_failed"]
    assert "metered_budget" not in rep["gates_failed"]


def test_override_is_inflate_only_below_baseline():
    # an override below baseline does NOT trip the budget gate but the run is
    # still test-marked via warnings (overrides can trip gates, never mask)
    rc, rep = run_sentinel({"SENTINEL_TEST_METERED_USD": "0.5"})
    assert "metered_budget" not in rep["gates_failed"]
    assert any("test override" in w for w in rep["warnings"])


def test_json_shape_complete():
    _, rep = run_sentinel({"SENTINEL_TEST_METERED_USD": "0"})
    for k in ("metered_llm_usd_today", "baseline_usd", "kill_switch_paused",
              "flags_match_tokens", "ok"):
        assert k in rep, k


def test_wrapper_preflight_breach_selects_digest_only():
    env = dict(os.environ)
    env.update({"SENTINEL_TEST_METERED_USD": "99", "AWAY_SESSION_TEST_PREFLIGHT": "1"})
    r = subprocess.run(["bash", str(WRAPPER), "am"], env=env, capture_output=True,
                       text=True, timeout=180, cwd=REPO)
    assert r.returncode == 0
    assert "PREFLIGHT_PROMPT=digest_only" in r.stdout


@pytest.mark.requires_live
def test_healthy_path_live_bq():
    rc, rep = run_sentinel()
    assert rep["metered_llm_usd_today"] is not None
    assert rep["baseline_usd"] == 8.0
