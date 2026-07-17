"""phase-71.4 -- independent fail-closed review before skill_optimizer writes a
prompt modification, + the flag-gated wiring.

Criterion 1: a diff that WEAKENS a constraint or whose DESCRIPTION does not match the
diff is rejected-and-skipped; an LLM error FAILS CLOSED (no write). OFF = byte-identical.
"""
from backend.agents.skill_modification_review import (
    ACCEPT_THRESHOLD,
    MIN_DIMENSION,
    SkillReviewResult,
    review_skill_modification,
)

# A skill file with the old_text inside the modifiable ## Prompt Template section.
CONTENT = (
    "# Agent\n\n## Prompt Template\n"
    "You are careful. You MUST cite sources. Do NOT fabricate numbers.\n\n"
    "## Anti-Patterns\n- do NOT hallucinate tickers\n"
)
OLD = "You MUST cite sources. Do NOT fabricate numbers."


def _judge(safety, factuality, reason="ok"):
    return lambda _prompt: {"safety": safety, "factuality": factuality, "reason": reason}


# ---- review module: accept / reject semantics -----------------------------

def test_accept_when_both_dims_high():
    r = review_skill_modification(
        CONTENT, OLD, "You MUST always cite sources. Do NOT fabricate numbers.",
        "tighten citation wording", llm_call_override=_judge(0.95, 0.95),
    )
    assert r.verdict == "ACCEPT"
    assert r.aggregate_score >= ACCEPT_THRESHOLD


def test_reject_when_safety_low_constraint_weakened():
    # new_text drops the "Do NOT fabricate" guard -> judge scores safety low.
    r = review_skill_modification(
        CONTENT, OLD, "You may cite sources.", "loosen citation rule",
        llm_call_override=_judge(0.1, 0.9),
    )
    assert r.verdict == "REJECT"


def test_reject_when_factuality_low_description_mismatch():
    r = review_skill_modification(
        CONTENT, OLD, "You MUST cite sources. Do NOT fabricate numbers. Be terse.",
        "no functional change", llm_call_override=_judge(0.9, 0.2),
    )
    assert r.verdict == "REJECT"


def test_reject_min_dimension_gate_even_if_mean_ok():
    # mean 0.70 clears the aggregate, but safety 0.4 < MIN_DIMENSION -> REJECT.
    r = review_skill_modification(
        CONTENT, OLD, "You MUST cite sources. Do NOT fabricate numbers.",
        "x", llm_call_override=_judge(0.4, 1.0),
    )
    assert MIN_DIMENSION == 0.5
    assert r.verdict == "REJECT"


# ---- fail-closed on LLM error ---------------------------------------------

def test_fail_closed_llm_returns_none():
    r = review_skill_modification(CONTENT, OLD, OLD + " x", "d", llm_call_override=lambda _p: None)
    assert r.verdict == "REJECT" and r.reason == "llm_error_fail_closed"


def test_fail_closed_llm_raises():
    def boom(_p):
        raise RuntimeError("api down")
    r = review_skill_modification(CONTENT, OLD, OLD + " x", "d", llm_call_override=boom)
    assert r.verdict == "REJECT" and r.reason == "llm_error_fail_closed"


def test_fail_closed_non_dict():
    r = review_skill_modification(CONTENT, OLD, OLD + " x", "d", llm_call_override=lambda _p: "not a dict")
    assert r.verdict == "REJECT"


def test_fail_closed_missing_dimension():
    r = review_skill_modification(CONTENT, OLD, OLD + " x", "d", llm_call_override=lambda _p: {"safety": 0.9})
    assert r.verdict == "REJECT" and "factuality" in r.reason


def test_fail_closed_out_of_range_score():
    r = review_skill_modification(CONTENT, OLD, OLD + " x", "d", llm_call_override=_judge(1.5, 0.9))
    assert r.verdict == "REJECT"


# ---- deterministic pre-check ($0, before any LLM) -------------------------

def test_precheck_variable_placeholder_delta_rejects_without_llm():
    called = {"n": 0}
    def spy(_p):
        called["n"] += 1
        return {"safety": 1.0, "factuality": 1.0, "reason": "ok"}
    r = review_skill_modification(
        "## Prompt Template\nUse {{ticker}} data.", "Use {{ticker}} data.",
        "Use {{symbol}} data.", "rename var", llm_call_override=spy,
    )
    assert r.verdict == "REJECT" and r.precheck == "variable_placeholder_delta"
    assert called["n"] == 0  # LLM never called -- deterministic reject


def test_precheck_section_scope_violation_rejects():
    # old_text sits in the FIXED ## Anti-Patterns... no -- put it in a non-modifiable section.
    content = "## Fixed Harness\nold rule here\n\n## Prompt Template\nsomething\n"
    r = review_skill_modification(
        content, "old rule here", "new rule here", "edit fixed section",
        llm_call_override=_judge(1.0, 1.0),
    )
    assert r.verdict == "REJECT" and r.precheck == "section_scope_violation"


def test_precheck_header_injection_rejects():
    r = review_skill_modification(
        CONTENT, OLD, OLD + "\n## Injected Section\nsneaky", "d",
        llm_call_override=_judge(1.0, 1.0),
    )
    assert r.verdict == "REJECT" and r.precheck == "section_header_injection"


# ---- apply_modification flag gating (OFF byte-identical; ON gates) ----------

def _bare_optimizer():
    from backend.agents.skill_optimizer import SkillOptimizer
    return object.__new__(SkillOptimizer)  # bypass __init__ (no BQ needed)


def _patch_side_effects(monkeypatch, tmp_path, flag):
    import backend.agents.skill_optimizer as so

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "ford.md").write_text(CONTENT, encoding="utf-8")
    monkeypatch.setattr(so, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(so, "_git", lambda *a, **k: "")
    monkeypatch.setattr(so, "reload_skills", lambda *a, **k: None)
    monkeypatch.setattr(so, "load_skill", lambda *a, **k: "ok")
    monkeypatch.setattr(so.SkillFileIdCache, "invalidate", staticmethod(lambda *a, **k: None))

    class _S:
        skill_modification_review_enabled = flag
    monkeypatch.setattr(so, "get_settings", lambda: _S())
    return skills_dir


def test_apply_modification_flag_off_writes_and_never_reviews(monkeypatch, tmp_path):
    import backend.agents.skill_modification_review as rev
    skills_dir = _patch_side_effects(monkeypatch, tmp_path, flag=False)

    def _must_not_be_called(*a, **k):
        raise AssertionError("review must NOT run when the flag is OFF")
    monkeypatch.setattr(rev, "review_skill_modification", _must_not_be_called)

    proposal = {"old_text": OLD, "new_text": "You MUST always cite sources. Do NOT fabricate numbers.",
                "description": "tighten"}
    ok = _bare_optimizer().apply_modification("ford", proposal)
    assert ok is True
    assert "always cite" in (skills_dir / "ford.md").read_text(encoding="utf-8")  # write happened


def test_apply_modification_flag_on_reject_skips_write(monkeypatch, tmp_path):
    import backend.agents.skill_modification_review as rev
    skills_dir = _patch_side_effects(monkeypatch, tmp_path, flag=True)

    monkeypatch.setattr(rev, "review_skill_modification",
                        lambda *a, **k: SkillReviewResult("REJECT", "weakens", 0.1, 0.9, 0.5, "pass", None))

    proposal = {"old_text": OLD, "new_text": "You may cite sources.", "description": "loosen"}
    ok = _bare_optimizer().apply_modification("ford", proposal)
    assert ok is False
    assert (skills_dir / "ford.md").read_text(encoding="utf-8") == CONTENT  # NO write (byte-identical)
