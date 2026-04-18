"""phase-3.7 step 3.7.7: capability-token + PII-filter regression tests.

Emits handoff/secret_leak_regression.json; exits 0 on PASS.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.agents.mcp_capabilities import (  # noqa: E402
    CapabilityError, ROLE_SCOPES, ScopeViolationError,
    TokenExpiredError, TokenInvalidError,
    TOKEN_TTL_SECONDS, has_scope, issue_token, scrub_args, verify_token,
)


def _test_unsigned_request_rejected() -> dict:
    raised = False
    try:
        verify_token(None, "data.read")
    except CapabilityError:
        raised = True
    return {
        "test": "unsigned_request_rejected",
        "raised": raised,
        "pass": raised,
    }


def _test_expired_token_rejected() -> dict:
    past = time.time() - 10.0
    tok = issue_token("sess-a", "researcher", ttl_seconds=1, now=past)
    raised = False
    try:
        verify_token(tok, "data.read")
    except TokenExpiredError:
        raised = True
    return {
        "test": "expired_token_rejected",
        "raised_expired": raised,
        "pass": raised,
    }


def _test_wrong_scope_rejected() -> dict:
    tok = issue_token("sess-b", "researcher")
    raised = False
    try:
        verify_token(tok, "trading.write")
    except ScopeViolationError:
        raised = True
    return {
        "test": "wrong_scope_rejected",
        "raised_scope": raised,
        "pass": raised,
    }


def _test_forged_token_rejected() -> dict:
    tok = issue_token("sess-c", "strategy")
    payload, sig = tok.rsplit(".", 1)
    forged_sig = ("A" if sig[0] != "A" else "B") + sig[1:]
    forged = f"{payload}.{forged_sig}"
    raised = False
    try:
        verify_token(forged, "data.read")
    except TokenInvalidError:
        raised = True
    return {
        "test": "forged_token_rejected",
        "raised_invalid": raised,
        "pass": raised,
    }


def _test_email_redacted() -> dict:
    logging.basicConfig(level=logging.WARNING, force=True)
    scrubbed, hits = scrub_args(
        {"note": "contact peder@example.com about AAPL"})
    return {
        "test": "email_redacted",
        "scrubbed": scrubbed,
        "hits": hits,
        "pass": "[REDACTED]" in scrubbed["note"]
                and "email" in hits
                and "peder@example.com" not in scrubbed["note"],
    }


def _test_phone_redacted() -> dict:
    scrubbed, hits = scrub_args({"note": "call +1-415-555-0123 ASAP"})
    return {
        "test": "phone_redacted",
        "scrubbed": scrubbed,
        "hits": hits,
        "pass": "[REDACTED]" in scrubbed["note"]
                and "phone" in hits
                and "555-0123" not in scrubbed["note"],
    }


def _test_anthropic_key_redacted() -> dict:
    leaked = "sk-ant-api03-" + "x" * 40
    scrubbed, hits = scrub_args({"payload": f"key is {leaked} ok?"})
    return {
        "test": "anthropic_key_redacted",
        "hits": hits,
        "pass": "[REDACTED]" in scrubbed["payload"]
                and "anthropic_key" in hits
                and leaked not in scrubbed["payload"],
    }


def _test_clean_args_passthrough() -> dict:
    clean = {"ticker": "AAPL", "date": "2026-01-15",
             "window": 30, "tags": ["momentum", "largecap"]}
    scrubbed, hits = scrub_args(clean)
    return {
        "test": "clean_args_passthrough",
        "hits": hits,
        "equal": scrubbed == clean,
        "pass": scrubbed == clean and hits == [],
    }


def _test_role_scope_map() -> dict:
    paper_has = has_scope("paper_trader", "trading.write")
    researcher_has = has_scope("researcher", "trading.write")
    all_others = {r: has_scope(r, "trading.write")
                    for r in ROLE_SCOPES if r != "paper_trader"}
    return {
        "test": "role_scope_map_structural",
        "paper_trader_has_trading_write": paper_has,
        "researcher_has_trading_write": researcher_has,
        "any_other_has_trading_write": any(all_others.values()),
        "pass": paper_has and not researcher_has
                and not any(all_others.values()),
    }


def _test_token_ttl_honored() -> dict:
    t0 = 1_700_000_000.0
    tok = issue_token("sess-ttl", "researcher", now=t0)
    import base64 as _b64, json as _json
    payload_b64 = tok.split(".", 1)[0]
    pad = "=" * (-len(payload_b64) % 4)
    payload = _json.loads(_b64.urlsafe_b64decode(payload_b64 + pad))
    delta = payload["exp"] - payload["iat"]
    return {
        "test": "token_ttl_honored",
        "iat": payload["iat"],
        "exp": payload["exp"],
        "delta_s": delta,
        "pass": abs(delta - TOKEN_TTL_SECONDS) < 1.0,
    }


def main() -> int:
    tests = [
        _test_unsigned_request_rejected(),
        _test_expired_token_rejected(),
        _test_wrong_scope_rejected(),
        _test_forged_token_rejected(),
        _test_email_redacted(),
        _test_phone_redacted(),
        _test_anthropic_key_redacted(),
        _test_clean_args_passthrough(),
        _test_role_scope_map(),
        _test_token_ttl_honored(),
    ]
    all_pass = all(t["pass"] for t in tests)
    result = {
        "step": "3.7.7",
        "verdict": "PASS" if all_pass else "FAIL",
        "tests": tests,
        "tests_passed": sum(1 for t in tests if t["pass"]),
        "tests_total": len(tests),
    }
    out = REPO / "handoff" / "secret_leak_regression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "verdict": result["verdict"],
        "tests_passed": result["tests_passed"],
        "tests_total": result["tests_total"],
    }))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
