"""phase-62.2: operator-token grammar, allowlist, dedupe, append, FO-2 cursor."""

import asyncio
import json

import pytest

from backend.slack_bot import operator_tokens as ot


OP = "U0A078KP4FQ"
CHANNELS = {"C0ANTGNNK8D", "C_DIGEST"}

# phase-75.2 (gap1-11): append_operator_token now REQUIRES the identity
# context (no fail-open default). These suites exercise the authorized
# path, so they pass the operator + the channels they use.
_APPEND_CHANNELS = CHANNELS | {"C1"}
AUTH = {"operator_user_id": OP, "allowed_channels": _APPEND_CHANNELS}


@pytest.fixture(autouse=True)
def isolate_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "TOKENS_PATH", tmp_path / "operator_tokens.jsonl")
    monkeypatch.setattr(ot, "CURSOR_PATH", tmp_path / "away_ops" / "tokens_cursor")
    monkeypatch.setattr(ot, "_seen_events", set())
    yield


# ── grammar ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,step,key,value", [
    ("65.2 EU SCREENER: ON", "65.2", "EU SCREENER", "ON"),
    ("KILL SWITCH: RESUME", None, "KILL SWITCH", "RESUME"),
    ("60.2 FLAG: KEEP OFF", "60.2", "FLAG", "KEEP OFF"),
    ("TEST TOKEN: PING", None, "TEST TOKEN", "PING"),
    ("HALT-DEV", None, "HALT-DEV", ""),
    ("RESUME-DEV", None, "RESUME-DEV", ""),
    ("  61.5 FEE TABLE: ON  ", "61.5", "FEE TABLE", "ON"),
])
def test_grammar_accepts(text, step, key, value):
    p = ot.parse_operator_token(text)
    assert p == {"step": step, "key": key, "value": value}


@pytest.mark.parametrize("text", [
    "fee table: on",                  # lowercase key = deliberate friction
    "what about the fee table?",      # prose
    "65.2 EU SCREENER:",              # no value
    "HALT-DEV now please",            # reserved word with trailing prose
    "TOKEN: line1\nline2: x",         # multiline (no MULTILINE flag)
    "", None,
])
def test_grammar_rejects(text):
    assert ot.parse_operator_token(text) is None


# ── allowlist matcher (criterion 2) ───────────────────────────────────

def msg(user=OP, channel="C0ANTGNNK8D", text="TEST TOKEN: PING", bot_id=None):
    m = {"user": user, "channel": channel, "text": text, "ts": "1.1"}
    if bot_id:
        m["bot_id"] = bot_id
    return m


def test_matcher_accepts_operator():
    assert ot.is_operator_token_message(msg(), OP, CHANNELS) is True


@pytest.mark.parametrize("m", [
    msg(user="U_SOMEONE_ELSE"),
    msg(bot_id="B123"),
    msg(channel="C_RANDOM"),
    msg(text="not a token"),
])
def test_matcher_rejects(m):
    assert ot.is_operator_token_message(m, OP, CHANNELS) is False


def test_matcher_fail_closed_when_unconfigured():
    assert ot.is_operator_token_message(msg(), "", CHANNELS) is False


# ── append + dedupe (criterion 1) ─────────────────────────────────────

def test_append_writes_structured_line():
    res = asyncio.run(ot.append_operator_token(
        **AUTH, text="65.2 EU SCREENER: ON", user=OP, channel="C0ANTGNNK8D",
        ts="171.001", event_id="Ev1"))
    assert res is not None
    line_no, rec = res
    assert line_no == 1
    on_disk = json.loads(ot.TOKENS_PATH.read_text().strip())
    for k in ("ts", "user", "channel", "raw", "step", "key", "value"):
        assert k in on_disk
    assert on_disk["raw"] == "65.2 EU SCREENER: ON"
    assert on_disk["key"] == "EU SCREENER"


def test_duplicate_event_id_not_rewritten():
    asyncio.run(ot.append_operator_token(
        **AUTH, text="TEST TOKEN: PING", user=OP, channel="C0ANTGNNK8D", ts="1.1", event_id="EvX"))
    dup = asyncio.run(ot.append_operator_token(
        **AUTH, text="TEST TOKEN: PING", user=OP, channel="C0ANTGNNK8D", ts="1.2", event_id="EvX"))
    assert dup is None
    assert len(ot.TOKENS_PATH.read_text().strip().splitlines()) == 1


def test_duplicate_channel_ts_not_rewritten():
    asyncio.run(ot.append_operator_token(
        **AUTH, text="TEST TOKEN: PING", user=OP, channel="C1", ts="9.9", event_id=None))
    dup = asyncio.run(ot.append_operator_token(
        **AUTH, text="TEST TOKEN: PING", user=OP, channel="C1", ts="9.9", event_id=None))
    assert dup is None
    assert len(ot.TOKENS_PATH.read_text().strip().splitlines()) == 1


def test_malformed_never_written():
    res = asyncio.run(ot.append_operator_token(
        **AUTH, text="not a token at all", user=OP, channel="C1", ts="2.2"))
    assert res is None
    assert not ot.TOKENS_PATH.exists()


def test_line_numbers_increment():
    for i, t in enumerate(["TEST TOKEN: ONE", "TEST TOKEN: TWO"], start=1):
        n, _ = asyncio.run(ot.append_operator_token(
            **AUTH, text=t, user=OP, channel="C1", ts=f"3.{i}", event_id=f"Ev{i}"))
        assert n == i


# ── FO-2 cursor semantics ─────────────────────────────────────────────

def test_cursor_absent_means_no_tokens_applied():
    assert ot.read_cursor() is None
    asyncio.run(ot.append_operator_token(
        **AUTH, text="TEST TOKEN: PING", user=OP, channel="C1", ts="4.1", event_id="Ev4"))
    pending = ot.unapplied_tokens()
    assert len(pending) == 1 and pending[0][0] == 1


def test_advance_cursor_semantic_fields_and_mtime():
    asyncio.run(ot.append_operator_token(
        **AUTH, text="65.2 EU SCREENER: ON", user=OP, channel="C1", ts="5.1", event_id="Ev5"))
    [(line_no, rec)] = ot.unapplied_tokens()
    payload = ot.advance_cursor(line_no, json.dumps(rec))
    assert payload["applied_line"] == 1
    assert payload["key"] == "EU SCREENER" and payload["value"] == "ON"
    assert len(payload["token_sha256"]) == 64
    cur = ot.read_cursor()
    assert cur == payload
    assert ot.unapplied_tokens() == []  # nothing past the cursor now
    assert ot.CURSOR_PATH.exists()      # fresh mtime = the 62.0 hook gate key


def test_corrupt_cursor_fails_closed():
    ot.CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ot.CURSOR_PATH.write_text("{not json")
    assert ot.read_cursor() is None  # gate stays closed


def test_env_map_keys_are_uppercase_grammar_compatible():
    for key in ot.KNOWN_TOKEN_ENV_MAP:
        assert ot.parse_operator_token(f"{key}: ON") is not None
