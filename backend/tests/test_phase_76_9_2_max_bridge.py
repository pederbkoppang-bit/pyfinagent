"""phase-76.9.2 -- durable Anthropic Max-rail routing for nightly autoresearch.

Covers the three shipped artifacts:
  1. scripts/ops/anthropic_max_bridge.py -- SSE->JSON aggregation (the
     anthropic SDK silently corrupts on SSE-to-non-streaming responses, so
     aggregation is load-bearing), SSE passthrough for streaming clients,
     /health relay. E2E tests run the REAL bridge process against a stub
     upstream via ANTHROPIC_BRIDGE_UPSTREAM.
  2. scripts/autoresearch/run_nightly.sh flag wiring -- default OFF is
     byte-identical; flag ON + bridge down = LOUD non-zero exit through the
     75.11 fail-state seam (never silent metered fallback); flag ON +
     healthy bridge exports both base-URL vars + the dummy key.
  3. The proxy-side reference copy stays in sync in spirit (registration
     asserts only repo-owned facts; ~/.openclaw is operator-deployed).

See handoff/current/contract.md (76.9.2) + research_brief_76.9.2.md.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import stat
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BRIDGE = REPO / "scripts" / "ops" / "anthropic_max_bridge.py"
NIGHTLY = REPO / "scripts" / "autoresearch" / "run_nightly.sh"

_spec = importlib.util.spec_from_file_location("max_bridge_under_test", BRIDGE)
max_bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(max_bridge)


def _sse_bytes(*events: dict) -> list[bytes]:
    return [f"data: {json.dumps(e)}\n\n".encode() for e in events]


# ─────────────────────────────────────────────────────────────────────
# 1a. sse_aggregate unit coverage
# ─────────────────────────────────────────────────────────────────────


def test_aggregate_accumulates_text_usage_and_stop():
    msg = max_bridge.sse_aggregate(_sse_bytes(
        {"type": "message_start", "message": {"id": "msg_1", "model": "m",
                                              "usage": {"input_tokens": 7}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "HELLO-"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "SSE"}},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"},
         "usage": {"output_tokens": 9}},
    ))
    assert msg["content"][0]["text"] == "HELLO-SSE"
    assert msg["id"] == "msg_1"
    assert msg["usage"]["input_tokens"] == 7
    assert msg["usage"]["output_tokens"] == 9
    assert msg["stop_reason"] == "max_tokens"


def test_aggregate_surfaces_proxy_error_events():
    """Proxy `error` events must become visible PROXY-ERROR text, never a
    silent empty string (the failure mode the first live test exposed)."""
    msg = max_bridge.sse_aggregate(_sse_bytes(
        {"type": "error", "error": {"type": "proxy_error", "message": "spawn claude ENOENT"}},
    ))
    assert msg["content"][0]["text"] == "PROXY-ERROR: spawn claude ENOENT"


def test_aggregate_ignores_garbage_lines():
    stream = [b"event: ping\n\n", b"data: not-json\n\n"] + _sse_bytes(
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}})
    assert max_bridge.sse_aggregate(stream)["content"][0]["text"] == "ok"


# ─────────────────────────────────────────────────────────────────────
# 1b. E2E: REAL bridge process against a stub upstream
# ─────────────────────────────────────────────────────────────────────


class _StubUpstream(BaseHTTPRequestHandler):
    """Plain-HTTP stand-in for the claude-code-proxy: /health JSON,
    /v1/messages ALWAYS SSE (mirroring the real proxy's behavior)."""
    serve_json_instead_of_sse = False  # mutation target M5

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"ok":true,"proxy":"stub"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        if type(self).serve_json_instead_of_sse:
            body = json.dumps({"content": [{"type": "text", "text": "RAW-JSON"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in _sse_bytes(
            {"type": "message_start", "message": {"id": "msg_stub", "model": "stub",
                                                  "usage": {"input_tokens": 3}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "HELLO-SSE"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 4}},
        ):
            self.wfile.write(chunk)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_bridge():
    up_port, br_port = _free_port(), _free_port()
    upstream = ThreadingHTTPServer(("127.0.0.1", up_port), _StubUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    env = dict(os.environ,
               ANTHROPIC_BRIDGE_UPSTREAM=f"http://127.0.0.1:{up_port}",
               ANTHROPIC_BRIDGE_PORT=str(br_port))
    proc = subprocess.Popen([str(REPO / ".venv" / "bin" / "python"), str(BRIDGE)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{br_port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("bridge never became healthy")
    yield base
    proc.kill()
    upstream.shutdown()
    _StubUpstream.serve_json_instead_of_sse = False


def test_e2e_health_relays_through_chain(live_bridge):
    body = urllib.request.urlopen(live_bridge + "/health", timeout=5).read()
    assert json.loads(body)["ok"] is True


def test_e2e_nonstreaming_gets_aggregated_json(live_bridge):
    """The load-bearing behavior: upstream answers SSE, the client asked
    non-streaming, the bridge must hand back a real Messages JSON body."""
    req = urllib.request.Request(
        live_bridge + "/v1/messages",
        data=json.dumps({"model": "claude-sonnet-5", "max_tokens": 10,
                         "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=10)
    assert "application/json" in resp.headers.get("Content-Type", "")
    msg = json.loads(resp.read())
    assert msg["content"][0]["text"] == "HELLO-SSE"
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"]["output_tokens"] == 4


def test_e2e_streaming_client_gets_sse_passthrough(live_bridge):
    req = urllib.request.Request(
        live_bridge + "/v1/messages",
        data=json.dumps({"model": "claude-sonnet-5", "stream": True, "max_tokens": 10,
                         "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=10)
    assert "text/event-stream" in resp.headers.get("Content-Type", "")
    assert b"HELLO-SSE" in resp.read()


def test_e2e_streaming_body_is_framed_for_a_KEEP_ALIVE_client(live_bridge):
    """phase-76.9.2 REGRESSION GUARD -- the defect that hung five run_memo attempts.

    Why the sibling urllib test above CANNOT catch this: urllib's
    AbstractHTTPHandler.do_open sets `headers["Connection"] = "close"`, so it
    forces the very close the production client never sends. It stays green for
    every possible state of the framing bug (vacuity shape #5 -- a harness that
    cannot represent the failure).

    The real client is httpx (anthropic SDK), which gpt_researcher reaches via
    stream=True at 6 sites in actions/report_generation.py. It speaks HTTP/1.1
    keep-alive. The bridge declares protocol_version = "HTTP/1.1", so per RFC
    7230 3.3.3 a response with neither Content-Length nor Transfer-Encoding is
    delimited ONLY by connection close -- such a client blocks forever.

    So: talk raw HTTP/1.1 keep-alive over a socket and require BOTH that the
    body arrives AND that the stream is terminated. `recv()` returning b"" is
    the server closing; a socket.timeout here IS the production hang.
    """
    host, port = live_bridge.rsplit("/", 1)[-1].split(":")
    payload = json.dumps({"model": "claude-sonnet-5", "stream": True, "max_tokens": 10,
                          "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = (b"POST /v1/messages HTTP/1.1\r\n"
           b"Host: 127.0.0.1\r\n"
           b"Content-Type: application/json\r\n"
           b"Connection: keep-alive\r\n"          # the production client's behavior
           b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)

    with socket.create_connection((host, int(port)), timeout=10) as s:
        s.sendall(req)
        s.settimeout(8.0)
        chunks, closed = [], False
        try:
            while True:
                b = s.recv(4096)
                if not b:
                    closed = True       # server closed => body is delimited
                    break
                chunks.append(b)
        except socket.timeout:
            closed = False              # server never terminated => the hang

    raw = b"".join(chunks)
    head = raw.split(b"\r\n\r\n", 1)[0].lower()
    assert b"text/event-stream" in head, f"not an SSE response: {head!r}"
    assert b"HELLO-SSE" in raw, "SSE payload did not arrive"
    # The actual regression guard: the body must be delimited somehow.
    framed = (b"connection: close" in head) or (b"transfer-encoding: chunked" in head)
    assert framed, (
        "SSE passthrough sent neither Content-Length, Transfer-Encoding nor "
        "Connection: close on an HTTP/1.1 response -- RFC 7230 3.3.3 leaves the "
        f"body delimited only by close, so a keep-alive client hangs. head={head!r}")
    assert closed, (
        "server never terminated the stream for a keep-alive client -- this is "
        "the exact wedge that hung run_memo attempts 3 and 5 at 0% CPU")


def test_e2e_upstream_down_is_loud_502():
    """Bridge with a dead upstream must return a typed 502 error body --
    never hang or fabricate success (loud-fail doctrine)."""
    br_port = _free_port()
    env = dict(os.environ,
               ANTHROPIC_BRIDGE_UPSTREAM=f"http://127.0.0.1:{_free_port()}",
               ANTHROPIC_BRIDGE_PORT=str(br_port))
    proc = subprocess.Popen([str(REPO / ".venv" / "bin" / "python"), str(BRIDGE)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        base = f"http://127.0.0.1:{br_port}"
        time.sleep(0.5)
        req = urllib.request.Request(base + "/v1/messages", data=b"{}",
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 502
        assert json.loads(exc.value.read())["error"]["type"] == "bridge_upstream_error"
    finally:
        proc.kill()


import urllib.error  # noqa: E402  (used by the loud-502 test)


# ─────────────────────────────────────────────────────────────────────
# 2. run_nightly.sh flag wiring (REAL script, fixture repo)
# ─────────────────────────────────────────────────────────────────────


def _make_nightly_fixture(root: Path, env_lines: str) -> None:
    (root / "backend").mkdir(parents=True)
    (root / "backend" / ".env").write_text(env_lines, encoding="utf-8")
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    shim = venv_bin / "python"
    shim.write_text("#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n", encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    activate = venv_bin / "activate"
    activate.write_text(f'export PATH="{venv_bin}:$PATH"\n', encoding="utf-8")
    ar = root / "scripts" / "autoresearch"
    ar.mkdir(parents=True)
    # Stub run_memo: records the Anthropic env it observed, exits 0.
    (ar / "run_memo.py").write_text(
        "import json, os, pathlib\n"
        "out = {k: os.environ.get(k) for k in\n"
        "       ('ANTHROPIC_API_URL', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_API_KEY')}\n"
        "pathlib.Path(__file__).parents[2].joinpath('observed_env.json').write_text(json.dumps(out))\n",
        encoding="utf-8")
    (root / "handoff").mkdir()


def _run_nightly(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(NIGHTLY)], capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
             "AUTORESEARCH_REPO": str(root)})


def test_nightly_flag_off_is_inert(tmp_path):
    # phase-82.11: the flag is now set EXPLICITLY to 0 here. Before 82.11 this
    # fixture omitted the flag and leaned on the script default being 0, so it
    # was silently doing double duty as a default-pin -- and 82.11 flipped that
    # default to 1 (operator instruction, see run_nightly.sh:71-92). Setting it
    # explicitly restores this guard to what its name claims: "flag OFF is
    # inert". The default itself stays pinned, by
    # test_nightly_default_documented_on below.
    root = tmp_path / "repo"
    _make_nightly_fixture(root, "SOME_KEY=1\nAUTORESEARCH_USE_MAX_RAIL=0\n")
    r = _run_nightly(root)
    assert r.returncode == 0, r.stderr
    observed = json.loads((root / "observed_env.json").read_text())
    assert observed["ANTHROPIC_API_URL"] is None, "flag OFF must not touch routing env"
    assert observed["ANTHROPIC_BASE_URL"] is None


def test_nightly_flag_on_bridge_down_fails_loud(tmp_path):
    root = tmp_path / "repo"
    dead_port = _free_port()  # bound then released -> nothing listens
    _make_nightly_fixture(
        root,
        "AUTORESEARCH_USE_MAX_RAIL=1\n"
        f"AUTORESEARCH_MAX_RAIL_URL=http://127.0.0.1:{dead_port}\n")
    r = _run_nightly(root)
    assert r.returncode == 78, f"expected loud rc=78, got {r.returncode}: {r.stderr}"
    assert not (root / "observed_env.json").exists(), \
        "run_memo must NEVER run when the rail is down (silent metered fallback)"
    log = (root / "handoff" / "autoresearch.log").read_text()
    assert "NOT falling back to metered" in log
    fail_state = json.loads(
        (root / "handoff" / "away_ops" / "autoresearch_fail_state.json").read_text())
    assert fail_state["consecutive_fails"] >= 1, "preflight failure must feed the 75.11 seam"


def test_nightly_flag_on_healthy_bridge_exports_routing(tmp_path):
    root = tmp_path / "repo"
    port = _free_port()
    health = ThreadingHTTPServer(("127.0.0.1", port), _StubUpstream)
    threading.Thread(target=health.serve_forever, daemon=True).start()
    try:
        _make_nightly_fixture(
            root,
            "AUTORESEARCH_USE_MAX_RAIL=1\n"
            f"AUTORESEARCH_MAX_RAIL_URL=http://127.0.0.1:{port}\n"
            "ANTHROPIC_API_KEY=real-metered-key-from-env\n")
        r = _run_nightly(root)
        assert r.returncode == 0, r.stderr
        observed = json.loads((root / "observed_env.json").read_text())
        assert observed["ANTHROPIC_API_URL"] == f"http://127.0.0.1:{port}"
        assert observed["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{port}"
        assert observed["ANTHROPIC_API_KEY"] == "max-rail-dummy-key", \
            "the sourced REAL key must be overridden by the dummy (leakage-proof)"
    finally:
        health.shutdown()


def test_nightly_default_documented_on():
    """phase-76.9.2 guard, repinned by phase-82.11.

    ORIGINAL INTENT (unchanged): the max-rail default must be an EXPLICIT,
    pinned value that cannot drift silently, and the loud-fail behaviour must
    live in executed code rather than in a comment.

    WHAT CHANGED AND WHY: the pinned value moved 0 -> 1. The metered direct API
    had exhausted its credit balance and failed the nightly run on 12
    consecutive dates; the operator's standing instruction (recorded verbatim
    in handoff/current/contract_82.11.md) is "$0-metered stands -> move
    autoresearch OFF it or disable it. Do NOT buy credits." Defaulting the rail
    ON is that move, in tracked code. This is a deliberate supersession of the
    76.9.2 default-OFF decision, not a silent edit.
    """
    text = NIGHTLY.read_text(encoding="utf-8")
    assert 'AUTORESEARCH_USE_MAX_RAIL:-1' in text, (
        "default must be ON (max rail) -- phase-82.11 operator decision")
    assert 'AUTORESEARCH_USE_MAX_RAIL:-0' not in text, (
        "the old metered-by-default value must not linger anywhere in the file")

    # phase-76.9.2 (76.9.2 Q/A finding): this assertion used to accept
    #   "NEVER silently fall" in text  OR  "NOT falling back to metered" in text
    # and the FIRST alternate is satisfied by the COMMENT at run_nightly.sh:75-76 --
    # so deleting the loud-fail behaviour while leaving the comment kept it green
    # (vacuity shape #8, the OR-escape-hatch / comment-token trap). Assert only on
    # EXECUTABLE lines: strip comment-only lines before matching.
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "NOT falling back to metered API" in code, (
        "the loud-fail message must live in an EXECUTED echo, not only in a comment")
    assert "exit 78" in code, "the dead-rail path must exit non-zero, not fall through"
