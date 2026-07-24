#!/usr/bin/env python3
"""scripts/ops/anthropic_max_bridge.py -- phase-76.9.2.

Durable, repo-versioned Anthropic Messages adapter that puts the nightly
autoresearch (and any future flag-gated caller) on the Claude Max
subscription rail at $0 metered:

    client (anthropic SDK / langchain, plain HTTP)
      -> this bridge (127.0.0.1:18797)
      -> claude-code-proxy (https://localhost:18796, ~/.openclaw)
      -> `claude -p` on the operator's Max plan

WHY a bridge instead of fixing the proxy cert (research_brief_76.9.2.md):

1. AGGREGATION IS MANDATORY: the proxy ALWAYS answers /v1/messages as SSE,
   and installed anthropic 0.96.0's non-streaming create() on an SSE
   response tries response.json(), swallows the failure, and returns raw
   response.text instead of a Message object -- silent type corruption
   (anthropic/_response.py:266-278; _strict_response_validation defaults
   False). A cert fix alone leaves every non-streaming SDK caller broken.
2. The proxy cert is a self-signed leaf (CN=api.anthropic.com, zero
   extensions); Python 3.13+ default-enables VERIFY_X509_STRICT, which
   requires keyUsage+SKI+CA:TRUE+SAN. Regenerating it edits un-versioned
   ~/.openclaw operator infra; this file is git-versioned instead.
3. upstream verify=False is acceptable HERE ONLY: loopback-to-loopback,
   nothing else anchors that cert, and the OpenClaw gateway itself runs
   NODE_TLS_REJECT_UNAUTHORIZED=0 (verified wf_3b9205bc-666 /
   wf_12c8ead9-d26 -- no live external client of the proxy exists).

Behavior:
- POST /v1/messages with "stream": true  -> upstream SSE passed through
  verbatim (the proxy's SSE is Anthropic-shaped).
- POST /v1/messages non-streaming       -> upstream SSE consumed and
  aggregated into a standard non-streaming Messages JSON body; proxy
  `error` events surfaced as PROXY-ERROR text, never silent empties.
- GET /health -> relayed through the chain (proxy answers {"ok":true...}).
- Upstream timeout 600s (> the proxy's 180s per-claude-call ceiling).
- stdlib only. ASCII-only log lines. Binds 127.0.0.1 exclusively.

Env:
  ANTHROPIC_BRIDGE_UPSTREAM  upstream base URL (default https://localhost:18796;
                             plain-HTTP allowed -- used by the test suite's stub)
  ANTHROPIC_BRIDGE_PORT      listen port (default 18797)

Service install: scripts/ops/templates/com.pyfinagent.anthropic-bridge.plist
(OPERATOR bootstraps -- OPS-BRIDGE-BOOTSTRAP token; the 62.0 rail blocks
launchctl bootstrap from agent sessions by design).
"""
import json
import os
import ssl
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("ANTHROPIC_BRIDGE_UPSTREAM", "https://localhost:18796")
PORT = int(os.environ.get("ANTHROPIC_BRIDGE_PORT", "18797"))
UPSTREAM_TIMEOUT_S = 600  # > proxy's 180s per-call claude -p ceiling

_INSECURE_CTX = ssl.create_default_context()
_INSECURE_CTX.check_hostname = False
_INSECURE_CTX.verify_mode = ssl.CERT_NONE


def sse_aggregate(resp):
    """Consume an Anthropic-shaped SSE stream; return a non-streaming
    Messages JSON dict. Surfaces proxy `error` events as PROXY-ERROR text
    so upstream failures are never silent empty strings."""
    msg = {"id": "msg_bridge", "type": "message", "role": "assistant",
           "model": "", "content": [{"type": "text", "text": ""}],
           "stop_reason": "end_turn", "stop_sequence": None,
           "usage": {"input_tokens": 0, "output_tokens": 0}}
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[5:].strip())
        except Exception:
            continue
        t = d.get("type")
        if t == "message_start":
            m = d.get("message", {})
            msg["id"] = m.get("id", msg["id"])
            msg["model"] = m.get("model", "")
            msg["usage"].update(m.get("usage") or {})
        elif t == "content_block_delta":
            delta = d.get("delta", {})
            if delta.get("type") == "text_delta":
                msg["content"][0]["text"] += delta.get("text", "")
        elif t == "message_delta":
            if d.get("delta", {}).get("stop_reason"):
                msg["stop_reason"] = d["delta"]["stop_reason"]
            msg["usage"].update(d.get("usage") or {})
        elif t == "error":
            err = (d.get("error") or {}).get("message", "unknown proxy error")
            msg["content"][0]["text"] = "PROXY-ERROR: " + err
            msg["stop_reason"] = "end_turn"
    return msg


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[max-bridge] " + fmt % args, flush=True)

    def do_GET(self):
        self._relay(None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self._relay(self.rfile.read(n) if n else b"")

    def _send(self, status, ctype, out):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _relay(self, body):
        client_wants_stream = False
        if body:
            try:
                client_wants_stream = bool(json.loads(body).get("stream"))
            except Exception:
                pass
        try:
            req = urllib.request.Request(
                UPSTREAM + self.path, data=body,
                headers={"Content-Type": "application/json"},
                method="POST" if body is not None else "GET")
            ctx = _INSECURE_CTX if UPSTREAM.startswith("https") else None
            with urllib.request.urlopen(req, context=ctx,
                                        timeout=UPSTREAM_TIMEOUT_S) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if "text/event-stream" in ctype and client_wants_stream:
                    # Client speaks SSE natively -> verbatim passthrough.
                    self.send_response(resp.status)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for raw in resp:
                        self.wfile.write(raw)
                        self.wfile.flush()
                    return
                if "text/event-stream" in ctype:
                    out = json.dumps(sse_aggregate(resp)).encode()
                else:
                    out = resp.read()
                self._send(resp.status, "application/json", out)
        except Exception as e:
            # Loud, typed error -- callers must see failure, never a
            # silent empty success (phase-76.9.2 loud-fail doctrine).
            err = json.dumps({"type": "error", "error": {
                "type": "bridge_upstream_error", "message": str(e)}}).encode()
            self._send(502, "application/json", err)


def main():
    print(f"[max-bridge] listening on http://127.0.0.1:{PORT} -> {UPSTREAM}",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
