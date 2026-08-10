#!/usr/bin/env python3
"""smoke_cc_rail_e2e.py -- phase-4000 E2E smoke for the Claude Code Max rail.

Modes:
  --dry   No mutation of any kind. Preflight (backend health, claude binary
          resolution, optional Max-auth probe), honest rail state (live flag
          via GET, 7d rail health, per-analysis call ESTIMATE from history).
  --live  Bracketed window: PUT the flag ON, drive <=2 single-ticker analyses
          through POST /api/analysis/, run checks E1-E6 per the frozen 4000.1
          baseline section (f), restore the flag in an ExitStack finally
          (always via PUT -- the PUT persists to backend/.env, so an in-memory
          reset would NOT survive; see contract_4000.2.md R7). --keep-on
          cancels the restore ONLY on all-checks-pass.

Exit codes: 0 all-pass; 1 check-fail; 2 preflight-fail; 3 budget-cap trip;
4 usage error. One JSON line per check verdict on stdout + a final summary.

Budget architecture (contract R3): the rail subprocesses are spawned by the
BACKEND, so no in-process wrapper here can count them; llm_call_log is
buffered (flush 100 rows / 60s, no flush endpoint), so the during-window
watcher is best-effort and the AUTHORITATIVE count is post-window after a
>=65s flush wait. A cap trip aborts THIS observer and restores the flag;
in-flight backend calls continue (no cancel endpoint exists) -- disclosed.

E5 surrogate (contract R5): rail-guard internals have no out-of-process
surface; E5 = zero in-window ok=false rail rows AND zero 'rail_guard_skipped:'
markers in analysis output. The breaker P1 page is not observable here
(reported as a note, never a pass leg).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

BQ_PROJECT = "sunny-might-477607-p8"
LLM_LOG_TABLE = f"{BQ_PROJECT}.pyfinagent_data.llm_call_log"
MAX_TICKERS = 2
DEFAULT_MAX_RAIL_CALLS = 30
FLUSH_WAIT_S = 65  # llm_call_log buffer: 100 rows / 60s, piggybacked flush
FLAG = "paper_use_claude_code_route"

# USD list rates per Mtok (input, output) for the E6 would-be-cost basis.
# Source: research_brief_4000.1.md F0 (Agent SDK credit priced at API list).
E6_LIST_RATES = {
    "claude-opus": (5.0, 25.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
}
E6_MONTHLY_CREDIT_USD = 100.0  # Max 5x figure from the PAUSED policy (W1)


# ── row predicates (4000.1 baseline section (c); mutation target m1) ─────────
def is_rail_row(row: dict) -> bool:
    provider = row.get("provider")
    agent = row.get("agent") or ""
    return provider == "claude-code" or agent == "cc_rail" or agent.startswith("cc_rail:")


def is_metered_row(row: dict) -> bool:
    provider = row.get("provider")
    agent = row.get("agent")
    if provider != "anthropic":
        return False
    return agent is None or (agent != "cc_rail" and not str(agent).startswith("cc_rail:"))


def is_claude_role_row(row: dict) -> bool:
    return str(row.get("model") or "").startswith("claude")


# ── HTTP helpers (stdlib only; the tests stub the backend cross-process) ─────
def http_json(method: str, url: str, body: dict | None = None, timeout: int = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    # phase-86.27: this is the ONLY place this script opens an HTTP connection,
    # so it is the one place that knows the verb. The socket guard installed in
    # main() knows the RESOLVED address but never the verb -- `socket.connect`
    # fires before any verb is on the wire. Carrying it across here is what lets
    # the guard refuse a mutating call while leaving GETs working.
    with mutating_scope(str(method).upper() in MUTATING_VERBS):
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())


# ── log sources ──────────────────────────────────────────────────────────────
class HttpLogSource:
    """Test seam: GET <url>?since=<iso>&until=<iso> returns a JSON list of rows."""

    def __init__(self, url: str):
        self.url = url

    def rows(self, since: datetime, until: datetime) -> list[dict]:
        q = f"?since={since.isoformat()}&until={until.isoformat()}"
        out = http_json("GET", self.url + q)
        return out if isinstance(out, list) else out.get("rows", [])


class HttpSpendSource:
    """Test seam: GET <url> returns {"daily_usd": <float>}."""

    def __init__(self, url: str):
        self.url = url

    def daily_usd(self) -> float:
        return float(http_json("GET", self.url).get("daily_usd", 0.0))


class BigQuerySpendSource:
    """Production: the REAL spend-breaker function, so the E2 spend leg uses the
    production exclusion logic rather than this script's reimplemented predicate
    (frozen baseline (f) E2 second conjunct). CAVEAT disclosed in the verdict:
    fetch_llm_spend is fail-open to (0.0, 0.0), so a zero delta where BOTH
    bracketing reads are 0.0 cannot distinguish 'metered dark' from 'fetch
    failed' -- the row-count leg is the primary evidence either way."""

    def daily_usd(self) -> float:
        from backend.services.observability.spend import fetch_llm_spend

        daily, _monthly = fetch_llm_spend()
        return float(daily)


class BigQueryLogSource:
    def rows(self, since: datetime, until: datetime) -> list[dict]:
        from google.cloud import bigquery  # deferred import: tests never need it

        client = bigquery.Client(project=BQ_PROJECT)
        q = f"""
        SELECT ts, provider, model, agent, ok, ticker, latency_ms,
               input_tok, output_tok, cache_creation_tok, cache_read_tok
        FROM `{LLM_LOG_TABLE}`
        WHERE ts >= TIMESTAMP('{since.isoformat()}')
          AND ts <= TIMESTAMP('{until.isoformat()}')
        LIMIT 2000"""
        return [dict(r) for r in client.query(q).result(timeout=30)]


# ── verdict emission ─────────────────────────────────────────────────────────
class Verdicts:
    def __init__(self):
        self.checks: list[dict] = []

    def emit(self, check: str, ok: bool, **detail):
        rec = {"check": check, "ok": bool(ok), **detail}
        self.checks.append(rec)
        print(json.dumps(rec, default=str), flush=True)

    @property
    def all_ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


# ── preflight ────────────────────────────────────────────────────────────────
def resolve_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    # Criterion 3: imported production resolution logic, never reimplemented.
    from backend.agents.claude_code_client import _resolve_claude_binary

    return _resolve_claude_binary("claude")


def run_probe(binary: str, model: str, timeout_s: int = 120) -> dict:
    """One CLI call in an EMPTY temp cwd (avoids the repo-context overhead
    measured in 4000.1 P3). Returns the parsed envelope."""
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.run(
            [binary, "--print", "--output-format", "json", "--model", model,
             "reply with exactly: OK"],
            capture_output=True, text=True, timeout=timeout_s, cwd=d,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"probe exit {proc.returncode}: {proc.stderr[:300]}")
    return json.loads(proc.stdout)


def probe_envelope_check(env: dict, requested_model: str) -> tuple[bool, str]:
    """E3a: iterate ALL modelUsage keys (the phase-78 defect class -- a
    last-wins collapse under-counts; 4000.1 probe P1/P2)."""
    mu = env.get("modelUsage")
    if not isinstance(mu, dict) or not mu:
        return False, "modelUsage absent or not a map"
    total = env.get("total_cost_usd")
    cost_sum = 0.0
    foreign = []
    base = requested_model.rsplit("-2", 1)[0]  # tolerate dated ids
    for key, entry in mu.items():
        cost_sum += float((entry or {}).get("costUSD") or 0.0)
        canonical = str((entry or {}).get("canonicalModel") or key)
        if not canonical.startswith(base):
            foreign.append(canonical)
    if foreign:
        return False, f"foreign model(s) in modelUsage: {foreign}"
    if total is None or abs(float(total) - cost_sum) >= 1e-6:
        return False, f"total_cost_usd {total} != sum(costUSD) {cost_sum}"
    return True, f"{len(mu)} entries, cost sum {cost_sum:.6f} == total"


# ── E6 report ────────────────────────────────────────────────────────────────
def e6_report(window_rows: list[dict], n_analyses: int, cycles_per_day: float) -> dict:
    rail = [r for r in window_rows if is_rail_row(r)]
    tok_in = sum(int(r.get("input_tok") or 0) + int(r.get("cache_creation_tok") or 0)
                 + int(r.get("cache_read_tok") or 0) for r in rail)
    tok_out = sum(int(r.get("output_tok") or 0) for r in rail)
    usd = 0.0
    for r in rail:
        model = str(r.get("model") or "")
        for prefix, (rin, rout) in E6_LIST_RATES.items():
            if model.startswith(prefix):
                usd += (int(r.get("input_tok") or 0) + int(r.get("cache_creation_tok") or 0)) / 1e6 * rin
                usd += int(r.get("output_tok") or 0) / 1e6 * rout
                break
    per_analysis_usd = usd / max(n_analyses, 1)
    daily_usd = per_analysis_usd * 13 * cycles_per_day
    return {
        "basis": "USD at API list rates over window rail rows (cache_read priced as input; "
                 "session_cost_usd is a gauge and is never summed); "
                 "13-ticker cycle x --cycles-per-day extrapolation",
        "window_rail_calls": len(rail),
        "window_tokens_in_incl_cache": tok_in,
        "window_tokens_out": tok_out,
        "would_be_usd_per_analysis": round(per_analysis_usd, 4),
        "would_be_usd_per_day_13_tickers": round(daily_usd, 4),
        "pct_of_100usd_monthly_credit": round(daily_usd * 30 / E6_MONTHLY_CREDIT_USD * 100, 2),
        "weekly_pool_note": "no token-denominated weekly-pool figure is published; "
                            "the USD figures above are the stated proxy (contract R9/E6)",
    }


# ── live window ──────────────────────────────────────────────────────────────
def poll_analysis(base: str, analysis_id: str, deadline_s: int) -> dict:
    """AIP-151-style poll with AWS Full Jitter (contract R8)."""
    t0 = time.monotonic()
    attempt = 0
    while True:
        body = http_json("GET", f"{base}/api/analysis/{analysis_id}")
        if body.get("status") in ("completed", "failed"):
            return body
        if time.monotonic() - t0 > deadline_s:
            raise TimeoutError(f"analysis {analysis_id} not terminal after {deadline_s}s")
        time.sleep(random.uniform(0, min(30.0, 1.0 * 2 ** attempt)))
        attempt += 1


def run_live(args, verdicts: Verdicts) -> int:
    base = args.backend_url.rstrip("/")
    source = HttpLogSource(args.llm_log_source) if args.llm_log_source != "bigquery" \
        else BigQueryLogSource()

    pre = http_json("GET", f"{base}/api/settings/")
    pre_flag = bool(pre.get(FLAG))
    print(json.dumps({"event": "pre_flag", FLAG: pre_flag}), flush=True)

    spend = HttpSpendSource(args.spend_source) if args.spend_source != "bigquery" \
        else BigQuerySpendSource()
    spend_before = spend.daily_usd()
    print(json.dumps({"event": "spend_bracket_before", "daily_usd": spend_before}), flush=True)

    cap = args.max_rail_calls
    abort = threading.Event()
    since = datetime.now(timezone.utc) - timedelta(seconds=5)

    def watcher():
        while not abort.is_set():
            time.sleep(args.watch_interval)
            try:
                n = sum(1 for r in source.rows(since, datetime.now(timezone.utc))
                        if is_rail_row(r))
            except Exception:
                continue  # best-effort; authoritative count is post-window
            if n > cap:
                print(json.dumps({"event": "budget_cap_tripped", "observed": n,
                                  "cap": cap}), flush=True)
                abort.set()
                return

    with contextlib.ExitStack() as stack:
        # Restore is ALWAYS a PUT (persists to .env); registered before the flip
        # so any later crash unwinds through it (criterion 6).
        stack.callback(lambda: http_json("PUT", f"{base}/api/settings/", {FLAG: pre_flag}))
        http_json("PUT", f"{base}/api/settings/", {FLAG: True})
        print(json.dumps({"event": "flag_put", FLAG: True}), flush=True)

        wt = threading.Thread(target=watcher, daemon=True)
        wt.start()

        results = []
        used_estimate = 0
        for ticker in args.tickers:
            used_estimate += args.expected_calls_per_analysis
            if used_estimate > cap:
                print(json.dumps({"event": "budget_cap_refusal",
                                  "message": f"expected rail calls {used_estimate} would exceed "
                                             f"the <=%d-rail-call cap; refusing to start {ticker}" % cap,
                                  "cap": cap}), flush=True)
                abort.set()
                return 3
            start = http_json("POST", f"{base}/api/analysis/", {"ticker": ticker})
            aid = start["analysis_id"]
            print(json.dumps({"event": "analysis_started", "ticker": ticker,
                              "analysis_id": aid}), flush=True)
            body = poll_analysis(base, aid, args.analysis_deadline)
            results.append((ticker, body))
            if abort.is_set():
                print(json.dumps({"event": "aborted_on_cap",
                                  "note": "observer aborted; in-flight backend calls "
                                          "continue (no cancel endpoint)"}), flush=True)
                return 3

        abort.set()
        print(json.dumps({"event": "flush_wait", "seconds": args.flush_wait_s}), flush=True)
        time.sleep(args.flush_wait_s)
        until = datetime.now(timezone.utc) + timedelta(seconds=5)
        rows = source.rows(since, until)
        rail = [r for r in rows if is_rail_row(r)]

        # AUTHORITATIVE budget gate (contract R3): the watcher is best-effort
        # early-abort only; correctness of the <=cap guarantee lives HERE, on
        # the post-flush count, with no timing dependence.
        if len(rail) > cap:
            print(json.dumps({"event": "budget_cap_exceeded_postwindow",
                              "observed": len(rail), "cap": cap,
                              "message": f"window used {len(rail)} rail calls, over the "
                                         f"<={cap}-rail-call cap"}), flush=True)
            return 3

        claude_rows = [r for r in rows if is_claude_role_row(r)]
        # No model clause here: the frozen baseline section (c) metered-complement
        # rule is provider/agent-only, and any metered hit is a fail regardless
        # of which model it billed.
        metered = [r for r in rows if is_metered_row(r)]

        # E1: N-of-N claude-role rows match the rail rule; N > 0.
        n, n_rail = len(claude_rows), sum(1 for r in claude_rows if is_rail_row(r))
        verdicts.emit("E1", n > 0 and n == n_rail, matched=n_rail, claude_role_rows=n,
                      rule="provider='claude-code' OR agent='cc_rail' OR agent LIKE 'cc_rail:%'")

        # E2: BOTH frozen-baseline conjuncts -- zero metered-complement rows AND
        # zero spend-metric delta (production exclusion logic) -- with the rail
        # positive control (contract R6). Q/A 4000.2 cycle-1 finding 1.
        spend_after = spend.daily_usd()
        spend_delta = spend_after - spend_before
        verdicts.emit("E2", len(metered) == 0 and spend_delta == 0.0 and len(rail) > 0,
                      metered_rows=len(metered), rail_rows_positive_control=len(rail),
                      spend_before=spend_before, spend_after=spend_after,
                      spend_delta=spend_delta,
                      fail_open_note="fetch_llm_spend fails open to 0.0; a 0.0/0.0 "
                                     "bracket cannot distinguish dark from fetch-failure "
                                     "-- the row-count leg is primary")

        # E3a probe envelope truth (raw-map iteration).
        if args.no_probe:
            verdicts.emit("E3", False, detail="probe disabled but E3a requires it in live mode")
        else:
            binary = resolve_binary(args.claude_binary)
            env = run_probe(binary, args.probe_model)
            ok3a, d3a = probe_envelope_check(env, args.probe_model)
            # E3b: window row models within the configured expectation set.
            expected = {m for m in (pre.get("gemini_model"), pre.get("deep_think_model"))
                        if str(m or "").startswith("claude")}
            expected |= {"claude-haiku-4-5", "claude-sonnet-4-6"}  # overlay + lite pins (79.55)
            stray = sorted({str(r.get("model")) for r in rail
                            if not any(str(r.get("model") or "").startswith(e) for e in expected)})
            verdicts.emit("E3", ok3a and not stray, probe=d3a, stray_models=stray,
                          expected_set=sorted(expected))

        # E4: frozen baseline (f) legs 1-3 -- terminal-completed, schema-compliant
        # non-absent report, and NO synthetic 0.0/HOLD shape (the 61.2 defect
        # class). Leg 4 (persisted row) is DISCLOSED as not provable here and
        # queued (Q/A 4000.2 cycle-2 finding 1).
        bad = []
        for ticker, body in results:
            report = body.get("report") or {}
            rec_raw = report.get("recommendation")
            rec = str(rec_raw.get("action") if isinstance(rec_raw, dict) else rec_raw or "")
            score = report.get("final_weighted_score", report.get("final_score"))
            if body.get("status") != "completed" or body.get("error"):
                bad.append((ticker, body.get("status"), str(body.get("error"))[:120]))
            elif not report:
                bad.append((ticker, "completed", "report absent (degraded/unparseable synthesis)"))
            elif score == 0.0 and rec.lower() == "hold":
                bad.append((ticker, "completed", f"synthetic {score}/{rec} shape (61.2 defect class)"))
        verdicts.emit("E4", not bad, failures=bad, analyses=len(results),
                      persisted_row_leg="NOT PROVABLE HERE: the sync poll reads the in-memory "
                                        "task dict (analysis.py:392), which is not persistence "
                                        "evidence -- frozen-baseline E4 leg 4 is queued as its "
                                        "own step and owed as 4000.3 live_check evidence")

        # E5 surrogate predicate (contract R5; direct guard read impossible).
        failed_rail = [r for r in rail if r.get("ok") is False]
        skipped_markers = sum("rail_guard_skipped:" in json.dumps(body, default=str)
                              for _, body in results)
        verdicts.emit("E5", not failed_rail and skipped_markers == 0,
                      failed_rail_rows=len(failed_rail), rail_guard_skipped_markers=skipped_markers,
                      note="breaker P1 page not observable from here; "
                           "direct consecutive_failures read has no out-of-process surface (gap queued as 4000.8)")

        # E6: report-shaped, never a failing leg in 4000.2 (contract).
        verdicts.emit("E6", True, **e6_report(rows, len(results), args.cycles_per_day))

        if verdicts.all_ok and args.keep_on:
            stack.pop_all()  # cancel the restore -- operator-authorized keep-ON
            print(json.dumps({"event": "keep_on", "restore_cancelled": True}), flush=True)

    print(json.dumps({"summary": True, "mode": "live",
                      "all_ok": verdicts.all_ok,
                      "checks": [(c["check"], c["ok"]) for c in verdicts.checks]}), flush=True)
    return 0 if verdicts.all_ok else 1


# ── dry mode ─────────────────────────────────────────────────────────────────
def run_dry(args, verdicts: Verdicts) -> int:
    base = args.backend_url.rstrip("/")
    health = http_json("GET", f"{base}/api/health")
    settings = http_json("GET", f"{base}/api/settings/")
    flag = bool(settings.get(FLAG))
    binary = resolve_binary(args.claude_binary)
    print(json.dumps({"event": "preflight", "health": health.get("status"),
                      "binary": binary, FLAG: flag,
                      "rail_state": "ON" if flag else "OFF"}), flush=True)

    probe_detail = "skipped (--no-probe)"
    if not args.no_probe:
        env = run_probe(binary, args.probe_model)
        ok, probe_detail = probe_envelope_check(env, args.probe_model)
        if not ok:
            print(json.dumps({"event": "probe_failed", "detail": probe_detail}), flush=True)
            return 2

    estimate = None
    if args.llm_log_source == "bigquery":
        try:
            src = BigQueryLogSource()
            now = datetime.now(timezone.utc)
            rows = src.rows(now - timedelta(days=7), now)
            per_bucket: dict[tuple, int] = {}
            for r in rows:
                if is_rail_row(r) and r.get("ticker"):
                    ts = r.get("ts")
                    hour = str(ts)[:13]
                    per_bucket[(r["ticker"], hour)] = per_bucket.get((r["ticker"], hour), 0) + 1
            counts = sorted(per_bucket.values())
            estimate = counts[len(counts) // 2] if counts else 0
        except Exception as e:
            estimate = f"unavailable: {type(e).__name__}"
    print(json.dumps({"summary": True, "mode": "dry", "rail_state": "ON" if flag else "OFF",
                      "flag_mutations": 0, "probe": probe_detail,
                      "per_analysis_rail_call_ESTIMATE_from_7d_history": estimate,
                      "estimate_note": "ticker+hour bucket median; the REAL count is "
                                       "measured in the first 4000.3 window"}), flush=True)
    return 0


# phase-86.6: ONE definition of "the live backend", imported rather than
# re-spelled. See scripts/qa/live_backend_origin.py for why it is a separate
# module and how drift against conftest's copy is alarmed.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from live_backend_origin import (  # noqa: E402
    MUTATING_VERBS,
    install_socket_guard,
    is_live_backend,
    mutating_scope,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--live", action="store_true")
    p.add_argument("--ticker", dest="tickers", action="append", default=[])
    # phase-86.6 Part B: NO DEFAULT. This used to default to
    # "http://localhost:8000" -- the operator's live backend -- and this script
    # MUTATES (PUT /api/settings/, POST /api/analysis/). A caller that forgot
    # the flag would have flipped a live setting. All 12 call sites happened to
    # pass it, so the risk was latent; the defect is the FUTURE 13th.
    p.add_argument("--backend-url", default=None,
                   help="REQUIRED. No default: see --allow-live-backend.")
    p.add_argument("--allow-live-backend", action="store_true",
                   help="opt in to targeting the operator's RUNNING backend "
                        "(loopback:8000). Required for a real --live window; "
                        "without it that target is refused.")
    p.add_argument("--llm-log-source", default="bigquery",
                   help="'bigquery' or an http URL serving canned rows (test seam)")
    p.add_argument("--spend-source", default="bigquery",
                   help="'bigquery' (production fetch_llm_spend) or an http URL "
                        "returning {'daily_usd': x} (test seam)")
    p.add_argument("--claude-binary", default=None,
                   help="explicit path; omitted => imported production resolution")
    p.add_argument("--probe-model", default="claude-haiku-4-5")
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--max-rail-calls", type=int, default=DEFAULT_MAX_RAIL_CALLS)
    p.add_argument("--expected-calls-per-analysis", type=int, default=None)
    p.add_argument("--cycles-per-day", type=float, default=1.0)
    p.add_argument("--analysis-deadline", type=int, default=1800)
    p.add_argument("--watch-interval", type=int, default=20)
    p.add_argument("--flush-wait-s", type=int, default=FLUSH_WAIT_S,
                   help="post-window llm_call_log buffer wait; test seam, default 65")
    p.add_argument("--keep-on", action="store_true")
    args = p.parse_args()

    # ── phase-86.6 Part B: the subprocess channel ────────────────────────
    # A conftest guard lives in the pytest process; this script runs in a
    # CHILD, which loads no conftest. So the guard has to be here, at the
    # seam every invocation must pass through, rather than at each call site
    # -- a per-call-site convention is regressed by simply writing a new one.
    if args.backend_url is None:
        print("usage error: --backend-url is REQUIRED and has no default. "
              "This script mutates (PUT /api/settings/, POST /api/analysis/); "
              "a default of http://localhost:8000 would aim those at the "
              "operator's running backend. Pass a stub URL for tests, or "
              "--backend-url http://localhost:8000 --allow-live-backend for a "
              "real window.", file=sys.stderr)
        return 4
    if is_live_backend(args.backend_url) and not args.allow_live_backend:
        print(f"refusing to target the LIVE backend at {args.backend_url} "
              f"without --allow-live-backend. This script mutates settings and "
              f"starts analyses; pointing it at the operator's running book by "
              f"accident is the failure this guard exists to prevent. Re-run "
              f"with --allow-live-backend if that is genuinely intended.",
              file=sys.stderr)
        return 4

    # ── phase-86.27: the string check above is BEST EFFORT ────────────────
    # `is_live_backend` reads `urlsplit().hostname`, and urllib does NOT hand
    # the socket that string unchanged -- measured, `urlsplit` reports
    # `%31%32%37%2e%30%2e%30%2e%31` while urllib connects to 127.0.0.1:8000.
    # So the check above can be walked past by a spelling it cannot resolve,
    # and this child process loads no conftest, so nothing else covers it.
    # Install the AUTHORITATIVE guard, which decides on the address the socket
    # is actually about to use. Skipped only under the explicit opt-in, which
    # is the one case where reaching the live backend is the point.
    if not args.allow_live_backend:
        install_socket_guard()

    if args.max_rail_calls > DEFAULT_MAX_RAIL_CALLS:
        print(f"usage error: --max-rail-calls may only lower the <= "
              f"{DEFAULT_MAX_RAIL_CALLS}-rail-call cap, never raise it", file=sys.stderr)
        return 4
    if args.live:
        if not args.tickers:
            print("usage error: --live requires at least one --ticker", file=sys.stderr)
            return 4
        if len(args.tickers) > MAX_TICKERS:
            print(f"usage error: at most {MAX_TICKERS} tickers per window "
                  f"(the <= {MAX_TICKERS}-ticker cap)", file=sys.stderr)
            return 4
        if args.expected_calls_per_analysis is None:
            print("usage error: --live requires --expected-calls-per-analysis "
                  "(measure it with --dry first; 4000.1 baseline section (e))", file=sys.stderr)
            return 4

    verdicts = Verdicts()
    try:
        return run_dry(args, verdicts) if args.dry else run_live(args, verdicts)
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as e:
        print(json.dumps({"event": "preflight_or_window_error",
                          "error": f"{type(e).__name__}: {e}"}), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
