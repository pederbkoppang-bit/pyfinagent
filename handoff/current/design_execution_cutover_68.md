# Design: Execution Cutover (phase-68 Real-Fill Runway)

Authored 2026-07-10 by Main (Fable 5) from research_brief_68.0.md (13 full reads, 48
URLs, 19 internal files; every file:line below is anchored in the brief). Governs
68.1 (wiring), 68.2 (shadow), 68.3 (cutover). NO code changes in this step.

## 1. Config precedence and propagation (68.1)

Resolution chain (first hit wins), resolved AT CONSTRUCTOR TIME (execution_router.py
:65-71 reads os.getenv per-construction, :268-269 -- so a plist env + process reload
is sufficient; the "import time" docstring there is stale and 68.1 fixes the comment):

1. `os.environ["EXECUTION_BACKEND"]` -- the launchd plist EnvironmentVariables block
   (precedent: the 2026-07-08 setup-token wiring added keys to 4 plists + bootout/
   bootstrap reload; today the backend plist carries only 4 keys, none execution-
   related).
2. NEW `settings.execution_backend` pydantic field (backend/config/settings.py) --
   the .env channel. pydantic env_file loads into the settings MODEL ONLY (never
   exports to os.environ), which is exactly why the current router never sees .env
   values: the router reads os.getenv, not settings. 68.1 bridges by having the
   router consult settings as the second link, NOT by exporting env.
3. Default: `bq_sim` -- byte-identical behavior when nothing is set (68.1 immutable
   criterion).

Startup observability (68.1): ONE unmissable log line at router construction:
`execution_backend=<mode> source=<env|settings|default>`. If mode is alpaca_paper
and creds are absent: a LOUD single startup error naming the exact missing keys --
never a silent fall to `_alpaca_mock_fill`.

Creds: the router reads os.environ today; settings carries SecretStr fields for the
news-channel Alpaca keys. If 68.1 unifies on settings, the unwrap_secret trap applies
(SecretStr is truthy; never `or ""` it -- project_secretstr memory). OPERATOR-VERIFY
ITEM: backend/.env is agent-locked; the operator confirms which ALPACA_* keys it
holds before 68.2 (the design does not assume).

## 2. Shadow-mode isolation (68.2)

Authority: bq_sim remains the ONLY writer of position state. The existing structure
is already shaped right (brief: alpaca leg wrapped in try/except and swallowed);
what is missing is PERSISTENCE of the paired fill -- today it is discarded.

- Shadow leg = fire-and-forget: submit the alpaca-paper order AFTER the bq_sim fill
  is booked; any shadow exception is caught, logged, and NEVER fails the cycle.
- Paired-fill persistence: NEW table `paper_shadow_fills` created via
  scripts/migrations/ (house rule: migrations, not ad-hoc). Do NOT add a source
  column to paper_trades in 68.2 -- its dynamic INSERT rejects unknown keys and the
  table is live; schema change belongs to 68.3's cutover commit if needed.
- Join key: `client_order_id == trade_id` (see §3) links each shadow fill to its
  bq_sim decision row. Drift report = join on trade_id; deltas on fill price + time.
- Invariant tests (68.2 criteria): bq_sim book byte-identical with shadow on/off;
  shadow outage (bad creds, network) leaves the cycle green.
- PRE-STEP: flatten the stray shorts (-13,842.89, the 2026-06-10 MCP-drill artifact)
  with `close_all_positions(cancel_orders=True)`. NEVER use account reset -- reset
  INVALIDATES the API keys (brief, alpaca docs). Token: ALPACA-RESET: APPROVED.
- PDT: paper accounts <$25k are PDT-constrained. The engine trades a few orders/day
  max; low risk, but the drift report should count any PDT rejections explicitly.

## 3. Order-id idempotency and the fill lifecycle (68.2/68.3)

- `client_order_id = trade_id` (<=128 chars, fits). Alpaca enforces uniqueness
  against ACTIVE orders: duplicate submit -> 422 "client_order_id must be unique".
  Recovery: on 422 or timeout-ambiguity, `get_order_by_client_id(trade_id)` -- never
  blind-resubmit. This makes retries idempotent end-to-end.
- Lifecycle poll: accepted -> filled|rejected|canceled.
- **CUTOVER TRAP (must fix before 68.3 flips):** today's fill-poll
  (execution_router.py:239-244) treats rejected/canceled orders as "use reference
  price" and paper_trader.py:260 BOOKS it -- fine for a mock, catastrophic for real
  execution (an unfilled order would enter the book). In alpaca_paper mode:
  rejected/canceled/unfilled => NO book entry; log + skip + reconcile next cycle.
  This lands dark in 68.1 (gated on mode) so 68.3 inherits it tested.
- Reconciliation (68.3): every alpaca order maps to exactly one decision row and
  vice versa (zero orphans); price drift vs bq_sim expectation <2%; any mismatch
  pages P1.

## 4. Rollback = single env flip (68.3 drill)

Set `EXECUTION_BACKEND=bq_sim` (or remove the key) in the plist + `launchctl
kickstart -k` the backend; the next scheduled cycle runs fully on bq_sim. bq_sim is
never removed -- it stays the compiled-in default forever. The 68.3 drill: flip
back, one clean bq_sim cycle, flip forward; all three states evidenced in
live_check_68.3.md.

## 5. Paper-only guards -- kept, named, strengthened (68.1)

Kept (named, from the brief): the existing try/except mock-fill isolation; the
settings-level paper flags; the in-session MCP deny-list on all alpaca trade-mutation
tools (settings.json deny block -- unrelated to the backend path but part of the
fence).

STRENGTHENED -- the "PKLIVE prefix" assumption is folklore (live keys reputedly
"AK"-prefixed; not documented): replace prefix-blocklist thinking with
triple-enforcement, each independently sufficient:
(a) `TradingClient(paper=True)` pinned as a code constant -- not configurable;
(b) key-shape allowlist: reject any key NOT matching the paper "PK" shape with a
    LOUD startup error (allowlist, not blocklist);
(c) no code path constructs a live client: no live base URL string exists in the
    repo; any `paper=False` literal is test-only and the 68.1 test suite asserts the
    production path cannot reach one.

## 6. Sequencing consequences from the research (register)

- **68.5 premise partially overturned -- OPERATOR DECISION PENDING:** AMD/MU fills
  were REAL prices (AMD close $546.72 / MU $991.64 on 2026-07-09; MU ATH $1213.37 on
  06-25). Criteria 1-2 ("root-cause the defect", "correct the rows") are
  unsatisfiable as written -- correcting a CORRECT book would corrupt it; criterion
  4 (DESC phantom) was already fixed (9262ed36 + regression tests). Still fully
  valid from 68.5: the pre-persist fill-price sanity gate targeting the REAL holes
  (unguarded SELL fill price with hash-synthetic fallback; tolerance gate fails open
  without price_at_analysis), FX-1 root-cause handoff to parked 61.3, and the 63.3
  defect seeds. The sanity gate's independent quote MUST NOT be yfinance-vs-yfinance
  (the stale-anchor class that produced the false premise is the cautionary tale).
- The live_check_66.2.md:402 "~$150/~$110" line is a recorded-evidence error; do not
  edit the archived artifact -- this doc + the 68.0 log entry are the correction of
  record.
- DON'T-RE-FIX list re-verified standing (alerting imports; cc-rail guard).
