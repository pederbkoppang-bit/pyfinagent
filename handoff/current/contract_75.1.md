# Contract — Step 75.1: Audit75 S1 — backend auth surface fail-closed

- **Step id:** 75.1 (phase-75, P0, executor: sonnet-4.6/high — run this cycle on the Fable rail per operator budget directive 2026-07-20)
- **Date:** 2026-07-20
- **Boundary:** no `.env` edits — allowlist enforcement ships DARK behind default-OFF flag `auth_enforce_allowlist`.
- **Findings remediated:** security-01 (P1), security-03 (P2), security-04 (P2), gap2-03 (P2), api-design-12 (P3), pysvc-08 (P3)

## Research-gate summary

Gate PASSED (moderate tier, envelope in `handoff/current/research_brief_75.1.md`): 6 official/canonical sources read in full (FastAPI conditional-OpenAPI, FastAPI CORS, Tailscale CGNAT KB / RFC 6598, OWASP Fail Securely, FastAPI Path() reference, Pydantic v2 Literal), 40 URLs across the 3-variant query discipline, recency scan performed (regex=→pattern= rename settled ≥FastAPI 0.100; no superseding patterns). Internal audit line-anchored all six findings across 16 files; consumer grep exhaustive. Verified: the proposed CGNAT regex accepts exactly octets 64..127 (100.64.0.0/10). Key risk is NOT code: immutable masterplan curls + `smoke_test_4_17_6.py` + the Slack bot's existing non-public calls all ride the `DEV_LOCALHOST_BYPASS` rail (auth.py:150-153) — the GENERATE must live-probe and document that rail, never claim "nothing breaks" without it.

## Hypothesis

Pruning the 8 drifted prefixes from `_PUBLIC_PATHS`, gating docs on `settings.debug`, sharing one compiled CGNAT origin-predicate across both CORS seams, adding the DARK `auth_enforce_allowlist` flag + startup WARNING, and adding POST validation (Path pattern + Literal) closes all six audit findings **without breaking any live consumer**, because every frontend caller of the affected prefixes already sends credentials via `apiFetch`, the Slack bot calls none of them, and localhost tooling rides the pre-existing `DEV_LOCALHOST_BYPASS` rail.

## Immutable success criteria (verbatim from .claude/masterplan.json step 75.1)

1. _PUBLIC_PATHS in backend/main.py contains none of: /api/harness/monthly-approval, /api/sovereign, /api/signals, /api/observability, /api/cost-budget, /docs, /openapi.json, /redoc; every remaining public prefix carries an inline justification and .claude/rules/security.md lists exactly the final set
2. CORS allow_origin_regex uses exactly the CGNAT block 100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+ and the permissive 100\.\d+\.\d+\.\d+ pattern plus the startswith('http://100.') 401-echo shortcut are gone -- one shared origin predicate for both seams
3. Every HTTP consumer of a newly-authed prefix is enumerated by grep in experiment_results.md and each either sends credentials or was moved to an explicitly-public sub-route -- no frontend page or slack_bot caller left silently 401ing (curl-level evidence for at least /api/sovereign and /api/signals consumers)
4. monthly-approval POST: non-^\d{4}-\d{2}$ month_key and non-Literal action both return 422; an invalid action can no longer produce HTTP 200 with status 'rejected'
5. allowed_emails empty at startup emits WARNING; auth_enforce_allowlist defaults False (byte-identical today, executor edits no .env); with the flag True an empty allowlist rejects all authenticated users
6. python -c 'import ast; ast.parse(...)' passes on every touched backend file

## Plan steps (shapes per research brief §Application)

1. **(a)+(b) prune** — remove the 8 target entries from `_PUBLIC_PATHS` (main.py:406-423); add inline justification comments on each survivor (`/api/health`, `/api/changelog`, `/api/auth`, `/api/jobs/status`, 4 read-only harness dashboards). No other entries touched (pitfall 9: scope guard).
2. **(b) docs gating** — `FastAPI(..., docs_url=... if debug, redoc_url=..., openapi_url=... if debug else None)` driven by `get_settings().debug` (settings.py:20). `openapi_url=None` cascades 404 to /docs+/redoc.
3. **(c) shared predicate** — module-level `_TAILSCALE_ORIGIN_RE = re.compile(r"^http://(localhost|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+):\d+$")`; `allow_origin_regex=_TAILSCALE_ORIGIN_RE.pattern`; 401-echo uses `_TAILSCALE_ORIGIN_RE.match(origin)` (replaces both startswith checks). Intended tightening: echo now requires an explicit `:port` (browser origins always carry it here).
4. **(d) DARK flag** — `auth_enforce_allowlist: bool = Field(False, ...)` in settings.py Authentication section; auth.py allowlist leg becomes: empty+flag-True → 401 reject-all; empty+flag-False → today's fail-open (byte-identical); non-empty → unchanged membership check. Startup WARNING for empty allowlist emitted ONCE in `main.py::lifespan` (ASCII-only), NOT a model_validator (Settings multi-instantiation, pitfall 4).
5. **(e) POST validation** — `month_key: str = Path(pattern=r"^\d{4}-\d{2}$")` + `ApprovalActionBody.action: Literal["approved","rejected"]` (monthly_approval_api.py:58-59, 184-193). Intended tightening: mixed-case actions now 422 (no repo caller sends them; no POST callers exist in-repo).
6. **Doc rewrite** — `.claude/rules/security.md` auth section lists exactly the 8-entry post-prune public set with per-prefix justification.
7. **Consumer evidence** — grep-enumerate all consumers in experiment_results.md; curl-level evidence for /api/sovereign + /api/signals consumers; live-probe the `DEV_LOCALHOST_BYPASS` state (tokenless localhost curl to a non-public endpoint, record 200-bypass vs 401) and state consequences for smoke_test_4_17_6.py + immutable masterplan curls.
8. **Verify** — run the step's immutable verification command (exit 0) + ast.parse on every touched backend file; write experiment_results_75.1.md + live_check_75.1.md; spawn Q/A (Workflow structured-output launch); log-last; flip.

## References

- `handoff/current/research_brief_75.1.md` (source tables: FastAPI docs conditional-openapi + CORS + Path reference, Tailscale KB CGNAT, OWASP Fail Securely, Pydantic Literal; 9 pitfalls; exact shapes)
- `handoff/current/audit_phase75/register.md` — security-01/03/04, gap2-03, api-design-12, pysvc-08
- `.claude/masterplan.json` step 75.1 (immutable criteria + verification command)
