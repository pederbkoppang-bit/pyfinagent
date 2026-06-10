# UX / Control-Surface Roadmap (2026-05-28)

Source: `ux-control-surface-audit` workflow (4 agents) vs `.claude/rules/frontend.md` +
`frontend-layout.md`. Governs priority-7 of `active_goal.md`. Companion: `roadmap_master.md`.

## State of the UX
Structurally sound (20/20 routes use the canonical flex shell except `/login`; Phosphor-only;
no emojis), but (a) the design system exists on paper (tokens in tailwind.config.js +
globals.css) yet isn't enforced — text colors / hover / focus / button classes hand-composed
across ~120 files; the Framer Motion preset lib `@/lib/motion.ts` (v12.38.0, 6 presets) is
ORPHANED (zero imports); and (b) the operator lacks "full control" — **promote paper->live is
a disabled button wired to nothing with no backend**, plus stop-scheduler / position-close /
manual-order / per-job control / deposit gaps.

Page offenders: `/performance` uses Tremor not Recharts (HIGH); `/learnings` missing Tier-1
fixed header (WARN); `/paper-trading/exit-quality` no loading/error/empty wrapper (WARN);
`/reports` 3x inline `style={{backgroundColor}}` (WARN); `EmptyState.tsx` zinc not navy/slate (WARN).

## Target design system
- Shell: keep canonical two-zone (`flex h-screen` + flex-shrink-0 header / overflow-y-auto scrollbar-thin body). Fix `/learnings`.
- NEW `@/lib/design-tokens.ts`: semantic maps — text (slate-100 primary..slate-500 dim), surface
  (navy-800/70 card..), hover (navy-700/40 row), ONE focus ring
  (`focus-visible:ring-2 ring-sky-400 ring-offset-2 ring-offset-navy-800`), transition (200/300/150),
  status (emerald/amber/rose/slate /15 + /300 text). navy/slate only, JIT-safe literals.
- NEW components: `ui/Button` (primary|secondary|ghost|danger, bakes focus+whileTap; replaces 30+
  inline strings; danger = Pause/Flatten/Stop), `ui/StatusBadge`, `ui/TextInput`+`ui/Select`,
  `ConfirmActionModal` (typed-confirm "FLATTEN_ALL"/"PROMOTE_LIVE", reused by all destructive controls).
  Migrate ~50 card divs to existing `BentoCard` (+isHoverable/isPressable). Fix `EmptyState` zinc->slate.
- Animation: adopt `@/lib/motion.ts` (motion/react for chrome; CSS @keyframes stay for charts;
  honor prefers-reduced-motion). Page/tab fade+slide 200-300ms; drawer/modal slide-up 300ms spring;
  stagger 40ms; button whileTap 0.95 100ms; hover lift 1.02 150ms; kill-switch ACTIVE pulse 2s;
  gate-check stagger 50ms; keep NumberFlow (don't double-animate); enable Recharts isAnimationActive
  (currently false RedLineMonitor.tsx:185). Timing: micro100/short200/standard300/long500/macro2-3s.

## Operator full-control gaps (ranked)
| # | Control | Backend | UI | Priority |
|---|---|---|---|---|
| 1 | Promote paper->live capital | NEEDS BUILDING | disabled button, onClick undefined | **P0 BLOCKING** |
| 2 | Stop scheduler | POST /api/paper-trading/stop EXISTS | none | P1 |
| 3 | Manual position close | mock only (portfolio.py DELETE not wired) | none | P1 |
| 4 | Manual order entry | needs building | none | P1 |
| 5 | Per-cron-job control (pause/resume/trigger) | only GET /api/jobs/all | read-only | P1 |
| 6 | Deposit funds | POST /api/paper-trading/deposit EXISTS | none | P2 |
| 7 | In-cockpit risk-limit override | PUT /api/settings EXISTS | Settings page only | P2 |
| 8 | Alerts/breach-history + muting | needs event store | none | P2 |
| 9-11 | run-now override params / bulk attribution export / live cycle SSE | partial | partial | P3 |
Already wired: Pause/Resume/Flatten-All (confirm-gated), gate read, run-now, kill-switch read, jobs read, log tail, settings.

## UX roadmap (dependency order W1 -> W2 -> (W3,W4) -> W5 -> W6 -> W7 -> W8)
- W1 [P0] Promote paper->live: build `POST /api/paper-trading/promote` (typed-confirm PROMOTE_LIVE,
  gate-pass precondition -> 409 if not all 5 green, audit row); wire GoLiveGateWidget onPromote ->
  ConfirmActionModal. SAFETY-CRITICAL, blocking. (M)
- W2 Design-system extraction: design-tokens.ts + ui/Button + ui/StatusBadge + ui/TextInput + ui/Select;
  EmptyState zinc->slate. Built before cosmetic work so pages inherit. (M)
- W3 [P1] money/safety controls: stop-scheduler toggle; manual position-close (real backend); manual
  order-entry; all via ConfirmActionModal. (L)
- W4 page-consistency: /performance Tremor->Recharts; /learnings Tier-1 header; /exit-quality states;
  /reports inline-style->JIT map. (S-M)
- W5 component migration to tokens (~120 files; ~50 cards -> BentoCard). (L, depends W2)
- W6 animation rollout (adopt motion.ts; enable chart anim). (M, depends W2/W5)
- W7 [P2] deposit form; in-cockpit risk override; alerts/breach page + muting. (M-L)
- W8 [P3] per-job control; run-now overrides; bulk export; live cycle SSE. (defer)

All are NEW masterplan steps (none map to existing phases). W1 + W2 land first.

## First UX step (when priority-7 cycles begin)
phase-XX "Paper->Live promotion control (P0)" = W1. Establishes ConfirmActionModal that W3 depends on.
Verify: ast.parse paper_trading.py; grep promote endpoint; grep onPromote in GoLiveGateWidget; no
direct @phosphor import in ConfirmActionModal; `cd frontend && npm run build`.
Key files: GoLiveGateWidget.tsx, backend/api/paper_trading.py, new design-tokens.ts + components/ui/,
motion.ts (adopt), performance/learnings/exit-quality/reports pages, EmptyState.tsx, RedLineMonitor.tsx:185.
