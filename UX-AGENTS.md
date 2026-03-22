# UX-agent.md — Comprehensive Report Dashboard

This document describes the UX architecture, component inventory, data flow, and design conventions for the PyFinAgent comprehensive analysis report dashboard. It serves as a reference for AI agents and developers working on the frontend.

---

## Design Philosophy

The report dashboard follows a **"Glass Box" institutional analyst** design — every data point, agent decision, and recommendation is traceable back to its originating source. The visual language is benchmarked against Bloomberg Terminal, Morningstar, Goldman Sachs, and FactSet.

**Core UX Principles:**

| Principle | Implementation |
|-----------|---------------|
| **Source Provenance** | Every signal card, evidence citation, and metric displays the originating data source (SEC EDGAR, yfinance, Alpha Vantage, FRED, USPTO, Vertex AI) |
| **Glass Box Transparency** | Full decision audit trail — every agent's inputs, reasoning steps, confidence, and output are visible in the Audit tab |
| **Progressive Disclosure** | Dashboard uses expandable cards and a 5-tab layout so users see the summary first, then drill into details on demand |
| **Visual Hierarchy** | Bloomberg-style hero header → tabbed sections → expandable card grids — information density scales with user intent |

---

## Workspace Environment

The frontend and backend share one VS Code workspace, but the backend Python tooling is standardized on `.venv312`.

| Concern | Canonical Setting |
|--------|-------------------|
| Backend interpreter | `.venv312/Scripts/python.exe` |
| VS Code Python setting | `.vscode/settings.json` → `python.defaultInterpreterPath` |
| Pyright/Pylance env | `pyrightconfig.json` → `venv = ".venv312"` |
| Backend debug config | `.vscode/launch.json` → `Backend: FastAPI (.venv312)` |
| Frontend debug config | `.vscode/launch.json` → `Frontend: Next.js` |
| Full-stack launch | `.vscode/launch.json` → `Full Stack: Backend + Frontend` |

Use this interpreter for backend imports, FastAPI startup, and editor diagnostics.

---

## Design System (v3.0)

### Typography
*   **Primary font:** Geist Sans (`geist/font/sans`) — self-hosted via `next/font`, applied as `var(--font-geist-sans)` CSS variable
*   **Mono font:** Geist Mono (`geist/font/mono`) — applied as `var(--font-geist-mono)` CSS variable
*   **Global:** `font-feature-settings: "tnum"` (tabular-nums for aligned financial numbers), `-webkit-font-smoothing: antialiased`

### Icons — Phosphor Icons
*   **Library:** `@phosphor-icons/react` v2.1.10 — 9000+ icons, 6 weights (thin/light/regular/bold/fill/duotone)
*   **Centralized exports:** All icons imported via `frontend/src/lib/icons.ts` (~120 aliased re-exports)
*   **TypeScript type:** `Icon` from `@phosphor-icons/react` used for all icon props in component interfaces
*   **Rendering pattern:** `<meta.icon size={20} weight="duotone" className="text-slate-400" />` (JSX component, not string interpolation)
*   **Active/inactive weight:** Tabs and nav use `weight={active ? "fill" : "regular"}` for state indication
*   **Domain-prefixed aliases:** `Nav*`, `Step*`, `Signal*`, `Debate*`, `Risk*`, `Bias*`, `Pillar*`, `Macro*`, `Settings*`, `Tab*`, `Icon*`
*   **Tree-shaking:** Enabled via `next.config.js` → `experimental.optimizePackageImports: ["@phosphor-icons/react"]`

### Animation (prepared, not yet applied)
*   **Library:** Motion (`motion/react`) v12.38.0
*   **Variants:** `fadeIn`, `slideUp`, `staggerContainer`, `staggerItem` in `frontend/src/lib/motion.ts`
*   **Spring presets:** `springSnappy` (stiffness: 400, damping: 30), `springGentle` (stiffness: 120, damping: 20)

### Tailwind Design Tokens

| Token | Value | Usage |
|-------|-------|-------|
| `navy-500` | `#243352` | Alternative dark surface |
| `navy-600` | `#1a2744` | Deeper dark surface |
| `shadow-card` | `0 1px 3px rgba(0,0,0,.12), 0 0 0 1px rgba(255,255,255,.04)` | Card elevation |
| `shadow-card-hover` | `0 4px 12px rgba(0,0,0,.2), 0 0 0 1px rgba(255,255,255,.06)` | Card hover state |
| `rounded-card` | `12px` | Card border radius |
| `rounded-button` | `8px` | Button border radius |
| `rounded-badge` | `6px` | Badge border radius |

---

## Page Layout & Tab Architecture

The main dashboard (`/`) renders in three visual layers:

```
┌──────────────────────────────────────────────────────────┐
│  Ticker Input Bar + "Analyze" Button                      │
├──────────────────────────────────────────────────────────┤
│  AnalysisProgress (15-step tracker, visible during run)   │
├──────────────────────────────────────────────────────────┤
│  ReportHeader (always visible when report exists)         │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Score Ring │ Action Badge │ Metrics │ 52w Range     │  │
│  └────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│  ReportTabs                                               │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┐ │
│  │Overview│Signals │ Debate │  Risk  │ Audit  │  Cost  │ │
│  │ [icon] │[icon]11│[icon]BUY│[icon]3 │[icon]1 │[icon]$ │ │
│  └────────┴────────┴────────┴────────┴────────┴────────┘ │
│                                                           │
│  [ Active Tab Content ]                                   │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Tab Definitions

| Tab ID | Label | Icon | Badge Source | Components Rendered |
|--------|-------|------|-------------|---------------------|
| `overview` | Overview | `TabOverview` (ClipboardText) | — | InvestmentThesisCard, EvaluationTable, ValuationRange, RisksCard, ScoringMatrixCard, PdfDownload |
| `signals` | Signals | `TabSignals` (Broadcast) | Count of enrichment signals (0–12) | SignalDashboard |
| `debate` | Debate | `TabDebate` (Scales) | Consensus label (e.g. "BUY") | DebateView |
| `risk` | Risk | `TabRisk` (Crosshair) | Anomaly count (e.g. "3 anomalies") | RiskDashboard, StockChart |
| `audit` | Audit | `TabAudit` (MagnifyingGlass) | Bias flag count (e.g. "1 flags") | BiasReport, DecisionTraceView, ResearchInvestigator |
| `cost` | Cost | `TabCost` (CurrencyDollar) | Total cost (e.g. "$0.12") | CostDashboard |

---

## Data Flow: Backend → Frontend

```
Orchestrator (15-step pipeline)
    │
    ▼
SynthesisReport (Pydantic model, backend/api/models.py)
    │
    ├── scoring_matrix          → EvaluationTable, ScoringMatrixCard
    ├── recommendation          → ReportHeader (action badge), InvestmentThesisCard
    ├── final_summary           → InvestmentThesisCard (multi-paragraph)
    ├── key_risks               → RisksCard
    ├── final_weighted_score    → ReportHeader (score ring)
    ├── enrichment_signals      → SignalDashboard (12 signal cards + consensus bar)
    ├── debate_result           → DebateView (bull/bear/DA + multi-round timeline)
    ├── decision_traces         → DecisionTraceView (XAI timeline)
    ├── risk_data               → RiskDashboard (Monte Carlo + anomalies)
    ├── risk_assessment         → RiskAssessmentPanel (judge verdict + analyst cards)
    ├── info_gap_report         → (data quality tracking, stored to BQ)
    ├── bias_report             → BiasReport (bias flags + raw/adjusted score)
    ├── conflict_report         → BiasReport (knowledge conflict table)
    └── cost_summary            → CostDashboard (per-agent token/cost breakdown)

Skill Optimizer (autonomous background loop)
    │
    ├── GET /api/skills/status       → Running status, experiment counts, keep rate
    ├── GET /api/skills/experiments  → Full experiment history (TSV rows)
    ├── GET /api/skills/analysis     → Summary stats, delta chain, running best, top hits
    ├── GET /api/skills/agents       → List of optimizable agents with skill file status
    ├── GET /api/skills/{agent}      → Agent's current skills.md content
    ├── POST /api/skills/optimize    → Start the optimization loop (background task)
    └── POST /api/skills/stop        → Stop the loop gracefully

Feedback Loop (wired in outcome_tracker.py)
    │
    evaluate_all_pending() → generate_reflection() → save_agent_memory()
    │  (4 agent types: bull, bear, moderator, risk_judge)
    └── Persists to BQ agent_memories table → BM25 retrieval in future prompts

Paper Trading (autonomous daily cycle)
    │
    ├── POST /api/paper-trading/start       → Initialize fund + start daily scheduler
    ├── POST /api/paper-trading/stop        → Pause the daily scheduler
    ├── GET /api/paper-trading/status       → NAV, P&L, scheduler state, loop status
    ├── GET /api/paper-trading/portfolio    → All open positions with unrealized P&L
    ├── GET /api/paper-trading/trades       → Trade history (buys/sells with reasons)
    ├── GET /api/paper-trading/snapshots    → Daily NAV history (for LineChart)
    ├── GET /api/paper-trading/performance  → Sharpe ratio, win rate, alpha, costs
    └── POST /api/paper-trading/run-now     → Trigger manual daily cycle (async)

Walk-Forward Backtesting (quant-only, $0 LLM cost, 5 strategies)
    │
    ├── POST /api/backtest/run              → Start walk-forward backtest (async, auto-ingests if BQ empty)
    ├── GET /api/backtest/status            → Poll backtest progress
    ├── GET /api/backtest/results           → Full results with per-window analytics
    ├── GET /api/backtest/results/{id}      → Per-window detail (trades, feature importance)
    ├── POST /api/backtest/ingest           → Bulk ingest historical data to BigQuery
    ├── GET /api/backtest/ingest/status     → Row counts for historical data tables
    ├── POST /api/backtest/optimize         → Start quant strategy optimizer (background, rotates 5 strategies)
    ├── POST /api/backtest/optimize/stop    → Stop optimizer gracefully
    ├── GET /api/backtest/optimize/status   → Optimizer state (iterations, best Sharpe, feature drift)
    ├── GET /api/backtest/optimize/experiments → Experiment history (quant_results.tsv + top5_mda)
    └── GET /api/backtest/optimize/best     → Best strategy params + feature importance

    Strategy Registry (5 strategies, configurable via QuantOptimizer):
    │  triple_barrier  — López de Prado Ch. 3 (default)
    │  quality_momentum — Asness et al. 2019 (6M momentum × quality_score)
    │  mean_reversion   — Lo & MacKinlay 1990 (SMA50 + RSI reversion)
    │  factor_model     — Fama-French 5-factor composite
    └  meta_label       — Secondary model layer (future)

MetaCoordinator (cross-loop sequencing, v4.3)
    │
    ├── gather_health()                     → Portfolio Sharpe, agent accuracy, API latency
    ├── decide(health)                      → Priority routing to QuantOpt/SkillOpt/PerfOpt/idle
    ├── update_mda_features(importances)    → Stores MDA from QuantOpt backtest
    └── _get_mda_target_agents()            → MDA→Agent bridge (top features → skill files)

PerformanceSkill (canonical metrics, v4.3)
    │  All P&L, Sharpe, benchmark, alpha, and scalar metrics are computed via
    │  backend/services/perf_metrics.py — single source of truth.
    │
    ├── compute_sharpe_from_snapshots()     → Paper trading Sharpe (risk-free adjusted)
    ├── compute_alpha()                     → Portfolio return − benchmark return
    ├── compute_position_pnl()              → Position-level unrealized P&L
    ├── get_scalar_metric()                 → THE unified optimization metric
    │       scalar = risk_adjusted_return × (1 − tx_cost_drag)
    └── beat_benchmark()                    → Geometric benchmark comparison

AnalysisStatusResponse (GET /api/analysis/{id}, polled every 3s)
    │
    ├── status                  → AnalysisProgress (running/completed/error)
    ├── progress                → AnalysisProgress (percentage bar)
    ├── current_step            → AnalysisProgress (active step highlight)
    ├── message                 → AnalysisProgress (live activity quote)
    └── step_log[]              → AnalysisProgress (expanding thought panel)
         ├── step               │  Step name
         ├── status             │  running / completed / error
         ├── message            │  Sub-step detail (e.g. "[3/11] Patent data collected")
         └── timestamp          │  ISO 8601 UTC

---

## Data Persistence: BigQuery (88-Column ML Schema)

After the orchestrator completes and the `SynthesisReport` is returned to the frontend, `backend/api/analysis.py` extracts quantitative features from the full report dict and persists an **88-column row** to BigQuery (`financial_reports.analysis_results`). This runs in the backend — the frontend does not interact with BigQuery directly. The same extraction logic is mirrored in `backend/tasks/analysis.py` for the Celery async path.

### Persistence Dataflow

```
Orchestrator (15-step pipeline)
    │
    ▼
Full report dict (in-memory Python dict with all agent outputs)
    │
    ▼
backend/api/analysis.py :: _run_sync_analysis()
    │
    ├── Phase 1: Financial Fundamentals ← report["quant"]["yf_data"]
    │   price_at_analysis, market_cap, pe_ratio, peg_ratio, debt_equity, sector, industry
    │
    ├── Phase 2: Risk Metrics ← report["risk_data"]["monte_carlo"] + ["anomalies"]
    │   annualized_volatility, var_95_6m, var_99_6m, expected_shortfall_6m, prob_positive_6m, anomaly_count
    │
    ├── Phase 3: Debate Dynamics ← report["debate"]
    │   bull_confidence, bear_confidence, bull_thesis, bear_thesis,
    │   contradiction_count, dissent_count, recommendation_confidence, key_risks
    │
    ├── Phase 4: Enrichment Signals ← report["insider"], ["options"], ["social_sentiment"], etc.
    │   insider_signal, options_signal, social_sentiment_score, nlp_sentiment_score,
    │   patent_signal, earnings_confidence, sector_signal
    │
    ├── Phase 5: Bias & Conflict Audit ← synthesis["bias_report"] + ["conflict_report"]
    │   bias_count, bias_adjusted_score, conflict_count, overall_reliability, decision_trace_count
    │
    ├── Phase 6: Macro Context ← report["macro"]["fred_data"]["indicators"]
    │   fed_funds_rate, cpi_yoy, unemployment_rate, yield_curve_spread
    │
    ├── Phase 7: v2.2 Enhancements ← report["info_gap_report"] + ["risk_assessment"]
    │   debate_rounds_count, devils_advocate_challenges, info_gap_count,
    │   info_gap_resolved_count, data_quality_score, risk_judge_decision,
    │   risk_adjusted_confidence, aggressive_analyst_confidence, conservative_analyst_confidence
    │
    ├── Phase 8: Cost Metrics ← synthesis["cost_summary"]
    │   total_tokens, total_cost_usd, deep_think_calls
    │
    └── Phase 9: Reflection Loop ← synthesis["synthesis_iterations"]
        synthesis_iterations
    │
    ├── Phase 10: Model Tracking ← synthesis["cost_summary"]
    │   standard_model, deep_think_model
    │
    └── Phase 11: Autoresearch Parity ← report["quant_model"], ["alt_data"], ["anomaly"], ["scenario"], debate DA, risk neutral
        consumer_sentiment, revenue_growth_yoy, quality_score, momentum_6m, rsi_14,
        alt_data_signal, alt_data_momentum_pct, anomaly_signal, monte_carlo_signal,
        quant_model_signal, quant_model_score, social_sentiment_velocity, nlp_sentiment_confidence,
        risk_level, recommended_position_pct, neutral_analyst_confidence, risk_debate_rounds_count,
        groupthink_flag, da_confidence_adjustment, grounded_calls
    │
    ▼
backend/db/bigquery_client.py :: save_report()  →  BigQuery INSERT (88 fields)
```

### Column Categories (68 total)

| Category | Count | Example Columns | Purpose |
|----------|-------|----------------|---------|
| Identity | 3 | `ticker`, `company_name`, `analysis_date` | Primary key |
| Output | 4 | `final_score`, `recommendation`, `summary`, `recommendation_justification` | LLM outputs |
| Pillar Scores | 5 | `pillar_1_corporate` … `pillar_5_governance` | 5-pillar scoring matrix |
| Financial Fundamentals | 7 | `price_at_analysis`, `pe_ratio`, `peg_ratio`, `debt_equity` | Point-in-time market data |
| Risk Metrics | 6 | `var_95_6m`, `var_99_6m`, `annualized_volatility`, `anomaly_count` | Monte Carlo + anomalies |
| Debate Dynamics | 10 | `bull_confidence`, `bear_confidence`, `contradiction_count`, `debate_rounds_count`, `devils_advocate_challenges` | Multi-agent debate features |
| Enrichment Signals | 7 | `insider_signal`, `options_signal`, `nlp_sentiment_score` | Individual signal values |
| Bias & Conflict | 5 | `bias_count`, `bias_adjusted_score`, `conflict_count`, `overall_reliability` | Validation audit |
| Info-Gap Quality | 3 | `info_gap_count`, `info_gap_resolved_count`, `data_quality_score` | AlphaQuanter data completeness |
| Risk Assessment | 4 | `risk_judge_decision`, `risk_adjusted_confidence`, `aggressive_analyst_confidence` | TradingAgents risk debate |
| Macro Context | 4 | `fed_funds_rate`, `cpi_yoy`, `unemployment_rate`, `yield_curve_spread` | FRED indicators |
| Cost Metrics | 3 | `total_tokens`, `total_cost_usd`, `deep_think_calls` | Per-analysis LLM cost tracking |
| Reflection Loop | 1 | `synthesis_iterations` | Evaluator-Optimizer iteration count |
| Full Report | 1 | `full_report_json` | Complete JSON fallback |

### What This Enables for Frontend

| Frontend Feature | BQ Data Source | Notes |
|-----------------|---------------|-------|
| `/reports` — Past Reports list | `ticker`, `analysis_date`, `final_score`, `recommendation` | Quick-list display |
| `/compare` — Multi-report comparison | All pillar scores + enrichment signals + debate dynamics | Side-by-side radar chart data |
| `/performance` — Historical accuracy | Joined with `outcome_tracking` table (return_pct, beat_benchmark) | Supervised feedback loop |
| `/portfolio` — Recommendation accuracy | `recommendation`, `price_at_analysis` + outcome tracking | P&L attribution |
| `/performance` — Cost history | `total_tokens`, `total_cost_usd`, `deep_think_calls` | Per-analysis cost table |
| ML model training | All 88 columns joined with outcome labels | Direct `SELECT`-based BQ ML queries |

### Schema Migration

The schema is managed by `migrate_bq_schema.py` (idempotent — safe to re-run). It checks existing columns and only adds missing ones via `client.update_table()`.

The agent memories table is managed by `migrate_agent_memories.py` (idempotent). Creates `financial_reports.agent_memories` with columns: `agent_type`, `ticker`, `situation`, `lesson`, `created_at`.

The paper trading tables are managed by `migrate_paper_trading.py` (idempotent). Creates 4 tables: `paper_portfolio` (8 cols), `paper_positions` (14 cols), `paper_trades` (11 cols), `paper_portfolio_snapshots` (11 cols). See AGENTS.md for full column definitions.

Financials (GET /api/charts/{ticker}/financials)
    │
    ├── company_name            → ReportHeader
    ├── sector / industry       → ReportHeader (tags)
    ├── valuation (P/E, PEG, …) → ReportHeader metrics strip, InvestmentThesisCard
    ├── week_52_high/low        → ReportHeader (range bar)
    ├── revenue / net_income    → InvestmentThesisCard (financial snapshot)
    └── efficiency / health     → PdfDownload (financial snapshot page)
```

### Pydantic Model: `SynthesisReport`

```python
class StepLogEntry(BaseModel):
    step: str                                 # Step name (e.g. "Data Enrichment")
    status: str                               # running | completed | error
    message: str                              # Sub-step detail text
    timestamp: str                            # ISO 8601 UTC

class AnalysisStatusResponse(BaseModel):
    task_id: str
    status: str                               # pending | running | completed | error
    progress: float                           # 0.0–100.0
    current_step: Optional[str]
    message: Optional[str]                    # Latest sub-step message
    step_log: list[StepLogEntry]              # Full timestamped log
    result: Optional[SynthesisReport]
```

```python
class SynthesisReport(BaseModel):
    scoring_matrix: ScoringMatrix             # 5-pillar scores (1-10)
    recommendation: RecommendationDetail      # action + justification
    final_summary: str                        # Multi-paragraph analyst summary
    key_risks: list[str]                      # Risk bullet points
    final_weighted_score: Optional[float]     # 0.0–10.0 weighted score
    # v2 enrichment fields
    enrichment_signals: Optional[dict]        # 12 signal results
    debate_result: Optional[dict]             # Bull/Bear/DA/Moderator consensus
    decision_traces: Optional[list]           # Agent XAI audit trail
    risk_data: Optional[dict]                 # Monte Carlo + anomalies
    bias_report: Optional[dict]              # LLM bias flags
    conflict_report: Optional[dict]          # Knowledge conflict flags
    # v2.2 fields
    risk_assessment: Optional[dict]          # Risk Assessment Team verdict
    info_gap_report: Optional[dict]          # Info-Gap detection quality report
    # v2.4 fields
    cost_summary: Optional[dict]             # Per-agent token/cost tracking
    # v2.6 fields
    synthesis_iterations: Optional[int]      # Reflection loop iteration count
```

---

## Component Registry

### 1. ReportHeader

**File:** `frontend/src/components/ReportHeader.tsx`
**Props:** `{ ticker, report: SynthesisReport, financials? }`

Bloomberg-style hero strip that always sits above the tab bar when a report exists.

| Element | Description | Data Source |
|---------|-------------|-------------|
| Company Name | Large bold title | `financials.company_name` |
| Ticker Badge | Pill badge next to name | `ticker` prop |
| Sector / Industry Tags | Small colored tags | `financials.sector`, `financials.industry` |
| Score Ring | Circular SVG progress (radius 40) color-coded by threshold | `report.final_weighted_score` |
| Action Badge | Color-coded recommendation pill (Strong Buy→emerald, Sell→rose) | `report.recommendation.action` |
| Metrics Strip | Price, Market Cap, P/E, PEG, Div Yield | `financials.valuation.*` |
| 52-Week Range Bar | Horizontal bar with current price indicator dot | `financials.week_52_high`, `week_52_low` |
| Data Source Tags | Small provenance badges at bottom | Static (yfinance, SEC, AV, FRED, USPTO, Vertex AI) |

**Score Color Thresholds:** ≥7 emerald, ≥5 sky, ≥3 amber, <3 rose

---

### 2. ReportTabs

**File:** `frontend/src/components/ReportTabs.tsx`
**Props:** `{ tabs: TabDef[], children: (activeTab: string) => ReactNode }`

Generic tab navigation wrapper. Uses a render-prop pattern — the parent passes tab definitions and a function that receives the active tab ID to render content.

```typescript
interface TabDef {
  id: string;       // "overview" | "signals" | "debate" | "risk" | "audit" | "cost"
  label: string;
  icon: Icon;       // Phosphor Icon component (from @phosphor-icons/react)
  badge?: string | number | null;
}
```

**Styling:** Active tab uses `bg-sky-500/15 text-sky-400` with border highlight. Icons render as `<tab.icon size={16} weight={active ? "fill" : "regular"} />`. Badges render inline with optional count/label.

---

### 3. SignalDashboard

**File:** `frontend/src/components/SignalDashboard.tsx`
**Props:** `{ signals: EnrichmentSignals }`

FactSet-style panel showing all 12 enrichment signals with source attribution, expandable detail, and a consensus divergence bar.

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│  Consensus Divergence Bar                             │
│  ████████████ bullish (5) ████ neutral (3) ██ bear(3)│
├──────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Insider │ │ Options │ │ Social  │ │ Patents │   │
│  │ SEC     │ │yfinance │ │   AV    │ │ USPTO   │   │
│  │ BULLISH │ │ BEARISH │ │ NEUTRAL │ │ ACCEL.  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Earnings │ │  Macro  │ │Alt Data │ │ Sector  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                │
│  │  NLP    │ │Anomaly  │ │M. Carlo │                │
│  └─────────┘ └─────────┘ └─────────┘                │
└──────────────────────────────────────────────────────┘
```

**12 Signal Cards:**

| Signal Key | Label | Source | Source Tag Color |
|-----------|-------|--------|-----------------|
| `insider` | Insider Activity | SEC EDGAR | blue |
| `options` | Options Flow | yfinance | purple |
| `social_sentiment` | Social Sentiment | Alpha Vantage | amber |
| `patent` | Innovation/Patents | BigQuery Patents | cyan |
| `earnings_tone` | Earnings Tone | API Ninjas | pink |
| `fred_macro` | Macro Climate | FRED | orange |
| `alt_data` | Alt Data | Google Trends | indigo |
| `sector` | Sector Strength | yfinance + 11 ETFs | violet |
| `nlp_sentiment` | NLP Sentiment | Vertex AI | teal |
| `anomaly` | Anomaly Scan | Multi-dim Z-score | red |
| `monte_carlo` | Risk Scenario | Monte Carlo VaR | slate |
| `quant_model` | Quant Model | MDA + yfinance | sky |

**Signal Color Mapping:**

| Signal Value | Color | Background |
|-------------|-------|-----------|
| `STRONG_BULLISH` / `INNOVATION_BREAKOUT` / `DOUBLE_TAILWIND` | emerald-400 | emerald-500/10 |
| `BULLISH` / `RISING_STRONG` / `ACCELERATING` / `CONFIDENT` | emerald-300 | emerald-500/10 |
| `NEUTRAL` / `STABLE` / `NORMAL` | slate-400 | slate-500/10 |
| `BEARISH` / `DECLINING` / `EVASIVE` / `LAGGING` | rose-400 | rose-500/10 |
| `STRONG_BEARISH` / `ANOMALY_RISK` | rose-500 | rose-500/15 |

**Consensus Bar:** Counts each signal as bullish, bearish, or neutral using `signalSide()` helper. Renders as a horizontal stacked bar with counts.

---

### 4. InvestmentThesisCard (GlassBoxCards.tsx)

**File:** `frontend/src/components/GlassBoxCards.tsx`
**Props:** `{ report: SynthesisReport, financials? }`

Enhanced executive summary card that combines recommendation justification, thesis narrative, debate-sourced catalysts/threats, and a financial snapshot.

| Section | Content | Source |
|---------|---------|--------|
| Justification | Bold italic recommendation reasoning | `report.recommendation.justification` |
| Summary | Multi-paragraph analysis (split on `\n`) | `report.final_summary` |
| Key Catalysts | Up to 4 bullish catalysts from debate (emerald) | `report.debate_result.bull_case.key_catalysts` |
| Key Threats | Up to 4 bearish threats from debate (rose) | `report.debate_result.bear_case.key_threats` |
| Consensus Meter | Horizontal confidence bar | `report.debate_result.consensus_confidence` |
| Financial Snapshot | Revenue, Net Income, Market Cap | `financials.revenue`, `financials.net_income`, `financials.market_cap` |

---

### 5. EvaluationTable

**File:** `frontend/src/components/EvaluationTable.tsx`
**Props:** `{ scores: ScoringMatrix, previousScores?: ScoringMatrix | null }`

5-pillar scoring grid with expandable detail, weighted total, contributing source attribution, and optional historical delta comparison.

**5 Pillars:**

| Pillar | Key | Weight | Icon | Contributing Sources |
|--------|-----|--------|------|---------------------|
| Corporate Profile | `pillar_1_corporate` | 35% | `PillarCorporate` (Buildings) | RAG Agent (10-K/10-Q), yfinance fundamentals |
| Industry & Macro | `pillar_2_industry` | 20% | `PillarIndustry` (GlobeHemisphereWest) | Sector Analysis, FRED Macro, Competitor Agent |
| Valuation | `pillar_3_valuation` | 20% | `PillarValuation` (CurrencyDollar) | yfinance valuation metrics, quant-agent data |
| Market Sentiment | `pillar_4_sentiment` | 15% | `PillarSentiment` (ChartBar) | NLP Sentiment, Social Sentiment, Insider Activity, Options Flow |
| Governance | `pillar_5_governance` | 10% | `PillarGovernance` (Bank) | RAG Agent governance section, insider ownership |

**Visual Features:**
- Progress bar per pillar (color: emerald ≥8, sky ≥6, amber ≥4, rose <4)
- Delta indicators (▲/▼ with green/red) when `previousScores` is provided
- Weighted total displayed in header
- Click-to-expand reveals description + contributing agent/source list

---

### 6. DebateView

**File:** `frontend/src/components/DebateView.tsx`
**Props:** `{ debate: DebateResult }`

Adversarial debate visualization showing the Bull vs Bear argument structure, Devil's Advocate stress-test, multi-round timeline, evidence provenance, contradictions, and dissent registry.

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│  [DebateConsensus] Consensus: BUY  ████████ 72%      │
│  2 rounds · Devil's Advocate reviewed                 │
├─────────────────────────┬────────────────────────────┤
│  [DebateBull] BULL (78%)│  [DebateBear] BEAR (45%)  │
│  ─────────────────────  │  ─────────────────────     │
│  Thesis paragraph...    │  Thesis paragraph...       │
│                         │                            │
│  Key Catalysts:         │  Key Threats:              │
│  • Catalyst 1           │  • Threat 1                │
│  • Catalyst 2           │  • Threat 2                │
│                         │                            │
│  Evidence:              │  Evidence:                 │
│  [SEC EDGAR] data → ..  │  [yfinance] data → ..     │
│  [USPTO] data → interp  │  [FRED] data → interp     │
├─────────────────────────┴────────────────────────────┤
│  [DebateDevilsAdvocate] Devil's Advocate (violet)     │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Challenges: [challenge 1], [challenge 2]        │ │
│  │  Hidden Risks: [risk 1], [risk 2]                │ │
│  │  Groupthink: No groupthink detected              │ │
│  │  Confidence Adjustment: -0.05                    │ │
│  └──────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│  [DebateRounds] Debate Timeline (collapsible)         │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Round 1:  Bull argument   │  Bear argument      │ │
│  │  Round 2:  Bull rebuttal   │  Bear rebuttal      │ │
│  └──────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│  [DebateContradiction] Contradictions                 │
│  ┌──────────┬──────────────┬──────────────┬────────┐ │
│  │  Topic   │  Bull View   │  Bear View   │ Winner │ │
│  ├──────────┼──────────────┼──────────────┼────────┤ │
│  │ Margins  │ Expanding..  │ Pressure..   │ BULL   │ │
│  └──────────┴──────────────┴──────────────┴────────┘ │
├──────────────────────────────────────────────────────┤
│  [DebateDissent] Dissent: [Options Flow: BEARISH]    │
└──────────────────────────────────────────────────────┘
```

**Note:** All debate icons are Phosphor Icons from `frontend/src/lib/icons.ts` (aliased as `Debate*` names). Rendered as JSX components with `weight="duotone"`.

**New v2.2 Sections:**
- **Devil's Advocate** (violet/purple theme): Challenges list, hidden risks, bull/bear weaknesses, groupthink flag (boolean), confidence adjustment badge
- **Multi-Round Timeline** (collapsible): Shows round-by-round Bull/Bear arguments in a side-by-side grid. Only visible when `debate_rounds` array exists.

**Evidence Format:** Each evidence item shows `[source badge]` → `data_point` → `interpretation` with color-coded background (emerald for bull, rose for bear).

---

### 7. RiskDashboard + RiskAssessmentPanel

**File:** `frontend/src/components/RiskDashboard.tsx`

#### RiskDashboard
**Props:** `{ data: RiskDataPayload }`

Monte Carlo simulation results and multi-dimensional anomaly alerts.

**Sections:**

| Section | Content |
|---------|---------|
| Header | Current price + annualized volatility |
| VaR Gauges (3M, 6M, 1Y) | VaR(95%), VaR(99%), Expected Shortfall, Median Return — horizontal bars with color severity |
| Probability Grid | P(Positive), P(≥+20%), P(≤−20%) per horizon |
| Anomaly Alerts | Grid of anomaly cards: metric name, Z-score badge, value, note (rose=risk, emerald=opportunity) |

**Gauge Color Thresholds:** <30% emerald, <60% amber, ≥60% rose

#### RiskAssessmentPanel (v2.2, updated v2.3)
**Props:** `{ data: RiskAssessment }`

TradingAgents-style risk assessment team verdict panel. Rendered in the Risk tab below RiskDashboard. v2.3 adds multi-round risk debate with cross-visibility — each analyst sees the other two's prior arguments. Rounds configurable via `MAX_RISK_DEBATE_ROUNDS` (1-3).

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│  [RiskJudge] Risk Assessment Team                     │
├──────────────────────────────────────────────────────┤
│  Judge Verdict: MODERATE_POSITION                     │
│  ┌─────────────┬─────────────┬─────────────┐        │
│  │ Position: 5%│ Risk: MED   │ Conf: 0.72  │        │
│  └─────────────┴─────────────┴─────────────┘        │
├──────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │[RiskAggr.]   │ │[RiskConserv.]│ │[RiskNeutral] │ │
│  │ emerald theme│ │ rose theme   │ │ sky theme    │ │
│  │ Position: 8% │ │ Position: 2% │ │ Position: 5% │ │
│  │ Conf: 0.80   │ │ Conf: 0.65   │ │ Conf: 0.72   │ │
│  │ Max: 10%     │ │ Max: 3%      │ │ Max: 6%      │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ │
├──────────────────────────────────────────────────────┤
│  Risk Limits: Stop Loss: -8% │ Max Drawdown: -15%   │
├──────────────────────────────────────────────────────┤
│  [IconWarning] Unresolved Risks: [risk1], [risk2]    │
└──────────────────────────────────────────────────────┘
```

**Note:** Risk icons are Phosphor Icons from `frontend/src/lib/icons.ts`: `RiskAggressive` (Sword), `RiskConservative` (Shield), `RiskNeutral` (Scales), `RiskJudge` (Gavel).

**Analyst Card Colors:** Aggressive=emerald, Conservative=rose, Neutral=sky

---

### 8. DecisionTraceView

**File:** `frontend/src/components/DecisionTraceView.tsx`
**Props:** `{ traces: DecisionTrace[] }`

XAI audit timeline showing every agent's decision process with expandable detail.

**Timeline Layout:** Vertical line with numbered dots (color = confidence level). Each step is an expandable card.

**Per Trace Card:**

| Field | Description |
|-------|-------------|
| Agent Name | Step label on the timeline |
| Confidence | Horizontal bar + percentage (emerald ≥70%, amber ≥40%, rose <40%) |
| Signal | Color-coded badge (BULLISH/BEARISH/NEUTRAL/etc.) |
| Latency | Milliseconds elapsed |
| Evidence Citations | Bulleted list of specific data points used |
| Reasoning Steps | Numbered chain-of-thought steps |
| Source URL | Clickable link to raw data source (if available) |
| Input Data Hash | Truncated hash for reproducibility verification |

**Header Summary:** Total pipeline latency + step count.

---

### 9. BiasReport

**File:** `frontend/src/components/BiasReport.tsx`
**Props:** `{ biasReport?: BiasReportData, conflictReport?: ConflictReportData }`

LLM bias detection flags and knowledge conflict table.

**Bias Audit Section:**
- Header badge: 0 flags → emerald, ≤2 → amber, >2 → rose
- Raw vs Adjusted Score display (shows impact of bias correction)
- Per flag: icon by type, severity badge, description, evidence, adjustment suggestion

**Bias Type Icons:**

| Bias Type | Icon | Description |
|-----------|------|-------------|
| `tech_bias` | `BiasTech` (Monitor) | Systematic tech/large-cap favoritism |
| `confirmation_bias` | `BiasConfirmation` (ArrowsClockwise) | Ignoring contradictory signals |
| `recency_bias` | `BiasRecency` (Clock) | Over-weighting recent data |
| `anchoring` | `BiasAnchoring` (Anchor) | Over-relying on first data point |
| `source_diversity` | `BiasDiversity` (ChartBar) | Insufficient source variety |

**Knowledge Conflicts Table:**

| Column | Content |
|--------|---------|
| Field | Data field in question |
| LLM Belief | What Gemini's parametric knowledge says (amber background) |
| Actual Data | Real-time data from yfinance/SEC (emerald background) |
| Severity | HIGH/MEDIUM/LOW badge |
| Explanation | Why the conflict matters |

---

### 10. PdfDownload

**File:** `frontend/src/components/PdfDownload.tsx`
**Props:** `{ ticker, report: SynthesisReport, financials?, className? }`

Multi-page Goldman Sachs-style PDF export using jsPDF.

**Page Structure:**

| Page | Section | Content |
|------|---------|---------|
| 1 | Cover | PyFinAgent branding, ticker, score, action, generation date |
| 2 | Executive Summary | Justification + final summary text (auto-wrapped) |
| 3 | Scoring Matrix | 5 pillars with scores + weights |
| 3 | Key Risks | Bulleted risk items |
| 4 | Financial Snapshot | Valuation metrics + efficiency metrics |
| 5 | Enrichment Signals | Signal key:value pairs + summaries (truncated 200 chars) |
| 6 | Agent Debate | Consensus + bull/bear excerpts (300 chars max) |
| 7 | Decision Audit Trail | Agent name, signal, confidence per trace step |
| Last | Footer | Disclaimer + data source list on every page |

---

### Supporting Components

| Component | File | Purpose |
|-----------|------|---------|
| `AnalysisProgress` | `AnalysisProgress.tsx` | Gemini Deep Research-style expanding thought panel — 15-step tracker with live sub-step messages, elapsed timer, auto-expanding active step, timestamped log entries, duration badges |
| `StockChart` | `StockChart.tsx` | Candlestick + volume with toggleable SMA50/SMA200/RSI, 5 period options |
| `AlphaScoreCard` | `GlassBoxCards.tsx` | Large score display with recommendation bar (used in legacy/compare views) |
| `ScoringMatrixCard` | `GlassBoxCards.tsx` | 5 horizontal progress bars (one per pillar) with weights |
| `RisksCard` | `GlassBoxCards.tsx` | Rose-themed risk bullet list |
| `ValuationRange` | `GlassBoxCards.tsx` | Valuation metric range visualization |
| `ResearchInvestigator` | `ResearchInvestigator.tsx` | RAG-powered investigation query against 10-K documents |
| `Sidebar` | `Sidebar.tsx` | Navigation sidebar for all routes + user auth UI (avatar, email, passkey registration, logout) |
| `MacroDashboard` | `MacroDashboard.tsx` | 7-indicator FRED grid + macro warnings (used on /signals page) |
| `SectorDashboard` | `SectorDashboard.tsx` | Sector rotation chart + relative strength table (used on /signals page) |
| `CostDashboard` | `CostDashboard.tsx` | LLM cost analytics: 4 summary cards (total cost, tokens, calls, deep think), token distribution bar, cost by model breakdown, per-agent table sorted by cost |
| `Skeleton` | `Skeleton.tsx` | Reusable loading skeleton components: `SkeletonPulse` (atomic pulse), `SkeletonCard` (card-sized), `SkeletonGrid` (N-card grid), `PageSkeleton` (full page layout). Used in backtest, paper-trading, and settings pages |
| `AuthProvider` | `AuthProvider.tsx` | SessionProvider wrapper with 15-minute refetch interval (wraps entire app in layout.tsx) |

---

### 11. CostDashboard (v2.4)

**File:** `frontend/src/components/CostDashboard.tsx`
**Props:** `{ costSummary?: CostSummary }`

LLM cost and token analytics panel, rendered in the Cost tab. Shows per-agent breakdown across standard and deep-think models.

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│  LLM Cost Analytics                                       │
├──────────────┬─────────────┬─────────────┬─────────────┤
│ Total Cost   │ Total Tokens│  LLM Calls  │ Deep Think  │
│  $0.1234     │   245.3K    │     37      │   4 calls   │
└──────────────┴─────────────┴─────────────┴─────────────┘
┌──────────────────────────────────────────────────────┐
│  Token Distribution: ████████ input (180K) ███ output (65K) │
├──────────────────────────────────────────────────────┤
│  Cost by Model:                                          │
│    gemini-2.0-flash:  $0.0834 (33 calls)                 │
│    gemini-2.5-pro:    $0.0400 (4 calls [Sparkle])        │
├──────────────────────────────────────────────────────┤
│  Per-Agent Breakdown (sorted by cost):                    │
│  ┌──────────────────┬────────────┬───────┬─────────┐ │
│  │ Agent            │ Model       │ Tokens│ Cost    │ │
│  ├──────────────────┼────────────┼───────┼─────────┤ │
│  │ Synthesis [dt]   │ 2.5-pro     │  12.1K│ $0.0150 │ │
│  │ Moderator [dt]   │ 2.5-pro     │   8.3K│ $0.0120 │ │
│  │ Bull Agent       │ 2.0-flash   │  15.2K│ $0.0045 │ │
│  │ ...              │ ...         │  ...  │ ...     │ │
│  └──────────────────┴────────────┴───────┴─────────┘ │
└──────────────────────────────────────────────────────┘
```

**Deep Think Badge:** Deep-think agents (Moderator, Risk Judge, Synthesis, Critic) display `IconDeepThink` (Sparkle) icon next to their model name.

**Sections:**

| Section | Content |
|---------|---------|
| Summary Cards (4) | Total Cost (amber), Total Tokens (sky), LLM Calls (violet), Deep Think Calls (emerald) |
| Token Distribution | Horizontal stacked bar: input tokens (sky) vs output tokens (violet) |
| Cost by Model | Per-model cost breakdown with call counts; deep think badge on applicable models |
| Per-Agent Table | Sortable table: agent name, model, input/output tokens, total cost, deep think badge |

---

### AnalysisProgress — Gemini Deep Research Expanding Thought Panel

**File:** `frontend/src/components/AnalysisProgress.tsx`
**Props:** `{ status: AnalysisStatusResponse, ticker: string }`

**Design:** Inspired by Gemini Deep Research's real-time thinking panel. Shows every orchestrator sub-step as it happens, with expandable sections per pipeline step.

**Layout:**
```
┌──────────────────────────────────────────────────────────┐
│  ▍▍▍▍ Analyzing AAPL             ⏱ 2m 34s         72%  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━░░░░░░░░░░░░░░  │
│  Running adversarial debate (Bull vs Bear)...            │
├──────────────────────────────────────────────────────────┤
│  ✓ [Newspaper] Market Intel                      2.1s ▾  │
│  ✓ [CloudArrowDown] Ingestion                    1.8s ▸  │
│  ✓ [HashStraight] Quant Data                     3.2s ▸  │
│  ✓ [FileText] Document Analysis                  5.4s ▸  │
│  ✓ [Crosshair] Sentiment Analysis                4.1s ▸  │
│  ✓ [Trophy] Competitor Analysis                  3.8s ▸  │
│  ✓ [Broadcast] Data Enrichment                  12.3s ▾  │
│  │  [14:32:01] [1/11] Insider data collected             │
│  │  [14:32:02] [2/11] Options data collected             │
│  │  [14:32:03] [3/11] Patent data collected              │
│  │  ...                                                  │
│  ● [Brain] Enrichment Analysis                       ▾   │
│  │  [14:32:15] [1/10] Gemini analyzing Insider...        │
│  │  [14:32:18] [1/10] Insider Activity → BULLISH         │
│  │  thinking...                                          │
│  ○ [Scales] Agent Debate                             ▸   │
│  ○ [Globe] Enhanced Macro                            ▸   │
│  ○ [Binoculars] Deep Dive                            ▸   │
│  ○ [Flask] Synthesis                                 ▸   │
│  ○ [ShieldCheck] Critic Review                       ▸   │
└──────────────────────────────────────────────────────────┘
```

**Note:** All step icons are Phosphor Icons from `frontend/src/lib/icons.ts` (aliased as `Step*` names, e.g., `StepMarketIntel`, `StepIngestion`). Rendered as `<step.icon size={16} weight="duotone" />`.

**Visual States:**

| State | Icon | Style | Behavior |
|-------|------|-------|----------|
| Completed | `CheckCircle` (Phosphor) | `text-emerald-400` | Green check, duration badge, click to expand/collapse log |
| Active | `●` (pulsing) | `text-sky-400 bg-sky-500/10` | Blue ping dot, auto-expanded, `thinking...` pulse at bottom |
| Pending | `○` | `text-zinc-600` | Dimmed, collapsed |
| Error | `XCircle` (Phosphor) | `text-rose-400` | Red X, auto-expanded |

**Features:**
- **Header:** Gemini 4-bar spinner animation + ticker name + elapsed timer (`⏱ 2m 34s`) + percentage
- **Progress bar:** 1.5px tall, `sky-500 → cyan-400` gradient with smooth transition
- **Live activity quote:** Latest `message` from `AnalysisStatusResponse` displayed in italics below progress bar
- **Expandable steps:** Click any completed step to reveal timestamped sub-messages; active step auto-expands
- **Auto-scroll:** Container scrolls to keep active step visible
- **Auto-expand/collapse:** When active step advances, previous step collapses and new step expands
- **Duration badges:** Completed steps show elapsed time (e.g. `3.8s`) in a zinc-600 badge
- **Timestamped messages:** Each log entry shows `[HH:MM:SS]` prefix; completed in emerald, running in zinc

**Orchestrator Sub-Step Messages (30+ messages):**

| Pipeline Step | Example Messages |
|---------------|------------------|
| Data Enrichment (Step 6) | `[1/12] Insider data collected`, `[2/12] Options data collected`, ... `[12/12] Quant model signal complete`. **Sector routing**: irrelevant tools skipped (e.g., patents for Financial Services) with `[X/12] Patents skipped (sector routing)` |
| Info-Gap Detection (Step 6b) | `Scanning 12 enrichment sources for data gaps...`, `Found N critical gaps, retrying...`, `Data quality score: 85/100` |
| Enrichment Analysis (Step 7) | `[1/10] Gemini analyzing Insider Activity...`, `[1/10] Insider Activity → BULLISH` |
| Agent Debate (Step 8) | `Round 1/2: Bull building strongest case...`, `Round 1/2: Bear challenging...`, `Round 2/2: Bull rebutting Bear's argument...`, `Devil's Advocate stress-testing both sides...`, `Moderator resolving contradictions with full debate history...`. **Quality gate**: if data quality < threshold, shows `Skipping debate (data quality too low)` |
| Synthesis (Step 11) | `Building synthesis prompt from all agent outputs...`, `Gemini generating structured JSON report...`, `Parsing and validating report structure...`. **Reflection loop**: `Critic returned REVISE — re-running synthesis (iteration 2/2)...` |
| Bias Audit (Step 12b) | `Checking for tech bias, confirmation bias, recency bias...`, `Found N bias flags. Checking knowledge conflicts...` |
| Risk Assessment (Step 12c) | `Aggressive analyst arguing position...`, `Conservative analyst arguing position...`, `Neutral analyst synthesizing...`, `Risk Judge delivering verdict...`. **Quality gate**: if data quality < threshold, shows `Skipping risk assessment (data quality too low)` |

---

## TypeScript Type Reference

All types are defined in `frontend/src/lib/types.ts`.

### Analysis Status Types

```typescript
interface StepLogEntry {
  step: string;                // Step name (e.g. "Data Enrichment")
  status: string;              // running | completed | error
  message: string;             // Sub-step detail text
  timestamp: string;           // ISO 8601 UTC
}

interface AnalysisStatusResponse {
  task_id: string;
  status: string;              // pending | running | completed | error
  progress: number;            // 0–100
  current_step?: string;
  message?: string;            // Latest sub-step message
  step_log?: StepLogEntry[];   // Full timestamped log
  result?: SynthesisReport;
}
```

### Core Report Types

```typescript
interface ScoringMatrix {
  pillar_1_corporate: number;   // 1-10
  pillar_2_industry: number;
  pillar_3_valuation: number;
  pillar_4_sentiment: number;
  pillar_5_governance: number;
}

interface RecommendationDetail {
  action: string;               // "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell"
  justification: string;
}

interface SynthesisReport {
  scoring_matrix: ScoringMatrix;
  recommendation: RecommendationDetail;
  final_summary: string;
  key_risks: string[];
  final_weighted_score?: number;
  enrichment_signals?: EnrichmentSignals;
  debate_result?: DebateResult;
  decision_traces?: DecisionTrace[];
  risk_data?: RiskData;
  bias_report?: BiasReportData;
  conflict_report?: ConflictReportData;
  // v2.2 additions
  risk_assessment?: RiskAssessment;
  info_gap_report?: InfoGapReport;
  // v2.4 additions
  cost_summary?: CostSummary;
}
```

### Enrichment Signal Types

```typescript
interface SignalSummary {
  ticker: string;
  signal: string;               // BULLISH, BEARISH, NEUTRAL, etc.
  summary: string;
  data?: Record<string, unknown>;
}

interface EnrichmentSignals {
  insider: SignalSummary;
  options: SignalSummary;
  social_sentiment: SignalSummary;
  patent: SignalSummary;
  earnings_tone: SignalSummary;
  fred_macro: SignalSummary;
  alt_data: SignalSummary;
  sector: SignalSummary;
  nlp_sentiment: SignalSummary;
  anomaly: SignalSummary;
  monte_carlo: SignalSummary;
}
```

### Debate Types

```typescript
interface DebateCaseDetail {
  thesis: string;
  confidence: number;           // 0.0–1.0
  key_catalysts?: string[];     // Bull case only
  key_threats?: string[];       // Bear case only
  evidence?: Array<{
    source: string;
    data_point: string;
    interpretation: string;
  }>;
}

interface DebateContradiction {
  topic: string;
  bull_view: string;
  bear_view: string;
  resolution: string;
  winner?: string;
}

interface DebateDissent {
  agent: string;
  position: string;
  reason: string;
}

interface DebateResult {
  bull_case: DebateCaseDetail;
  bear_case: DebateCaseDetail;
  consensus: string;            // STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
  consensus_confidence: number; // 0.0–1.0
  contradictions: DebateContradiction[];
  dissent_registry: DebateDissent[];
  moderator_analysis?: string;
  // v2.2 additions
  debate_rounds?: DebateRound[];
  total_rounds?: number;
  devils_advocate?: DevilsAdvocate;
}

// v2.2 — Multi-round debate
interface DebateRound {
  round: number;
  bull_argument: string;
  bear_argument: string;
}

// v2.2 — Devil's Advocate stress-test
interface DevilsAdvocate {
  challenges: string[];
  hidden_risks: string[];
  bull_weakness: string;
  bear_weakness: string;
  groupthink_flag: boolean;
  confidence_adjustment: number;
  summary: string;
}
```

### Risk & Monte Carlo Types

```typescript
interface MonteCarloHorizon {
  var_95: number;
  var_99: number;
  expected_shortfall_95: number;
  prob_gain_20_pct: number;
  prob_loss_20_pct: number;
  prob_positive: number;
  median_return: number;
  mean_return: number;
  std_return: number;
}

interface MonteCarloData {
  ticker: string;
  signal: string;
  summary: string;
  current_price?: number;
  annualized_volatility?: number;
  horizons?: Record<string, MonteCarloHorizon>;  // "3M" | "6M" | "1Y"
}

interface AnomalyItem {
  metric: string;
  z_score: number;
  value: number;
  note: string;
}

interface AnomalyData {
  ticker: string;
  signal: string;
  summary: string;
  anomalies: AnomalyItem[];
}
```

### Decision Trace & Bias Types

```typescript
interface DecisionTrace {
  agent_name: string;
  timestamp: string;
  input_data_hash: string;
  output_signal: string;
  confidence: number;           // 0.0–1.0
  evidence_citations: string[];
  reasoning_steps: string[];
  latency_ms: number;
  source_url?: string;
}

interface BiasFlag {
  bias_type: string;            // tech_bias | confirmation_bias | recency_bias | anchoring | source_diversity
  severity: string;             // HIGH | MEDIUM | LOW
  description: string;
  evidence: string;
  adjustment_suggestion?: string;
}

interface BiasReportData {
  flags: BiasFlag[];
  raw_score?: number;
  adjusted_score?: number;
  bias_count: number;
}

interface KnowledgeConflict {
  field: string;
  llm_belief: string;
  actual_value: string;
  severity: string;
  category: string;
  explanation: string;
}

interface ConflictReportData {
  conflicts: KnowledgeConflict[];
  conflict_count: number;
  overall_reliability: string;  // HIGH | MEDIUM | LOW
}
```

### v2.2/v2.3 Types — Risk Assessment, Info-Gap & Agent Memory

```typescript
// Risk Assessment Team (TradingAgents pattern)
interface RiskAnalystCase {
  position: string;
  confidence: number;
  max_position_pct: number;
}

interface RiskJudgeVerdict {
  decision: string;
  risk_adjusted_confidence: number;
  recommended_position_pct: number;
  risk_level: string;
  reasoning: string;
  risk_limits: Record<string, string>;
  unresolved_risks: string[];
  summary: string;
}

interface RiskAssessment {
  aggressive: RiskAnalystCase;
  conservative: RiskAnalystCase;
  neutral: RiskAnalystCase;
  judge: RiskJudgeVerdict;
  risk_debate_rounds?: RiskDebateRound[];  // v2.3: multi-round history
  total_risk_rounds?: number;              // v2.3: total rounds executed
}

// v2.3: Multi-round risk debate history
interface RiskDebateRound {
  round: number;
  aggressive_argument: string;
  conservative_argument: string;
  neutral_argument: string;
}

// Info-Gap Detection (AlphaQuanter pattern)
interface InfoGap {
  source: string;
  status: string;               // SUFFICIENT | PARTIAL | MISSING
  criticality: string;          // high | medium | low
  impact: string;
}

interface InfoGapReport {
  gaps: InfoGap[];
  data_quality_score: number;   // 0–100
  critical_gaps: number;
  recommendation_at_risk: boolean;
  summary: string;
}
```

### v2.4 Types — Cost Analytics & Model Configuration

```typescript
// Per-agent cost entry from CostTracker
interface AgentCostEntry {
  agent_name: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
  is_deep_think: boolean;
}

// Per-model aggregation
interface ModelBreakdown {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
  calls: number;
}

// Full cost summary attached to SynthesisReport
interface CostSummary {
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_cost_usd: number;
  total_calls: number;
  deep_think_calls: number;
  model_breakdown: Record<string, ModelBreakdown>;
  agents: AgentCostEntry[];
}

// Cost history entry from BQ (used on /performance page)
interface CostHistoryEntry {
  ticker: string;
  analysis_date: string;
  total_tokens: number | null;
  total_cost_usd: number | null;
  deep_think_calls: number | null;
}

// Model configuration from /api/settings/models
interface ModelConfig {
  gemini_model: string;
  deep_think_model: string;
  max_debate_rounds: number;
  max_risk_debate_rounds: number;
}

// Available model pricing from /api/settings/models/available
// v3.4: Added provider field for multi-provider grouping
interface ModelPricing {
  model: string;
  provider?: string;             // "Gemini" | "GitHub Models" | "Anthropic" | "OpenAI"
  input_per_1m: number;
  output_per_1m: number;
}

// v2.7: Full settings (GET /api/settings/)
// v3.3: Added enable_thinking + thinking budget fields
// v3.4: Added provider key-configured booleans (read-only; never exposes key values)
interface FullSettings {
  gemini_model: string;
  deep_think_model: string;
  max_debate_rounds: number;
  max_risk_debate_rounds: number;
  pillar_1_weight: number;
  pillar_2_weight: number;
  pillar_3_weight: number;
  pillar_4_weight: number;
  pillar_5_weight: number;
  data_quality_min: number;
  lite_mode: boolean;
  max_analysis_cost_usd: number;
  max_synthesis_iterations: number;
  // v3.3 — Extended thinking (Gemini 2.5+)
  enable_thinking: boolean;
  thinking_budget_critic: number;
  thinking_budget_moderator: number;
  thinking_budget_risk_judge: number;
  thinking_budget_synthesis: number;
  // v3.4 — Multi-provider key status (read-only)
  anthropic_key_configured?: boolean;
  openai_key_configured?: boolean;
  github_token_configured?: boolean;
}

// v2.7: Latest cost summary for live cost estimator
interface LatestCostSummary extends CostSummary {
  ticker: string;
  analysis_date: string;
}

// v4.2: Performance monitoring types (GET /api/perf/*)
interface EndpointLatency {
  endpoint: string;
  count: number;
  p50: number;
  p95: number;
  p99: number;
}

interface PerfSummary {
  endpoints: EndpointLatency[];
  overall: { p50: number; p95: number; p99: number };
}

interface CacheStats {
  entries: number;
  hits: number;
  misses: number;
  hit_rate: number;
}

interface PerfOptimizerStatus {
  running: boolean;
  iterations: number;
  kept: number;
  discarded: number;
}

interface PerfExperiment {
  iteration: number;
  endpoint: string;
  old_ttl: number;
  new_ttl: number;
  p95_before: number;
  p95_after: number;
  decision: string;
  timestamp: string;
}
```

### v2.9 Types — Paper Trading

```typescript
// Paper trading fund state
interface PaperPortfolio {
  portfolio_id: string;
  starting_capital: number;
  current_cash: number;
  total_nav: number;
  total_pnl_pct: number;
  benchmark_return_pct: number;
  created_at: string;
  updated_at: string;
}

// Open position
interface PaperPosition {
  position_id: string;
  ticker: string;
  quantity: number;
  avg_entry_price: number;
  cost_basis: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  entry_date: string;
  last_analysis_date: string;
  recommendation: string;
  risk_judge_position_pct: number;
  stop_loss_price: number;
}

// Trade history entry
interface PaperTrade {
  trade_id: string;
  ticker: string;
  action: string;              // BUY | SELL
  quantity: number;
  price: number;
  total_value: number;
  reason: string;              // new_buy_signal | stop_loss | signal_flip | rebalance
  analysis_id: string;
  risk_judge_decision: string;
  pnl_pct: number | null;     // Realized P&L % (sells only)
  created_at: string;
}

// Daily NAV snapshot
interface PaperSnapshot {
  snapshot_date: string;
  total_nav: number;
  cash: number;
  positions_value: number;
  daily_pnl_pct: number;
  cumulative_pnl_pct: number;
  benchmark_pnl_pct: number;
  alpha_pct: number;
  position_count: number;
  trades_today: number;
  analysis_cost_today: number;
}

// Aggregated status response
interface PaperTradingStatus {
  portfolio: PaperPortfolio | null;
  scheduler_running: boolean;
  last_run: string | null;
  last_result: string | null;
  loop_running: boolean;
}

// Performance metrics
interface PaperPerformance {
  sharpe_ratio: number | null;
  total_return_pct: number;
  alpha_pct: number;
  win_rate: number | null;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  max_drawdown_pct: number | null;
  cumulative_cost_usd: number;
  days_active: number;
}
```

---

## Color & Styling Conventions

### Global Design Tokens (Tailwind CSS)

| Purpose | Color Classes | Usage |
|---------|--------------|-------|
| Strongly bullish | `emerald-400`, `emerald-500/10` bg | Signals, score bars, anomaly opportunity |
| Mildly bullish | `emerald-300` | Secondary bullish UI |
| Neutral / stable | `slate-400`, `slate-500/10` bg | Neutral signals, default states |
| Mildly bearish | `rose-300` | Secondary bearish UI |
| Strongly bearish | `rose-500`, `rose-500/15` bg | Signals, anomaly risk, high bias severity |
| Primary accent | `sky-400`, `sky-500/15` bg | Active tabs, links, primary metrics |
| Warning | `amber-400`, `amber-500/10` bg | Contradictions, medium severity |
| Card surface | `zinc-800/60`, `border-zinc-700/50` | All card containers |
| Text primary | `zinc-100` | Headings, key values |
| Text secondary | `zinc-400` | Labels, descriptions |

### Score Thresholds

| Context | ≥7 or ≥8 | ≥5 or ≥6 | ≥3 or ≥4 | <3 or <4 |
|---------|----------|----------|----------|----------|
| Score Ring | emerald | sky | amber | rose |
| Pillar Bars | emerald (≥8) | sky (≥6) | amber (≥4) | rose (<4) |
| Confidence | emerald (≥0.7) | — | amber (≥0.4) | rose (<0.4) |

### Source Badge Colors

| Source | Color |
|--------|------|
| SEC EDGAR | `sky-400/20 text-sky-300` |
| yfinance | `violet-400/20 text-violet-300` |
| Alpha Vantage | `amber-400/20 text-amber-300` |
| FRED | `orange-400/20 text-orange-300` |
| USPTO / BigQuery Patents | `cyan-400/20 text-cyan-300` |
| Vertex AI | `teal-400/20 text-teal-300` |
| API Ninjas | `pink-400/20 text-pink-300` |
| Google Trends | `indigo-400/20 text-indigo-300` |
| Multi-dim Z-score | `red-400/20 text-red-300` |
| Monte Carlo | `slate-400/20 text-slate-300` |

---

## User Interaction Patterns

### Progressive Disclosure

1. **Top level:** ReportHeader shows score, action, and key metrics at a glance
2. **Tab level:** User selects area of interest (Overview, Signals, Debate, Risk, Audit)
3. **Card level:** Cards are expandable — click to reveal contributing sources, evidence, reasoning
4. **Detail level:** Decision traces expand to show full chain-of-thought, evidence citations, and source URLs

### Expand / Collapse

| Component | Expand Trigger | Expanded Content |
|-----------|---------------|-----------------|
| SignalDashboard cards | Click card | Full signal summary + raw data dump |
| EvaluationTable pillars | Click pillar | Description + contributing data sources |
| DecisionTraceView steps | Click timeline step | Evidence, reasoning, source URL, hash |
| DebateView contradictions | Always visible | Topic → bull view → bear view → resolution |

### PDF Export

The download button in the Overview tab generates a multi-page PDF containing all key sections of the report. The PDF uses `jsPDF` with automatic pagination, color-coded section headers, and a Goldman Sachs-style layout.

---

## Page Routes Summary

| Route | Page File | Key Components | Description |
|-------|-----------|---------------|-------------|
| `/login` | `login/page.tsx` | Google SSO + Passkey login, PyFinAgent branding | Auth gateway |
| `/` | `page.tsx` | **Home** — Portfolio Snapshot hero (4 cards), Recent Reports table, Quick Actions | Overview dashboard |
| `/analyze` | `analyze/page.tsx` | **Deep Analysis** — 15-step pipeline, 6-tab report (Overview/Signals/Debate/Risk/Audit/Cost) | Primary analysis page |
| `/signals` | `signals/page.tsx` | SignalDashboard, SectorDashboard, MacroDashboard | Standalone signal explorer |
| `/reports` | `reports/page.tsx` | **Tabbed** — History (searchable report list) + Compare (multi-report radar/price/pillar) | Reports + comparison |
| `/performance` | `performance/page.tsx` | Historical accuracy metrics + LLM cost history | Performance tracking |
| `/portfolio` | `portfolio/page.tsx` | Position tracking, P&L, allocation | Portfolio management |
| `/paper-trading` | `paper-trading/page.tsx` | Autonomous fund dashboard, NAV chart, positions, trades | Paper trading |
| `/backtest` | `backtest/page.tsx` | **Walk-Forward Backtest** — 5 strategies, 4 tabs (Results/Equity/Features/Optimizer), auto-ingest, DSR guard, feature drift detection, ingestion result banner with row counts, cost info section, button tooltips | ML backtesting |
| `/settings` | `settings/page.tsx` | **3-tab sub-navigation** (Models & Analysis \| Cost & Weights \| Performance): Model Config, Debate Depth, Cost Estimator, Pillar Weights, Cache Health, TTL Optimizer, API Latency | User preferences + monitoring |

### Sidebar Navigation (Grouped)

The sidebar uses grouped sections instead of a flat list:

```
Analyze
  Home          /             NavHome (House)
  Deep Analysis /analyze      NavAnalyze (Brain)
  Signals       /signals      NavSignals (Broadcast)

Reports
  Reports       /reports      NavReports (Files)
  Performance   /performance  NavPerformance (ChartLineUp)

Trading
  Portfolio     /portfolio    NavPortfolio (Wallet)
  Paper Trading /paper-trading NavPaperTrading (Robot)
  Backtest      /backtest     NavBacktest (ClockCounterClockwise)

[pinned at bottom]
  Settings      /settings     NavSettings (Gear)
```

Section headers render as `text-[10px] uppercase tracking-widest text-slate-600`. Settings is pinned above the user auth section.

---

## Conventions for Modifications

### Settings Page (v4.2 — 3-Tab Sub-Navigation)

**File:** `frontend/src/app/settings/page.tsx`

Settings management page with 3-tab pill-style sub-navigation. Loads current settings via `GET /api/settings/` and latest cost data via `GET /api/reports/latest-cost-summary`. Performance tab loads from `GET /api/perf/*` endpoints. Saves changed fields as a diff via `PUT /api/settings/` (Models & Cost tabs only — Performance tab is read-only).

**Tab Architecture:**

```
┌──────────────────────────────────────────────────────────────┐
│  [Models & Analysis]  [Cost & Weights]  [Performance]         │
│                                           Save All Settings ↗ │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Tab: Models & Analysis                                        │
│  ┌─────────────────────┬──────────────────────────────────────┐│
│  │  Analysis Mode       │  Model Configuration (col-span-2)   ││
│  │  [Full] [Lite]       │  ModelPicker + CostBadge             ││
│  ├─────────────────────┤                                      ││
│  │  Debate Depth        │  Standard + Deep Think pickers       ││
│  │  Bull↔Bear + Risk    │                                      ││
│  └─────────────────────┴──────────────────────────────────────┘│
│                                                                │
│  Tab: Cost & Weights                                           │
│  ┌─────────────────────┬──────────────────────────────────────┐│
│  │  Live Cost Estimator │  Pillar Weights (col-span-2)        ││
│  │  Est. $/analysis     │  5 weight sliders (must = 100%)     ││
│  ├─────────────────────┤                                      ││
│  │  Cost Controls       │                                      ││
│  │  Budget, Synth Iters │                                      ││
│  └─────────────────────┴──────────────────────────────────────┘│
│                                                                │
│  Tab: Performance                                              │
│  ┌─────────────────────┬──────────────────────────────────────┐│
│  │  Cache Health        │  API Latency (col-span-2)           ││
│  │  Hit/miss, clear btn │  Overall p50/p95/p99 + endpoint tbl ││
│  ├─────────────────────┤                                      ││
│  │  TTL Optimizer       │                                      ││
│  │  Status + start/stop │                                      ││
│  └─────────────────────┴──────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

**Tab Definitions:**

| Tab ID | Label | Cards | Editable |
|--------|-------|-------|----------|
| `models` | Models & Analysis | Analysis Mode, Debate Depth, Model Configuration (col-span-2) | Yes (Save button visible) |
| `cost` | Cost & Weights | Live Cost Estimator, Cost Controls, Pillar Weights (col-span-2) | Yes (Save button visible) |
| `performance` | Performance | Cache Health, TTL Optimizer, Optimization Progress Chart (col-span-2), API Latency (col-span-2) | No (Save button hidden) |

**Performance Tab Cards:**

| Card | Icon | Data Source | Key Behavior |
|------|------|-------------|-------------|
| **Cache Health** | `SettingsCache` (Database) | `GET /api/perf/cache` | Hit/miss counts, hit rate %, entry count. Clear Cache button → `POST /api/perf/cache/clear` with confirmation feedback |
| **TTL Optimizer** | `SettingsOptimizer` (GearSix) | `GET /api/perf/optimize/status` | Running/idle badge, iteration count, kept/discarded stats. Start → `POST /api/perf/optimize`, Stop → `POST /api/perf/optimize/stop` |
| **Optimization Progress** | — (chart) | `GET /api/perf/optimize/experiments` | `PerfProgressChart` component (col-span-2). Autoresearch-style Recharts ComposedChart: green dots (kept), gray dots (discarded), green step line (running best p95), text annotations on kept dots (karpathy/autoresearch `analysis.ipynb` style). Hover tooltip shows endpoint, TTL change, p95 change. Click expands detail panel below chart with timestamp, TTL before→after, p95 before→after (color-coded), hit rate |
| **API Latency** | `SettingsLatency` (Timer) | `GET /api/perf/summary` | 3 overall metric cards (p50/p95/p99 in ms), per-endpoint table sorted by p95 desc with color-coded latency badges (green <100ms, amber <500ms, red ≥500ms) |

**Settings-Specific Icons (10 aliases in `icons.ts`):**

| Alias | Phosphor Icon | Used In |
|-------|--------------|--------|
| `SettingsMode` | Lightning | Analysis Mode card header |
| `SettingsDebate` | ChatTeardropDots | Debate Depth card header |
| `SettingsModel` | Brain | Model Configuration card header |
| `SettingsCostControls` | ShieldCheck | Cost Controls card header |
| `SettingsEstimator` | CurrencyDollar | Cost Estimator card header |
| `SettingsPillars` | ChartBar | Pillar Weights card header |
| `SettingsCache` | Database | Cache Health card header |
| `SettingsOptimizer` | GearSix | TTL Optimizer card header |
| `SettingsLatency` | Timer | API Latency card header |
| `SettingsRefresh` | ArrowClockwise | Refresh buttons (with `animate-spin` when loading) |

**Data Flow:**
```
Page load:
  GET /api/settings/              → form state (all 19 fields incl. thinking budgets)
  GET /api/reports/latest-cost-summary → costData (per-agent tokens)
  GET /api/settings/models/available   → model list with pricing

Performance tab switch:
  Promise.all([
    GET /api/perf/cache             → cacheStats
    GET /api/perf/summary           → perfSummary (endpoints + overall)
    GET /api/perf/optimize/status   → optimizerStatus
    GET /api/perf/optimize/experiments → perfExperiments (for chart)
  ])

Save (Models & Cost tabs only):
  Compute diff (form vs saved settings) → PUT /api/settings/ (only changed fields)
  Backend validates, updates .env file, reloads Settings singleton
```

**v3.4 — VS Code Copilot Model Picker + Quota Multipliers**: Model Configuration BentoCard redesigned. `<select>/<optgroup>` replaced with `ModelPicker` — a searchable list where the selected model shows a checkmark. `CostBadge` replaces raw price display: for GitHub Models shows Copilot premium quota multiplier (`0.33x` light / `1x` standard / `3x` premium), for Gemini/Anthropic shows `$X.XX/1M`. Live Cost Estimator gains `~N Copilot premium requests` estimate when a GitHub Models model is selected. 24 models total: GPT-4.1/4.1-mini, o1/o1-mini/o3/o3-mini/o4-mini, Claude Sonnet 4/Opus 4, plus existing models. o1/o1-mini/o3-mini reclassified from OpenAI direct → GitHub Models.

**v3.4 Bug Fixes**: (1) Live activity step message "Gemini analyzing {name}..." now reflects the actual selected model (`_model_label = settings.gemini_model`). (2) o-series reasoning models (o1/o3/o4-prefix) use `max_completion_tokens` instead of `max_tokens` and omit `temperature` — fixes 400 errors from OpenAI/GitHub Models. (3) Small-context model token guard: `_MODEL_MAX_INPUT_CHARS` registry + proactive `enrichment_for_debate` compaction (drops `analysis` field, caps `summary` by mode) when model input limit < 30K chars; in Full Mode caps are 200 chars / 1,500 char fact ledger; in **Lite Mode** caps halve to 100 chars / 800 chars and ERROR/UNAVAILABLE/N/A signals are stripped entirely; safety-net truncation in `OpenAIClient` as a final fallback — fixes 413 `tokens_limit_reached` errors on o3-mini and gpt-4.1-mini. `context_limited: bool` added to `ModelPricing` (backend + TS types); `ModelPicker` shows amber `ctx limit` chip on restricted models and an amber warning banner when a context-limited model is selected as Standard Model.

**v3.4 Additional Token Hardening**: Standard GitHub Models like `gpt-4.1` also needed model-aware compaction in later pipeline stages because the API uses namespaced IDs such as `openai/gpt-4.1`. `llm_client.py` now normalizes those names before applying input budgets, and constrained runs use deterministic compact state instead of replaying full conversation history. In practice this means the debate Moderator receives a compact round summary plus shortened Bull/Bear/Devil's Advocate payloads, while Critic and synthesis revision receive a typed compact draft reference and reduced quant snapshot. UX implication: analyses on GitHub Models should fail less often at the Moderator/Critic stages, but highly constrained models may still see a compressed debate context in exchange for staying within provider limits.

**v3.3 — Extended Thinking toggle** (`enable_thinking`): Added to the **Model Configuration** BentoCard. When toggled on, reveals 4 thinking budget sliders (Critic, Moderator, Risk Judge, Synthesis). Only available when `deep_think_model` is set to `gemini-2.5-flash` or later; a warning is shown if enabled with `gemini-2.0-flash`.

### Modifying an Agent's Prompt (Skills System)

1. Edit the `## Prompt Template` section in the agent's skills.md file under `backend/agents/skills/`
2. Use `{{variable}}` syntax for runtime values (variables are injected by `format_skill()` in `prompts.py`)
3. Do NOT change `{{variable}}` placeholder names — they must match what the wrapper function provides
4. Do NOT change the output JSON schema requirements
5. The skill cache auto-invalidates on file modification time — no restart needed
6. For autonomous optimization, use `POST /api/skills/optimize` to start the experiment loop
7. The optimizer only modifies `## Prompt Template`, `## Skills & Techniques`, and `## Anti-Patterns` sections

### Adding a New Signal

1. Add the tool in `backend/tools/` returning `{ ticker, signal, summary, data }`
2. Add the enrichment agent prompt in `backend/config/prompts.py`
3. Wire into the orchestrator parallel enrichment step (Step 6-7) in `backend/agents/orchestrator.py`
4. Add the signal key to `EnrichmentSignals` in `frontend/src/lib/types.ts`
5. Add metadata entry to `SIGNAL_META` in `SignalDashboard.tsx` (label, icon, description, source, sourceTag)
6. Add signal value mapping to `signalColor()` and `signalSide()` if new signal values are introduced
7. Update `PdfDownload.tsx` if the signal should appear in the PDF

### Adding a New Tab

1. Add a `TabDef` entry to the `tabs` array in `page.tsx`
2. Add a `case` branch in the tab content switch inside `ReportTabs` children
3. Create or import the component to render
4. Extract any needed data from `report` for the badge and component props

### Modifying the Scoring Matrix

1. Update `ScoringMatrix` in both `backend/api/models.py` and `frontend/src/lib/types.ts`
2. Update pillar config in `EvaluationTable.tsx` (weights, icons, descriptions, sources)
3. Update `ScoringMatrixCard` in `GlassBoxCards.tsx`
4. Update the scoring section in `PdfDownload.tsx`
5. Ensure the synthesis prompt in `backend/config/prompts.py` reflects new pillar definitions

### Adding a New BigQuery Column

1. Add the column tuple `(name, BQ_type)` to `NEW_COLUMNS` in `migrate_bq_schema.py`
2. Run `python migrate_bq_schema.py` (idempotent — skips existing columns)
3. Add the corresponding parameter to `save_report()` in `backend/db/bigquery_client.py` (with `Optional` default)
4. Add the extraction logic in `backend/api/analysis.py :: _run_sync_analysis()` to populate the field from the report dict
5. Update `AGENTS.md` → "Data Persistence & ML Training" section with the new column
6. Update `UX-agent.md` → "Data Persistence: BigQuery" section with the new column

### Style Guidelines

- All cards use `bg-zinc-800/60 border border-zinc-700/50 rounded-xl` base styling
- Use semantic colors consistently: emerald=bullish, rose=bearish, amber=warning, sky=primary
- Never hardcode hex colors — use Tailwind utility classes only
- Source badges use the `SOURCE_COLORS` map from `SignalDashboard.tsx`
- All text uses `zinc-100` (primary) or `zinc-400` (secondary) — never `white` or `gray`

---

## Authentication UX (v2.8)

### Login Page

**File:** `frontend/src/app/login/page.tsx`

Full-page login screen with PyFinAgent branding. Two authentication methods:

| Method | Button | Flow |
|--------|--------|------|
| **Google SSO** | "Sign in with Google" (SVG icon) | NextAuth `signIn("google")` → Google OAuth → callback → redirect to `/` |
| **Passkey/WebAuthn** | "Sign in with Passkey" (Key icon) | `webAuthnSignIn()` → browser credential prompt → callback → redirect to `/` |

**Visual:** Dark theme (`bg-zinc-950`), centered card, PyFinAgent logo + tagline "AI-Powered Financial Analysis", generic error messages (never leaks auth details).

### Auth Middleware

**File:** `frontend/src/middleware.ts`

Route protection using Edge-compatible `auth.config.ts` (no Prisma adapter in Edge Runtime):
- **Protected**: All routes by default
- **Public**: `/login`, `/api/auth/*`, `/_next/*`, `/favicon.ico`
- **Behavior**: Unauthenticated → redirect to `/login`

### Sidebar Auth UI

**File:** `frontend/src/components/Sidebar.tsx`

Bottom section of the sidebar shows authenticated user info:
- User avatar (from Google) or initials fallback
- Display name + email
- "Register Passkey" button (calls `webAuthnSignIn("register")`)
- "Sign out" button (red hover, calls `signOut()`)

### API Client Auth

**File:** `frontend/src/lib/api.ts`

All API calls include `Authorization: Bearer <session-token>` header. On 401 response, redirects to `/login`. Uses `Cache-Control: no-store` to prevent stale auth data.

---

## Slack Bot UX (v2.8)

### Slash Commands

| Command | Response | Block Kit Format |
|---------|----------|-----------------|
| `/analyze <TICKER>` | Starts analysis, polls 5s intervals, posts formatted result | Score badge emoji, recommendation, investment thesis, key risks, debate consensus |
| `/portfolio` | Portfolio P&L summary | Total value, P&L with color, position list (up to 10) |
| `/report <TICKER>` | Latest report card | Score, recommendation, date, summary |

### Morning Digest

Automated cron job (configurable hour, default 8 AM) posts to configured Slack channel:
- Portfolio summary section (total value, P&L)
- Recent analyses section (last 7 days, up to 5 reports)

### Proactive Alerts

After each analysis completes, `send_analysis_alert()` posts the result to the configured channel with score, recommendation, and key metrics.

---

## Autonomous Paper Trading Dashboard (v2.9)

### Overview

**File:** `frontend/src/app/paper-trading/page.tsx`

Full-page dashboard for managing the autonomous AI trading fund. Displays fund status, open positions, trade history, and NAV performance chart. Controls for initializing the fund, starting/stopping the daily scheduler, and triggering manual trading cycles.

### Data Flow

```
Page load:
  GET /api/paper-trading/status      → portfolio state, scheduler status
  GET /api/paper-trading/portfolio    → open positions with unrealized P&L
  GET /api/paper-trading/trades       → trade history (latest 50)
  GET /api/paper-trading/snapshots    → daily NAV for chart data
  GET /api/paper-trading/performance  → Sharpe, win rate, alpha, costs

Actions:
  POST /api/paper-trading/start      → Init fund ($10,000) + start cron scheduler
  POST /api/paper-trading/stop        → Pause daily scheduler
  POST /api/paper-trading/run-now     → Trigger immediate daily cycle (async)
```

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  [NavPaperTrading] Autonomous Paper Trading                 │
├──────────────────────────────────────────────────────────┤
│  Status Banner                                            │
│  Scheduler: Running ● │ Last Run: 2026-03-10 10:00       │
│  [Initialize Fund] [Start Scheduler] [Pause] [Run Now]   │
├──────────────────────────────────────────────────────────┤
│  Summary Cards (6)                                        │
│  ┌─────────┬─────────┬─────────┬──────────┬────────┬────┐│
│  │  NAV    │  Cash   │  P&L    │  vs SPY  │ Sharpe │Pos ││
│  │$10,432  │ $2,100  │ +4.32%  │ +1.2%    │  1.45  │ 5  ││
│  └─────────┴─────────┴─────────┴──────────┴────────┴────┘│
├──────────────────────────────────────────────────────────┤
│  Tabs: [Positions] [Trades] [NAV Chart]                   │
│                                                           │
│  === Positions Tab ===                                    │
│  ┌────────┬─────┬────────┬─────────┬──────┬─────┬──────┐ │
│  │ Ticker │ Qty │Avg Cost│Mkt Value│ P&L  │ P&L%│ Days │ │
│  ├────────┼─────┼────────┼─────────┼──────┼─────┼──────┤ │
│  │ AAPL   │ 12  │$178.50 │$2,202   │+$60  │+2.8%│  14  │ │
│  │ MSFT   │  8  │$415.00 │$3,400   │+$80  │+2.4%│   7  │ │
│  └────────┴─────┴────────┴─────────┴──────┴─────┴──────┘ │
│                                                           │
│  === Trades Tab ===                                       │
│  ┌────────┬────────┬─────┬────────┬───────┬────────────┐ │
│  │ Date   │ Ticker │ Act │ Price  │ Value │ Reason     │ │
│  ├────────┼────────┼─────┼────────┼───────┼────────────┤ │
│  │ Mar 10 │ AAPL   │ BUY │$178.50 │$2,142 │ new_signal │ │
│  │ Mar 09 │ TSLA   │SELL │$245.00 │$1,225 │ stop_loss  │ │
│  └────────┴────────┴─────┴────────┴───────┴────────────┘ │
│                                                           │
│  === NAV Chart Tab ===                                    │
│  Recharts LineChart (3 lines):                            │
│  ── Portfolio NAV (sky-500)                               │
│  ── SPY Benchmark (zinc-500)                              │
│  ── Alpha (emerald-500)                                   │
└──────────────────────────────────────────────────────────┘
```

### Summary Cards (6)

| Card | Value Source | Color Logic |
|------|-------------|-------------|
| Net Asset Value | `status.portfolio.total_nav` | Always sky |
| Cash Available | `status.portfolio.current_cash` | Always slate |
| Total P&L | `status.portfolio.total_pnl_pct` | emerald if positive, rose if negative |
| vs SPY (Alpha) | `performance.alpha_pct` | emerald if positive, rose if negative |
| Sharpe Ratio | `performance.sharpe_ratio` | emerald if ≥1, amber if ≥0, rose if <0 |
| Positions | `positions.length` | Always violet |

### Positions Table

| Column | Source | Formatting |
|--------|--------|------------|
| Ticker | `position.ticker` | Bold, uppercase |
| Quantity | `position.quantity` | 2 decimal places |
| Avg Cost | `position.avg_entry_price` | Currency |
| Market Value | `position.market_value` | Currency |
| P&L | `position.unrealized_pnl` | Currency, green/red |
| P&L % | `position.unrealized_pnl_pct` | Percentage, green/red |
| Days Held | Computed from `position.entry_date` | Integer |

### Trades Table

| Column | Source | Formatting |
|--------|--------|------------|
| Date | `trade.created_at` | Short date |
| Ticker | `trade.ticker` | Bold |
| Action | `trade.action` | BUY = emerald badge, SELL = rose badge |
| Price | `trade.price` | Currency |
| Value | `trade.total_value` | Currency |
| Reason | `trade.reason` | Styled pill (new_buy_signal, stop_loss, signal_flip, rebalance) |
| P&L | `trade.pnl_pct` | Percentage, green/red (sells only) |

### NAV Chart

Recharts `LineChart` with responsive container. Three lines:
- **Portfolio** (`sky-500`): `snapshot.total_nav / starting_capital * 100`
- **SPY** (`zinc-500`): `100 + snapshot.benchmark_pnl_pct`
- **Alpha** (`emerald-500`): `snapshot.alpha_pct`

X-axis: `snapshot_date`, Y-axis: percentage return. Tooltip shows all three values.

### Action Buttons

| Button | Condition | Action | Style |
|--------|-----------|--------|-------|
| Initialize Fund | No portfolio exists | `POST /start` | sky (primary) |
| Start Scheduler | Portfolio exists, scheduler stopped | `POST /start` | emerald |
| Pause Scheduler | Scheduler running | `POST /stop` | amber |
| Run Now | Portfolio exists | `POST /run-now` | violet |

### Sidebar Entry

```typescript
// NAV_SECTIONS in Sidebar.tsx (grouped layout)
{ href: "/paper-trading", label: "Paper Trading", icon: NavPaperTrading }
{ href: "/backtest", label: "Backtest", icon: NavBacktest }
```

Added to the **Trading** section in `NAV_SECTIONS` grouped array in `Sidebar.tsx`. All nav icons are Phosphor Icon components (type `Icon`) from `@/lib/icons`.

---

## Walk-Forward Backtest Dashboard (v4.0 + v5.3)

### Overview

**File:** `frontend/src/app/backtest/page.tsx`

Full-page dashboard for walk-forward ML backtesting. Displays ingestion status, backtest results with per-window analytics, equity curve, feature importance, and quant strategy optimizer. Data is sourced from yfinance + FRED (free), stored in BigQuery (<$0.05), and backtests are ML-only ($0 LLM cost).

### Data Flow

```
Page load:
  GET /api/backtest/status              → backtest progress (status, run_id)
  GET /api/backtest/ingest/status       → row counts for 3 BQ tables
  GET /api/backtest/results             → full results with per-window analytics
  GET /api/backtest/optimize/status     → optimizer state
  GET /api/backtest/optimize/experiments → experiment history
  GET /api/backtest/optimize/best       → best strategy params

Actions:
  POST /api/backtest/ingest             → Bulk ingest historical data (auto-creates BQ tables)
  POST /api/backtest/run                → Start walk-forward backtest (async)
  POST /api/backtest/optimize           → Start quant strategy optimizer (background)
  POST /api/backtest/optimize/stop      → Stop optimizer gracefully
```

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  [NavBacktest] Walk-Forward Backtest                      │
├──────────────────────────────────────────────────────────┤
│  Ingestion Metrics (3 cards)                              │
│  ┌──────────────┬──────────────────┬────────────────────┐│
│  │ Prices Rows  │ Fundamentals Rows│  Macro Rows        ││
│  │ 125,432      │ 4,200            │  2,100             ││
│  └──────────────┴──────────────────┴────────────────────┘│
│  Data: yfinance + FRED (free) · BQ storage <$0.05 ·      │
│  Backtest: ML only, $0 LLM cost                          │
├──────────────────────────────────────────────────────────┤
│  [Ingest Data ⓘ] [Run Backtest ⓘ]                        │
├──────────────────────────────────────────────────────────┤
│  Ingestion Result Banner (conditional)                    │
│  ┌─ ✓ emerald ─────────────────────────────────────── ×─┐│
│  │ Ingestion complete: 125,432 prices, 4,200              │
│  │ fundamentals, 2,100 macro rows inserted               ││
│  └──────────────────────────────────────────────────────┘│
│  ┌─ ✗ rose ─────────────────────────────────────── ×────┐│
│  │ Ingestion failed: <error message>                     ││
│  └──────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────┤
│  Tabs: [Results] [Equity Curve] [Features] [Optimizer]    │
│                                                           │
│  === Results Tab ===                                      │
│  Analytics summary cards + per-window results table       │
│                                                           │
│  === Equity Curve Tab ===                                 │
│  Recharts LineChart (portfolio vs baselines)              │
│                                                           │
│  === Features Tab ===                                     │
│  Feature importance bar chart (MDA + MDI)                 │
│                                                           │
│  === Optimizer Tab ===                                    │
│  PerfProgressChart + experiment table + best params       │
└──────────────────────────────────────────────────────────┘
```

### Ingestion Result Banner (v5.3)

Displayed after `POST /api/backtest/ingest` completes. Managed by `ingestResult` state (`{ type: "success" | "error"; message: string } | null`).

| State | Style | Content |
|-------|-------|---------|
| Success | `bg-emerald-500/10 border-emerald-500/30 text-emerald-400` | "Ingestion complete: X prices, Y fundamentals, Z macro rows inserted" |
| Error | `bg-rose-500/10 border-rose-500/30 text-rose-400` | "Ingestion failed: {error message}" |

Dismiss button (`×`) sets `ingestResult` to `null`.

### Cost Info Section (v5.3)

Inline text below ingestion metric cards:

```
Data: yfinance + FRED (free) · BQ storage <$0.05 · Backtest: ML only, $0 LLM cost
```

Styled as `text-xs text-zinc-500 mt-2`. Appears always (not conditional).

### Button Tooltips (v5.3)

| Button | Tooltip (`title` attribute) |
|--------|----------------------------|
| Ingest Data | "Download S&P 500 price history, quarterly fundamentals, and FRED macro data from yfinance + FRED (free APIs). Stores in BigQuery (<$0.05). Takes ~5-15 minutes on first run." |
| Run Backtest | "Run walk-forward ML backtest using GradientBoosting. No LLM cost — uses only quantitative features. BQ reads <$0.01 per run. Takes ~2-5 minutes." |

### Auto-Table Creation (v5.3)

`POST /api/backtest/ingest` now auto-creates missing BQ tables (`historical_prices`, `historical_fundamentals`, `historical_macro`) before ingestion begins. No need to run `migrate_backtest_data.py` manually for new deployments. Existing tables get the new `dividends_per_share` column via the migration script.
