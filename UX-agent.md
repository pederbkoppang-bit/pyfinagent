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

## Page Layout & Tab Architecture

The main dashboard (`/`) renders in three visual layers:

```
┌──────────────────────────────────────────────────────────┐
│  Ticker Input Bar + "Analyze" Button                      │
├──────────────────────────────────────────────────────────┤
│  AnalysisProgress (13-step tracker, visible during run)   │
├──────────────────────────────────────────────────────────┤
│  ReportHeader (always visible when report exists)         │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Score Ring │ Action Badge │ Metrics │ 52w Range     │  │
│  └────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│  ReportTabs                                               │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐    │
│  │Overview │ Signals │ Debate  │  Risk   │  Audit  │    │
│  │   📋    │  📡 11  │ ⚖️ BUY │ 🎯 3   │ 🔍 1   │    │
│  └─────────┴─────────┴─────────┴─────────┴─────────┘    │
│                                                           │
│  [ Active Tab Content ]                                   │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Tab Definitions

| Tab ID | Label | Icon | Badge Source | Components Rendered |
|--------|-------|------|-------------|---------------------|
| `overview` | Overview | 📋 | — | InvestmentThesisCard, EvaluationTable, ValuationRange, RisksCard, ScoringMatrixCard, PdfDownload |
| `signals` | Signals | 📡 | Count of enrichment signals (0–11) | SignalDashboard |
| `debate` | Debate | ⚖️ | Consensus label (e.g. "BUY") | DebateView |
| `risk` | Risk | 🎯 | Anomaly count (e.g. "3 anomalies") | RiskDashboard, StockChart |
| `audit` | Audit | 🔍 | Bias flag count (e.g. "1 flags") | BiasReport, DecisionTraceView, ResearchInvestigator |

---

## Data Flow: Backend → Frontend

```
Orchestrator (13-step pipeline)
    │
    ▼
SynthesisReport (Pydantic model, backend/api/models.py)
    │
    ├── scoring_matrix          → EvaluationTable, ScoringMatrixCard
    ├── recommendation          → ReportHeader (action badge), InvestmentThesisCard
    ├── final_summary           → InvestmentThesisCard (multi-paragraph)
    ├── key_risks               → RisksCard
    ├── final_weighted_score    → ReportHeader (score ring)
    ├── enrichment_signals      → SignalDashboard (11 signal cards + consensus bar)
    ├── debate_result           → DebateView, InvestmentThesisCard (catalysts/threats)
    ├── decision_traces         → DecisionTraceView (XAI timeline)
    ├── risk_data               → RiskDashboard (Monte Carlo + anomalies)
    ├── bias_report             → BiasReport (bias flags + raw/adjusted score)
    └── conflict_report         → BiasReport (knowledge conflict table)

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
class SynthesisReport(BaseModel):
    scoring_matrix: ScoringMatrix             # 5-pillar scores (1-10)
    recommendation: RecommendationDetail      # action + justification
    final_summary: str                        # Multi-paragraph analyst summary
    key_risks: list[str]                      # Risk bullet points
    final_weighted_score: Optional[float]     # 0.0–10.0 weighted score
    # v2 enrichment fields
    enrichment_signals: Optional[dict]        # 11 signal results
    debate_result: Optional[dict]             # Bull/Bear/Moderator consensus
    decision_traces: Optional[list]           # Agent XAI audit trail
    risk_data: Optional[dict]                 # Monte Carlo + anomalies
    bias_report: Optional[dict]              # LLM bias flags
    conflict_report: Optional[dict]          # Knowledge conflict flags
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
  id: string;       // "overview" | "signals" | "debate" | "risk" | "audit"
  label: string;
  icon: string;     // Emoji icon
  badge?: string | number | null;
}
```

**Styling:** Active tab uses `bg-sky-500/15 text-sky-400` with border highlight. Badges render inline with optional count/label.

---

### 3. SignalDashboard

**File:** `frontend/src/components/SignalDashboard.tsx`
**Props:** `{ signals: EnrichmentSignals }`

FactSet-style panel showing all 11 enrichment signals with source attribution, expandable detail, and a consensus divergence bar.

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

**11 Signal Cards:**

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
| Corporate Profile | `pillar_1_corporate` | 35% | 🏢 | RAG Agent (10-K/10-Q), yfinance fundamentals |
| Industry & Macro | `pillar_2_industry` | 20% | 🌐 | Sector Analysis, FRED Macro, Competitor Agent |
| Valuation | `pillar_3_valuation` | 20% | 💰 | yfinance valuation metrics, quant-agent data |
| Market Sentiment | `pillar_4_sentiment` | 15% | 📊 | NLP Sentiment, Social Sentiment, Insider Activity, Options Flow |
| Governance | `pillar_5_governance` | 10% | 🏛️ | RAG Agent governance section, insider ownership |

**Visual Features:**
- Progress bar per pillar (color: emerald ≥8, sky ≥6, amber ≥4, rose <4)
- Delta indicators (▲/▼ with green/red) when `previousScores` is provided
- Weighted total displayed in header
- Click-to-expand reveals description + contributing agent/source list

---

### 6. DebateView

**File:** `frontend/src/components/DebateView.tsx`
**Props:** `{ debate: DebateResult }`

Adversarial debate visualization showing the Bull vs Bear argument structure, evidence provenance, contradictions, and dissent registry.

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│  Consensus: BUY  ████████████░░░░ 72% confidence     │
├─────────────────────────┬────────────────────────────┤
│  🐂 BULL CASE (78%)    │  🐻 BEAR CASE (45%)       │
│  ─────────────────────  │  ─────────────────────     │
│  Thesis paragraph...    │  Thesis paragraph...       │
│                         │                            │
│  Key Catalysts:         │  Key Threats:              │
│  • Catalyst 1           │  • Threat 1                │
│  • Catalyst 2           │  • Threat 2                │
│                         │                            │
│  📎 Evidence:           │  📎 Evidence:              │
│  [SEC EDGAR] data → ..  │  [yfinance] data → ..     │
│  [USPTO] data → interp  │  [FRED] data → interp     │
├─────────────────────────┴────────────────────────────┤
│  ⚠️ Contradictions                                    │
│  ┌──────────┬──────────────┬──────────────┬────────┐ │
│  │  Topic   │  Bull View   │  Bear View   │ Winner │ │
│  ├──────────┼──────────────┼──────────────┼────────┤ │
│  │ Margins  │ Expanding..  │ Pressure..   │ BULL   │ │
│  └──────────┴──────────────┴──────────────┴────────┘ │
├──────────────────────────────────────────────────────┤
│  Dissent: [Options Flow: BEARISH — "P/C ratio 1.4"]  │
└──────────────────────────────────────────────────────┘
```

**Evidence Format:** Each evidence item shows `[source badge]` → `data_point` → `interpretation` with color-coded background (emerald for bull, rose for bear).

---

### 7. RiskDashboard

**File:** `frontend/src/components/RiskDashboard.tsx`
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
| `tech_bias` | 🖥️ | Systematic tech/large-cap favoritism |
| `confirmation_bias` | 🔄 | Ignoring contradictory signals |
| `recency_bias` | ⏰ | Over-weighting recent data |
| `anchoring` | ⚓ | Over-relying on first data point |
| `source_diversity` | 📊 | Insufficient source variety |

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
| `AnalysisProgress` | `AnalysisProgress.tsx` | 13-step real-time tracker with progress bar, spinner, emoji status icons |
| `StockChart` | `StockChart.tsx` | Candlestick + volume with toggleable SMA50/SMA200/RSI, 5 period options |
| `AlphaScoreCard` | `GlassBoxCards.tsx` | Large score display with recommendation bar (used in legacy/compare views) |
| `ScoringMatrixCard` | `GlassBoxCards.tsx` | 5 horizontal progress bars (one per pillar) with weights |
| `RisksCard` | `GlassBoxCards.tsx` | Rose-themed risk bullet list |
| `ValuationRange` | `GlassBoxCards.tsx` | Valuation metric range visualization |
| `ResearchInvestigator` | `ResearchInvestigator.tsx` | RAG-powered investigation query against 10-K documents |
| `Sidebar` | `Sidebar.tsx` | Navigation sidebar for all routes |
| `MacroDashboard` | `MacroDashboard.tsx` | 7-indicator FRED grid + macro warnings (used on /signals page) |
| `SectorDashboard` | `SectorDashboard.tsx` | Sector rotation chart + relative strength table (used on /signals page) |

---

## TypeScript Type Reference

All types are defined in `frontend/src/lib/types.ts`.

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

| Route | Page File | Key Components | Shares Report Data |
|-------|-----------|---------------|-------------------|
| `/` | `page.tsx` | **Full Report Dashboard** — ReportHeader + 5 tabs | Primary analysis page |
| `/signals` | `signals/page.tsx` | SignalDashboard, SectorDashboard, MacroDashboard | Standalone signal explorer |
| `/compare` | `compare/page.tsx` | Multi-report comparison with radar chart | Past report comparison |
| `/reports` | `reports/page.tsx` | Searchable report list | Historical reports |
| `/performance` | `performance/page.tsx` | Historical accuracy metrics | Performance tracking |
| `/portfolio` | `portfolio/page.tsx` | Position tracking, P&L, allocation | Portfolio management |
| `/settings` | `settings/page.tsx` | Pillar weight configuration | User preferences |

---

## Conventions for Modifications

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

### Style Guidelines

- All cards use `bg-zinc-800/60 border border-zinc-700/50 rounded-xl` base styling
- Use semantic colors consistently: emerald=bullish, rose=bearish, amber=warning, sky=primary
- Never hardcode hex colors — use Tailwind utility classes only
- Source badges use the `SOURCE_COLORS` map from `SignalDashboard.tsx`
- All text uses `zinc-100` (primary) or `zinc-400` (secondary) — never `white` or `gray`
