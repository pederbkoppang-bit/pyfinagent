/**
 * Shared TypeScript types for the PyFinAgent frontend.
 */

export type AnalysisStatus = "pending" | "running" | "completed" | "failed";

export interface ScoringMatrix {
  pillar_1_corporate: number;
  pillar_2_industry: number;
  pillar_3_valuation: number;
  pillar_4_sentiment: number;
  pillar_5_governance: number;
}

export interface RecommendationDetail {
  action: string;
  justification: string;
}

export interface SynthesisReport {
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

export interface AnalysisResponse {
  analysis_id: string;
  ticker: string;
  status: AnalysisStatus;
}

export interface AnalysisStatusResponse {
  analysis_id: string;
  ticker: string;
  status: AnalysisStatus;
  current_step?: string;
  steps_completed: string[];
  error?: string;
  report?: SynthesisReport;
}

export interface ReportSummary {
  ticker: string;
  company_name?: string;
  analysis_date: string;
  final_score: number;
  recommendation: string;
  summary: string;
}

export interface PerformanceStats {
  total_recommendations: number;
  wins: number;
  losses: number;
  avg_return: number;
  win_rate: number;
  benchmark_beat_rate: number;
}

// ── Portfolio Types ──────────────────────────────────────────────

export interface PortfolioPosition {
  id: string;
  ticker: string;
  quantity: number;
  avg_entry_price: number;
  cost_basis: number;
  current_price?: number;
  market_value?: number;
  unrealized_pnl?: number;
  unrealized_pnl_pct?: number;
  recommendation?: string;
  recommendation_score?: number;
  added_at: string;
}

export interface PortfolioPerformance {
  total_cost_basis: number;
  total_market_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  positions_count: number;
  recommendation_accuracy?: number;
  allocation: Array<{
    ticker: string;
    market_value: number;
    pnl: number;
    pnl_pct: number;
  }>;
}

// ── Enrichment Signal Types ─────────────────────────────────────

export interface SignalSummary {
  signal: string;
  summary: string;
}

export interface EnrichmentSignals {
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

export interface InsiderData {
  ticker: string;
  signal: string;
  summary: string;
  cluster_buys: Array<{ name: string; date: string; shares: number; value: number }>;
  total_buy_value: number;
  total_sell_value: number;
  buy_sell_ratio: number;
  recent_trades: Array<Record<string, unknown>>;
}

export interface OptionsData {
  ticker: string;
  signal: string;
  summary: string;
  overall_pc_ratio: number;
  expirations_analyzed: number;
  unusual_activity: Array<Record<string, unknown>>;
}

export interface SectorData {
  ticker: string;
  company_name: string;
  sector: string;
  industry: string;
  sector_etf: string | null;
  stock_returns: Record<string, number>;
  sector_returns: Record<string, number>;
  spy_returns: Record<string, number>;
  relative_vs_sector: Record<string, number>;
  relative_vs_market: Record<string, number>;
  sector_performance: Record<string, number>;
  peers: Array<{
    ticker: string;
    name: string;
    market_cap: number;
    pe_ratio: number;
    revenue_growth: number;
    profit_margin: number;
    roe: number;
  }>;
  signal: string;
  summary: string;
}

export interface AllSignals {
  ticker: string;
  company_name: string;
  insider: InsiderData;
  options: OptionsData;
  social_sentiment: Record<string, unknown>;
  patent: Record<string, unknown>;
  earnings_tone: Record<string, unknown>;
  fred_macro: Record<string, unknown>;
  alt_data: Record<string, unknown>;
  sector: SectorData;
}

// ── Debate Types ─────────────────────────────────────────────────

export interface DebateContradiction {
  topic: string;
  bull_view: string;
  bear_view: string;
  resolution: string;
  winner?: string;
}

export interface DebateDissent {
  agent: string;
  position: string;
  reason: string;
}

export interface DebateCaseDetail {
  thesis: string;
  confidence: number;
  key_catalysts?: string[];
  key_threats?: string[];
  evidence?: Array<{ source: string; data_point: string; interpretation: string }>;
}

export interface DebateResult {
  bull_case: DebateCaseDetail;
  bear_case: DebateCaseDetail;
  consensus: string;
  consensus_confidence: number;
  contradictions: DebateContradiction[];
  dissent_registry: DebateDissent[];
  moderator_analysis?: string;
}

// ── Risk & Monte Carlo Types ─────────────────────────────────────

export interface MonteCarloHorizon {
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

export interface MonteCarloData {
  ticker: string;
  signal: string;
  summary: string;
  current_price?: number;
  annualized_volatility?: number;
  horizons?: Record<string, MonteCarloHorizon>;
}

export interface AnomalyItem {
  metric: string;
  value: number;
  z_score: number;
  mean?: number;
  std?: number;
  note?: string;
}

export interface AnomalyData {
  ticker: string;
  signal: string;
  summary: string;
  anomaly_count?: number;
  anomalies?: AnomalyItem[];
}

export interface RiskData {
  monte_carlo: MonteCarloData;
  anomalies: AnomalyData;
}

// ── Decision Trace (XAI) ─────────────────────────────────────────

export interface DecisionTrace {
  agent_name: string;
  timestamp: string;
  input_data_hash: string;
  output_signal: string;
  confidence: number;
  evidence_citations: string[];
  reasoning_steps: string[];
  latency_ms: number;
  source_url?: string;
}

// ── Bias & Conflict Detection ────────────────────────────────────

export interface BiasFlag {
  bias_type: string;
  severity: string;
  description: string;
  evidence: string;
  adjustment_suggestion?: string;
}

export interface BiasReportData {
  flags: BiasFlag[];
  raw_score?: number;
  adjusted_score?: number;
  bias_count: number;
}

export interface KnowledgeConflict {
  field: string;
  llm_belief: string;
  actual_value: string;
  severity: string;
  category: string;
  explanation: string;
}

export interface ConflictReportData {
  conflicts: KnowledgeConflict[];
  conflict_count: number;
  overall_reliability: string;
}
