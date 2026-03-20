/**
 * Shared TypeScript types for the PyFinAgent frontend.
 */

export interface AuthUser {
  id: string;
  name?: string | null;
  email?: string | null;
  image?: string | null;
}

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
  citations?: Citation[];
  grounding_sources?: {
    market?: GroundingSource[];
    competitor?: GroundingSource[];
    enhanced_macro?: GroundingSource[];
    deep_dive?: GroundingSource[];
  };
  final_weighted_score?: number;
  enrichment_signals?: EnrichmentSignals;
  debate_result?: DebateResult;
  decision_traces?: DecisionTrace[];
  risk_data?: RiskData;
  bias_report?: BiasReportData;
  conflict_report?: ConflictReportData;
  risk_assessment?: RiskAssessment;
  info_gap_report?: InfoGapReport;
  cost_summary?: CostSummary;
}

export interface AnalysisResponse {
  analysis_id: string;
  ticker: string;
  status: AnalysisStatus;
}

export interface StepLogEntry {
  step: string;
  status: string;
  message: string;
  timestamp: string;
}

export interface AnalysisStatusResponse {
  analysis_id: string;
  ticker: string;
  status: AnalysisStatus;
  current_step?: string;
  steps_completed: string[];
  message?: string;
  step_log?: StepLogEntry[];
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
  // Multi-round debate additions (backward-compatible)
  debate_rounds?: DebateRound[];
  total_rounds?: number;
  devils_advocate?: DevilsAdvocate;
}

export interface DebateRound {
  round: number;
  bull_argument: string;
  bear_argument: string;
}

export interface DevilsAdvocate {
  challenges: string[];
  hidden_risks?: string[];
  bull_weakness?: string;
  bear_weakness?: string;
  groupthink_flag?: string;
  confidence_adjustment?: number;
  summary?: string;
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
  thoughts?: string;          // Phase 5: extended thinking output
  grounding_sources?: GroundingSource[]; // Phase 4: web grounding sources
}

// ── Citation & Grounding Types (Phase 6) ────────────────────────

/** A traceable data citation linking a synthesis claim to its source system. */
export interface Citation {
  claim: string;
  source: string; // YFIN | SEC | FRED | AV | URL
  value: string;
}

/** A web grounding source returned by Google Search Retrieval (Phase 4). */
export interface GroundingSource {
  title?: string;
  uri?: string;
  text?: string;
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

// ── Risk Assessment Team Types ───────────────────────────────────

export interface RiskAnalystCase {
  position: string;
  confidence: number;
  max_position_pct?: number;
  [key: string]: unknown;
}

export interface RiskJudgeVerdict {
  decision: string;
  risk_adjusted_confidence: number;
  recommended_position_pct: number;
  risk_level: string;
  reasoning: string;
  risk_limits?: { stop_loss_pct?: number; max_drawdown_pct?: number };
  unresolved_risks?: string[];
  summary: string;
}

export interface RiskAssessment {
  aggressive: RiskAnalystCase;
  conservative: RiskAnalystCase;
  neutral: RiskAnalystCase;
  judge: RiskJudgeVerdict;
  risk_debate_rounds?: RiskDebateRound[];
  total_risk_rounds?: number;
}

export interface RiskDebateRound {
  round: number;
  aggressive_argument: string;
  conservative_argument: string;
  neutral_argument: string;
}

// ── Info-Gap Detection Types ─────────────────────────────────────

export interface InfoGap {
  source: string;
  status: string;
  criticality: string;
  impact: string;
}

export interface InfoGapReport {
  gaps: InfoGap[];
  data_quality_score: number;
  critical_gaps: string[];
  recommendation_at_risk: boolean;
  summary: string;
}

// ── Cost Tracking Types ──────────────────────────────────────────

export interface AgentCostEntry {
  agent_name: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
  is_deep_think: boolean;
}

export interface ModelBreakdown {
  calls: number;
  tokens: number;
  cost_usd: number;
}

export interface CostSummary {
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_cost_usd: number;
  total_calls: number;
  deep_think_calls: number;
  model_breakdown: Record<string, ModelBreakdown>;
  agents: AgentCostEntry[];
}

export interface CostHistoryEntry {
  ticker: string;
  analysis_date: string;
  total_tokens: number | null;
  total_cost_usd: number | null;
  deep_think_calls: number | null;
}

export interface ModelPricing {
  model: string;
  input_per_1m: number;
  output_per_1m: number;
}

export interface ModelConfig {
  gemini_model: string;
  deep_think_model: string;
  max_debate_rounds: number;
  max_risk_debate_rounds: number;
}

export interface FullSettings {
  gemini_model: string;
  deep_think_model: string;
  max_debate_rounds: number;
  max_risk_debate_rounds: number;
  weight_corporate: number;
  weight_industry: number;
  weight_valuation: number;
  weight_sentiment: number;
  weight_governance: number;
  data_quality_min: number;
  lite_mode: boolean;
  max_analysis_cost_usd: number;
  max_synthesis_iterations: number;
  // Phase 5 — Extended thinking (Gemini 2.5+)
  enable_thinking?: boolean;
  thinking_budget_critic?: number;
  thinking_budget_moderator?: number;
  thinking_budget_risk_judge?: number;
  thinking_budget_synthesis?: number;
}

export interface LatestCostSummary extends CostSummary {
  ticker?: string;
  analysis_date?: string;
}

// ── Paper Trading Types ──────────────────────────────────────────

export interface PaperPortfolio {
  portfolio_id: string;
  starting_capital: number;
  current_cash: number;
  total_nav: number | null;
  total_pnl_pct: number | null;
  benchmark_return_pct: number | null;
  inception_date: string;
  updated_at: string | null;
}

export interface PaperPosition {
  position_id: string;
  ticker: string;
  quantity: number;
  avg_entry_price: number;
  cost_basis: number | null;
  current_price: number | null;
  market_value: number | null;
  unrealized_pnl: number | null;
  unrealized_pnl_pct: number | null;
  entry_date: string;
  last_analysis_date: string | null;
  recommendation: string | null;
  risk_judge_position_pct: number | null;
  stop_loss_price: number | null;
}

export interface PaperTrade {
  trade_id: string;
  ticker: string;
  action: "BUY" | "SELL";
  quantity: number;
  price: number;
  total_value: number | null;
  transaction_cost: number | null;
  reason: string | null;
  analysis_id: string | null;
  risk_judge_decision: string | null;
  created_at: string;
}

export interface PaperSnapshot {
  snapshot_date: string;
  total_nav: number;
  cash: number;
  positions_value: number | null;
  daily_pnl_pct: number | null;
  cumulative_pnl_pct: number | null;
  benchmark_pnl_pct: number | null;
  alpha_pct: number | null;
  position_count: number | null;
  trades_today: number | null;
  analysis_cost_today: number | null;
}

export interface PaperTradingStatus {
  status: string;
  portfolio: {
    nav: number | null;
    cash: number | null;
    starting_capital: number | null;
    pnl_pct: number | null;
    benchmark_return_pct: number | null;
    inception_date: string | null;
    updated_at: string | null;
  };
  position_count: number;
  scheduler_active: boolean;
  loop: {
    running: boolean;
    last_run: string | null;
    last_result: Record<string, unknown> | null;
  };
}

export interface PaperPerformance {
  nav: number | null;
  starting_capital: number | null;
  pnl_pct: number | null;
  benchmark_return_pct: number | null;
  alpha_pct: number | null;
  sharpe_ratio: number;
  total_sell_trades: number;
  total_buy_trades: number;
  total_analysis_cost: number;
  days_active: number;
}
