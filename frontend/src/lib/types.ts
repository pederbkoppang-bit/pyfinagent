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
