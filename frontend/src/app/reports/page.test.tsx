import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen, fireEvent, waitFor } from "@testing-library/react";
import ReportsPage from "./page";
import type { ReportSummary } from "@/lib/types";

// phase-75.12 (frontend-03): reports/page.tsx's compare-charts flow
// previously raw-`fetch`ed /api/charts/<ticker> and swallowed both
// non-ok responses and thrown errors ("ignore chart failures"), leaving
// a silently empty/partial chart with no operator-visible signal. This
// test drives the real compare flow (tab switch -> drawer selection ->
// Compare click) and asserts a visible partial-failure notice renders
// when one ticker's series fails while another succeeds.

function rs(ticker: string, analysis_date: string): ReportSummary {
  return {
    ticker,
    company_name: `${ticker} Inc.`,
    analysis_date,
    final_score: 7.5,
    recommendation: "BUY",
    summary: "test",
  };
}

const REPORTS = [rs("AAPL", "2026-05-20"), rs("MSFT", "2026-05-19")];

vi.mock("@/components/Sidebar", () => ({
  Sidebar: () => <div data-testid="sidebar-stub" />,
}));

// Stable reference across renders -- useURLState's effect is keyed on the
// `params` object's identity (its dep array is [key, params]), so a mock
// that returns `new URLSearchParams()` fresh each render makes that effect
// fire on EVERY render and immediately revert any tab-click state change.
const stableSearchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => stableSearchParams,
  usePathname: () => "/reports",
}));

vi.mock("@/lib/api", () => ({
  listReports: vi.fn(() => Promise.resolve(REPORTS)),
  getReport: vi.fn((ticker: string, analysisDate?: string) =>
    Promise.resolve({
      ticker,
      company_name: `${ticker} Inc.`,
      analysis_date: analysisDate,
      full_report_json: {
        final_synthesis: { final_weighted_score: 7.5, scoring_matrix: {} },
      },
    }),
  ),
  getChartData: vi.fn((ticker: string) =>
    ticker === "AAPL"
      ? Promise.resolve([{ Date: "2026-01-01", Close: 100 }])
      : Promise.reject(new Error("no chart data")),
  ),
}));

afterEach(() => cleanup());

describe("Reports compare partial-failure notice (phase-75.12 frontend-03)", () => {
  it("renders a visible notice naming the ticker whose chart failed, never a silently empty chart", async () => {
    render(<ReportsPage />);

    await waitFor(() => {
      expect(screen.getAllByText("AAPL").length).toBeGreaterThan(0);
    });

    fireEvent.click(screen.getByRole("tab", { name: /compare/i }));
    fireEvent.click(screen.getByRole("button", { name: /select reports to compare/i }));

    const dialog = await screen.findByRole("dialog");
    const aaplRow = Array.from(dialog.querySelectorAll("button[aria-pressed]")).find((b) =>
      b.textContent?.includes("AAPL"),
    )!;
    const msftRow = Array.from(dialog.querySelectorAll("button[aria-pressed]")).find((b) =>
      b.textContent?.includes("MSFT"),
    )!;
    fireEvent.click(aaplRow);
    fireEvent.click(msftRow);

    const compareBtn = Array.from(dialog.querySelectorAll("button")).find(
      (b) => b.textContent === "Compare",
    )!;
    fireEvent.click(compareBtn);

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
    expect(screen.getByRole("alert").textContent).toMatch(/MSFT/);
    expect(screen.getByRole("alert").textContent).not.toMatch(/AAPL/);
  });
});
