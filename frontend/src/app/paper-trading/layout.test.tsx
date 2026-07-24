import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen, waitFor } from "@testing-library/react";
import PaperTradingLayout from "./layout";

// phase-75.12 (fe-ts-01): the backend's not_initialized payload
// ({status, message} -- NO loop/portfolio/scheduler_active/next_run keys,
// paper_trading.py:134) previously crashed this layout with a TypeError
// on every fresh/reset install, because `status?.loop.running` only
// guarded `status` being null, not `loop` being undefined. This test
// reproduces that exact payload shape and asserts the layout renders
// its "not initialized" placeholder instead of throwing.

vi.mock("@/lib/api", () => ({
  getPaperTradingStatus: vi.fn(() =>
    Promise.resolve({ status: "not_initialized", message: "Call POST /start first" }),
  ),
  getPaperPortfolio: vi.fn(() => Promise.resolve(null)),
  getPaperTrades: vi.fn(() => Promise.resolve({ trades: [] })),
  getPaperSnapshots: vi.fn(() => Promise.resolve({ snapshots: [] })),
  getPaperPerformance: vi.fn(() => Promise.resolve(null)),
  startPaperTrading: vi.fn(),
  stopPaperTrading: vi.fn(),
  triggerPaperTradingCycle: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => "/paper-trading/positions",
}));

vi.mock("@/lib/useTickerMeta", () => ({
  useTickerMeta: () => ({ meta: {} }),
}));

vi.mock("@/lib/live-portfolio-context", () => ({
  useLivePortfolio: () => ({ livePrices: {}, liveNav: null, liveTotalPnlPct: null }),
}));

vi.mock("@/components/Sidebar", () => ({
  Sidebar: () => <div data-testid="sidebar-stub" />,
}));

vi.mock("@/components/AgentRationaleDrawer", () => ({
  AgentRationaleDrawer: () => null,
}));

vi.mock("@/components/Skeleton", () => ({
  PageSkeleton: () => <div data-testid="skeleton-stub" />,
}));

afterEach(() => cleanup());

describe("PaperTradingLayout not_initialized payload (phase-75.12 fe-ts-01)", () => {
  it("renders the placeholder without throwing", async () => {
    render(<PaperTradingLayout>{null}</PaperTradingLayout>);

    await waitFor(() => {
      expect(screen.getByText(/No paper portfolio initialized/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /initialize fund/i })).toBeInTheDocument();
  });
});
