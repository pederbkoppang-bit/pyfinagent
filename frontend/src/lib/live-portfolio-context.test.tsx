import { describe, it, expect, afterEach, vi } from "vitest";
import { renderHook, cleanup } from "@testing-library/react";
import {
  LivePortfolioProvider,
  useLivePortfolio,
  useLivePortfolioOptional,
} from "./live-portfolio-context";
import type { ReactNode } from "react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

vi.mock("@/lib/api", () => ({
  getPaperTradingStatus: vi.fn(() => Promise.resolve(null)),
  getPaperPortfolio: vi.fn(() => Promise.resolve(null)),
  getPaperSnapshots: vi.fn(() => Promise.resolve({ snapshots: [] })),
}));

vi.mock("@/lib/useLivePrices", () => ({
  useLivePrices: () => ({ prices: {} }),
}));

vi.mock("@/lib/useTickerMeta", () => ({
  useTickerMeta: () => ({ meta: {} }),
}));

vi.mock("@/lib/useLiveNav", () => ({
  useLiveNav: () => ({ liveNav: null, liveTotalPnlPct: null }),
}));

const wrapper = ({ children }: { children: ReactNode }) => (
  <LivePortfolioProvider>{children}</LivePortfolioProvider>
);

describe("LivePortfolioProvider (phase-72 SSOT)", () => {
  it("useLivePortfolio returns the context value when wrapped", () => {
    const { result } = renderHook(() => useLivePortfolio(), { wrapper });
    expect(result.current).not.toBeNull();
    expect(result.current.loading).toBe(true);
    expect(result.current.liveNav).toBeNull();
    expect(result.current.freshnessBand).toBe("unknown");
  });

  it("useLivePortfolio throws when outside the provider", () => {
    expect(() => renderHook(() => useLivePortfolio())).toThrow(
      /LivePortfolioProvider/,
    );
  });

  it("useLivePortfolioOptional returns null outside the provider", () => {
    const { result } = renderHook(() => useLivePortfolioOptional());
    expect(result.current).toBeNull();
  });

  it("exposes a `refresh` callable", () => {
    const { result } = renderHook(() => useLivePortfolio(), { wrapper });
    expect(typeof result.current.refresh).toBe("function");
  });

  it("derives freshness band=unknown when no live prices", () => {
    const { result } = renderHook(() => useLivePortfolio(), { wrapper });
    expect(result.current.freshnessBand).toBe("unknown");
    expect(result.current.freshnessAgeSec).toBeNull();
  });

  it("pnlTodayPct/Dollars are null when liveNav or snapshots missing", () => {
    const { result } = renderHook(() => useLivePortfolio(), { wrapper });
    expect(result.current.pnlTodayPct).toBeNull();
    expect(result.current.pnlTodayDollars).toBeNull();
  });
});
