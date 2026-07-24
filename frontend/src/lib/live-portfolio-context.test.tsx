import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { renderHook, cleanup, waitFor } from "@testing-library/react";
import {
  LivePortfolioProvider,
  useLivePortfolio,
  useLivePortfolioOptional,
} from "./live-portfolio-context";
import type { ReactNode } from "react";

beforeEach(() => {
  // vi.restoreAllMocks() (afterEach, below) only affects vi.spyOn() mocks
  // -- these are plain vi.fn() instances, so their call-count history
  // leaks across tests unless cleared here too (caught the /login-gate
  // test asserting toHaveBeenCalledTimes(0) but seeing 4, carried over
  // from earlier tests in this file).
  vi.clearAllMocks();
  usePathnameMock.mockReturnValue("/");
  getPaperTradingStatusMock.mockResolvedValue(null);
  getPaperPortfolioMock.mockResolvedValue(null);
  getPaperSnapshotsMock.mockResolvedValue({ snapshots: [] });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const {
  getPaperTradingStatusMock,
  getPaperPortfolioMock,
  getPaperSnapshotsMock,
  usePathnameMock,
} = vi.hoisted(() => ({
  getPaperTradingStatusMock: vi.fn(() => Promise.resolve(null)),
  getPaperPortfolioMock: vi.fn(() => Promise.resolve(null)),
  getPaperSnapshotsMock: vi.fn(() => Promise.resolve({ snapshots: [] })),
  // Default: NOT /login, so the pre-existing tests below (written before
  // the phase-75.12 gate existed) keep polling as they always have.
  usePathnameMock: vi.fn(() => "/"),
}));

vi.mock("@/lib/api", () => ({
  getPaperTradingStatus: getPaperTradingStatusMock,
  getPaperPortfolio: getPaperPortfolioMock,
  getPaperSnapshots: getPaperSnapshotsMock,
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

vi.mock("next/navigation", () => ({
  usePathname: usePathnameMock,
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

describe("LivePortfolioProvider /login gate (phase-75.12 frontend-02)", () => {
  it("fires ZERO polls on /login", async () => {
    usePathnameMock.mockReturnValue("/login");
    const { result } = renderHook(() => useLivePortfolio(), { wrapper });

    // Give any (incorrectly) queued microtask/effect a chance to run.
    await new Promise((r) => setTimeout(r, 0));

    expect(getPaperTradingStatusMock).not.toHaveBeenCalled();
    expect(getPaperPortfolioMock).not.toHaveBeenCalled();
    expect(getPaperSnapshotsMock).not.toHaveBeenCalled();
    // Mutation M3 (un-gate the provider) would leave `loading` stuck at
    // `true` forever without the isLoginPage-aware effect below; assert
    // it resolves to false instead of hanging.
    expect(result.current.loading).toBe(false);
  });

  it("fires its normal poll trio on a non-/login pathname", async () => {
    usePathnameMock.mockReturnValue("/");
    renderHook(() => useLivePortfolio(), { wrapper });

    await waitFor(() => {
      expect(getPaperTradingStatusMock).toHaveBeenCalled();
    });
    expect(getPaperPortfolioMock).toHaveBeenCalled();
    expect(getPaperSnapshotsMock).toHaveBeenCalled();
  });
});
