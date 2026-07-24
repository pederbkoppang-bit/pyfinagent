import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, cleanup, act } from "@testing-library/react";
import { useLivePrices } from "./useLivePrices";

const { getPaperLivePricesMock } = vi.hoisted(() => ({
  getPaperLivePricesMock: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getPaperLivePrices: getPaperLivePricesMock,
}));

beforeEach(() => {
  // vi.restoreAllMocks() below only affects vi.spyOn() mocks -- plain
  // vi.fn() instances (like this hoisted one) keep BOTH their queued
  // .mockResolvedValueOnce() implementations AND their call-count history
  // across tests unless explicitly reset here.
  vi.resetAllMocks();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("useLivePrices circuit breaker (phase-75.12 frontend-06)", () => {
  it("stops polling at EXACTLY 5 consecutive failures (4 fails -> still polling)", async () => {
    vi.useFakeTimers();
    getPaperLivePricesMock.mockRejectedValue(new Error("backend down"));

    renderHook(() => useLivePrices(["AAPL"], true));

    // Mount fire (fail #1).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(getPaperLivePricesMock).toHaveBeenCalledTimes(1);

    // 3 more interval ticks -> fail #2, #3, #4. Still polling.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000 * 3);
    });
    expect(getPaperLivePricesMock).toHaveBeenCalledTimes(4);

    // 5th tick -> fail #5, circuit trips, interval cleared.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(getPaperLivePricesMock).toHaveBeenCalledTimes(5);

    // A 6th interval tick must NOT fire -- mutation M6 (restore
    // non-stopping behavior) would make this call count 6.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(getPaperLivePricesMock).toHaveBeenCalledTimes(5);
  });

  it("resets the failure counter and keeps polling on success", async () => {
    vi.useFakeTimers();
    getPaperLivePricesMock
      .mockResolvedValueOnce({ prices: { AAPL: { price: 100, age_sec: 1, cached: false } } })
      .mockResolvedValueOnce({ prices: { AAPL: { price: 101, age_sec: 1, cached: false } } });

    const { result } = renderHook(() => useLivePrices(["AAPL"], true));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.prices.AAPL?.price).toBe(100);
    expect(result.current.error).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(getPaperLivePricesMock).toHaveBeenCalledTimes(2);
    expect(result.current.prices.AAPL?.price).toBe(101);
  });
});
