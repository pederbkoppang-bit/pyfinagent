import { describe, it, expect } from "vitest";
import { latestTradeIdForTicker, bandFromAgeSec } from "./paper-trading-utils";
import type { PaperTrade } from "./types";

function trade(
  trade_id: string,
  ticker: string,
  action: "BUY" | "SELL",
  created_at: string,
): PaperTrade {
  return {
    trade_id,
    ticker,
    action,
    quantity: 1,
    price: 1,
    total_value: null,
    transaction_cost: null,
    reason: null,
    analysis_id: null,
    risk_judge_decision: null,
    created_at,
  };
}

describe("latestTradeIdForTicker", () => {
  it("returns null when no trades match the ticker", () => {
    const trades: PaperTrade[] = [trade("t1", "AAPL", "BUY", "2026-05-20T10:00:00Z")];
    expect(latestTradeIdForTicker(trades, "MSFT")).toBeNull();
  });

  it("returns null when only SELL trades exist for the ticker", () => {
    const trades: PaperTrade[] = [trade("t1", "AAPL", "SELL", "2026-05-20T10:00:00Z")];
    expect(latestTradeIdForTicker(trades, "AAPL")).toBeNull();
  });

  it("returns the only BUY trade", () => {
    const trades: PaperTrade[] = [trade("t1", "AAPL", "BUY", "2026-05-20T10:00:00Z")];
    expect(latestTradeIdForTicker(trades, "AAPL")).toBe("t1");
  });

  it("returns the most recent BUY when multiple exist", () => {
    const trades: PaperTrade[] = [
      trade("old", "AAPL", "BUY", "2026-05-19T10:00:00Z"),
      trade("new", "AAPL", "BUY", "2026-05-20T10:00:00Z"),
      trade("mid", "AAPL", "BUY", "2026-05-19T15:00:00Z"),
    ];
    expect(latestTradeIdForTicker(trades, "AAPL")).toBe("new");
  });

  it("ignores SELL trades even when more recent", () => {
    const trades: PaperTrade[] = [
      trade("buy", "AAPL", "BUY", "2026-05-19T10:00:00Z"),
      trade("sell", "AAPL", "SELL", "2026-05-20T10:00:00Z"),
    ];
    expect(latestTradeIdForTicker(trades, "AAPL")).toBe("buy");
  });

  it("isolates lookups by ticker", () => {
    const trades: PaperTrade[] = [
      trade("aapl_buy", "AAPL", "BUY", "2026-05-19T10:00:00Z"),
      trade("msft_buy", "MSFT", "BUY", "2026-05-20T10:00:00Z"),
    ];
    expect(latestTradeIdForTicker(trades, "AAPL")).toBe("aapl_buy");
    expect(latestTradeIdForTicker(trades, "MSFT")).toBe("msft_buy");
  });
});

describe("bandFromAgeSec", () => {
  it("returns unknown for null age", () => {
    expect(bandFromAgeSec(null)).toBe("unknown");
  });
  it("returns unknown for undefined age", () => {
    expect(bandFromAgeSec(undefined)).toBe("unknown");
  });
  it("returns green under 90s", () => {
    expect(bandFromAgeSec(0)).toBe("green");
    expect(bandFromAgeSec(89)).toBe("green");
  });
  it("returns amber between 90s and 300s", () => {
    expect(bandFromAgeSec(90)).toBe("amber");
    expect(bandFromAgeSec(299)).toBe("amber");
  });
  it("returns red at or above 300s", () => {
    expect(bandFromAgeSec(300)).toBe("red");
    expect(bandFromAgeSec(3600)).toBe("red");
  });
});
