import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, act } from "@testing-library/react";
import { AlphaLeaderboard } from "./AlphaLeaderboard";
import type { SovereignLeaderboardEntry } from "@/lib/api";

// Local fireEvent shim wrapped in `act()` so React 19 flushes state
// updates before assertions. Testing-library's `fireEvent` isn't
// exported from the install on this stack (same as RedLineMonitor).
function clickEl(el: Element) {
  act(() => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  });
}

const ENTRIES: SovereignLeaderboardEntry[] = [
  {
    strategy_id: "alpha_a",
    sharpe: 1.2,
    dsr: 0.95,
    pbo: 0.10,
    max_dd: 0.08,
    status: "champion",
    allocation_pct: 0.6,
    notes: null,
  },
  {
    strategy_id: "alpha_b",
    sharpe: 0.8,
    dsr: 0.70,
    pbo: 0.15,
    max_dd: 0.12,
    status: "challenger",
    allocation_pct: 0.3,
    notes: null,
  },
  {
    strategy_id: "alpha_c",
    sharpe: 1.5,
    dsr: 0.99,
    pbo: 0.05,
    max_dd: 0.06,
    status: "champion",
    allocation_pct: 0.1,
    notes: null,
  },
];

const EXPECTED_HEADERS = [
  "Strategy",
  "Sharpe",
  "DSR",
  "PBO",
  "Max DD",
  "Status",
  "Alloc %",
];

describe("AlphaLeaderboard", () => {
  afterEach(cleanup);

  it("columns_match_spec", () => {
    const { container } = render(<AlphaLeaderboard entries={ENTRIES} />);
    const ths = Array.from(container.querySelectorAll<HTMLElement>("th"));
    const labels = ths.map((th) => (th.textContent || "").trim());
    // Each TH may include a CaretUp/Down + count suffix; check inclusion.
    for (const h of EXPECTED_HEADERS) {
      expect(labels.some((l) => l.includes(h))).toBe(true);
    }
    expect(ths.length).toBe(EXPECTED_HEADERS.length);
  });

  it("status_pill_phosphor_only", () => {
    const { container } = render(<AlphaLeaderboard entries={ENTRIES} />);
    const pills = Array.from(
      container.querySelectorAll<HTMLElement>('[data-testid="status-pill"]'),
    );
    expect(pills.length).toBe(ENTRIES.length);
    for (const pill of pills) {
      // Phosphor icons render as <svg> children; assert presence.
      const svg = pill.querySelector("svg");
      expect(svg).not.toBeNull();
      // No emoji codepoints in the text content.
      const text = pill.textContent || "";
      // eslint-disable-next-line no-misleading-character-class
      const emoji = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u;
      expect(emoji.test(text)).toBe(false);
    }
  });

  it("sort_persists_client_side", () => {
    const { container } = render(<AlphaLeaderboard entries={ENTRIES} />);
    // Default sort is sharpe DESC -> expect [1.5, 1.2, 0.8].
    const sharpeCellsDesc = Array.from(
      container.querySelectorAll<HTMLElement>('[data-cell="sharpe"]'),
    ).map((td) => (td.textContent || "").trim());
    expect(sharpeCellsDesc).toEqual(["1.5000", "1.2000", "0.8000"]);

    // Click the sharpe header -> toggle to ASC -> expect [0.8, 1.2, 1.5].
    const sharpeBtn = container.querySelector<HTMLElement>('button[data-col="sharpe"]');
    expect(sharpeBtn).not.toBeNull();
    clickEl(sharpeBtn!);
    const sharpeCellsAsc = Array.from(
      container.querySelectorAll<HTMLElement>('[data-cell="sharpe"]'),
    ).map((td) => (td.textContent || "").trim());
    expect(sharpeCellsAsc).toEqual(["0.8000", "1.2000", "1.5000"]);
  });

  it("filter_by_status_pill_row", () => {
    const { container } = render(<AlphaLeaderboard entries={ENTRIES} />);
    // Click the challenger pill -> only 1 row (alpha_b) remains.
    const challengerPill = container.querySelector<HTMLElement>(
      '[data-testid="status-pill"][data-status="challenger"]',
    );
    expect(challengerPill).not.toBeNull();
    clickEl(challengerPill!);

    const rowsAfter = container.querySelectorAll('[data-row]');
    expect(rowsAfter.length).toBe(1);
    expect(rowsAfter[0].getAttribute("data-row")).toBe("alpha_b");

    // Active-filter chip is visible.
    const chip = container.querySelector('[data-testid="status-filter-chip"]');
    expect(chip).not.toBeNull();

    // Clicking the chip clears the filter -> all 3 rows back.
    clickEl(chip!);
    const rowsCleared = container.querySelectorAll('[data-row]');
    expect(rowsCleared.length).toBe(ENTRIES.length);
  });
});
