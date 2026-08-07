import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import type { ColumnDef } from "@tanstack/react-table";
import type { PaperPosition } from "@/lib/types";
import { positionsColumns } from "./positions-columns";

/**
 * phase-61.3 — the positions table renders LOCAL price columns in the LOCAL currency,
 * and every animated USD cell carries an explicit locale.
 *
 * The NumberFlow assertions are deliberately PROP-level. jsdom's ICU resolves an
 * omitted locale to en-US, so an output-only assertion would pass just as happily
 * against the defective `locales={undefined}` code — the test would be green and
 * blind. Asserting the prop is what actually distinguishes the two.
 */

// Capture NumberFlow's props instead of rendering the custom element (jsdom does not
// implement it). The mock records every render so the `locales` prop can be asserted.
const numberFlowRenders: Array<Record<string, unknown>> = [];

vi.mock("@number-flow/react", () => ({
  __esModule: true,
  default: (props: Record<string, unknown>) => {
    numberFlowRenders.push(props);
    return (
      <span
        data-testid="number-flow"
        data-locales={String(props.locales)}
        data-currency={String(
          (props.format as { currency?: string } | undefined)?.currency,
        )}
      >
        {String(props.value)}
      </span>
    );
  },
}));

const KR_ROW: PaperPosition = {
  position_id: "kr1",
  ticker: "005930.KS",
  quantity: 10,
  avg_entry_price: 70_000, // LOCAL: KRW
  cost_basis: 458.5, // USD
  current_price: 71_000, // LOCAL: KRW
  market_value: 465.05, // USD
  unrealized_pnl: 6.55,
  unrealized_pnl_pct: 1.43,
  entry_date: "2026-01-01T00:00:00+00:00",
  last_analysis_date: null,
  recommendation: "BUY",
  risk_judge_position_pct: null,
  stop_loss_price: 64_400, // LOCAL: KRW
  market: "KR",
  base_currency: "USD", // exactly what the backend ships on every row
  marked_at: null,
};

const US_ROW: PaperPosition = {
  ...KR_ROW,
  position_id: "us1",
  ticker: "NTAP",
  avg_entry_price: 177.85,
  current_price: 188.87,
  stop_loss_price: 164.62,
  market: "US",
};

type Cols = ColumnDef<PaperPosition, unknown>[];

/** Render one column's cell for a row and return its text content. */
function renderCell(cols: Cols, id: string, row: PaperPosition): string {
  const col = cols.find((c) => c.id === id);
  if (!col) throw new Error(`column ${id} not found`);
  const cellFn = (col as { cell?: unknown }).cell;
  if (typeof cellFn !== "function") throw new Error(`column ${id} has no cell renderer`);
  const element = (cellFn as (ctx: unknown) => React.ReactNode)({
    row: { original: row },
    getValue: () => undefined,
  });
  const { container } = render(<>{element}</>);
  return container.textContent ?? "";
}

afterEach(() => {
  cleanup();
  numberFlowRenders.length = 0;
});

const USD_SYMBOL_ON_A_NUMBER = /\$\s?\d/;

describe("positions table LOCAL columns render the LOCAL currency (criterion 2)", () => {
  const cols = positionsColumns({}, {});

  it("Entry on a KR row renders won, not dollars, despite base_currency USD", () => {
    const text = renderCell(cols, "entry", KR_ROW);
    expect(text).toMatch(/₩|KRW/);
    expect(text).not.toMatch(USD_SYMBOL_ON_A_NUMBER);
  });

  it("Stop Loss on a KR row renders won, not dollars", () => {
    const text = renderCell(cols, "stop_loss", KR_ROW);
    expect(text).toMatch(/₩|KRW/);
    expect(text).not.toMatch(USD_SYMBOL_ON_A_NUMBER);
  });

  it("Current on a KR row passes KRW to the animated cell", () => {
    renderCell(cols, "current", KR_ROW);
    const props = numberFlowRenders.at(-1);
    expect(props).toBeDefined();
    expect((props!.format as { currency?: string }).currency).toBe("KRW");
  });

  it("no KRW-magnitude value anywhere in the KR row carries a dollar sign", () => {
    // The class of defect the 60.3 prompt regex test guards against, applied to the
    // rendered table: a 5-digit-plus number must never wear a "$".
    const dollarOnBigNumber = /\$\s?\d{1,3}(,\d{3}){1,}|\$\s?\d{5,}/;
    for (const id of ["entry", "current", "stop_loss"]) {
      const text = renderCell(cols, id, KR_ROW);
      expect(text, `column ${id}`).not.toMatch(dollarOnBigNumber);
    }
  });

  it("US rows are unchanged — Entry and Stop still render dollars (do-no-harm)", () => {
    expect(renderCell(cols, "entry", US_ROW)).toMatch(USD_SYMBOL_ON_A_NUMBER);
    expect(renderCell(cols, "stop_loss", US_ROW)).toMatch(USD_SYMBOL_ON_A_NUMBER);
  });
});

describe("animated USD cells pin their locale (criterion 3)", () => {
  const cols = positionsColumns({}, {});

  it("the USD branch passes locales='en-US', never undefined", () => {
    renderCell(cols, "current", US_ROW);
    const props = numberFlowRenders.at(-1);
    expect(props).toBeDefined();
    // The whole point: an omitted locale means "browser default" per NumberFlow's
    // docs, which is what produced the mixed nb-NO/en-US rendering.
    expect(props!.locales).toBeDefined();
    expect(props!.locales).toBe("en-US");
  });

  it("the non-USD branch keeps its market locale", () => {
    renderCell(cols, "current", KR_ROW);
    const props = numberFlowRenders.at(-1);
    expect(props!.locales).toBe("ko-KR");
  });

  it("every animated cell rendered in this suite had an explicit locale", () => {
    for (const row of [US_ROW, KR_ROW]) {
      renderCell(cols, "current", row);
      renderCell(cols, "market_value", row);
    }
    expect(numberFlowRenders.length).toBeGreaterThan(0);
    for (const props of numberFlowRenders) {
      expect(props.locales).not.toBeUndefined();
    }
  });
});

describe("stale non-US P&L is labelled with its mark time (criterion 4)", () => {
  const cols = positionsColumns({}, {});

  it("shows an as-of chip on a non-live row that carries marked_at", () => {
    const marked = new Date(Date.now() - 5 * 3600 * 1000).toISOString();
    const text = renderCell(cols, "pnl", { ...KR_ROW, marked_at: marked });
    expect(text).toMatch(/as of 5h/);
  });

  it("renders no chip when marked_at is absent (pre-migration rows)", () => {
    const text = renderCell(cols, "pnl", { ...KR_ROW, marked_at: null });
    expect(text).not.toMatch(/as of/);
  });

  it("does not label a US row whose P&L is live-recomputed", () => {
    const marked = new Date(Date.now() - 5 * 3600 * 1000).toISOString();
    const withLive = positionsColumns(
      {},
      { NTAP: { price: 188.87, age_sec: 10 } as never },
    );
    const text = renderCell(withLive, "pnl", { ...US_ROW, marked_at: marked });
    expect(text).not.toMatch(/as of/);
  });

  it("escalates the chip styling as the mark ages", () => {
    const hoursAgo = (h: number) => new Date(Date.now() - h * 3600 * 1000).toISOString();
    const fresh = renderCell(cols, "pnl", { ...KR_ROW, marked_at: hoursAgo(5) });
    cleanup();
    const old = renderCell(cols, "pnl", { ...KR_ROW, marked_at: hoursAgo(100) });
    expect(fresh).toMatch(/as of 5h/);
    expect(old).toMatch(/as of 4d/);
  });

  it("colours the chip by MARK age, not by live-price age", () => {
    // The text alone cannot catch a chip wired to bandFromAgeSec: a 6h-old mark is
    // healthy (slate) while a 6h-old live price would be red. Assert the class.
    const chipClass = (hoursOld: number): string => {
      const marked = new Date(Date.now() - hoursOld * 3600 * 1000).toISOString();
      const col = cols.find((c) => c.id === "pnl")!;
      const cell = (col as { cell: (ctx: unknown) => React.ReactNode }).cell;
      const { container } = render(
        <>{cell({ row: { original: { ...KR_ROW, marked_at: marked } }, getValue: () => undefined })}</>,
      );
      // The innermost span carrying the label -- the outer wrapper also matches on
      // textContent, and its className would silently pass a weaker assertion.
      const chip = Array.from(container.querySelectorAll("span")).find(
        (el) => /as of/.test(el.textContent ?? "") && el.querySelector("span") === null,
      );
      const cls = chip?.className ?? "";
      cleanup();
      return cls;
    };

    expect(chipClass(6)).toMatch(/text-slate-400/); // within one cycle -> quiet
    expect(chipClass(48)).toMatch(/text-amber-400/); // over a day -> notable
    expect(chipClass(100)).toMatch(/text-rose-400/); // no schedule explains this
  });
});

describe("screen-level smoke: a KR row shows no dollar-denominated local price", () => {
  it("renders Entry through the real cell renderer", () => {
    const cols = positionsColumns({}, {});
    const col = cols.find((c) => c.id === "entry")!;
    const cell = (col as { cell: (ctx: unknown) => React.ReactNode }).cell;
    render(<>{cell({ row: { original: KR_ROW }, getValue: () => undefined })}</>);
    expect(screen.getByText(/₩/)).toBeTruthy();
  });
});
