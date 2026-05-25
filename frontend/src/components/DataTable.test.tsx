import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, act } from "@testing-library/react";
import { DataTable } from "./DataTable";
import type { ColumnDef } from "@tanstack/react-table";

afterEach(() => cleanup());

type Row = { ticker: string; pnl: number };

const ROWS: Row[] = [
  { ticker: "AAPL", pnl: 12.5 },
  { ticker: "MSFT", pnl: -3.2 },
  { ticker: "NVDA", pnl: 24.8 },
];

const COLUMNS: ColumnDef<Row, unknown>[] = [
  { accessorKey: "ticker", header: "Ticker" },
  { accessorKey: "pnl", header: "P&L" },
];

describe("DataTable (phase-44.0 foundation)", () => {
  it("renders all rows", () => {
    const { container } = render(<DataTable data={ROWS} columns={COLUMNS} />);
    expect(container.textContent).toContain("AAPL");
    expect(container.textContent).toContain("MSFT");
    expect(container.textContent).toContain("NVDA");
  });

  it("renders headers with sort indicators", () => {
    const { container } = render(<DataTable data={ROWS} columns={COLUMNS} />);
    const headers = container.querySelectorAll("th");
    expect(headers.length).toBe(2);
    expect(headers[0].textContent).toContain("Ticker");
    expect(headers[0].getAttribute("aria-sort")).toBe("none");
  });

  it("renders empty state when data is empty", () => {
    const { container } = render(<DataTable data={[]} columns={COLUMNS} />);
    expect(container.textContent).toContain("No rows.");
  });

  it("renders custom empty state when provided", () => {
    const { container } = render(
      <DataTable data={[]} columns={COLUMNS} emptyState="no positions yet" />,
    );
    expect(container.textContent).toContain("no positions yet");
  });

  it("renders global filter input when placeholder provided", () => {
    const { container } = render(
      <DataTable
        data={ROWS}
        columns={COLUMNS}
        globalFilterPlaceholder="Search ticker..."
      />,
    );
    const input = container.querySelector("input");
    expect(input).not.toBeNull();
    expect(input?.getAttribute("placeholder")).toBe("Search ticker...");
  });

  it("does NOT render filter input when placeholder undefined", () => {
    const { container } = render(<DataTable data={ROWS} columns={COLUMNS} />);
    expect(container.querySelector("input")).toBeNull();
  });

  it("applies aria-label to table", () => {
    const { container } = render(
      <DataTable data={ROWS} columns={COLUMNS} ariaLabel="positions table" />,
    );
    const table = container.querySelector("table");
    expect(table?.getAttribute("aria-label")).toBe("positions table");
  });

  it("rows are clickable when onRowClick provided", () => {
    let clickedTicker = "";
    const onRowClick = (row: Row) => {
      clickedTicker = row.ticker;
    };
    const { container } = render(
      <DataTable data={ROWS} columns={COLUMNS} onRowClick={onRowClick} />,
    );
    const firstDataRow = container.querySelectorAll("tbody tr")[0];
    expect(firstDataRow?.className).toContain("cursor-pointer");
    act(() => {
      firstDataRow?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(clickedTicker).toBe("AAPL");
  });
});
