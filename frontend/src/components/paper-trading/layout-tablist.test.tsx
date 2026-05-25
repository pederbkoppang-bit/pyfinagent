import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";
import { DataTable } from "@/components/DataTable";
import type { ColumnDef } from "@tanstack/react-table";

// Smoke tests for phase-44.2 DataTable meta.className + meta.align support.
// Full layout tablist a11y is exercised end-to-end via Playwright in a
// separate operator-side cycle; here we cover the deterministic pieces.

afterEach(() => cleanup());

interface Row {
  ticker: string;
  qty: number;
}

const rows: Row[] = [
  { ticker: "AAPL", qty: 10 },
  { ticker: "MSFT", qty: 5 },
];

const columns: ColumnDef<Row, unknown>[] = [
  { id: "ticker", accessorKey: "ticker", header: "Ticker", meta: { align: "left" } },
  {
    id: "qty",
    accessorKey: "qty",
    header: "Qty",
    cell: ({ row }) => row.original.qty.toFixed(2),
    meta: { align: "right", className: "tabular-nums custom-marker" },
  },
];

describe("DataTable meta support (phase-44.2 augmentation)", () => {
  it("applies meta.align=right to numeric column header", () => {
    const { container } = render(<DataTable data={rows} columns={columns} />);
    const ths = container.querySelectorAll("th");
    expect(ths[1].className).toContain("text-right");
    expect(ths[0].className).toContain("text-left");
  });

  it("applies meta.align=right to numeric column cells", () => {
    const { container } = render(<DataTable data={rows} columns={columns} />);
    const cells = container.querySelectorAll("tbody tr td");
    expect(cells.length).toBe(4);
    // ticker cells (index 0, 2) -> left; qty cells (index 1, 3) -> right.
    expect(cells[0].className).toContain("text-left");
    expect(cells[1].className).toContain("text-right");
    expect(cells[3].className).toContain("text-right");
  });

  it("applies meta.className token to th + td", () => {
    const { container } = render(<DataTable data={rows} columns={columns} />);
    const qtyHeader = container.querySelectorAll("th")[1];
    const qtyCells = Array.from(container.querySelectorAll("tbody tr td")).filter(
      (_, idx) => idx % 2 === 1,
    );
    expect(qtyHeader.className).toContain("tabular-nums");
    expect(qtyHeader.className).toContain("custom-marker");
    for (const c of qtyCells) {
      expect(c.className).toContain("tabular-nums");
      expect(c.className).toContain("custom-marker");
    }
  });

  it("invokes onRowClick with the row's original data", () => {
    const onRowClick = vi.fn();
    const { container } = render(
      <DataTable data={rows} columns={columns} onRowClick={onRowClick} />,
    );
    const tr = container.querySelector("tbody tr");
    expect(tr).not.toBeNull();
    fireEvent.click(tr!);
    expect(onRowClick).toHaveBeenCalledWith(rows[0]);
  });
});
