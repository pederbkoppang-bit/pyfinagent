"use client";

// phase-44.0 foundation -- TanStack v8 DataTable wrapper
//
// Generic sortable + filterable table used by phase-44.2 cockpit
// (positions + trades), phase-44.4 reports, and phase-44.5 trading
// surfaces. Replaces the raw `<table>` markup flagged in the
// phase-44.2 audit (master_design Section 3.7).
//
// Single source of truth for table styling, accessibility, and
// keyboard nav so all consumers stay consistent. Phosphor icons
// (per .claude/rules/frontend.md no-emoji policy).

import { useState } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { CaretDown, CaretUp } from "@/lib/icons";

export interface DataTableProps<TData> {
  data: TData[];
  columns: ColumnDef<TData, unknown>[];
  // Optional global-filter input; pass undefined to hide.
  globalFilterPlaceholder?: string;
  // Optional row click handler (used by AgentRationaleDrawer wire).
  onRowClick?: (row: TData) => void;
  // Optional empty-state slot.
  emptyState?: React.ReactNode;
  // Optional aria-label for the table element.
  ariaLabel?: string;
}

export function DataTable<TData>({
  data,
  columns,
  globalFilterPlaceholder,
  onRowClick,
  emptyState,
  ariaLabel,
}: DataTableProps<TData>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  const rows = table.getRowModel().rows;

  return (
    <div className="space-y-3">
      {globalFilterPlaceholder !== undefined && (
        <input
          type="text"
          value={globalFilter}
          onChange={(e) => setGlobalFilter(e.target.value)}
          placeholder={globalFilterPlaceholder}
          aria-label={`Filter ${ariaLabel ?? "table"}`}
          className="w-full max-w-md px-3 py-2 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500/40"
        />
      )}
      <div className="overflow-x-auto scrollbar-thin">
        <table
          aria-label={ariaLabel}
          className="min-w-full text-sm border-collapse"
        >
          <thead className="border-b border-zinc-200 dark:border-zinc-800">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => {
                  const sort = header.column.getIsSorted();
                  const canSort = header.column.getCanSort();
                  const meta = header.column.columnDef.meta;
                  const alignClass =
                    meta?.align === "right"
                      ? "text-right"
                      : meta?.align === "center"
                        ? "text-center"
                        : "text-left";
                  return (
                    <th
                      key={header.id}
                      onClick={canSort ? header.column.getToggleSortingHandler() : undefined}
                      aria-sort={
                        sort === "asc" ? "ascending" : sort === "desc" ? "descending" : "none"
                      }
                      scope="col"
                      className={`px-3 py-2 text-xs font-medium uppercase tracking-wider text-zinc-700 dark:text-slate-400 ${alignClass} ${meta?.className ?? ""} ${
                        canSort ? "cursor-pointer select-none hover:text-zinc-900 dark:hover:text-slate-200" : ""
                      }`}
                    >
                      <span className="inline-flex items-center gap-1">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {canSort && (
                          <span aria-hidden="true" className="text-zinc-400">
                            {sort === "asc" ? (
                              <CaretUp weight="bold" size={12} />
                            ) : sort === "desc" ? (
                              <CaretDown weight="bold" size={12} />
                            ) : (
                              <span className="inline-flex flex-col leading-none">
                                <CaretUp weight="bold" size={8} />
                                <CaretDown weight="bold" size={8} />
                              </span>
                            )}
                          </span>
                        )}
                      </span>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-3 py-8 text-center text-zinc-500">
                  {emptyState ?? "No rows."}
                </td>
              </tr>
            ) : (
              rows.map((row) => (
                <tr
                  key={row.id}
                  onClick={onRowClick ? () => onRowClick(row.original) : undefined}
                  className={`border-b border-zinc-100 dark:border-zinc-800/50 ${
                    onRowClick ? "cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-900/50" : ""
                  }`}
                >
                  {row.getVisibleCells().map((cell) => {
                    const meta = cell.column.columnDef.meta;
                    const alignClass =
                      meta?.align === "right"
                        ? "text-right"
                        : meta?.align === "center"
                          ? "text-center"
                          : "text-left";
                    return (
                      <td
                        key={cell.id}
                        className={`px-3 py-2 text-zinc-800 dark:text-slate-200 ${alignClass} ${meta?.className ?? ""}`}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    );
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
