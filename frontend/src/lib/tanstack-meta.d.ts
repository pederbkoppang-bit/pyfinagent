// phase-44.2 -- module augmentation for TanStack Table v8 ColumnMeta.
//
// Adds `align` + `className` to every column's `meta` field so the
// generic DataTable wrapper can right-align numeric columns and apply
// per-column className tokens without touching the column definition
// shape consumers rely on. Pattern: TanStack Table v8 community-canonical
// (TanStack/table discussion #4097, #4439, #5319).

import "@tanstack/react-table";

declare module "@tanstack/react-table" {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ColumnMeta<TData extends unknown, TValue> {
    align?: "left" | "right" | "center";
    className?: string;
  }
}
