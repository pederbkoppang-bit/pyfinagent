/**
 * phase-44.1: useDebounced -- shared debounce primitive for search inputs.
 *
 * Returns the debounced value `delayMs` after the last input change. Cancels
 * pending timer on unmount. Standard React pattern; zero deps beyond React.
 */
"use client";

import { useEffect, useState } from "react";

export function useDebounced<T>(value: T, delayMs = 200): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(t);
  }, [value, delayMs]);
  return debounced;
}
