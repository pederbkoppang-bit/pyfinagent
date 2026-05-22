/**
 * phase-44.1: useURLState -- sync component state to URL ?search-params.
 *
 * Enables deep-linking across /reports tabs, /backtest run selection,
 * /paper-trading sub-tabs. Reads initial value from URL on mount; updates
 * URL via router.replace on state change (no scroll jump, no history pile-up).
 *
 * Default-value sentinel: when value === defaultValue, the param is REMOVED
 * from the URL (keeps shareable URLs compact).
 */
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams, usePathname } from "next/navigation";

type Parser<T> = (raw: string | null) => T;
type Serializer<T> = (value: T) => string | null;

interface UseURLStateOptions<T> {
  parser?: Parser<T>;
  serializer?: Serializer<T>;
  /** when true, replace history (default); when false, push new entry */
  replace?: boolean;
}

const stringParser: Parser<string> = (raw) => raw ?? "";
const stringSerializer: Serializer<string> = (v) => (v === "" ? null : v);

export function useURLState<T = string>(
  key: string,
  defaultValue: T,
  options?: UseURLStateOptions<T>,
): [T, (next: T) => void] {
  const router = useRouter();
  const params = useSearchParams();
  const pathname = usePathname();

  const parser = (options?.parser ?? (stringParser as unknown as Parser<T>));
  const serializer = (options?.serializer ?? (stringSerializer as unknown as Serializer<T>));
  const replace = options?.replace !== false;

  const initial = useMemo(() => {
    const raw = params?.get(key) ?? null;
    if (raw === null) return defaultValue;
    try {
      return parser(raw);
    } catch {
      return defaultValue;
    }
  }, [key, params, defaultValue, parser]);

  const [value, setValue] = useState<T>(initial);

  // keep state in sync if URL changes externally (back/forward nav)
  useEffect(() => {
    const raw = params?.get(key) ?? null;
    const fromUrl = raw === null ? defaultValue : (() => { try { return parser(raw); } catch { return defaultValue; }})();
    setValue(fromUrl);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, params]);

  const update = useCallback((next: T) => {
    setValue(next);
    if (!params || !pathname) return;
    const sp = new URLSearchParams(params.toString());
    const serialized = serializer(next);
    if (serialized === null || serialized === undefined) {
      sp.delete(key);
    } else {
      sp.set(key, serialized);
    }
    const qs = sp.toString();
    const url = qs.length > 0 ? `${pathname}?${qs}` : pathname;
    if (replace) router.replace(url, { scroll: false });
    else router.push(url, { scroll: false });
  }, [key, params, pathname, router, serializer, replace]);

  return [value, update];
}
