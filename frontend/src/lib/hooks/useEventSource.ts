/**
 * phase-44.1: useEventSource -- shared SSE consumer.
 *
 * Wraps the EventSource API with reconnect-on-error + last-event-timestamp
 * tracking. Replaces inline EventSource in /agents (only SSE consumer today
 * per research_brief Section A.12) and enables phase-44.10 (SSE everywhere).
 *
 * Reconnect strategy: exponential backoff capped at 30s; max 3 consecutive
 * failures before giving up + surfacing "disconnected" state. Matches the
 * existing /agents pattern (`failCountRef` at agents/page.tsx).
 */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseEventSourceState<T = unknown> {
  /** Last event payload, parsed by the optional parser. */
  data: T | null;
  /** Connection state. */
  status: "connecting" | "connected" | "disconnected" | "error";
  /** Last received event ts (epoch ms), or null. */
  lastEventAt: number | null;
  /** Consecutive failure count -- exposed for UI badges. */
  failures: number;
  /** Force reconnect now (resets failure count). */
  reconnect: () => void;
}

interface UseEventSourceOptions<T> {
  /** Parse the raw event.data string into your shape. Defaults to JSON.parse. */
  parser?: (raw: string) => T;
  /** Override event-type filter (default: all "message" events). */
  eventType?: string;
  /** Disable entirely (e.g. when waiting for auth). */
  enabled?: boolean;
  /** Cap on consecutive failures before stopping. Default 3. */
  maxFailures?: number;
  /**
   * Optional per-event callback fired synchronously when an event arrives.
   * Used by buffer-accumulating consumers (e.g. /agents Live Stream) that
   * need every event, not just the last. The default `data` state still
   * carries the latest event for non-buffered consumers.
   */
  onEvent?: (event: T) => void;
}

const DEFAULT_PARSER = <T>(raw: string): T => {
  try {
    return JSON.parse(raw) as T;
  } catch {
    // graceful fallback: pass raw string through as unknown
    return raw as unknown as T;
  }
};

export function useEventSource<T = unknown>(
  url: string | null,
  options?: UseEventSourceOptions<T>,
): UseEventSourceState<T> {
  const enabled = options?.enabled !== false;
  const parser = options?.parser ?? (DEFAULT_PARSER<T>);
  const eventType = options?.eventType ?? "message";
  const maxFailures = options?.maxFailures ?? 3;
  // phase-44.7: ref the callback so the connect closure doesn't capture
  // a stale version of it across renders.
  const onEventRef = useRef(options?.onEvent);
  useEffect(() => {
    onEventRef.current = options?.onEvent;
  }, [options?.onEvent]);

  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<UseEventSourceState["status"]>("connecting");
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [failures, setFailures] = useState(0);

  const sourceRef = useRef<EventSource | null>(null);
  const backoffRef = useRef(1000); // start at 1s

  const cleanup = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (!enabled || !url || typeof window === "undefined") return;
    cleanup();
    try {
      setStatus("connecting");
      const es = new EventSource(url, { withCredentials: false });
      sourceRef.current = es;

      const onMessage = (event: MessageEvent) => {
        setStatus("connected");
        setFailures(0);
        backoffRef.current = 1000;
        setLastEventAt(Date.now());
        try {
          const parsed = parser(event.data);
          setData(parsed);
          // phase-44.7: per-event callback for buffer-accumulating consumers.
          if (onEventRef.current) onEventRef.current(parsed);
        } catch {
          // parser-internal error -- swallow; raw event preserved at next iter
        }
      };

      es.addEventListener(eventType, onMessage as EventListener);

      es.onerror = () => {
        setStatus("error");
        cleanup();
        setFailures((prev) => {
          const next = prev + 1;
          if (next < maxFailures) {
            const delay = Math.min(backoffRef.current, 30_000);
            backoffRef.current = Math.min(backoffRef.current * 2, 30_000);
            window.setTimeout(connect, delay);
          } else {
            setStatus("disconnected");
          }
          return next;
        });
      };
    } catch (err) {
      setStatus("error");
      setFailures((p) => p + 1);
    }
  }, [enabled, url, parser, eventType, maxFailures, cleanup]);

  useEffect(() => {
    if (!enabled || !url) {
      cleanup();
      setStatus("disconnected");
      return;
    }
    connect();
    return cleanup;
  }, [enabled, url, connect, cleanup]);

  const reconnect = useCallback(() => {
    setFailures(0);
    backoffRef.current = 1000;
    connect();
  }, [connect]);

  return { data, status, lastEventAt, failures, reconnect };
}
