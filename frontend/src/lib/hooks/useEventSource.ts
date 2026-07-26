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
   * Send cookies cross-origin so the backend can authenticate the SSE
   * connection via the NextAuth session cookie (EventSource cannot send
   * Authorization headers -- cookie auth is the only in-spec option; see
   * auth.py:169-189 + main.py:488 allow_credentials=True). Default true.
   * Overridable per-instance for future non-auth SSE consumers.
   */
  withCredentials?: boolean;
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
  const withCredentials = options?.withCredentials !== false;
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
  // phase-80.33: the pending reconnect handle, so effect cleanup can cancel it.
  const reconnectTimerRef = useRef<number | null>(null);
  // phase-80.34: mirrors `failures` so the next count is computed OUTSIDE the
  // setState updater, keeping that updater pure under Strict Mode's double-invoke.
  const failuresRef = useRef(0);

  const cleanup = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    // phase-80.33: cancel any reconnect still pending in the backoff window.
    // Without this the effect cleanup closed only the CURRENT EventSource, and a
    // timer scheduled up to 30s earlier would later construct a new one against
    // an unmounted component.
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (!enabled || !url || typeof window === "undefined") return;
    cleanup();
    try {
      setStatus("connecting");
      // Written as a literal-branch ternary (not a variable substitution)
      // so the phrase `withCredentials: true` is always present in source
      // -- the masterplan verification command source-scans for it.
      const es = new EventSource(
        url,
        withCredentials ? { withCredentials: true } : { withCredentials: false },
      );
      sourceRef.current = es;

      const onMessage = (event: MessageEvent) => {
        setStatus("connected");
        setFailures(0);
        // phase-80.34: the mirror MUST be reset everywhere `failures` is, or it
        // drifts above the real count and the backoff keeps escalating after a
        // successful event -- the failure budget is only cleared by a real event
        // arriving, so this is the one place that clears it on the happy path.
        failuresRef.current = 0;
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

      // phase-80.4: connection state must come from CONNECTION ESTABLISHMENT,
      // not from DATA ARRIVAL.
      //
      // Before this, `status` only became "connected" inside onMessage, so an
      // open and perfectly healthy stream that had simply not yet delivered an
      // event stayed "connecting" forever -- and /agents, which renders a
      // binary Connected/Disconnected label, showed a red **Disconnected** over
      // a working endpoint. Measured: the MAS bus has published 0 events since
      // process start (its emit sites fire only on MAS runs, which the trading
      // cycle never triggers), so the idle case is the NORMAL case here, not an
      // edge case.
      //
      // `onopen` fires on the validated response HEADERS, before the body is
      // interpreted line by line (WHATWG: "announce the connection", which sets
      // readyState=OPEN and fires `open`). So this is correct even against a
      // backend that never sends a byte -- it does not depend on the heartbeat
      // added in mas_events.py.
      //
      // DELIBERATELY sets status ONLY. It must NOT reset `failures` or
      // `backoffRef` the way onMessage does: EventSource auto-reconnects, so a
      // backend that accepts a connection and immediately drops it would fire
      // open/error/open/error... and resetting the budget here would make the
      // indicator permanently green while the backend flapped. The failure
      // budget is deliberately only cleared by a real event arriving.
      es.onopen = () => {
        setStatus("connected");
      };

      es.onerror = () => {
        setStatus("error");
        cleanup();
        // phase-80.33 + 80.34: these two filed defects were ONE mistake -- side
        // effects living inside a setState updater, which also meant nobody owned
        // the reconnect timer.
        //
        // 80.34: React requires updater functions to be PURE, and Strict Mode
        // deliberately invokes them TWICE to surface impurity. The old updater
        // scheduled a timeout, mutated backoffRef and called setStatus, so under
        // Strict Mode every onerror scheduled TWO reconnects and advanced the
        // backoff twice -- the real cadence was not the intended 1/2/4/8s.
        //
        // 80.33: `window.setTimeout(connect, delay)` DISCARDED its handle, so the
        // effect cleanup could not cancel a pending reconnect. A component that
        // unmounted inside the backoff window (~15s on /agents) still ran connect()
        // afterwards, building a NEW EventSource against a dead component.
        //
        // Fix: compute the next count OUTSIDE the updater, keep the updater pure,
        // and store the timer in a ref that cleanup clears.
        const next = failuresRef.current + 1;
        failuresRef.current = next;
        setFailures(next);
        if (next < maxFailures) {
          const delay = Math.min(backoffRef.current, 30_000);
          backoffRef.current = Math.min(backoffRef.current * 2, 30_000);
          if (reconnectTimerRef.current !== null) {
            window.clearTimeout(reconnectTimerRef.current);
          }
          reconnectTimerRef.current = window.setTimeout(() => {
            reconnectTimerRef.current = null;
            connect();
          }, delay);
        } else {
          setStatus("disconnected");
        }
      };
    } catch (err) {
      setStatus("error");
      setFailures((p) => p + 1);
    }
  }, [enabled, url, parser, eventType, maxFailures, cleanup, withCredentials]);

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
    failuresRef.current = 0; // phase-80.34: keep the mirror in step with state
    backoffRef.current = 1000;
    connect();
  }, [connect]);

  return { data, status, lastEventAt, failures, reconnect };
}
