import { describe, it, expect, afterEach, vi } from "vitest";
import { renderHook, cleanup, act } from "@testing-library/react";
import { useEventSource } from "./useEventSource";

// phase-75.12 (frontend-01): the masterplan's immutable verification
// command source-SCANS for the literal string "withCredentials: true" --
// a weak guard, trivially satisfied by writing the string anywhere even
// if the runtime default is wrong (research_brief_75.12.md Section 6).
// This test is the discriminating check: it inspects the ACTUAL
// EventSource constructor argument at runtime.
class MockEventSource {
  static instances: MockEventSource[] = [];
  url: string;
  withCredentials: boolean;
  listeners: Record<string, EventListener[]> = {};
  onerror: (() => void) | null = null;
  // phase-80.4: jsdom ships no EventSource, so the hook's `onopen` handler
  // is only reachable through this mock.
  onopen: (() => void) | null = null;

  constructor(url: string, init?: EventSourceInit) {
    this.url = url;
    this.withCredentials = init?.withCredentials ?? false;
    MockEventSource.instances.push(this);
  }
  addEventListener(type: string, cb: EventListener) {
    (this.listeners[type] ??= []).push(cb);
  }
  close() {}
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  MockEventSource.instances.length = 0;
});

describe("useEventSource withCredentials default (phase-75.12 frontend-01)", () => {
  it("opens with withCredentials:true by default (cookie-based SSE auth)", () => {
    vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);
    renderHook(() => useEventSource("http://localhost:8000/api/mas/events"));
    expect(MockEventSource.instances.length).toBe(1);
    expect(MockEventSource.instances[0].withCredentials).toBe(true);
  });

  it("honors an explicit withCredentials:false override", () => {
    vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);
    renderHook(() =>
      useEventSource("http://localhost:8000/api/mas/events", { withCredentials: false }),
    );
    expect(MockEventSource.instances.length).toBe(1);
    expect(MockEventSource.instances[0].withCredentials).toBe(false);
  });
});

// ── phase-80.4 ─────────────────────────────────────────────────────────
// The MAS dashboard showed a red "Disconnected" over a HEALTHY stream,
// because `status` only became "connected" inside onMessage -- so a
// connection that was open but had not yet delivered an event stayed
// "connecting". Measured live: the MAS bus has published 0 events since
// process start, so open-but-silent is the NORMAL case, not an edge case.
describe("useEventSource open-but-no-events (phase-80.4)", () => {
  it("reports connected on OPEN, before any event has arrived", () => {
    vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);
    const { result } = renderHook(() =>
      useEventSource("http://localhost:8000/api/mas/events")
    );

    // precondition: this is the exact failing state -- open, zero events
    expect(result.current.status).toBe("connecting");

    const es = MockEventSource.instances[0];
    expect(es.onopen, "the hook never registered an onopen handler").not.toBeNull();

    act(() => es.onopen!());

    expect(result.current.status).toBe("connected");
    expect(result.current.lastEventAt, "no event arrived, so this must stay null").toBeNull();
  });

  it("does NOT reset the failure budget on open -- a flapping backend must still go Disconnected", () => {
    // THE CRITERION-4 TRAP. EventSource auto-reconnects, so a backend that
    // accepts a connection and immediately drops it fires open/error
    // repeatedly. If onopen cleared `failures`, the budget would never be
    // exhausted and the indicator would be permanently green over a dead
    // backend -- strictly worse than the bug being fixed.
    vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);
    const { result } = renderHook(() =>
      useEventSource("http://localhost:8000/api/mas/events", { maxFailures: 2 })
    );

    const flap = () => {
      const es = MockEventSource.instances[MockEventSource.instances.length - 1];
      act(() => es.onopen!());
      act(() => es.onerror!());
    };

    flap();
    expect(result.current.failures).toBe(1);
    flap();

    expect(result.current.failures).toBeGreaterThanOrEqual(2);
    expect(result.current.status).toBe("disconnected");
  });
});
