import { describe, it, expect, afterEach, vi } from "vitest";
import { renderHook, cleanup } from "@testing-library/react";
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
