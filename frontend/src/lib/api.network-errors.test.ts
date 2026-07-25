/**
 * phase-80.2 — browser network-failure message detection.
 *
 * The operator's Safari showed the raw `Network error calling
 * /api/signals/AAPL: Load failed` because the detection branch only knew
 * Chromium's and Firefox's strings. This asserts the real predicate, not
 * the source text, so removing an engine's string fails the test.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  NETWORK_FAILURE_MESSAGES,
  isNetworkFailureMessage,
  listReports,
} from "./api";

describe("isNetworkFailureMessage", () => {
  it.each([
    ["Safari / WebKit", "Load failed"],
    ["Chromium / Edge", "Failed to fetch"],
    ["Firefox", "NetworkError when attempting to fetch resource."],
    ["axios", "Network Error"],
  ])("matches %s: %j", (_engine, message) => {
    expect(isNetworkFailureMessage(message)).toBe(true);
  });

  it("matches the exact string the operator saw in Safari", () => {
    // The message apiFetch receives is err.message, i.e. the bare engine
    // string; this is the value that reached the `Network error calling
    // ...: Load failed` fallback in the 2026-07-25 screenshot.
    expect(isNetworkFailureMessage("Load failed")).toBe(true);
  });

  it("does not match unrelated errors", () => {
    // A server error must NOT be reported as 'backend unreachable' — that
    // misdirection is the whole defect.
    expect(isNetworkFailureMessage("Server error on /api/signals/AAPL")).toBe(
      false
    );
    expect(isNetworkFailureMessage("Session expired.")).toBe(false);
    expect(isNetworkFailureMessage("")).toBe(false);
  });

  it("covers all four engines", () => {
    expect(NETWORK_FAILURE_MESSAGES).toContain("Load failed");
    expect(NETWORK_FAILURE_MESSAGES.length).toBe(4);
  });
});

/**
 * phase-80.2 cycle 2 — the guard the Q/A proved was missing.
 *
 * The tests above import the predicate directly, so they cannot tell a
 * predicate that is CORRECT from one that is WIRED IN. Q/A reverted the
 * call site at `api.ts` from `isNetworkFailureMessage(msg)` back to
 * `msg.includes("Failed to fetch") || msg.includes("NetworkError")` --
 * re-introducing the exact operator-visible Safari bug -- and all 7 tests
 * above, the whole 49-test src/lib suite, and `tsc` stayed green. That is
 * sole-coverage vacuity on a behavioural criterion.
 *
 * These tests drive the REAL `apiFetch` branch through an exported caller
 * (`apiFetch` itself is module-private), so they fail when the wiring
 * breaks, not merely when the array changes.
 */
describe("apiFetch network-failure branch (defect-site binding)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  /**
   * Reject EVERY fetch with `message`. `getAuthToken` probes
   * `/api/auth/session` first and swallows its own failure (returns null),
   * so the rejection under test is the one raised by the real request.
   */
  function stubFetchRejecting(message: string) {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new TypeError(message)))
    );
  }

  it.each([
    ["Safari / WebKit", "Load failed"],
    ["Chromium / Edge", "Failed to fetch"],
    ["Firefox", "NetworkError when attempting to fetch resource."],
  ])(
    "reports an unreachable backend (not the raw engine string) on %s",
    async (_engine, message) => {
      stubFetchRejecting(message);
      await expect(listReports()).rejects.toThrow(/Cannot reach backend/);
    }
  );

  it("does NOT fall through to the raw 'Network error calling ...' message", async () => {
    // This is precisely what the operator's Safari screenshot showed:
    // "Network error calling /api/signals/AAPL: Load failed".
    stubFetchRejecting("Load failed");
    await expect(listReports()).rejects.not.toThrow(/Network error calling/);
  });

  it("still surfaces an unrecognised rejection through the generic fallback", async () => {
    // Correctness must not depend on the string set alone -- the engine
    // messages are implementation-defined, not spec-stable.
    stubFetchRejecting("some brand-new engine wording");
    await expect(listReports()).rejects.toThrow(/Network error calling/);
    await expect(listReports()).rejects.not.toThrow(/Cannot reach backend/);
  });

  it("does not misreport a real server error as an unreachable backend", async () => {
    // A 500 RESOLVES (it is a response, not a network failure), so it must
    // take the !res.ok path -- this is the whole point of the 80.2 backend
    // fix, asserted from the frontend side.
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(JSON.stringify({ detail: "Internal Server Error" }), {
            status: 500,
            headers: { "Content-Type": "application/json" },
          })
        )
      )
    );
    await expect(listReports()).rejects.toThrow(/Server error on/);
    await expect(listReports()).rejects.not.toThrow(/Cannot reach backend/);
  });
});
