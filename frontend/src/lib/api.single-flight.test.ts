/**
 * phase-80.11 -- the session-fetch stampede.
 *
 * MEASURED: one 20s view of /paper-trading/positions fired ELEVEN requests to
 * /api/auth/session, against 1-2 for every backend endpoint (raw log:
 * handoff/current/captures_ui_audit_2026-07-25/audit-net-positions-20s.txt).
 *
 * getAuthToken memoised the RESOLVED value with a 60s TTL but wrote the cache
 * only AFTER `await fetch(...)` returned, so N concurrent apiFetch calls each
 * observed an empty cache and each fired its own probe. A cache stampede.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

const SESSION_URL = "/api/auth/session";

function installFetchSpy(sessionStatus = 200) {
  const calls: string[] = [];
  const spy = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : String(input);
    calls.push(url);
    if (url.includes(SESSION_URL)) {
      // deliberately async so concurrent callers overlap in flight
      await new Promise((r) => setTimeout(r, 5));
      return {
        ok: sessionStatus === 200,
        status: sessionStatus,
        json: async () => ({ user: { email: "x@y.z" } }),
      } as unknown as Response;
    }
    return {
      ok: true, status: 200,
      headers: { get: () => "application/json" },
      json: async () => ({}),
    } as unknown as Response;
  });
  vi.stubGlobal("fetch", spy);
  return { calls, spy };
}

beforeEach(() => {
  vi.resetModules();
  vi.unstubAllGlobals();
});
afterEach(() => vi.unstubAllGlobals());

describe("getAuthToken single-flight (phase-80.11)", () => {
  it("collapses N concurrent probes into ONE request", async () => {
    const { calls } = installFetchSpy();
    const api = await import("./api");
    api.__resetSessionTokenCacheForTests();

    // 11 concurrent callers -- the exact count measured on the live page
    await Promise.all(Array.from({ length: 11 }, () => api.getPaperPortfolio().catch(() => null)));

    const probes = calls.filter((u) => u.includes(SESSION_URL));
    // THE ASSERTION. Pre-fix this is 11 (one per caller); single-flight makes it 1.
    expect(probes.length, `expected ONE shared probe, got ${probes.length}`).toBe(1);
  });

  it("clears the in-flight promise, so a lapsed TTL probes again", async () => {
    // THE OPPOSITE FAILURE: a single-flight that never releases serves a
    // permanently stale token and masks a real re-login.
    //
    // An earlier version of this test called __resetSessionTokenCacheForTests()
    // to "simulate the TTL lapse" -- but that helper nulls sessionTokenInflight
    // ITSELF, i.e. it performed the very clear the mutation removes. It passed
    // against a build with no `.finally()` at all. Verified by mutation, not
    // assumed. The TTL is lapsed by advancing the clock instead, so the ONLY
    // thing that can release the in-flight ref is the production `.finally()`.
    const { calls } = installFetchSpy();
    const api = await import("./api");
    api.__resetSessionTokenCacheForTests();

    const realNow = Date.now;
    try {
      let t = realNow.call(Date);
      vi.stubGlobal("Date", { ...Date, now: () => t } as unknown as DateConstructor);

      await api.getPaperPortfolio().catch(() => null);
      expect(calls.filter((u) => u.includes(SESSION_URL)).length).toBe(1);

      t += 61_000; // past SESSION_TOKEN_TTL_MS -- cache is stale, nothing else touched

      await api.getPaperPortfolio().catch(() => null);
      expect(
        calls.filter((u) => u.includes(SESSION_URL)).length,
        "in-flight ref was never released -- a stale token would be served forever",
      ).toBe(2);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("sends an abort signal on the probe -- a stall must not block every caller", async () => {
    // Created BY the single-flight fix: previously a stalled probe blocked one
    // caller; shared, it would block all of them. The 30s AbortController in
    // apiFetch guards only the backend fetch, never this probe.
    const { spy } = installFetchSpy();
    const api = await import("./api");
    api.__resetSessionTokenCacheForTests();
    await api.getPaperPortfolio().catch(() => null);

    const probeCall = (spy.mock.calls as unknown as unknown[][]).find(
      (c) => String(c[0]).includes(SESSION_URL),
    );
    expect(probeCall, "no session probe was made").toBeDefined();
    const init = probeCall![1] as RequestInit | undefined;
    expect(init?.signal, "probe has no AbortSignal -- a stall blocks every apiFetch").toBeDefined();
  });

  it("a 401 during an IN-FLIGHT probe is not undone when that probe resolves", async () => {
    // THE POISON SEQUENCE, and it only reproduces if the invalidation lands WHILE
    // the probe is still in flight. An earlier version of this test invalidated
    // AFTER the probe had resolved -- so the epoch was never consulted and the
    // test passed against a build with no epoch at all. Verified by mutation.
    //
    // Sequence: probe P starts -> a 401 from an earlier request invalidates ->
    // P resolves and writes the just-invalidated token back with a fresh 60s ts,
    // silently reverting the invalidation. Normally the /login redirect cuts this
    // short, but that redirect is skipped when already ON /login, so there it is
    // unbounded. Clearing the in-flight ref alone is NOT sufficient.
    const { calls } = installFetchSpy();
    const api = await import("./api");
    api.__resetSessionTokenCacheForTests();

    // start a probe but do NOT await it -- it stalls 5ms inside the fetch spy
    const inFlight = api.getPaperPortfolio().catch(() => null);

    // the 401 lands mid-flight
    api.__invalidateSessionTokenForTests();

    await inFlight;
    const afterPoison = calls.filter((u) => u.includes(SESSION_URL)).length;

    // the resolved probe must NOT have written a token. If it did, this next call
    // is served from that resurrected cache and issues no new probe.
    await api.getPaperPortfolio().catch(() => null);
    expect(
      calls.filter((u) => u.includes(SESSION_URL)).length,
      "the in-flight probe resurrected a token that a 401 had just invalidated",
    ).toBe(afterPoison + 1);
  });

  it("issues ZERO session probes when the auth cookie is present", async () => {
    // The residual probe. When the token cookie exists, the token IS the cookie
    // value and the fetched session object is never read -- the request was pure
    // overhead on every TTL lapse for a logged-in operator.
    //
    // This is not a weakened auth check: the cookie is the credential the backend
    // validates, and a stale one is still caught by the 401 path.
    const { calls } = installFetchSpy();
    Object.defineProperty(document, "cookie", {
      value: "authjs.session-token=abc123; other=x",
      configurable: true,
      writable: true,
    });
    const api = await import("./api");
    api.__resetSessionTokenCacheForTests();

    await Promise.all(Array.from({ length: 11 }, () => api.getPaperPortfolio().catch(() => null)));

    expect(
      calls.filter((u) => u.includes(SESSION_URL)).length,
      "a cookie was present, so no session probe should have been needed",
    ).toBe(0);
  });

  it("still probes when NO auth cookie is present", async () => {
    // The other half -- the skip-auth rig and the logged-out case. Without this the
    // optimisation above could silently disable authentication discovery entirely.
    const { calls } = installFetchSpy();
    Object.defineProperty(document, "cookie", {
      value: "", configurable: true, writable: true,
    });
    const api = await import("./api");
    api.__resetSessionTokenCacheForTests();

    await Promise.all(Array.from({ length: 5 }, () => api.getPaperPortfolio().catch(() => null)));

    expect(
      calls.filter((u) => u.includes(SESSION_URL)).length,
      "no cookie means the probe is the only way to discover a session",
    ).toBe(1);
  });
});
