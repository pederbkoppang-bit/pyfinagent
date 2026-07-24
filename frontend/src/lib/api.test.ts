import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";

// phase-75.12: api.ts's module-level sessionTokenCache is a singleton, so
// every test gets a fresh module instance via vi.resetModules() + a
// dynamic import -- otherwise cache state would leak across tests.

function mockLocation(pathname: string) {
  const setHref = vi.fn();
  Object.defineProperty(window, "location", {
    value: {
      pathname,
      get href() {
        return `http://localhost:3000${pathname}`;
      },
      set href(v: string) {
        setHref(v);
      },
    },
    writable: true,
    configurable: true,
  });
  return setHref;
}

function jsonResponse(body: unknown, init?: { ok?: boolean; status?: number }): Response {
  return {
    ok: init?.ok ?? true,
    status: init?.status ?? 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

function mockFetchByUrl(handlers: Record<string, () => Response>) {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      for (const [key, handler] of Object.entries(handlers)) {
        if (url.includes(key)) return Promise.resolve(handler());
      }
      throw new Error(`Unmocked fetch: ${url}`);
    }),
  );
}

beforeEach(() => {
  vi.resetModules();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("apiFetch 401 branch (phase-75.12 frontend-02)", () => {
  it("does NOT navigate when already on /login", async () => {
    const setHref = mockLocation("/login");
    mockFetchByUrl({
      "/api/auth/session": () => jsonResponse({ user: { email: "x@y.com" } }),
      "/api/health": () => jsonResponse({ detail: "Unauthorized" }, { ok: false, status: 401 }),
    });
    const { healthCheck } = await import("./api");
    await expect(healthCheck()).rejects.toThrow(/Session expired/);
    expect(setHref).not.toHaveBeenCalled();
  });

  it("DOES navigate to /login when on a different page", async () => {
    const setHref = mockLocation("/reports");
    mockFetchByUrl({
      "/api/auth/session": () => jsonResponse({ user: { email: "x@y.com" } }),
      "/api/health": () => jsonResponse({ detail: "Unauthorized" }, { ok: false, status: 401 }),
    });
    const { healthCheck } = await import("./api");
    await expect(healthCheck()).rejects.toThrow(/Session expired/);
    expect(setHref).toHaveBeenCalledWith("/login");
  });
});

describe("sessionTokenCache (phase-75.12 frontend-09)", () => {
  it("a TTL hit avoids a second /api/auth/session probe", async () => {
    mockLocation("/reports");
    let sessionCalls = 0;
    mockFetchByUrl({
      "/api/auth/session": () => {
        sessionCalls += 1;
        return jsonResponse({ user: { email: "x@y.com" } });
      },
      "/api/health": () => jsonResponse({ status: "ok", service: "backend" }),
    });
    const { healthCheck } = await import("./api");
    await healthCheck();
    await healthCheck();
    expect(sessionCalls).toBe(1);
  });

  it("a 401 invalidates the cache so the next call re-probes the session", async () => {
    mockLocation("/reports");
    let sessionCalls = 0;
    let healthCalls = 0;
    mockFetchByUrl({
      "/api/auth/session": () => {
        sessionCalls += 1;
        return jsonResponse({ user: { email: "x@y.com" } });
      },
      "/api/health": () => {
        healthCalls += 1;
        return healthCalls === 1
          ? jsonResponse({ detail: "Unauthorized" }, { ok: false, status: 401 })
          : jsonResponse({ status: "ok" });
      },
    });
    const { healthCheck } = await import("./api");
    await expect(healthCheck()).rejects.toThrow();
    expect(sessionCalls).toBe(1);
    await healthCheck();
    expect(sessionCalls).toBe(2);
  });
});
