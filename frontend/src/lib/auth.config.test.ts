import { describe, it, expect, afterEach, vi } from "vitest";

/**
 * phase-75.6 -- SEMANTIC proof of the auth.config fail-closed behaviour.
 *
 * The step's immutable verification command only string-checks the source, and its
 * `'profile' within 250 chars of 'signIn('` assert is mutation-weak: it ALREADY PASSED on
 * the buggy pre-fix file (`const profile = account` sits inside that window). So the
 * string assert is NOT evidence that gap2-04 is fixed. This test drives the REAL exported
 * `signIn` callback and asserts the behaviour, and its `reads email_verified from profile
 * NOT account` case is the load-bearing one -- it fails on the exact aliasing bug the
 * immutable command cannot see.
 *
 * `allowedEmails` / `enforceAllowlist` are read at MODULE LOAD, so each scenario resets
 * modules, sets env, and dynamically re-imports the config.
 */

const ORIG_ENV = { ...process.env };

type SignIn = (params: unknown) => boolean | Promise<boolean>;

async function loadSignIn(
  env: Record<string, string | undefined>
): Promise<SignIn> {
  vi.resetModules();
  for (const k of [
    "ALLOWED_EMAILS",
    "AUTH_ENFORCE_ALLOWLIST",
    "AUTH_GOOGLE_ID",
    "AUTH_GOOGLE_SECRET",
  ]) {
    delete process.env[k];
  }
  Object.assign(process.env, env);
  const mod = await import("./auth.config");
  const cfg = mod.default as { callbacks?: { signIn?: SignIn } };
  if (!cfg.callbacks?.signIn) throw new Error("signIn callback missing");
  return cfg.callbacks.signIn;
}

const googleAccount = { provider: "google" };
const user = (email: string | null | undefined) => ({ email });

afterEach(() => {
  process.env = { ...ORIG_ENV };
  vi.resetModules();
});

describe("auth.config signIn -- fail-closed (phase-75.6)", () => {
  // gap2-02 + criterion 2: flag ON + empty allowlist -> deny ALL
  it("denies all sign-ins when AUTH_ENFORCE_ALLOWLIST is on and the allowlist is empty", async () => {
    const signIn = await loadSignIn({
      ALLOWED_EMAILS: "",
      AUTH_ENFORCE_ALLOWLIST: "true",
    });
    const ok = await signIn({
      user: user("anyone@example.com"),
      account: googleAccount,
      profile: { email_verified: true },
    });
    expect(ok).toBe(false);
  });

  // gap2-02: flag OFF + empty allowlist -> admit (byte-equivalent legacy behaviour)
  it("admits when the allowlist is empty and the flag is off (fail-open default)", async () => {
    const signIn = await loadSignIn({ ALLOWED_EMAILS: "", AUTH_ENFORCE_ALLOWLIST: "" });
    const ok = await signIn({
      user: user("anyone@example.com"),
      account: googleAccount,
      profile: { email_verified: true },
    });
    expect(ok).toBe(true);
  });

  // gap2-02: the frontend flag must accept the same truthy forms as backend pydantic
  it("accepts '1' as an on-value for the flag (no layer desync with pydantic)", async () => {
    const signIn = await loadSignIn({ ALLOWED_EMAILS: "", AUTH_ENFORCE_ALLOWLIST: "1" });
    const ok = await signIn({
      user: user("anyone@example.com"),
      account: googleAccount,
      profile: { email_verified: true },
    });
    expect(ok).toBe(false);
  });

  // gap2-04 + criterion 3: PROFILE email_verified:false -> reject
  it("rejects a Google sign-in whose PROFILE email_verified is false", async () => {
    const signIn = await loadSignIn({ ALLOWED_EMAILS: "" });
    const ok = await signIn({
      user: user("x@example.com"),
      account: googleAccount,
      profile: { email_verified: false },
    });
    expect(ok).toBe(false);
  });

  // gap2-04 LOAD-BEARING (kills M2, the account-alias bug the string assert cannot catch):
  // false on ACCOUNT, true on PROFILE. The fix reads PROFILE -> admit. The bug reads
  // ACCOUNT -> reject. So `true` distinguishes fixed from buggy.
  it("reads email_verified from profile NOT account (the aliasing bug would reject here)", async () => {
    const signIn = await loadSignIn({ ALLOWED_EMAILS: "" });
    const ok = await signIn({
      user: user("x@example.com"),
      account: { provider: "google", email_verified: false },
      profile: { email_verified: true },
    });
    expect(ok).toBe(true);
  });

  // OIDC: an omitted claim (undefined) is permitted, not rejected
  it("admits when email_verified is undefined (provider omitted the claim)", async () => {
    const signIn = await loadSignIn({ ALLOWED_EMAILS: "" });
    const ok = await signIn({
      user: user("x@example.com"),
      account: googleAccount,
      profile: {},
    });
    expect(ok).toBe(true);
  });

  // gap2-06 + criterion 4: active allowlist + emailless principal -> reject
  it("rejects an emailless principal when an allowlist is active", async () => {
    const signIn = await loadSignIn({ ALLOWED_EMAILS: "keep@example.com" });
    const ok = await signIn({
      user: user(null),
      account: { provider: "passkey" },
      profile: {},
    });
    expect(ok).toBe(false);
  });

  // active allowlist: non-member rejected, member admitted (case-insensitive)
  it("enforces membership against a non-empty allowlist", async () => {
    const denied = await (await loadSignIn({ ALLOWED_EMAILS: "keep@example.com" }))({
      user: user("other@example.com"),
      account: googleAccount,
      profile: { email_verified: true },
    });
    expect(denied).toBe(false);

    const allowed = await (await loadSignIn({ ALLOWED_EMAILS: "keep@example.com" }))({
      user: user("KEEP@example.com"),
      account: googleAccount,
      profile: { email_verified: true },
    });
    expect(allowed).toBe(true);
  });
});

describe("auth.config session hardening (phase-75.6 gap2-05)", () => {
  it("sets a 7-day maxAge with an explicit updateAge and no 30-day value", async () => {
    vi.resetModules();
    const mod = await import("./auth.config");
    const cfg = mod.default as { session?: { maxAge?: number; updateAge?: number } };
    expect(cfg.session?.maxAge).toBe(7 * 24 * 60 * 60);
    expect(cfg.session?.updateAge).toBe(24 * 60 * 60);
    expect(cfg.session?.maxAge).not.toBe(30 * 24 * 60 * 60);
  });
});
