// @vitest-environment node
/**
 * phase-16.37 (#51) regression tests for lighthouse-wrapper extractUrl.
 *
 * Verifies the argv translator converts `--url X` and `--url=X` to a
 * positional URL, leaves all other flags in `rest`, and handles the
 * trailing-flag edge case correctly.
 *
 * The wrapper exports extractUrl only when require.main !== module,
 * so importing it from this test file does NOT spawn lighthouse.
 *
 * Uses .mjs (ESM) because vitest is ESM-only; wrapper itself is CJS,
 * imported here via the default-export interop pattern.
 */
import { describe, it, expect } from "vitest";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { extractUrl } = require("./lighthouse-wrapper.js");

describe("extractUrl", () => {
  it("extracts --url X positional form and preserves rest", () => {
    const result = extractUrl([
      "--url", "http://localhost:3000",
      "--output", "json",
      "--output-path", "/tmp/x.json",
    ]);
    expect(result.url).toBe("http://localhost:3000");
    expect(result.rest).toEqual(["--output", "json", "--output-path", "/tmp/x.json"]);
  });

  it("extracts --url=X equals form and preserves rest", () => {
    const result = extractUrl([
      "--url=https://example.com/path",
      "--quiet",
    ]);
    expect(result.url).toBe("https://example.com/path");
    expect(result.rest).toEqual(["--quiet"]);
  });

  it("returns null url when no --url arg present", () => {
    const result = extractUrl(["--output", "json", "--quiet"]);
    expect(result.url).toBeNull();
    expect(result.rest).toEqual(["--output", "json", "--quiet"]);
  });

  it("treats trailing --url as a rest arg when no value follows", () => {
    // Loop bound `i + 1 < argv.length` is false for the last position,
    // so a dangling `--url` should fall through to rest, not consume nothing.
    const result = extractUrl(["--quiet", "--url"]);
    expect(result.url).toBeNull();
    expect(result.rest).toEqual(["--quiet", "--url"]);
  });

  it("handles --url=X mixed with other flags before and after", () => {
    const result = extractUrl([
      "--quiet",
      "--url=https://app.example.com",
      "--chrome-flags=--headless",
    ]);
    expect(result.url).toBe("https://app.example.com");
    expect(result.rest).toEqual(["--quiet", "--chrome-flags=--headless"]);
  });
});
