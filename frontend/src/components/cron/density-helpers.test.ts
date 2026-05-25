import { describe, it, expect } from "vitest";
import {
  LINE_HEIGHT_CLASS,
  LINE_FONT_CLASS,
  parseLevel,
  levelColorClass,
  type LogDensity,
  type LogLevel,
} from "./density-helpers";

describe("density-helpers (phase-44.7)", () => {
  it("LINE_HEIGHT_CLASS covers both densities", () => {
    expect(LINE_HEIGHT_CLASS.comfortable).toContain("min-h-[32px]");
    expect(LINE_HEIGHT_CLASS.compact).toContain("min-h-[16px]");
  });

  it("LINE_FONT_CLASS covers both densities", () => {
    expect(LINE_FONT_CLASS.comfortable).toContain("text-[11px]");
    expect(LINE_FONT_CLASS.compact).toContain("text-[10px]");
  });

  it("density type covers comfortable + compact", () => {
    const a: LogDensity = "comfortable";
    const b: LogDensity = "compact";
    expect(a).toBe("comfortable");
    expect(b).toBe("compact");
  });
});

describe("parseLevel", () => {
  it("matches ERROR", () => {
    expect(parseLevel("2026-05-25 ERROR Something broke")).toBe("ERROR");
  });
  it("matches WARN", () => {
    expect(parseLevel("WARN: warning here")).toBe("WARN");
  });
  it("matches WARNING -> WARN", () => {
    expect(parseLevel("WARNING: warning here")).toBe("WARN");
  });
  it("matches INFO", () => {
    expect(parseLevel("INFO: starting up")).toBe("INFO");
  });
  it("matches DEBUG", () => {
    expect(parseLevel("DEBUG: details")).toBe("DEBUG");
  });
  it("returns null for unknown lines", () => {
    expect(parseLevel("plain text")).toBe(null);
  });
  it("is case-insensitive", () => {
    expect(parseLevel("error something")).toBe("ERROR");
  });
});

describe("levelColorClass", () => {
  const cases: Array<[LogLevel, string]> = [
    ["ERROR", "text-rose-300"],
    ["WARN", "text-amber-300"],
    ["INFO", "text-sky-300"],
    ["DEBUG", "text-slate-500"],
    [null, "text-slate-300"],
  ];
  for (const [lvl, expected] of cases) {
    it(`returns ${expected} for ${lvl ?? "null"}`, () => {
      expect(levelColorClass(lvl)).toBe(expected);
    });
  }
});
