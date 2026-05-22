/**
 * phase-44.1: useKeyboardShortcut -- generalize the KillSwitchShortcut pattern.
 *
 * Binds a (modifier+key) combo to a handler at the document level. Cleans up
 * on unmount. Combo string format: "mod+k", "ctrl+shift+h", "cmd+k", "?".
 *
 * Cross-platform: "mod" maps to metaKey on darwin, ctrlKey elsewhere.
 */
"use client";

import { useEffect } from "react";

function isMac(): boolean {
  if (typeof navigator === "undefined") return false;
  return /Mac|iPhone|iPad/i.test(navigator.platform || navigator.userAgent || "");
}

interface ParsedCombo {
  mod: boolean;
  ctrl: boolean;
  alt: boolean;
  shift: boolean;
  key: string;
}

function parseCombo(combo: string): ParsedCombo {
  const parts = combo.toLowerCase().split("+").map((p) => p.trim());
  const last = parts[parts.length - 1] || "";
  return {
    mod: parts.includes("mod"),
    ctrl: parts.includes("ctrl"),
    alt: parts.includes("alt"),
    shift: parts.includes("shift"),
    key: last,
  };
}

export function useKeyboardShortcut(
  combo: string,
  handler: (event: KeyboardEvent) => void,
  options?: { enabled?: boolean; preventDefault?: boolean },
): void {
  const enabled = options?.enabled !== false;
  const preventDefault = options?.preventDefault !== false;

  useEffect(() => {
    if (!enabled) return;
    const parsed = parseCombo(combo);
    const onKey = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (key !== parsed.key) return;
      const modActive = isMac() ? event.metaKey : event.ctrlKey;
      if (parsed.mod && !modActive) return;
      if (parsed.ctrl && !event.ctrlKey) return;
      if (parsed.alt && !event.altKey) return;
      if (parsed.shift && !event.shiftKey) return;
      if (preventDefault) event.preventDefault();
      handler(event);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [combo, handler, enabled, preventDefault]);
}
