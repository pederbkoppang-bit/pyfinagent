"use client";

import { useCallback, useEffect, useState } from "react";
import { postPaperKillSwitchAction } from "@/lib/api";

const SHORTCUT_DESC = "Ctrl/Cmd+Shift+H";

export function KillSwitchShortcut() {
  const [status, setStatus] = useState<string | null>(null);

  const halt = useCallback(async () => {
    const ok = window.confirm(
      `Emergency halt: flatten all paper-trading positions and pause new orders?\n\nShortcut: ${SHORTCUT_DESC}`,
    );
    if (!ok) return;
    setStatus("Halting...");
    try {
      await postPaperKillSwitchAction("FLATTEN_ALL");
      await postPaperKillSwitchAction("PAUSE");
      setStatus("Halted: positions flattened, trading paused.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Kill-switch call failed";
      setStatus(`Halt failed: ${msg}`);
    }
    window.setTimeout(() => setStatus(null), 4000);
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.shiftKey && (e.key === "H" || e.key === "h")) {
        e.preventDefault();
        halt();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [halt]);

  return (
    <div
      aria-live="polite"
      role="status"
      className="sr-only"
      data-testid="kill-switch-shortcut"
      data-shortcut={SHORTCUT_DESC}
    >
      {status ?? `Keyboard shortcut active: ${SHORTCUT_DESC} = emergency halt`}
    </div>
  );
}
