/**
 * phase-44.1: feature-flag registry.
 *
 * Per `/goal` integration gate 3: "Every new UI/operator-visible backend
 * feature behind featureFlags.ts / env var (default OFF)".
 *
 * Reading order: NEXT_PUBLIC_FEATURE_<KEY>=true env var > localStorage > default.
 * Server-side renders see only env + defaults; client hydration overlays
 * localStorage. Toggle in /settings (phase-44.8 will surface this).
 */

export type FeatureFlagKey =
  | "command_palette" // phase-44.1: Cmd-K palette
  | "states_library" // phase-44.1: extracted states components
  | "sidebar_a11y_v2" // phase-44.1: ARIA + localStorage + skip link
  | "cockpit_v2" // phase-44.2
  | "decision_drawer_v2" // phase-44.3
  | "url_state_deep_link" // phase-44.4
  | "trade_history_datatable" // phase-44.4 / 44.5
  | "trace_tree" // phase-44.7
  | "settings_search" // phase-44.8
  | "settings_audit_log" // phase-44.8
  | "sse_live_updates"; // phase-44.10

const DEFAULTS: Record<FeatureFlagKey, boolean> = {
  command_palette: true, // phase-44.1: ships ON (operator-approved 2026-05-22)
  states_library: true, // phase-44.1: pure utility, no risk
  sidebar_a11y_v2: true, // phase-44.1: a11y improvement, pure-additive
  cockpit_v2: false,
  decision_drawer_v2: false,
  url_state_deep_link: false,
  trade_history_datatable: false,
  trace_tree: false,
  settings_search: false,
  settings_audit_log: false,
  sse_live_updates: false,
};

function readEnv(key: FeatureFlagKey): boolean | null {
  const envKey = `NEXT_PUBLIC_FEATURE_${key.toUpperCase()}`;
  if (typeof process !== "undefined" && process.env && envKey in process.env) {
    const raw = process.env[envKey];
    if (raw === "true" || raw === "1") return true;
    if (raw === "false" || raw === "0") return false;
  }
  return null;
}

function readLocalStorage(key: FeatureFlagKey): boolean | null {
  if (typeof window === "undefined" || !window.localStorage) return null;
  const v = window.localStorage.getItem(`pyfinagent.featureFlag.${key}`);
  if (v === "true") return true;
  if (v === "false") return false;
  return null;
}

export function isFeatureEnabled(key: FeatureFlagKey): boolean {
  const envOverride = readEnv(key);
  if (envOverride !== null) return envOverride;
  const lsOverride = readLocalStorage(key);
  if (lsOverride !== null) return lsOverride;
  return DEFAULTS[key];
}

export function setFeatureFlag(key: FeatureFlagKey, value: boolean): void {
  if (typeof window === "undefined" || !window.localStorage) return;
  window.localStorage.setItem(`pyfinagent.featureFlag.${key}`, String(value));
}

export function listFeatureFlags(): Array<{ key: FeatureFlagKey; enabled: boolean; default: boolean }> {
  const keys = Object.keys(DEFAULTS) as FeatureFlagKey[];
  return keys.map((key) => ({
    key,
    enabled: isFeatureEnabled(key),
    default: DEFAULTS[key],
  }));
}
