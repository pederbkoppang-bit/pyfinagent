"""phase-86.108 (criterion 4): make "committed is not in force" checkable.

## The gap

`GET /api/settings/` returns `response_model=FullSettings`, and FastAPI filters
the payload to exactly that model's fields. `FullSettings` declares **45**
fields. `Settings` declares **264**. Everything in the difference is invisible
to the API -- including every operator-gated dark flag the harness is required
to check before claiming a change is live. There was no read path at all: an
operator could confirm a flag was committed to `.env` and had no way to ask the
RUNNING process what value it actually holds.

## The population is DERIVED, not listed

The step that filed this named seven flags. The real population is **168**. A
hardcoded list of seven would have been wrong the day it shipped and would rot
on the next flag added, so the population here comes from a stated rule:

    a "gated scalar" is a `Settings` field whose annotation is bool / int /
    float and whose name is NOT among `FullSettings`' fields.

Two properties fall out of that rule and both are load-bearing:

1. **It cannot go stale.** Add a flag to `Settings` and it appears here with no
   edit. Expose one via `FullSettings` and it drops out, because it is no
   longer invisible.
2. **It cannot leak a secret.** The rule admits only bool/int/float. Every API
   key, token, path and connection string on `Settings` is a `str` or a
   `SecretStr`, so no such field can enter this payload by construction --
   not by policy, not by a denylist that someone has to remember to update.
   `test_no_string_typed_field_can_enter_the_payload` pins it.

## In force vs committed

For each gated scalar this module reports both:

- `in_force` -- read from the running process's `Settings` object, i.e. the
  exact value production code reads.
- `env_file` -- the raw text in `backend/.env`, or `None` when the key is
  absent (the field then holds its default).

They disagree whenever `.env` was edited without a restart. That divergence is
the whole point: it is the state in which a change is committed, looks correct
in the diff, and is not running.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

_SCALAR_TYPES = (bool, int, float)

# `KEY=value`, tolerating leading whitespace and an `export ` prefix. Anchored
# so a commented-out line (`# KEY=...`) cannot match.
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*?)\s*$")


def gated_scalar_names() -> list[str]:
    """The derived population, sorted. See the module docstring for the rule.

    Imports are function-local and deliberate: `settings_api` imports FastAPI
    and this module is imported by it, so a module-level import would be
    circular.
    """
    from backend.api.settings_api import FullSettings
    from backend.config.settings import Settings

    exposed = set(FullSettings.model_fields)
    return sorted(
        name
        for name, field in Settings.model_fields.items()
        if name not in exposed and field.annotation in _SCALAR_TYPES
    )


def _read_env_raw(keys: set[str]) -> dict[str, str]:
    """Raw `.env` text for EXACTLY the requested keys.

    Never returns a key that was not asked for, so this cannot become a
    file-dump primitive even if a future caller passes a wider set. Missing or
    unreadable file -> `{}` (the fields then simply report their defaults);
    the read error is logged rather than swallowed.
    """
    out: dict[str, str] = {}
    try:
        if not _ENV_FILE.exists():
            return out
        for line in _ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _ENV_LINE.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key in keys:
                out[key] = val.strip().strip('"').strip("'")
    except Exception as exc:
        logger.error(
            "phase-86.108: gated-flag .env read FAILED (%r). in_force values are "
            "still exact; env_file will read as absent for every key, which "
            "UNDER-reports divergence -- treat this run as inconclusive.", exc,
        )
    return out


def _coerce_like(raw: str, current: Any) -> Any:
    """Best-effort parse of a raw `.env` string into the field's type.

    Returns the raw string unchanged when it cannot be parsed, because an
    unparseable `.env` value is itself worth showing rather than hiding behind
    a coerced default.
    """
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        return raw
    try:
        if isinstance(current, int):
            return int(raw)
        if isinstance(current, float):
            return float(raw)
    except (TypeError, ValueError):
        return raw
    return raw


def gated_flag_report(only: Optional[list[str]] = None) -> dict[str, Any]:
    """Live values of every gated scalar, beside what `.env` currently says.

    Args:
        only: restrict to these field names. Unknown names are reported under
            `requested_but_unknown` rather than dropped -- a typo that silently
            returns an empty set is how an observability endpoint lies.
    """
    from backend.config.settings import get_settings

    names = gated_scalar_names()
    unknown: list[str] = []
    if only:
        wanted = set(only)
        unknown = sorted(wanted - set(names))
        names = [n for n in names if n in wanted]

    settings = get_settings()
    env_raw = _read_env_raw({n.upper() for n in names})

    flags: dict[str, Any] = {}
    divergent: list[str] = []
    for name in names:
        in_force = getattr(settings, name, None)
        env_key = name.upper()
        has_env = env_key in env_raw
        env_val = _coerce_like(env_raw[env_key], in_force) if has_env else None
        is_div = bool(has_env and env_val != in_force)
        if is_div:
            divergent.append(name)
        flags[name] = {
            "in_force": in_force,
            "env_file": env_val,
            "env_file_present": has_env,
            "divergent": is_div,
        }

    return {
        "population_rule": (
            "a Settings field whose annotation is bool/int/float and whose name "
            "is NOT among FullSettings' fields. Derived at call time, never a "
            "hardcoded list, so it cannot go stale and cannot admit a str-typed "
            "(i.e. secret-bearing) field."
        ),
        "count": len(flags),
        "population_total": len(gated_scalar_names()),
        "divergent": divergent,
        "divergent_count": len(divergent),
        "divergent_meaning": (
            "backend/.env holds a different value than the running process. The "
            "change is committed and NOT in force; only a restart closes it."
        ),
        "requested_but_unknown": unknown,
        "env_file": str(_ENV_FILE),
        "env_file_present": _ENV_FILE.exists(),
        "pid": os.getpid(),
        "flags": flags,
    }
