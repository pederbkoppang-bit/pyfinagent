"""phase-86.30 -- the live-origin guard's DEGRADED branch must refuse, not allow.

THE DEFECT. `scripts/qa/live_backend_origin.py::_is_this_machine` falls into a
degraded branch when it cannot enumerate this host's interfaces (psutil is
present only TRANSITIVELY, via gpt-researcher / unstructured, and appears in NO
requirements file). That branch used to `return not ip.is_global` and its own
comment called this "over-refusal, which is the safe direction".

IT WAS THE OPPOSITE, for exactly the addresses a modern host carries.
`is_global` answers ROUTABILITY, not OWNERSHIP: a host's own global unicast
addresses are `is_global == True`, so the expression answered "provably remote"
for this machine's own IPv6. MEASURED 2026-08-10 with psutil forced to fail:
**6 of 6** of this host's global IPv6 addresses were classified REMOTE -- an
UNDER-refusal hiding inside a branch documented as erring the other way.

WHY IT STAYS FIXED. RFC 8981 temporary addresses are global-scope and ROTATE, so
no enumeration is ever complete and `is_global` can never stand in for "mine".
Saltzer & Schroeder: an exclusion-mechanism mistake "tends to fail by allowing
access, a failure which may go unnoticed" (CWE-636).

THESE TESTS DERIVE THIS MACHINE'S ADDRESSES AT RUNTIME rather than hardcoding
them -- a hardcoded address would pass on this laptop and mean nothing anywhere
else, and RFC 8981 rotation would break it here too.
"""
from __future__ import annotations

import builtins
import importlib.util
import ipaddress
import pathlib
import re
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "qa" / "live_backend_origin.py"


def _load(tag: str):
    spec = importlib.util.spec_from_file_location(f"lbo_{tag}", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _own_global_v6() -> list[str]:
    """This machine's globally-routable IPv6 addresses, derived now."""
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=20).stdout
    except Exception:                                          # noqa: BLE001
        return []
    found = []
    for m in re.finditer(r"inet6 ([0-9a-f:]+)(?:%\w+)? prefixlen", out):
        try:
            ip = ipaddress.ip_address(m.group(1))
        except ValueError:
            continue
        if ip.version == 6 and ip.is_global:
            found.append(m.group(1))
    return found


class _NoPsutil:
    """Force the degraded branch.

    Blocking the import is not enough on its own: `_enumerate_interface_addresses`
    imports psutil LAZILY, so a module already in `sys.modules` is served from
    cache and the block is inert. The first version of this probe missed that and
    reported the degraded branch as unreachable. The eviction is the load-bearing
    half.
    """

    def __enter__(self):
        self._real = builtins.__import__
        self._saved = sys.modules.pop("psutil", None)

        def blocked(name, *a, **k):
            if name == "psutil":
                raise ImportError("psutil unavailable (phase-86.30 probe)")
            return self._real(name, *a, **k)

        builtins.__import__ = blocked
        mod = _load("degraded")
        mod._own_cache = None
        mod._own_enumerable = None
        return mod

    def __exit__(self, *exc):
        builtins.__import__ = self._real
        if self._saved is not None:
            sys.modules["psutil"] = self._saved
        return False


class TestDegradedBranchRefuses:
    def test_the_branch_is_actually_reached(self):
        """A positive control. If this fails the other degraded tests are
        vacuous -- they would be exercising the NORMAL path."""
        with _NoPsutil() as mod:
            assert mod.interfaces_enumerable() is False, (
                "the degraded branch was not reached, so every assertion below "
                "would be measuring the normal path instead"
            )

    def test_this_machines_own_global_ipv6_is_never_called_remote(self):
        own = _own_global_v6()
        if not own:
            pytest.skip("this host has no global IPv6 address to test with")
        with _NoPsutil() as mod:
            remote = [a for a in own if mod._is_this_machine(a) is False]
        assert remote == [], (
            f"{len(remote)}/{len(own)} of THIS MACHINE's own global IPv6 addresses "
            f"were classified REMOTE in degraded mode: {remote[:3]}. That is the "
            "under-refusal phase-86.30 removed (measured 6/6 before the fix)."
        )

    def test_loopback_is_still_this_machine(self):
        with _NoPsutil() as mod:
            assert mod._is_this_machine("127.0.0.1") is True
            assert mod._is_this_machine("::1") is True

    def test_a_genuinely_remote_address_is_OVER_refused_not_allowed(self):
        """The safe direction, stated as a deliberate consequence.

        In degraded mode we cannot tell Cloudflare from ourselves, so we refuse.
        This is the SAME direction as both sibling degraded paths: conftest.py
        falls back to port-only refusal, and `_canonical_addresses` returning
        None yields `verdict = True`.
        """
        with _NoPsutil() as mod:
            assert mod._is_this_machine("2606:4700:4700::1111") is True, (
                "degraded mode must refuse an address it cannot disprove"
            )


class TestNormalPathIsUntouched:
    """The frozen row is graded on the NORMAL path and must not move."""

    def test_frozen_example_row_still_allows(self):
        mod = _load("normal")
        if not mod.interfaces_enumerable():
            pytest.skip("interfaces not enumerable here; the normal path cannot be tested")
        assert mod.is_live_backend("https://example.com:8000") is False, (
            "the frozen row `https://example.com:8000 -> allow` "
            "(test_phase_86_6_subprocess_channel.py) moved on the NORMAL path"
        )

    def test_loopback_on_the_backend_port_still_refuses(self):
        mod = _load("normal2")
        if not mod.interfaces_enumerable():
            pytest.skip("interfaces not enumerable here")
        assert mod.is_live_backend("http://127.0.0.1:8000/api/x") is True

    def test_own_global_ipv6_is_this_machine_on_the_normal_path_too(self):
        own = _own_global_v6()
        if not own:
            pytest.skip("no global IPv6 on this host")
        mod = _load("normal3")
        if not mod.interfaces_enumerable():
            pytest.skip("interfaces not enumerable here")
        assert mod._is_this_machine(own[0]) is True

def _all_own_addresses() -> list[str]:
    """EVERY address on this host -- v4 and v6, global and link-local.

    Criterion 2 asks for the FULL set, not the global-IPv6 subset that happened
    to expose the defect. Derived from psutil when available (the same source
    the guard uses in its healthy path) so the two agree by construction.
    """
    try:
        import psutil
        import socket as _s
    except Exception:                                          # noqa: BLE001
        return []
    out = []
    for _iface, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if a.family in (_s.AF_INET, _s.AF_INET6):
                out.append(str(a.address).split("%")[0])
    return sorted(set(out))


#: Genuinely remote addresses. These MUST stay classified remote on the healthy
#: path -- the fix must not turn the guard into "refuse everything always",
#: which would pass every degraded-mode assertion while destroying the guard.
GENUINELY_REMOTE = [
    "2606:4700:4700::1111",   # Cloudflare DNS
    "8.8.8.8",                # Google DNS
    "1.1.1.1",                # Cloudflare DNS v4
    "93.184.216.34",          # example.com (the frozen row's address family)
]


class TestCriterion2FullAddressSet:
    def test_degraded_mode_calls_NO_own_address_remote(self):
        """The full set: v4, v6, global AND link-local."""
        own = _all_own_addresses()
        if not own:
            pytest.skip("psutil unavailable, so the full own-address set cannot be derived")
        with _NoPsutil() as mod:
            remote = [a for a in own if mod._is_this_machine(a) is False]
        assert remote == [], (
            f"{len(remote)}/{len(own)} of this machine's OWN addresses were called "
            f"remote in degraded mode: {remote[:5]}"
        )

    def test_healthy_path_still_calls_remote_addresses_remote(self):
        """The other half of criterion 2, and the anti-vacuity control.

        Without this, 'refuse everything unconditionally' would satisfy every
        degraded-mode assertion above while making the guard useless.
        """
        mod = _load("remote_check")
        if not mod.interfaces_enumerable():
            pytest.skip("interfaces not enumerable here")
        wrong = [a for a in GENUINELY_REMOTE if mod._is_this_machine(a) is not False]
        assert wrong == [], (
            f"these genuinely remote addresses were called THIS MACHINE on the "
            f"healthy path: {wrong}"
        )
