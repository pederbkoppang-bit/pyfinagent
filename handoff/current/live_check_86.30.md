# live_check -- step 86.30

Verbatim machine output. Captured 2026-08-10.

## 1. Criterion 1 -- REPRODUCE: the defect BEFORE the fix
```
# with psutil blocked AND evicted from sys.modules, against the PRE-FIX expression
  interfaces_enumerable() = False
  UNDER-REFUSALS: 6/6 of this machine's own global IPv6 addresses
  e.g. _is_this_machine('2001:4654:6451:0:31:6467:1ea6:1852') = False   <-- THIS MACHINE CALLED REMOTE
       address_is_live_backend((that,8000))                    = False   (guard does NOT refuse)
  controls: 127.0.0.1 -> True ; 2606:4700:4700::1111 (Cloudflare) -> False
```

## 2. AFTER the fix -- both paths measured
```
NORMAL path (psutil present):
  interfaces_enumerable()                     = True
  is_live_backend('https://example.com:8000') = False   <- frozen row UNCHANGED
  is_live_backend('http://127.0.0.1:8000/x')  = True
  _is_this_machine(own global IPv6)           = True

DEGRADED path (psutil blocked + evicted):
  interfaces_enumerable()                     = False
  own global IPv6 called REMOTE               = 0/6   (was 6/6)
  _is_this_machine('127.0.0.1')               = True
  _is_this_machine('2606:4700:4700::1111')    = True   <- OVER-refuse, the safe direction
  is_live_backend('https://example.com:8000') = True   <- degraded mode ONLY, disclosed
```

## 3. Criterion 3 -- the non-degraded path is untouched
```
$ git diff --stat -- backend/tests/test_phase_86_6_subprocess_channel.py
(empty = byte-unchanged)
$ md5 -q backend/tests/test_phase_86_6_subprocess_channel.py
d9f3650c4054c2504c1bbfaccea25629

$ bash -c "... pytest backend/tests/test_phase_86_27_live_origin_class.py -q"   # IMMUTABLE COMMAND
..................................................                       [100%]
50 passed in 9.82s
```

## 4. Criterion 5 -- the bind family, measured
```
$ lsof -nP -iTCP:8000 -sTCP:LISTEN
COMMAND   PID USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
Python  43839 ford   10u  IPv4 0xe9558776222538c7      0t0  TCP *:8000 (LISTEN)

IPv4 only. psutil IS importable in today's venv. The defect was therefore
LATENT ON BOTH COUNTS and not reachable in practice at the time of the fix.
```

## 5. Criterion 6 -- mutation
```
CONTROL exit=0  9 passed
[D1] KILLED  revert to the is_global proxy -- own global IPv6 becomes 'provably remote'
      2 failed, 7 passed | FAILED ...test_this_machines_own_global_ipv6_is_never_called_remote
                          FAILED ...test_a_genuinely_remote_address_is_OVER_refused_not_allowed
[D2] KILLED  delete the degraded branch entirely -- fall through to 'return False' (allow)
      2 failed, 7 passed | same two named assertions
[D3] KILLED  a different wrong proxy (ip.is_private) -- also mis-answers global GUAs
      2 failed, 7 passed | same two named assertions

NOTE: D1's first anchor ('        return True') occurred 6x and was scored
ANCHOR-BAD rather than as a kill. Re-anchored on the preceding comment line.
```

## 6. Regression
```
$ python -m pytest backend/tests/ -q -k "86_27 or 86_6 or live_backend or subprocess_channel"
81 passed, 3337 deselected
$ python scripts/qa/mutation_matrix_86_27.py

All 7 mutants killed.
```
