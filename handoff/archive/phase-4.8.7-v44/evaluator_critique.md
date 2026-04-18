# Evaluator Critique -- Cycle 90 / phase-4.9 step 4.9.2

Step: 4.9.2 Startup loader with no hot-reload

## Dual-evaluator run (parallel, anti-rubber-stamp, research-gate
upheld 5th cycle)

## qa-evaluator: PASS

7-point substantive review:
1. SIGHUP really ignored: `signal.signal(SIGHUP, SIG_IGN)` in
   load_once() under `_init_lock`; audit's runtime
   getsignal(SIGHUP) confirms.
2. mutation_kills_process: `os._exit(2)` in watcher loop; no
   `sys.exit` (catchable). Exception handler on transient FS
   errors correctly log-and-continues but does NOT swallow
   os._exit (which bypasses Python exception machinery).
3. Daemon thread: `daemon=True`, so clean shutdown works.
4. Env-var disable narrowly scoped to watcher-spawn only;
   SIGHUP-ignore still installs regardless.
5. Digest exposed to /api/health: real return-dict inclusion,
   not just log.
6. Lifespan wires load_once() on main thread pre-fork; signal.
   signal() legal per Python docs.
7. `_init_lock` + `_initialized` guard ensures one-shot side
   effects under concurrent lifespan calls.

## harness-verifier: PASS

8/8 mechanical checks green:
- Immutable verification exits 0.
- Audit clean (8/8 teeth true).
- Artifact structure correct.
- load_once() returns same object twice.
- digest 64-char lowercase hex.
- **Mutation A**: os._exit -> sys.exit -> audit rc=1.
- **Mutation B**: SIGHUP line removed -> audit rc=1.
- **Mutation C**: limits_digest stripped from /api/health -> rc=1.

Three independent mutations prove each invariant has real teeth.

## Decision: PASS (evaluator-owned)

Both evaluators green with three mutation-resistance tests.
Research-gate discipline held for 5th cycle in a row.
