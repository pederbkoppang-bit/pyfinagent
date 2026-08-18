---
name: subprocess-drive-that-redeclares-the-entrypoint
description: A "real CLI in a subprocess" test that imports the module and re-declares argparse never runs __main__, so the default it claims to prove is its own; and an ast.dump substring cannot see a Not()
metadata:
  type: feedback
---

A subprocess test labelled "drives the real CLI, because the default lives in
argument parsing" proved nothing: its harness did `import backfill_handoff_archive
as m` (which does **not** execute `if __name__ == "__main__":`), then re-declared
its own `argparse` block and called `m.main(dry_run=not a.execute)`. The `not`
under test lives in the SCRIPT's `__main__`; the harness carried a private copy.
Its companion assertion was an AST scan: `assert "execute" in ast.dump(kw["dry_run"])`.
Measured, both spellings pass:

    dry_run=not args.execute -> UnaryOp(op=Not(), operand=Attribute(..., attr='execute'))
    dry_run=args.execute     -> Attribute(..., attr='execute')

so the substring is blind to the `Not()`. It DOES kill a full revert to
`dry_run=args.dry_run` -- name the shape a guard covers, not just "it has an
assert". Dropping the `not` inverted a safe-by-default tool into execute-by-default
and **all 19 tests passed**.

**Why:** the two halves looked complementary (behavioural + structural) and were
both inert for the one mutation that matters. 75.11.4 cycle 1.

**How to apply:** for any "runs the real entrypoint" claim, check what the harness
actually EXECUTES -- `runpy.run_path(p, run_name="__main__")` or invoking the file
as a script runs `__main__`; `import` does not. Re-point the module's path constant
at a mutated COPY (`pytest.main(argv, plugins=[Repoint(tmpdir)])`) and re-run the
whole suite; a copy under `Path(__file__).resolve().parents[N]` is hermetic when
the script derives REPO that way, so nothing in the repo is touched. And when a
guard is a substring over `ast.dump`, spell out the two ASTs and diff them before
crediting it. Related: [[feedback_slice_and_exec_with_the_collaborator_stubbed]],
[[feedback_a_fix_verifier_can_be_vacuous_too]].
