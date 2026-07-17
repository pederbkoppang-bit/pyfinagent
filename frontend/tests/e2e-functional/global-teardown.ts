import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import * as path from "node:path";

/**
 * phase-64.1: the functional :3100 server runs `next dev` with
 * distDir=.next-functional (next.config.js reads PLAYWRIGHT_DIST_DIR) so it
 * never shares `.next` with the operator's live :3000 dev server. As a side
 * effect, `next dev` rewrites the tracked `next-env.d.ts` + `tsconfig.json` to
 * reference `.next-functional`. This teardown restores them to the committed
 * (`.next`) state -- but ONLY if they were actually polluted (they contain
 * `.next-functional`), so a normal visual-regression run is untouched.
 *
 * Best-effort: never throws (a teardown failure must not fail the suite).
 */
export default function globalTeardown(): void {
  const repoRoot = path.resolve(process.cwd(), "..");
  for (const rel of ["frontend/next-env.d.ts", "frontend/tsconfig.json"]) {
    try {
      const abs = path.join(repoRoot, rel);
      if (readFileSync(abs, "utf8").includes(".next-functional")) {
        execSync(`git show HEAD:'${rel}' > '${rel}'`, { cwd: repoRoot, stdio: "ignore" });
      }
    } catch {
      /* best-effort; leave the file as-is if restore fails */
    }
  }
}
