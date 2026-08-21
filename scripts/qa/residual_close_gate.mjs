/**
 * phase-90.2 -- the CONSUMER half of severity routing.
 *
 * Routing a finding to `queue_residual` means nothing on its own. This module is
 * what makes it oblige something: when a step's Q/A routed to `queue_residual`,
 * the parent step's CLOSE is refused until the residual has actually been FILED
 * as its own masterplan step carrying its own immutable verification command.
 *
 *   import { refuseCloseOnUnfiledResidual } from './residual_close_gate.mjs'
 *
 * Criterion 5: "Main's required response to queue_residual is to FILE a
 * masterplan step carrying its own immutable verification command -- not an
 * in-place fix and not a fresh spawn -- and a checker refuses the parent step's
 * close when the filed residual is absent or does not parse."
 *
 * WHAT "DOES NOT PARSE" MEANS HERE, stated because the criterion does not define
 * it and an undefined predicate is an unfalsifiable one:
 *
 *   1. the masterplan file itself does not parse as JSON            -> REFUSE
 *   2. no step in it references the parent step id                  -> REFUSE
 *   3. a referencing step exists but carries no non-empty
 *      `verification.command`                                       -> REFUSE
 *   4. a referencing step exists but carries no non-empty
 *      `verification.success_criteria`                              -> REFUSE
 *
 * A residual step that merely NAMES the parent without a command is exactly the
 * failure this gate exists to catch: it looks filed and grades nothing.
 *
 * FAIL-CLOSED. Every error path REFUSES. That is the opposite of the house's
 * usual fail-open hook discipline, and deliberately so: a hook that breaks the
 * harness must fail open, but a CLOSE gate that cannot read its own evidence
 * must not let the close through -- fail-open here would hide its own breakage,
 * which is the defect class it is meant to prevent.
 *
 * This module NEVER writes. It takes the masterplan as a string or an object and
 * returns a verdict object; it opens no file and mutates nothing.
 */

/** Does this step object look like a filed residual for `parentStepId`? */
function referencesParent (step, parentStepId) {
  if (!step || typeof step !== 'object') return false
  if (String(step.id) === String(parentStepId)) return false // the parent is not its own residual
  const hay = [step.audit_basis, step.notes, step.name]
    .filter(v => typeof v === 'string').join('\n')
  if (!hay) return false
  // Word-boundary-ish match so parent "90.1" is not satisfied by "90.10".
  const esc = String(parentStepId).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return new RegExp('(?<![0-9.])' + esc + '(?![0-9.])').test(hay)
}

function collectSteps (node, out) {
  if (Array.isArray(node)) { for (const v of node) collectSteps(v, out); return }
  if (node && typeof node === 'object') {
    if ('id' in node && node.verification && typeof node.verification === 'object') out.push(node)
    for (const v of Object.values(node)) collectSteps(v, out)
  }
}

/**
 * @param {object} arg
 * @param {string} arg.parentStepId    the step whose close is being decided
 * @param {object} arg.routing         the object enforceSeverityRouting returned
 * @param {string|object} arg.masterplan  the plan of record, raw text or parsed
 * @returns {{allowed: boolean, reason: string, filed: string[], checked: number}}
 */
export function refuseCloseOnUnfiledResidual ({ parentStepId, routing, masterplan }) {
  const owed = routing && routing.route === 'queue_residual'
  if (!owed) {
    return {
      allowed: true,
      reason: 'no residual is owed: route is '
        + (routing && routing.route ? routing.route : 'ABSENT')
        + ' -- this gate binds only on queue_residual',
      filed: [],
      checked: 0,
    }
  }
  let plan = masterplan
  if (typeof plan === 'string') {
    try { plan = JSON.parse(plan) } catch (e) {
      return {
        allowed: false,
        reason: 'REFUSED: the plan of record does not parse as JSON (' + e.message
          + '), so the filed residual cannot be shown to exist. Fail-closed.',
        filed: [],
        checked: 0,
      }
    }
  }
  if (!plan || typeof plan !== 'object') {
    return {
      allowed: false,
      reason: 'REFUSED: no usable plan of record was supplied. Fail-closed.',
      filed: [],
      checked: 0,
    }
  }
  const steps = []
  collectSteps(plan, steps)
  const referencing = steps.filter(s => referencesParent(s, parentStepId))
  if (referencing.length === 0) {
    return {
      allowed: false,
      reason: 'REFUSED: the Q/A routed a finding to queue_residual for step '
        + parentStepId + ', and NO step in the plan of record references it. The '
        + 'required response is to FILE the residual as its own step with its own '
        + 'immutable verification command -- not to fix it in place, and not to '
        + 'spawn a fresh Q/A.',
      filed: [],
      checked: steps.length,
    }
  }
  const wellFormed = referencing.filter(s => {
    const v = s.verification || {}
    const cmd = typeof v.command === 'string' ? v.command.trim() : ''
    const crit = Array.isArray(v.success_criteria)
      ? v.success_criteria.filter(c => typeof c === 'string' && c.trim()) : []
    return cmd.length > 0 && crit.length > 0
  })
  if (wellFormed.length === 0) {
    return {
      allowed: false,
      reason: 'REFUSED: ' + referencing.length + ' step(s) reference ' + parentStepId
        + ' (' + referencing.map(s => s.id).join(', ') + ') but none carries BOTH a '
        + 'non-empty verification.command AND non-empty success_criteria. A residual '
        + 'that names its parent and grades nothing is the shape this gate exists to '
        + 'catch: it looks filed and cannot fail.',
      filed: referencing.map(s => String(s.id)),
      checked: steps.length,
    }
  }
  return {
    allowed: true,
    reason: 'allowed: residual(s) ' + wellFormed.map(s => s.id).join(', ')
      + ' are filed for ' + parentStepId + ', each carrying its own immutable '
      + 'verification command and non-empty criteria',
    filed: wellFormed.map(s => String(s.id)),
    checked: steps.length,
  }
}
