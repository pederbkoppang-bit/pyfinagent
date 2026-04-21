# Experiment Results — phase-8.5 / 8.5.5 REMEDIATION v1

Immutable: 4/4 PASS + exit 0. Gate frozen dataclass + pure evaluate + CPCV C(6,2)=15 folds all confirmed on re-run.

Researcher substantive findings:
- DSR>=0.95 threshold grounded in 95% CDF interpretation (Bailey-Lopez de Prado).
- PBO<=0.20 is conservative convention (0.50 is canonical majority-overfit line); defensible as stricter safety guard.
- Conjunction (DSR AND PBO) is the correct defense-in-depth posture; the two metrics measure orthogonal failure modes.

Carry-forward advisory: document PBO<=0.20 as project-specific tightening in the gate docstring; canonical threshold is 0.50.
