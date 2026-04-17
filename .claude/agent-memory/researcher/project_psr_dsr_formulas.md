---
name: PSR/DSR formulas for paper_metrics_v2
description: Exact PSR and DSR formulas from Bailey & Lopez de Prado, kurtosis convention (raw vs excess), Python pseudocode, anti-patterns. Researched for backend/services/paper_metrics_v2.py.
type: project
---

Researched April 2026 for paper_metrics_v2.py implementation.

**Why:** PSR and DSR are the core evaluation metrics (DSR 0.9984 is current best). Implementation requires exact formula to avoid kurtosis convention bugs.

**Key facts:**
- PSR formula uses raw kurtosis (γ4 default = 3 for normal). Term in denominator is (γ4 - 1)/4, NOT (γ4 - 3)/4.
- scipy.stats.kurtosis() returns EXCESS kurtosis by default (normal = 0). Must add 3 to get raw before using in PSR formula.
- DSR SR* threshold uses Euler-Mascheroni γ ≈ 0.5772 and Var[SR] across N independent trials.
- Denominator zero-division when SR_obs ≈ sqrt(4/(γ4-1)): guard with epsilon clamp.

**How to apply:** Always note kurtosis convention in code comments; add 3 to scipy excess kurtosis before plugging into PSR denominator.

Sources:
- https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf
- https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- https://quantresearch.org/PSR.py.txt (Lopez de Prado's own Python, 2012)
- https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio
