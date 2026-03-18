"""
Experiment analysis tooling — the "analysis.ipynb" equivalent from autoresearch.

Loads skill_results.tsv and computes:
- Keep rate (overall + per-agent)
- Delta chain (cumulative improvement from kept experiments)
- Running best chart data
- Top hits table
- Summary statistics

Can be run standalone or imported by the API for JSON output.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

RESULTS_TSV = Path(__file__).parent / "skill_results.tsv"


def load_experiments(tsv_path: Path = None) -> list[dict]:
    """Load all experiments from the TSV file."""
    path = tsv_path or RESULTS_TSV
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            # Parse numeric fields
            for field in ("metric_before", "metric_after", "delta"):
                try:
                    row[field] = float(row.get(field, 0))
                except (ValueError, TypeError):
                    row[field] = 0.0
            rows.append(row)
        return rows


def compute_keep_rate(experiments: list[dict]) -> dict:
    """Compute overall and per-agent keep rates."""
    overall = {"kept": 0, "discarded": 0, "crashed": 0, "pending": 0, "total": 0}
    per_agent: dict[str, dict] = defaultdict(lambda: {"kept": 0, "discarded": 0, "crashed": 0, "total": 0})

    for exp in experiments:
        status = exp.get("status", "")
        agent = exp.get("agent", "UNKNOWN")

        if status == "keep":
            overall["kept"] += 1
            per_agent[agent]["kept"] += 1
        elif status == "discard":
            overall["discarded"] += 1
            per_agent[agent]["discarded"] += 1
        elif status == "crash":
            overall["crashed"] += 1
            per_agent[agent]["crashed"] += 1
        elif status == "pending":
            overall["pending"] += 1

        overall["total"] += 1
        per_agent[agent]["total"] += 1

    decisions = overall["kept"] + overall["discarded"]
    overall["keep_rate"] = round(overall["kept"] / decisions, 3) if decisions > 0 else 0.0

    agent_rates = {}
    for agent, counts in per_agent.items():
        d = counts["kept"] + counts["discarded"]
        agent_rates[agent] = {
            **counts,
            "keep_rate": round(counts["kept"] / d, 3) if d > 0 else 0.0,
        }

    return {"overall": overall, "per_agent": agent_rates}


def compute_delta_chain(experiments: list[dict]) -> list[dict]:
    """
    Compute the cumulative improvement from kept experiments.

    Returns a list of kept experiments with cumulative_delta field.
    """
    chain = []
    cumulative = 0.0

    for exp in experiments:
        if exp.get("status") == "keep" and exp.get("agent") != "BASELINE":
            delta = exp.get("delta", 0.0)
            cumulative += delta
            chain.append({
                "agent": exp["agent"],
                "delta": delta,
                "cumulative_delta": round(cumulative, 4),
                "description": exp.get("description", ""),
                "timestamp": exp.get("timestamp", ""),
                "commit": exp.get("commit", ""),
            })

    return chain


def compute_running_best(experiments: list[dict]) -> list[dict]:
    """
    Compute running best chart data.

    X = experiment index, Y = metric_after for kept experiments (step function).
    All experiments included for the scatter plot.
    """
    points = []
    running_best = None

    for i, exp in enumerate(experiments):
        status = exp.get("status", "")
        metric = exp.get("metric_after", 0.0)

        if status == "keep":
            running_best = metric

        points.append({
            "index": i,
            "agent": exp.get("agent", ""),
            "status": status,
            "metric": metric,
            "running_best": running_best,
            "description": exp.get("description", ""),
        })

    return points


def compute_top_hits(experiments: list[dict], n: int = 10) -> list[dict]:
    """Get the top N experiments ranked by delta improvement."""
    kept = [
        {
            "agent": exp["agent"],
            "delta": exp.get("delta", 0.0),
            "description": exp.get("description", ""),
            "timestamp": exp.get("timestamp", ""),
            "commit": exp.get("commit", ""),
        }
        for exp in experiments
        if exp.get("status") == "keep" and exp.get("agent") != "BASELINE"
    ]
    kept.sort(key=lambda x: x["delta"], reverse=True)
    return kept[:n]


def compute_summary(experiments: list[dict]) -> dict:
    """
    Compute summary statistics.

    Returns baseline metric, current best, total improvement %,
    experiments per improvement, and best experiment description.
    """
    if not experiments:
        return {
            "baseline": 0.0,
            "current_best": 0.0,
            "total_improvement_pct": 0.0,
            "total_experiments": 0,
            "experiments_per_improvement": 0.0,
            "best_experiment": None,
        }

    baseline = None
    current_best = None

    for exp in experiments:
        metric = exp.get("metric_after", 0.0)
        if exp.get("agent") == "BASELINE":
            baseline = metric
        if exp.get("status") == "keep":
            if current_best is None or metric > current_best:
                current_best = metric

    baseline = baseline or 0.0
    current_best = current_best or baseline

    improvement = current_best - baseline
    improvement_pct = (improvement / baseline * 100) if baseline != 0 else 0.0

    kept_count = sum(1 for e in experiments if e.get("status") == "keep" and e.get("agent") != "BASELINE")
    total = len(experiments)

    top_hits = compute_top_hits(experiments, n=1)
    best_experiment = top_hits[0] if top_hits else None

    return {
        "baseline": baseline,
        "current_best": current_best,
        "total_improvement": round(improvement, 4),
        "total_improvement_pct": round(improvement_pct, 2),
        "total_experiments": total,
        "kept_experiments": kept_count,
        "experiments_per_improvement": round(total / kept_count, 1) if kept_count > 0 else 0.0,
        "best_experiment": best_experiment,
    }


def full_analysis(tsv_path: Path = None) -> dict:
    """
    Run the complete analysis and return a JSON-serializable dict.

    Used by the API endpoint: GET /api/skills/analysis
    """
    experiments = load_experiments(tsv_path)

    return {
        "summary": compute_summary(experiments),
        "keep_rates": compute_keep_rate(experiments),
        "delta_chain": compute_delta_chain(experiments),
        "running_best": compute_running_best(experiments),
        "top_hits": compute_top_hits(experiments),
        "total_experiments": len(experiments),
    }


# ── CLI Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json

    result = full_analysis()
    print(_json.dumps(result, indent=2))

    summary = result["summary"]
    print(f"\n{'='*50}")
    print(f"Baseline:          {summary['baseline']:.4f}")
    print(f"Current best:      {summary['current_best']:.4f}")
    print(f"Total improvement: {summary['total_improvement']:+.4f} ({summary['total_improvement_pct']:+.1f}%)")
    print(f"Experiments:       {summary['total_experiments']} total, {summary['kept_experiments']} kept")
    print(f"Efficiency:        {summary['experiments_per_improvement']:.1f} experiments per improvement")

    keep_rates = result["keep_rates"]
    print(f"\nOverall keep rate: {keep_rates['overall']['keep_rate']:.1%}")

    if result["top_hits"]:
        print(f"\nTop hit: {result['top_hits'][0]['description']} (delta={result['top_hits'][0]['delta']:+.4f})")
