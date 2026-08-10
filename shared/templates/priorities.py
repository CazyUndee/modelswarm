"""
Research Prioritization Framework.

Maintains three categories:
- EXPLOIT: Strong branches that deserve additional compute
- EXPLORE: New hypotheses with meaningful upside
- VERIFY: Results that require replication or robustness testing
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentPriority:
    """Priority assessment for an experiment."""
    experiment_id: str
    category: str  # EXPLOIT, EXPLORE, VERIFY
    expected_information_gain: float  # 0-1
    probability_of_improvement: float  # 0-1
    potential_impact: float  # 0-1
    compute_cost: float  # relative cost (hours)
    score: float = 0.0  # computed priority score

    def compute_score(self) -> float:
        """Compute priority score using information-gain principle."""
        if self.compute_cost <= 0:
            self.compute_score = 0.0
        self.score = (
            self.expected_information_gain
            * self.probability_of_improvement
            * self.potential_impact
            / self.compute_cost
        )
        return self.score


class ResearchPrioritizer:
    """Prioritize experiments using information-gain principles."""

    @staticmethod
    def categorize(experiment: dict, existing_results: list[dict]) -> str:
        """Categorize an experiment as EXPLOIT, EXPLORE, or VERIFY."""
        # Check if this is a replication
        for result in existing_results:
            if (result.get("hypothesis") == experiment.get("hypothesis")
                    and result.get("status") == "completed"):
                return "VERIFY"

        # Check if this builds on a known strong branch
        parent_id = experiment.get("parent_experiment_id")
        if parent_id:
            for result in existing_results:
                if (result.get("experiment_id") == parent_id
                        and result.get("decision") == "promoted"):
                    return "EXPLOIT"

        # Default: explore
        return "EXPLORE"

    @staticmethod
    def expected_information_gain(experiment: dict, existing_results: list[dict]) -> float:
        """Estimate expected information gain from an experiment.

        Higher gain for unexplored hypotheses and novel approaches.
        """
        base_gain = 0.5

        # Reduce gain if similar experiments exist
        similar = [
            r for r in existing_results
            if r.get("model") == experiment.get("model")
            and set(r.get("features", [])) & set(experiment.get("features", []))
        ]
        if similar:
            # Reduce gain proportionally to number of similar experiments
            base_gain *= 0.7 ** len(similar)

        # Increase gain for novel model families
        existing_models = {r.get("model") for r in existing_results}
        if experiment.get("model") not in existing_models:
            base_gain = min(base_gain * 1.3, 1.0)

        return round(base_gain, 3)

    @staticmethod
    def prioritize(experiments: list[dict], state: dict) -> list[ExperimentPriority]:
        """Prioritize a list of experiments.

        Returns sorted list (highest priority first).
        """
        existing = state.get("experiments", [])
        priorities = []

        for exp in experiments:
            category = ResearchPrioritizer.categorize(exp, existing)
            eig = ResearchPrioritizer.expected_information_gain(exp, existing)

            priority = ExperimentPriority(
                experiment_id=exp.get("experiment_id", ""),
                category=category,
                expected_information_gain=eig,
                probability_of_improvement=exp.get("prob_improvement", 0.5),
                potential_impact=exp.get("impact", 0.5),
                compute_cost=exp.get("compute_hours", 1.0),
            )
            priority.compute_score()
            priorities.append(priority)

        # Sort by score descending
        priorities.sort(key=lambda p: p.score, reverse=True)
        return priorities

    @staticmethod
    def allocate_compute(budget_hours: float,
                         candidates: list[ExperimentPriority]) -> dict[str, float]:
        """Allocate compute budget across categories.

        Dynamic allocation:
        - 40% EXPLOIT (strong branches)
        - 35% EXPLORE (new hypotheses)
        - 25% VERIFY (replication/robustness)
        """
        allocation = {"EXPLOIT": 0.0, "EXPLORE": 0.0, "VERIFY": 0.0}
        target = {"EXPLOIT": 0.4, "EXPLORE": 0.35, "VERIFY": 0.25}

        for category in allocation:
            cat_experiments = [c for c in candidates if c.category == category]
            if not cat_experiments:
                # Redistribute to other categories
                continue

            cat_budget = budget_hours * target[category]
            total_score = sum(c.score for c in cat_experiments)

            if total_score > 0:
                for c in cat_experiments:
                    c.compute_cost = (c.score / total_score) * cat_budget

            allocation[category] = cat_budget

        return allocation
