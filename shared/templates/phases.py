"""
12-Phase Research Orchestration Framework.

Phases are strategic states, not immutable walls.
A discovery in Phase 9 can justify returning to Phase 6.
The lead agent controls phase transitions.
"""

from dataclasses import dataclass, field
from typing import Any


PHASES: dict[int, dict[str, Any]] = {
    1: {
        "name": "Reconnaissance",
        "description": "Understand the competition before spending serious compute.",
        "goals": [
            "Understand dataset structure, feature types, missing values",
            "Establish baseline, validation strategy, experiment infrastructure",
            "Identify initial hypotheses and potential leakage",
        ],
        "exit_criteria": "Baseline established, validation strategy confirmed",
    },
    2: {
        "name": "Validation and Statistical Foundations",
        "description": "Establish trustworthy evaluation.",
        "goals": [
            "Determine fold count, stratification, repeated CV",
            "Assess seed sensitivity, fold variance, distribution shifts",
            "Determine whether apparent improvements are real",
        ],
        "exit_criteria": "Stable validation protocol confirmed, noise floor estimated",
    },
    3: {
        "name": "Core Model Exploration",
        "description": "Explore strong conventional model families.",
        "goals": [
            "Test LightGBM, XGBoost, CatBoost",
            "Identify strong and complementary model families",
            "Avoid enormous blind sweeps",
        ],
        "exit_criteria": "Strong model families identified",
    },
    4: {
        "name": "Feature Engineering",
        "description": "Hypothesis-driven feature construction.",
        "goals": [
            "Ratios, differences, products, log transforms",
            "Binning, frequency encoding, target encoding",
            "Interactions, missingness indicators, aggregations",
        ],
        "exit_criteria": "Feature families constructed and evaluated",
    },
    5: {
        "name": "Feature Selection and Regularization",
        "description": "Determine which features actually help.",
        "goals": [
            "Feature importance, permutation importance, stability",
            "Redundancy analysis, noisy feature identification",
            "Feature-family ablations, regularization tuning",
        ],
        "exit_criteria": "Optimal feature set identified",
    },
    6: {
        "name": "Advanced Feature Research",
        "description": "Investigate deeper structure.",
        "goals": [
            "Higher-order interactions, conditional transformations",
            "Group-level statistics, cross-feature encodings",
            "Fold-safe target statistics, distribution-aware transforms",
        ],
        "exit_criteria": "Deep feature structure explored",
    },
    7: {
        "name": "Model Specialization",
        "description": "Deeply optimize the strongest surviving model families.",
        "goals": [
            "Targeted hyperparameter searches",
            "Regularization, feature subsets, seeds, training regimes",
            "Kill clearly inferior model families",
        ],
        "exit_criteria": "Best models optimized, inferior families eliminated",
    },
    8: {
        "name": "Diversity and Error Analysis",
        "description": "Study model complementarity.",
        "goals": [
            "Prediction correlations, error overlap, fold disagreement",
            "Model-family diversity, feature-set diversity",
            "Search explicitly for complementary models",
        ],
        "exit_criteria": "Complementary models identified, diversity quantified",
    },
    9: {
        "name": "Ensembling",
        "description": "Build robust ensembles.",
        "goals": [
            "Probability averaging, rank averaging, weighted averaging",
            "OOF-based weight optimization, stacking where justified",
            "Cross-family ensembles",
        ],
        "exit_criteria": "Robust ensemble constructed, OOF verified",
    },
    10: {
        "name": "Adversarial Validation and Robustness",
        "description": "Attempt to break the current solution.",
        "goals": [
            "Train/test distribution shift, adversarial validation",
            "Feature drift, seed/fold/outlier sensitivity",
            "Identify suspiciously powerful features, leakage",
        ],
        "exit_criteria": "Solution validated against adversarial tests",
    },
    11: {
        "name": "Final Optimization",
        "description": "Concentrate on the strongest surviving branches.",
        "goals": [
            "Final hyperparameter refinement",
            "Strong ensemble combinations, robustness checks",
            "Repeated validation, OOF verification",
        ],
        "exit_criteria": "Final model/ensemble selected",
    },
    12: {
        "name": "Final Validation and Submission",
        "description": "Freeze the research direction and verify everything.",
        "goals": [
            "Best model, best ensemble, validation strategy confirmed",
            "Feature pipeline, reproducibility, OOF/test predictions",
            "Submission format, no leakage, no preprocessing differences",
        ],
        "exit_criteria": "Final submission produced and verified",
    },
}


@dataclass
class PhaseReview:
    """End-of-phase review document."""
    phase: int
    duration: str = ""
    compute_consumed: str = ""
    baseline: str = ""
    best_result: str = ""
    improvement: str = ""
    most_important_discovery: str = ""
    successful_hypotheses: list[str] = field(default_factory=list)
    failed_hypotheses: list[str] = field(default_factory=list)
    promising_branches: list[str] = field(default_factory=list)
    killed_branches: list[str] = field(default_factory=list)
    experiments_requiring_verification: list[str] = field(default_factory=list)
    outstanding_questions: list[str] = field(default_factory=list)
    recommended_next_phase: int = 0
    recommended_compute_allocation: dict[str, float] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the review as markdown."""
        lines = [
            f"# Phase {self.phase} Review: {PHASES[self.phase]['name']}",
            "",
            f"**Duration:** {self.duration}",
            f"**Compute consumed:** {self.compute_consumed}",
            "",
            "## Results",
            f"- Baseline: {self.baseline}",
            f"- Best result: {self.best_result}",
            f"- Improvement: {self.improvement}",
            "",
            "## Most Important Discovery",
            self.most_important_discovery,
            "",
            "## Successful Hypotheses",
        ]
        for h in self.successful_hypotheses:
            lines.append(f"- {h}")
        lines.append("")
        lines.append("## Failed Hypotheses")
        for h in self.failed_hypotheses:
            lines.append(f"- {h}")
        lines.append("")
        lines.append("## Promising Branches")
        for b in self.promising_branches:
            lines.append(f"- {b}")
        lines.append("")
        lines.append("## Killed Branches")
        for b in self.killed_branches:
            lines.append(f"- {b}")
        lines.append("")
        lines.append("## Experiments Requiring Verification")
        for e in self.experiments_requiring_verification:
            lines.append(f"- {e}")
        lines.append("")
        lines.append("## Outstanding Questions")
        for q in self.outstanding_questions:
            lines.append(f"- {q}")
        lines.append("")
        lines.append(f"## Recommended Next Phase: {self.recommended_next_phase}")
        lines.append("")
        lines.append("## Recommended Compute Allocation")
        for category, fraction in self.recommended_compute_allocation.items():
            lines.append(f"- {category}: {fraction:.0%}")
        return "\n".join(lines)


class PhaseManager:
    """Manage research phase transitions and reviews."""

    @staticmethod
    def get_phase(phase_num: int) -> dict[str, Any]:
        """Get phase definition."""
        if phase_num not in PHASES:
            raise ValueError(f"Invalid phase number: {phase_num}. Must be 1-12.")
        return PHASES[phase_num]

    @staticmethod
    def get_current_phase(state: dict) -> int:
        """Get the current phase from the research state."""
        return state.get("current_phase", 1)

    @staticmethod
    def can_transition(from_phase: int, to_phase: int, evidence: dict) -> bool:
        """Determine if a phase transition is justified.

        Phases are strategic states, not immutable walls.
        A discovery in Phase 9 can justify returning to Phase 6.
        """
        # Always allow forward progression
        if to_phase > from_phase:
            return True

        # Allow backward transitions only with strong evidence
        if to_phase < from_phase:
            # Require evidence of a discovery that justifies revisiting
            has_discovery = evidence.get("major_discovery", False)
            has_complementary_signal = evidence.get("complementary_signal", False)
            return has_discovery or has_complementary_signal

        return False  # Same phase

    @staticmethod
    def phase_review(phase_num: int, experiments: list[dict],
                     compute_used: str = "") -> PhaseReview:
        """Generate an end-of-phase review."""
        successful = [e for e in experiments if e.get("decision") == "promoted"]
        failed = [e for e in experiments if e.get("decision") in ("rejected", "failed")]

        return PhaseReview(
            phase=phase_num,
            compute_consumed=compute_used,
            successful_hypotheses=[e["hypothesis"] for e in successful],
            failed_hypotheses=[e["hypothesis"] for e in failed],
            recommended_next_phase=min(phase_num + 1, 12),
        )

    @staticmethod
    def recommend_next_phase(state: dict) -> int:
        """Recommend the next phase based on current state."""
        current = state.get("current_phase", 1)
        # Simple forward progression with check for max phase
        return min(current + 1, 12)
