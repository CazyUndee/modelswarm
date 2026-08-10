"""
Experiment operations — creation, claiming, completion.

This module provides convenience functions that wrap the Client methods.
For full control, use Client directly.
"""

from modelswarm.client import Client


def create(hypothesis: str, features: list[str] | None = None,
           model: str | None = None, phase: int | None = None,
           configuration: dict | None = None,
           api_url: str | None = None, **kwargs) -> dict:
    """Create a new experiment."""
    client = Client(api_url=api_url)
    kwargs["hypothesis"] = hypothesis
    if features:
        kwargs["features"] = features
    if model:
        kwargs["model"] = model
    if phase:
        kwargs["phase"] = phase
    if configuration:
        kwargs["configuration"] = configuration
    return client.create_experiment(**kwargs)


def claim(experiment_id: str, api_url: str | None = None) -> dict:
    """Claim an experiment for exclusive execution.

    Raises:
        ClaimError: If the experiment is already claimed.
    """
    client = Client(api_url=api_url)
    return client.claim_experiment(experiment_id)


def complete(experiment_id: str, oof_metric: float | None = None,
             fold_metrics: list[float] | None = None,
             decision: str | None = None, reasoning: str = "",
             api_url: str | None = None, **kwargs) -> dict:
    """Mark an experiment as completed with results."""
    client = Client(api_url=api_url)
    if oof_metric is not None:
        kwargs["oof_metric"] = oof_metric
    if fold_metrics:
        kwargs["fold_metrics"] = fold_metrics
    if decision:
        kwargs["decision"] = decision
    if reasoning:
        kwargs["reasoning"] = reasoning
    return client.complete_experiment(experiment_id, **kwargs)


def fail(experiment_id: str, reason: str, api_url: str | None = None) -> dict:
    """Mark an experiment as failed."""
    client = Client(api_url=api_url)
    return client.fail_experiment(experiment_id, reason)


def list_experiments(status: str | None = None, agent_id: str | None = None,
                     competition_id: str | None = None, phase: int | None = None,
                     api_url: str | None = None) -> list[dict]:
    """List experiments with optional filters."""
    client = Client(api_url=api_url)
    return client.get_experiments(status=status, agent_id=agent_id,
                                   competition_id=competition_id, phase=phase)


def get_experiment(experiment_id: str, api_url: str | None = None) -> dict:
    """Get experiment details."""
    client = Client(api_url=api_url)
    return client.get_experiment(experiment_id)
