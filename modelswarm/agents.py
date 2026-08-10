"""
Agent operations — registration, heartbeat, subagents.

This module provides convenience functions that wrap the Client methods.
For full control, use Client directly.
"""

from modelswarm.client import Client
from modelswarm.identity import load_identity


def register(name: str, model: str, role: str = "research",
             parent_agent_id: str | None = None,
             api_url: str | None = None) -> dict:
    """Register a new agent. Returns the registration result with agent_id and api_key."""
    client = Client(api_url=api_url)
    return client.register(name, model, role, parent_agent_id)


def heartbeat(api_url: str | None = None) -> dict:
    """Send a heartbeat to signal the agent is active."""
    client = Client(api_url=api_url)
    return client.heartbeat()


def list_agents(api_url: str | None = None) -> list[dict]:
    """List all registered agents."""
    client = Client(api_url=api_url)
    return client.get_agents()


def list_subagents(api_url: str | None = None) -> list[dict]:
    """List subagents of the current agent."""
    client = Client(api_url=api_url)
    return client.list_subagents()


def create_subagent(name: str, model: str, role: str = "research",
                     api_url: str | None = None) -> dict:
    """Create a subagent under the current agent."""
    client = Client(api_url=api_url)
    return client.create_subagent(name, model, role)
