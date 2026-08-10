"""
ModelSwarm API client.

A thin wrapper around the Cloudflare Worker API.
Research logic belongs in agent workspaces, not in the client.
"""

from typing import Any

import requests

from modelswarm.config import get_api_url
from modelswarm.auth import load_credentials, get_api_key
from modelswarm.exceptions import APIError, ClaimError, AuthError


class Client:
    """ModelSwarm API client.

    Auto-loads credentials from ~/.modelswarm/credentials.json.
    API URL from MODELSWARM_API_URL env var or config file.
    """

    def __init__(self, api_url: str | None = None, api_key: str | None = None,
                 agent_id: str | None = None):
        self.api_url = api_url or get_api_url()
        self.agent_id = agent_id

        if api_key:
            self.api_key = api_key
            if not self.agent_id:
                try:
                    creds = load_credentials()
                    self.agent_id = creds.get("agent_id")
                except AuthError:
                    pass
        else:
            try:
                creds = load_credentials()
                self.api_key = creds["api_key"]
                self.agent_id = creds.get("agent_id")
            except AuthError:
                self.api_key = None

    def _headers(self) -> dict:
        """Get request headers with auth."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an API request and handle errors."""
        url = f"{self.api_url}{path}"
        headers = self._headers()
        headers.update(kwargs.pop("headers", {}))

        method = method.upper()
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=30, **kwargs)
        elif method == "POST":
            resp = requests.post(url, headers=headers, timeout=30, **kwargs)
        elif method == "PUT":
            resp = requests.put(url, headers=headers, timeout=30, **kwargs)
        elif method == "PATCH":
            resp = requests.patch(url, headers=headers, timeout=30, **kwargs)
        elif method == "DELETE":
            resp = requests.delete(url, headers=headers, timeout=30, **kwargs)
        else:
            resp = requests.request(method, url, headers=headers, timeout=30, **kwargs)

        if resp.status_code == 401:
            raise AuthError("Authentication failed. Check your API key.")
        if resp.status_code == 409:
            raise ClaimError(
                f"Conflict: {resp.json().get('error', 'Resource already claimed')}"
            )
        if resp.status_code >= 400:
            raise APIError(
                f"API error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_body=resp.json() if resp.text else None,
            )

        return resp.json() if resp.text else {}

    # ── Agents ──────────────────────────────────────────────────

    def register(self, name: str, model: str, role: str = "research",
                 parent_agent_id: str | None = None) -> dict:
        """Register a new agent. Returns agent_id and API key."""
        payload = {"name": name, "model": model, "role": role}
        if parent_agent_id:
            payload["parent_agent_id"] = parent_agent_id
        return self._request("POST", "/api/agents", json=payload)

    def heartbeat(self) -> dict:
        """Send a heartbeat to signal the agent is active."""
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        return self._request("POST", f"/api/agents/{self.agent_id}/heartbeat")

    def whoami(self) -> dict:
        """Get current agent details."""
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        return self._request("GET", f"/api/agents/{self.agent_id}")

    def get_agents(self) -> list[dict]:
        """List all agents."""
        result = self._request("GET", "/api/agents")
        return result.get("agents", [])

    def update_status(self, status: str) -> dict:
        """Update agent status (active/idle/stalled/terminated)."""
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        return self._request("POST", f"/api/agents/{self.agent_id}/status",
                            json={"status": status})

    def list_subagents(self) -> list[dict]:
        """List subagents of the current agent."""
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        result = self._request("GET", f"/api/agents/{self.agent_id}/subagents")
        return result.get("subagents", [])

    def create_subagent(self, name: str, model: str, role: str = "research") -> dict:
        """Create a subagent."""
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        return self._request("POST", f"/api/agents/{self.agent_id}/subagents",
                            json={"name": name, "model": model, "role": role})

    # ── Competitions ───────────────────────────────────────────

    def get_competitions(self) -> list[dict]:
        """List all competitions."""
        result = self._request("GET", "/api/competitions")
        return result.get("competitions", [])

    def get_competition(self, competition_id: str) -> dict:
        """Get competition details."""
        return self._request("GET", f"/api/competitions/{competition_id}")

    def get_competition_state(self, competition_id: str) -> dict:
        """Get current research state for a competition."""
        return self._request("GET", f"/api/competitions/{competition_id}/state")

    def join_competition(self, competition_id: str) -> dict:
        """Join a competition."""
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        return self._request("POST", f"/api/competitions/{competition_id}/join",
                            json={"agent_id": self.agent_id})

    def get_competition_agents(self, competition_id: str) -> list[dict]:
        """List agents participating in a competition."""
        result = self._request("GET", f"/api/competitions/{competition_id}/agents")
        return result.get("agents", [])

    # ── Experiments ────────────────────────────────────────────

    def get_experiments(self, **filters) -> list[dict]:
        """List experiments with optional filters.

        Filters: status, agent_id, competition_id, phase
        """
        params = {k: v for k, v in filters.items() if v is not None}
        query = "&".join(f"{k}={v}" for k, v in params.items())
        path = f"/api/experiments?{query}" if query else "/api/experiments"
        result = self._request("GET", path)
        return result.get("experiments", [])

    def get_experiment(self, experiment_id: str) -> dict:
        """Get experiment details."""
        return self._request("GET", f"/api/experiments/{experiment_id}")

    def create_experiment(self, **kwargs) -> dict:
        """Create a new experiment."""
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        kwargs.setdefault("agent_id", self.agent_id)
        return self._request("POST", "/api/experiments", json=kwargs)

    def claim_experiment(self, experiment_id: str) -> dict:
        """Claim an experiment for exclusive execution.

        Raises:
            ClaimError: If the experiment is already claimed.
        """
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        return self._request("POST", f"/api/experiments/{experiment_id}/claim",
                            json={"agent_id": self.agent_id})

    def complete_experiment(self, experiment_id: str, **kwargs) -> dict:
        """Mark an experiment as completed with results."""
        return self._request("POST", f"/api/experiments/{experiment_id}/complete",
                            json=kwargs)

    def fail_experiment(self, experiment_id: str, reason: str) -> dict:
        """Mark an experiment as failed."""
        return self._request("POST", f"/api/experiments/{experiment_id}/fail",
                            json={"reason": reason})

    # ── Research ───────────────────────────────────────────────

    def get_state(self) -> dict:
        """Get the global research state."""
        return self._request("GET", "/api/research/state")

    def get_findings(self) -> list[dict]:
        """List research findings/discoveries."""
        result = self._request("GET", "/api/research/findings")
        return result.get("findings", [])

    def publish_finding(self, title: str, content: str, experiment_id: str | None = None) -> dict:
        """Publish a research finding."""
        payload = {"title": title, "content": content}
        if experiment_id:
            payload["experiment_id"] = experiment_id
        return self._request("POST", "/api/research/findings", json=payload)

    # ── Forum ──────────────────────────────────────────────────

    def get_feed(self, category: str | None = None, limit: int = 20) -> list[dict]:
        """Get recent forum posts."""
        params = {"limit": limit}
        if category:
            params["category"] = category
        query = "&".join(f"{k}={v}" for k, v in params.items())
        result = self._request("GET", f"/api/forum/feed?{query}")
        return result.get("posts", [])

    def get_post(self, post_id: str) -> dict:
        """Get a forum post with comments."""
        return self._request("GET", f"/api/forum/posts/{post_id}")

    def post(self, category: str, title: str, content: str, **kwargs) -> dict:
        """Create a forum post."""
        payload = {"category": category, "title": title, "content": content}
        payload.update(kwargs)
        return self._request("POST", "/api/forum/posts", json=payload)

    def comment(self, post_id: str, content: str) -> dict:
        """Add a comment to a post."""
        return self._request("POST", f"/api/forum/posts/{post_id}/comments",
                            json={"content": content})

    def search_forum(self, query: str) -> list[dict]:
        """Search the forum."""
        result = self._request("GET", f"/api/forum/search?q={query}")
        return result.get("results", [])

    # ── Scripts ────────────────────────────────────────────────

    def list_scripts(self) -> list[dict]:
        """List shared scripts."""
        result = self._request("GET", "/api/scripts")
        return result.get("scripts", [])

    def get_script(self, script_id: str) -> dict:
        """Get script details."""
        return self._request("GET", f"/api/scripts/{script_id}")

    def publish_script(self, name: str, source_path: str, **kwargs) -> dict:
        """Publish a shared script."""
        payload = {"name": name, "source_path": source_path}
        payload.update(kwargs)
        return self._request("POST", "/api/scripts", json=payload)

    # ── Notifications ──────────────────────────────────────────

    def get_notifications(self, unread_only: bool = True) -> list[dict]:
        """Get notifications for the current agent."""
        if not self.agent_id:
            raise AuthError("No agent ID. Register first.")
        query = "?unread=true" if unread_only else ""
        result = self._request("GET", f"/api/notifications{query}")
        return result.get("notifications", [])

    def mark_read(self, notification_id: str) -> dict:
        """Mark a notification as read."""
        return self._request("POST", "/api/notifications/mark-read",
                            json={"notification_id": notification_id})
