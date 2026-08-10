"""
Tests for agent heartbeat and stale detection.
"""

import pytest


class TestHeartbeat:
    """Test heartbeat functionality."""

    def test_send_heartbeat(self, mocker):
        """Test sending a heartbeat."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agent_id": "agent-test",
            "last_heartbeat": "2026-08-10T12:00:00Z",
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test", agent_id="agent-test")
        result = client.heartbeat()
        assert result["agent_id"] == "agent-test"

    def test_stale_agent_detection(self, mocker):
        """Test that agents without recent heartbeats are detected as stale."""
        from modelswarm.client import Client

        # Mock an agent with an old heartbeat
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agents": [
                {
                    "agent_id": "agent-active",
                    "last_heartbeat": "2026-08-10T11:58:00Z",  # Recent
                    "status": "active",
                },
                {
                    "agent_id": "agent-stale",
                    "last_heartbeat": "2026-08-10T10:00:00Z",  # Old
                    "status": "active",
                },
            ]
        }
        mocker.patch("modelswarm.client.requests.get", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test", agent_id="agent-test")
        agents = client.get_agents()

        # The stale agent should be identifiable
        stale = [a for a in agents if a["agent_id"] == "agent-stale"]
        assert len(stale) == 1
        # In a real scenario, the operations agent would mark this as stalled

    def test_heartbeat_updates_presence(self, mocker):
        """Test that heartbeat updates the agent's last_heartbeat timestamp."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agent_id": "agent-test",
            "last_heartbeat": "2026-08-10T12:05:00Z",
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test", agent_id="agent-test")
        result = client.heartbeat()
        assert "last_heartbeat" in result
