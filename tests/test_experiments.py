"""
Tests for experiment lifecycle, claiming, and concurrency.
"""

import pytest

from modelswarm.exceptions import ClaimError


class TestExperimentLifecycle:
    """Test experiment creation and lifecycle."""

    def test_create_experiment(self, mocker):
        """Test creating an experiment via the client."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "experiment_id": "EXP-014",
            "status": "queued",
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test", agent_id="agent-test")
        result = client.create_experiment(
            hypothesis="Test hypothesis",
            features=["f1"],
            model="lightgbm",
            phase=4,
        )
        assert result["experiment_id"] == "EXP-014"
        assert result["status"] == "queued"

    def test_claim_experiment_success(self, mocker):
        """Test successfully claiming an experiment."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiment_id": "EXP-014",
            "status": "claimed",
            "claimed_by": "agent-test",
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test", agent_id="agent-test")
        result = client.claim_experiment("EXP-014")
        assert result["status"] == "claimed"

    def test_claim_experiment_conflict(self, mocker):
        """Test that claiming an already-claimed experiment raises ClaimError."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 409
        mock_response.json.return_value = {"error": "Already claimed"}
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test", agent_id="agent-test")
        with pytest.raises(ClaimError):
            client.claim_experiment("EXP-014")

    def test_complete_experiment(self, mocker):
        """Test completing an experiment with results."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiment_id": "EXP-014",
            "status": "completed",
            "oof_metric": 0.965,
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test", agent_id="agent-test")
        result = client.complete_experiment(
            "EXP-014",
            oof_metric=0.965,
            fold_metrics=[0.964, 0.966, 0.965, 0.965, 0.965],
            decision="promoted",
        )
        assert result["status"] == "completed"
        assert result["oof_metric"] == 0.965

    def test_fail_experiment(self, mocker):
        """Test marking an experiment as failed."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiment_id": "EXP-014",
            "status": "failed",
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test", agent_id="agent-test")
        result = client.fail_experiment("EXP-014", reason="OOM")
        assert result["status"] == "failed"

    def test_two_agents_cannot_claim_same_experiment(self, mocker):
        """
        CRITICAL TEST: Two agents attempting to claim the same experiment.
        Only one should succeed.
        """
        from modelswarm.client import Client

        # First claim succeeds
        first_response = mocker.MagicMock()
        first_response.status_code = 200
        first_response.json.return_value = {"status": "claimed"}

        # Second claim fails
        second_response = mocker.MagicMock()
        second_response.status_code = 409
        second_response.json.return_value = {"error": "Already claimed"}

        post_mock = mocker.patch(
            "modelswarm.client.requests.post",
            side_effect=[first_response, second_response],
        )

        agent1 = Client(api_url="https://test.workers.dev", api_key="ms_key1", agent_id="agent-1")
        agent2 = Client(api_url="https://test.workers.dev", api_key="ms_key2", agent_id="agent-2")

        # Agent 1 claims successfully
        result1 = agent1.claim_experiment("EXP-014")
        assert result1["status"] == "claimed"

        # Agent 2 fails to claim
        with pytest.raises(ClaimError):
            agent2.claim_experiment("EXP-014")


class TestExperimentValidation:
    """Test experiment data validation."""

    def test_experiment_id_format(self):
        """Test that experiment IDs follow the EXP-NNN format."""
        import re
        pattern = r"^EXP-[0-9]{3,4}[a-z]?$"
        assert re.match(pattern, "EXP-000")
        assert re.match(pattern, "EXP-001")
        assert re.match(pattern, "EXP-014")
        assert re.match(pattern, "EXP-003b")
        assert re.match(pattern, "EXP-9999")
        assert not re.match(pattern, "EXP-01")  # Too few digits
        assert not re.match(pattern, "exp-001")  # Lowercase

    def test_agent_id_format(self):
        """Test that agent IDs follow the agent-XXXX format."""
        import re
        pattern = r"^agent-[a-z0-9]{4}(-[a-z0-9]+)?$"
        assert re.match(pattern, "agent-7f3c")
        assert re.match(pattern, "agent-7f3c-01")
        assert re.match(pattern, "agent-abcd")
        assert not re.match(pattern, "agent-7f")  # Too short
        assert not re.match(pattern, "Agent-7f3c")  # Uppercase
