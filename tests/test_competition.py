"""
Tests for competition management and isolation.
"""

import pytest


class TestCompetition:
    """Test competition operations."""

    def test_list_competitions(self, mocker):
        """Test listing competitions."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "competitions": [
                {"competition_id": "s6e8", "name": "S6E8", "status": "active"},
            ]
        }
        mocker.patch("modelswarm.client.requests.get", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test")
        comps = client.get_competitions()
        assert len(comps) == 1

    def test_join_competition(self, mocker):
        """Test joining a competition."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agent_id": "agent-test",
            "competition_id": "s6e8",
            "status": "joined",
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_key")
        result = client.join_competition("s6e8")
        assert result["status"] == "joined"

    def test_competition_isolation(self, mocker):
        """Test that experiments from different competitions are isolated."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiments": [
                {"experiment_id": "EXP-001", "competition_id": "s6e8"},
                {"experiment_id": "EXP-002", "competition_id": "s6e8"},
            ]
        }
        mocker.patch("modelswarm.client.requests.get", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test")
        exps = client.get_experiments(competition_id="s6e8")

        # All returned experiments should be for s6e8
        for exp in exps:
            assert exp["competition_id"] == "s6e8"

    def test_competition_config_loading(self, temp_dir, monkeypatch):
        """Test loading competition configuration from YAML."""
        monkeypatch.chdir(temp_dir)

        import yaml
        from pathlib import Path

        # Create a test competition.yaml
        config = {
            "competition": {
                "id": "test-comp",
                "name": "Test Competition",
                "target": "target_col",
                "metric": "roc_auc",
            }
        }
        Path("competition.yaml").write_text(yaml.dump(config))

        # Load it
        with open("competition.yaml") as f:
            loaded = yaml.safe_load(f)

        assert loaded["competition"]["id"] == "test-comp"
        assert loaded["competition"]["metric"] == "roc_auc"
