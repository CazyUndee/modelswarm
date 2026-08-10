"""
End-to-end tests for the ModelSwarm bootstrap flow.

Tests the complete flow described in the specification:
1. Agent receives /agents.md
2. pip install modelswarm
3. modelswarm login
4. modelswarm competitions
5. modelswarm join s6e8
6. Environment initialized
7. Agent registers
8. Agent heartbeat
9. Agent reads research state
10. Agent reads recent forum activity
11. Agent claims experiment
12. Experiment executes
13. OOF/artifacts recorded
14. Experiment completed
15. Discovery published
16. Research state updated
17. Operations agent creates follow-up work
"""

import pytest


class TestBootstrapFlow:
    """Test the complete agent bootstrap flow."""

    def test_full_research_workflow(self, mocker):
        """
        Test the complete research workflow from registration to discovery.

        This simulates:
        1. Agent registers
        2. Agent joins competition
        3. Agent reads state
        4. Agent reads forum
        5. Agent creates experiment
        6. Agent claims experiment
        7. Agent completes experiment
        8. Agent publishes discovery
        """
        from modelswarm.client import Client

        # Mock all API responses
        responses = {
            "register": {"agent_id": "agent-7f3c", "api_key": "ms_test_key_123"},
            "join": {"status": "joined"},
            "state": {"current_phase": 4, "champion": "EXP-006", "best_score": 0.96421},
            "feed": {"posts": [{"post_id": "POST-1", "title": "Welcome"}]},
            "create_exp": {"experiment_id": "EXP-014", "status": "queued"},
            "claim": {"experiment_id": "EXP-014", "status": "claimed"},
            "complete": {"experiment_id": "EXP-014", "status": "completed"},
            "post": {"post_id": "POST-new", "category": "discovery"},
        }

        def mock_post(url, **kwargs):
            m = mocker.MagicMock()
            if "register" in url:
                m.status_code = 201
                m.json.return_value = responses["register"]
            elif "join" in url:
                m.status_code = 200
                m.json.return_value = responses["join"]
            elif "experiments" in url and "claim" not in url and "complete" not in url and "fail" not in url:
                m.status_code = 201
                m.json.return_value = responses["create_exp"]
            elif "claim" in url:
                m.status_code = 200
                m.json.return_value = responses["claim"]
            elif "complete" in url:
                m.status_code = 200
                m.json.return_value = responses["complete"]
            elif "posts" in url:
                m.status_code = 201
                m.json.return_value = responses["post"]
            else:
                m.status_code = 200
                m.json.return_value = {}
            return m

        def mock_get(url, **kwargs):
            m = mocker.MagicMock()
            if "research/state" in url:
                m.status_code = 200
                m.json.return_value = responses["state"]
            elif "forum/feed" in url:
                m.status_code = 200
                m.json.return_value = responses["feed"]
            else:
                m.status_code = 200
                m.json.return_value = {}
            return m

        mocker.patch("modelswarm.client.requests.post", side_effect=mock_post)
        mocker.patch("modelswarm.client.requests.get", side_effect=mock_get)

        # Step 1: Register
        client = Client()
        reg = client.register("Happy", "claude-opus-5", "research")
        assert reg["agent_id"] == "agent-7f3c"
        client.agent_id = reg["agent_id"]
        client.api_key = reg["api_key"]

        # Step 2: Join competition
        join_result = client.join_competition("s6e8")
        assert join_result["status"] == "joined"

        # Step 3: Read research state
        state = client.get_state()
        assert state["current_phase"] == 4
        assert state["champion"] == "EXP-006"

        # Step 4: Read forum
        feed = client.get_feed(limit=10)
        assert len(feed) > 0

        # Step 5: Create experiment
        exp = client.create_experiment(
            hypothesis="Test hypothesis",
            features=["f1"],
            model="lightgbm",
            phase=4,
        )
        exp_id = exp["experiment_id"]
        assert exp_id == "EXP-014"

        # Step 6: Claim experiment
        claim_result = client.claim_experiment(exp_id)
        assert claim_result["status"] == "claimed"

        # Step 7: Complete experiment
        complete_result = client.complete_experiment(
            exp_id,
            oof_metric=0.965,
            decision="promising",
        )
        assert complete_result["status"] == "completed"

        # Step 8: Publish discovery
        post_result = client.post(
            category="discovery",
            title="Test discovery",
            content="Evidence: ...",
            experiment_id=exp_id,
        )
        assert post_result["post_id"] == "POST-new"


class TestConcurrentClaim:
    """Test that concurrent experiment claims are handled correctly."""

    def test_only_one_agent_claims(self, mocker):
        """Two agents try to claim the same experiment simultaneously."""
        from modelswarm.client import Client

        call_count = 0

        def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            m = mocker.MagicMock()
            if "claim" in url:
                if call_count == 1:
                    m.status_code = 200
                    m.json.return_value = {"status": "claimed"}
                else:
                    m.status_code = 409
                    m.json.return_value = {"error": "Already claimed"}
            else:
                m.status_code = 200
                m.json.return_value = {}
            return m

        mocker.patch("modelswarm.client.requests.post", side_effect=mock_post)

        agent1 = Client(api_url="https://test.workers.dev", api_key="ms_1", agent_id="agent-1")
        agent2 = Client(api_url="https://test.workers.dev", api_key="ms_2", agent_id="agent-2")

        # First claim succeeds
        result1 = agent1.claim_experiment("EXP-042")
        assert result1["status"] == "claimed"

        # Second claim fails
        from modelswarm.exceptions import ClaimError
        with pytest.raises(ClaimError):
            agent2.claim_experiment("EXP-042")


class TestStaleAgentReclaim:
    """Test that stale agent work becomes reclaimable."""

    def test_expired_claim_reclaimable(self, mocker):
        """An experiment with an expired claim can be reclaimed."""
        from modelswarm.client import Client

        # First claim (expired)
        first = mocker.MagicMock()
        first.status_code = 200
        first.json.return_value = {"status": "claimed"}

        # Reclaim succeeds because original claim expired
        second = mocker.MagicMock()
        second.status_code = 200
        second.json.return_value = {"status": "claimed", "reclaimed": True}

        mocker.patch("modelswarm.client.requests.post", side_effect=[first, second])

        agent1 = Client(api_url="https://test.workers.dev", api_key="ms_1", agent_id="agent-1")
        agent2 = Client(api_url="https://test.workers.dev", api_key="ms_2", agent_id="agent-2")

        # Agent 1 claims
        agent1.claim_experiment("EXP-050")

        # Agent 2 reclaims (after expiration)
        result = agent2.claim_experiment("EXP-050")
        assert result["status"] == "claimed"


class TestSubagentProvenance:
    """Test that subagent experiments record correct provenance."""

    def test_subagent_experiment_records_both_ids(self, mocker):
        """A subagent experiment records both executing_agent_id and parent_agent_id."""
        from modelswarm.client import Client

        # Mock create experiment
        create_resp = mocker.MagicMock()
        create_resp.status_code = 201
        create_resp.json.return_value = {
            "experiment_id": "EXP-060",
            "agent_id": "agent-7f3c",
            "executing_agent_id": "agent-7f3c-01",
            "parent_agent_id": "agent-7f3c",
        }

        mocker.patch("modelswarm.client.requests.post", return_value=create_resp)

        # Subagent creates experiment
        subagent = Client(api_url="https://test.workers.dev", api_key="ms_sub", agent_id="agent-7f3c-01")
        exp = subagent.create_experiment(
            hypothesis="Subagent experiment",
            agent_id="agent-7f3c-01",
            parent_agent_id="agent-7f3c",
        )

        # Verify provenance
        assert exp["executing_agent_id"] == "agent-7f3c-01"
        assert exp["parent_agent_id"] == "agent-7f3c"
        assert exp["executing_agent_id"] != exp["parent_agent_id"]


class TestCrossAgentEnsemble:
    """Test that OOF predictions from multiple agents can be combined."""

    def test_cross_agent_ensemble_alignment(self):
        """OOF predictions from different agents can be aligned and combined."""
        from shared.utilities.oof import cross_agent_ensemble, oof_correlation
        import numpy as np

        np.random.seed(42)
        n = 500
        target = np.random.randint(0, 2, size=n).astype(float)

        # Three agents with different models
        agent_oofs = {
            "agent-a": target + np.random.rand(n) * 0.15,
            "agent-b": target + np.random.rand(n) * 0.18,
            "agent-c": target + np.random.rand(n) * 0.12,
        }

        # Combine into cross-agent ensemble
        ensemble = cross_agent_ensemble(agent_oofs, target, method="rank")

        # Verify output
        assert len(ensemble) == n
        assert np.all(ensemble >= 0) and np.all(ensemble <= 1)

        # Verify ensemble is correlated with target (better than random)
        corr = oof_correlation(ensemble, target)
        assert corr > 0.3  # Should have meaningful correlation
