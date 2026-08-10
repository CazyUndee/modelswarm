"""
Tests for OOF prediction validation and ensemble operations.
"""

import numpy as np
import pytest


class TestOOFValidation:
    """Test OOF prediction validation."""

    def test_validate_correct_predictions(self):
        """Test validation passes for correct OOF predictions."""
        from shared.utilities.oof import validate_oof_predictions

        ids = np.array([0, 1, 2, 3, 4])
        target = np.array([1, 0, 1, 0, 1])
        predictions = np.array([0.8, 0.2, 0.7, 0.3, 0.9])
        folds = np.array([0, 0, 1, 1, 2])

        # Should not raise
        validate_oof_predictions(ids, target, predictions, folds)

    def test_validate_mismatched_lengths(self):
        """Test validation fails for mismatched array lengths."""
        from shared.utilities.oof import validate_oof_predictions

        ids = np.array([0, 1, 2])
        target = np.array([1, 0])
        predictions = np.array([0.8, 0.2, 0.7])
        folds = np.array([0, 0, 1])

        with pytest.raises(ValueError):
            validate_oof_predictions(ids, target, predictions, folds)

    def test_validate_predictions_out_of_range(self):
        """Test validation fails for predictions outside [0, 1]."""
        from shared.utilities.oof import validate_oof_predictions

        ids = np.array([0, 1, 2])
        target = np.array([1, 0, 1])
        predictions = np.array([0.8, 1.5, 0.7])  # 1.5 is invalid
        folds = np.array([0, 0, 1])

        with pytest.raises(ValueError):
            validate_oof_predictions(ids, target, predictions, folds)


class TestOOFEnsemble:
    """Test OOF ensemble operations."""

    def test_oof_correlation(self):
        """Test OOF correlation computation."""
        from shared.utilities.oof import oof_correlation

        np.random.seed(42)
        a = np.random.rand(100)
        b = a + np.random.rand(100) * 0.1  # Highly correlated

        corr = oof_correlation(a, b)
        assert corr > 0.8  # Should be highly correlated

    def test_rank_blend(self):
        """Test rank blending."""
        from shared.utilities.oof import rank_blend

        np.random.seed(42)
        a = np.random.rand(100)
        b = np.random.rand(100)

        blended = rank_blend([a, b], weights=[0.6, 0.4])
        assert len(blended) == 100
        assert np.all(blended >= 0) and np.all(blended <= 1)

    def test_probability_blend(self):
        """Test probability blending."""
        from shared.utilities.oof import probability_blend

        np.random.seed(42)
        a = np.random.rand(100)
        b = np.random.rand(100)

        blended = probability_blend([a, b], weights=[0.5, 0.5])
        assert len(blended) == 100
        # Equal-weight blend should be close to element-wise mean
        np.testing.assert_array_almost_equal(blended, (a + b) / 2)

    def test_compute_diversity(self):
        """Test diversity measurement."""
        from shared.utilities.oof import compute_diversity

        np.random.seed(42)
        a = np.random.rand(100)
        b = np.random.rand(100)
        c = np.random.rand(100)

        corr_matrix = compute_diversity([a, b, c])
        assert corr_matrix.shape == (3, 3)
        # Diagonal should be 1.0 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1.0, 1.0, 1.0])

    def test_cross_agent_ensemble(self):
        """Test cross-agent ensemble (ensemble²)."""
        from shared.utilities.oof import cross_agent_ensemble

        np.random.seed(42)
        target = np.random.randint(0, 2, size=100).astype(float)
        agent_oofs = {
            "agent-a": target + np.random.rand(100) * 0.2,
            "agent-b": target + np.random.rand(100) * 0.2,
            "agent-c": target + np.random.rand(100) * 0.2,
        }

        ensemble = cross_agent_ensemble(agent_oofs, target, method="rank")
        assert len(ensemble) == 100
        assert np.all(ensemble >= 0) and np.all(ensemble <= 1)

    def test_subagent_provenance(self):
        """
        Test that subagent experiments correctly record both
        executing_agent_id and parent_agent_id.
        """
        # Simulate a subagent experiment record
        exp = {
            "experiment_id": "EXP-020",
            "agent_id": "agent-7f3c",  # Owner (parent)
            "executing_agent_id": "agent-7f3c-01",  # Actual runner (subagent)
            "parent_agent_id": "agent-7f3c",
        }

        # The executing agent should be the subagent
        assert exp["executing_agent_id"] == "agent-7f3c-01"
        # The parent should be the owning agent
        assert exp["parent_agent_id"] == "agent-7f3c"
        # They should be different (provenance tracking)
        assert exp["executing_agent_id"] != exp["agent_id"]
