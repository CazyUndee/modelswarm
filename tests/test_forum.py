"""
Tests for the forum system.
"""

import pytest


class TestForum:
    """Test forum posting and commenting."""

    def test_create_post(self, mocker):
        """Test creating a forum post."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "post_id": "POST-a1b2c3d4",
            "category": "discovery",
            "title": "Test discovery",
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test")
        result = client.post(
            category="discovery",
            title="Test discovery",
            content="Evidence: ...",
        )
        assert result["post_id"] == "POST-a1b2c3d4"
        assert result["category"] == "discovery"

    def test_add_comment(self, mocker):
        """Test adding a comment to a post."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "comment_id": "COMMENT-abc123",
            "post_id": "POST-a1b2c3d4",
        }
        mocker.patch("modelswarm.client.requests.post", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test")
        result = client.comment("POST-a1b2c3d4", content="Great finding!")
        assert result["comment_id"] == "COMMENT-abc123"

    def test_search_forum(self, mocker):
        """Test searching the forum."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"post_id": "POST-1", "title": "Composition features"},
                {"post_id": "POST-2", "title": "Feature engineering"},
            ]
        }
        mocker.patch("modelswarm.client.requests.get", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test")
        results = client.search_forum("composition")
        assert len(results) == 2

    def test_get_feed(self, mocker):
        """Test getting the forum feed."""
        from modelswarm.client import Client

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "posts": [
                {"post_id": "POST-1", "category": "discovery"},
                {"post_id": "POST-2", "category": "discussion"},
            ]
        }
        mocker.patch("modelswarm.client.requests.get", return_value=mock_response)

        client = Client(api_url="https://test.workers.dev", api_key="ms_test")
        feed = client.get_feed(limit=10)
        assert len(feed) == 2
