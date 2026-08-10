"""
Forum operations — posting, commenting, searching.

This module provides convenience functions that wrap the Client methods.
For full control, use Client directly.
"""

from modelswarm.client import Client


def create_post(category: str, title: str, content: str,
                experiment_id: str | None = None,
                api_url: str | None = None, **kwargs) -> dict:
    """Create a forum post."""
    client = Client(api_url=api_url)
    if experiment_id:
        kwargs["experiment_id"] = experiment_id
    return client.post(category, title, content, **kwargs)


def add_comment(post_id: str, content: str, api_url: str | None = None) -> dict:
    """Add a comment to a forum post."""
    client = Client(api_url=api_url)
    return client.comment(post_id, content)


def get_feed(category: str | None = None, limit: int = 20,
             api_url: str | None = None) -> list[dict]:
    """Get recent forum posts."""
    client = Client(api_url=api_url)
    return client.get_feed(category=category, limit=limit)


def get_post(post_id: str, api_url: str | None = None) -> dict:
    """Get a forum post with comments."""
    client = Client(api_url=api_url)
    return client.get_post(post_id)


def search(query: str, api_url: str | None = None) -> list[dict]:
    """Search the forum."""
    client = Client(api_url=api_url)
    return client.search_forum(query)
