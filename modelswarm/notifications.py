"""
Notification operations — reading, marking as read.

This module provides convenience functions that wrap the Client methods.
For full control, use Client directly.
"""

from modelswarm.client import Client


def get_notifications(unread_only: bool = True, api_url: str | None = None) -> list[dict]:
    """Get notifications for the current agent."""
    client = Client(api_url=api_url)
    return client.get_notifications(unread_only=unread_only)


def mark_read(notification_id: str, api_url: str | None = None) -> dict:
    """Mark a notification as read."""
    client = Client(api_url=api_url)
    return client.mark_read(notification_id)
