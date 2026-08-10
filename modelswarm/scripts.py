"""
Shared script operations — listing, publishing, retrieving.

This module provides convenience functions that wrap the Client methods.
For full control, use Client directly.
"""

from modelswarm.client import Client


def list_scripts(api_url: str | None = None) -> list[dict]:
    """List shared scripts."""
    client = Client(api_url=api_url)
    return client.list_scripts()


def get_script(script_id: str, api_url: str | None = None) -> dict:
    """Get script details."""
    client = Client(api_url=api_url)
    return client.get_script(script_id)


def publish(name: str, source_path: str, description: str = "",
            version: str = "1.0.0", dependencies: list[str] | None = None,
            usage: str = "", api_url: str | None = None) -> dict:
    """Publish a shared script."""
    client = Client(api_url=api_url)
    kwargs = {"description": description, "version": version, "usage": usage}
    if dependencies:
        kwargs["dependencies"] = dependencies
    return client.publish_script(name, source_path, **kwargs)
