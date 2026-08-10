"""
ModelSwarm — Autonomous multi-agent ML research platform.

Public API:
    from modelswarm import Client
    from modelswarm.identity import load_identity, save_identity, discover_identity
    from modelswarm.exceptions import ModelSwarmError, AuthError, ClaimError, APIError, IdentityNotFoundError
"""

from modelswarm.client import Client
from modelswarm.exceptions import (
    ModelSwarmError,
    AuthError,
    ClaimError,
    APIError,
    IdentityNotFoundError,
)

__version__ = "0.1.1"
__all__ = [
    "Client",
    "ModelSwarmError",
    "AuthError",
    "ClaimError",
    "APIError",
    "IdentityNotFoundError",
]
