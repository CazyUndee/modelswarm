"""
Custom exceptions for ModelSwarm.
"""


class ModelSwarmError(Exception):
    """Base exception for all ModelSwarm errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuthError(ModelSwarmError):
    """Raised when authentication fails or credentials are missing."""
    pass


class ClaimError(ModelSwarmError):
    """Raised when an experiment cannot be claimed (already claimed by another agent)."""
    pass


class APIError(ModelSwarmError):
    """Raised when the API returns a non-200 response."""

    def __init__(self, message: str, status_code: int | None = None, response_body: dict | None = None):
        super().__init__(message, status_code)
        self.response_body = response_body


class IdentityNotFoundError(ModelSwarmError):
    """Raised when agent identity cannot be found."""
    pass


class ConfigError(ModelSwarmError):
    """Raised when configuration is invalid or missing."""
    pass
