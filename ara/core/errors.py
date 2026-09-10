"""Error taxonomy. Every failure mode the runtime can produce has a typed error so
callers/APIs/evals can react deterministically instead of string-matching."""
from __future__ import annotations


class AraError(Exception):
    """Base class. `retryable` tells the resilience layer whether a retry makes sense."""

    code = "ara_error"
    http_status = 500
    retryable = False

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ValidationError(AraError):
    code = "validation_error"
    http_status = 422


class AuthorizationError(AraError):
    code = "authorization_error"
    http_status = 403


class AuthenticationError(AraError):
    code = "authentication_error"
    http_status = 401


class NotFoundError(AraError):
    code = "not_found"
    http_status = 404


class PolicyViolation(AraError):
    code = "policy_violation"
    http_status = 403


class BudgetExceeded(AraError):
    code = "budget_exceeded"
    http_status = 429
    retryable = False


class ToolError(AraError):
    code = "tool_error"
    http_status = 502


class ToolTimeout(ToolError):
    code = "tool_timeout"
    retryable = True


class ToolUnavailable(ToolError):
    code = "tool_unavailable"
    retryable = True


class RateLimitError(AraError):
    code = "rate_limited"
    http_status = 429


class LLMError(AraError):
    code = "llm_error"
    http_status = 502
    retryable = True


class RetrievalError(AraError):
    code = "retrieval_error"
    http_status = 502
    retryable = True


class ApprovalRequired(AraError):
    """Raised by the tool pipeline when a step needs human approval (control-flow signal)."""

    code = "approval_required"
    http_status = 202


class UnsafeContentError(AraError):
    code = "unsafe_content"
    http_status = 400


class VerificationFailure(AraError):
    code = "verification_failure"
    http_status = 422
