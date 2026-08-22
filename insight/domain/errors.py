from __future__ import annotations

from typing import Any


class InsightError(Exception):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(InsightError):
    pass


class DataError(InsightError):
    pass


class ProviderError(InsightError):
    def __init__(
        self,
        provider: str,
        model: str,
        detail: str,
        *,
        status_code: int | None = None,
    ):
        super().__init__(f"[{provider}] {model}: {detail}")
        self.provider = provider
        self.model = model
        self.detail = detail
        self.status_code = status_code


class ProviderAuthError(ProviderError):
    pass


class RateLimitError(ProviderError):
    def __init__(
        self,
        provider: str,
        model: str,
        detail: str,
        retry_after_s: float | None = None,
    ):
        super().__init__(provider, model, detail)
        self.retry_after_s = retry_after_s


class AllProvidersFailedError(InsightError):
    def __init__(self, errors: list[Exception]):
        self.errors = errors
        super().__init__(
            "All providers failed: "
            + " | ".join(getattr(e, "message", str(e)) for e in errors)
        )


class PlanValidationError(InsightError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Plan validation failed: " + "; ".join(issues))


class OperationExecutionError(InsightError):
    pass
