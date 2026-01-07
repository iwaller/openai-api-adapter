class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, status_code: int, error_type: str, message: str):
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Authentication failed."""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(401, "authentication_error", message)


class RateLimitError(ProviderError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(429, "rate_limit_error", message)


class InvalidRequestError(ProviderError):
    """Invalid request parameters."""

    def __init__(self, message: str):
        super().__init__(400, "invalid_request_error", message)


class ModelNotFoundError(ProviderError):
    """Model not found."""

    def __init__(self, model: str):
        super().__init__(404, "model_not_found", f"Model '{model}' not found")


class ConnectionError(ProviderError):
    """Failed to connect to provider."""

    def __init__(self, message: str = "Failed to connect to provider"):
        super().__init__(502, "connection_error", message)


class ProviderAPIError(ProviderError):
    """Provider API returned an error."""

    def __init__(self, status_code: int, message: str):
        super().__init__(status_code, "api_error", message)
