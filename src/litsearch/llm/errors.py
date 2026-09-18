class LLMError(RuntimeError):
    """Provider-neutral base error."""


class LLMUnavailableError(LLMError):
    pass


class LLMAuthenticationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMModelNotFoundError(LLMError):
    pass


class LLMStructuredOutputError(LLMError):
    pass


class LLMConfigurationError(LLMError):
    pass