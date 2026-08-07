"""Typed failures raised by external PaperOps service adapters."""


class ExternalServiceError(RuntimeError):
    """Base error for a failed or malformed external service operation."""


class ExternalServiceTimeout(ExternalServiceError):
    """Report an operation that exceeded its configured deadline."""


class MinerUError(ExternalServiceError):
    """Report an invalid or failed MinerU operation."""


class MinerUTimeout(MinerUError, ExternalServiceTimeout):
    """Report a MinerU task that did not complete before its deadline."""


class RAGFlowError(ExternalServiceError):
    """Report an invalid or failed RAGFlow operation."""


class RAGFlowTimeout(RAGFlowError, ExternalServiceTimeout):
    """Report a RAGFlow indexing job that exceeded its deadline."""
