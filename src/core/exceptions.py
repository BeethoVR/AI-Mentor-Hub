class RAGQueryError(Exception):
    """Base exception for RAG query failures."""
    pass

class QuotaExceededError(RAGQueryError):
    """Raised when Google API quota is exceeded."""
    pass

class APIServiceUnavailableError(RAGQueryError):
    """Raised when the API service is temporarily unavailable."""
    pass