"""PaperOps domain package.

PR1 establishes the product boundary and shared domain types.  The executable
document workflow is introduced in PR2 so this package does not pretend that
the MinerU/RAGFlow integration already exists.
"""

from paperops.models import JobStatus, QualityDecision, QualityVerdict
from paperops.state import DocumentJobState

__all__ = [
    "DocumentJobState",
    "JobStatus",
    "QualityDecision",
    "QualityVerdict",
]
