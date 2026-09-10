from ara.db.database import Database
from ara.db.repositories import (
    ApprovalRepository,
    AuditLog,
    ConversationRepository,
    DocumentRepository,
    EvidenceRepository,
    EvalRunRepository,
    MemoryRepository,
    TaskRepository,
    ToolCallRepository,
)


class UnitOfWork:
    """Bundle of repositories over one database adapter."""

    def __init__(self, db: Database):
        self.db = db
        self.tasks = TaskRepository(db)
        self.documents = DocumentRepository(db)
        self.evidence = EvidenceRepository(db)
        self.tool_calls = ToolCallRepository(db)
        self.approvals = ApprovalRepository(db)
        self.memories = MemoryRepository(db)
        self.audit = AuditLog(db)
        self.conversations = ConversationRepository(db)
        self.evals = EvalRunRepository(db)
