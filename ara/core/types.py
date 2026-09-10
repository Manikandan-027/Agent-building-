"""Core enums and value types shared across all layers."""
from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return list(RiskLevel).index(self)

    def requires_approval(self) -> bool:
        return self.rank >= RiskLevel.HIGH.rank


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BUDGET_STOPPED = "BUDGET_STOPPED"

    @property
    def is_terminal(self) -> bool:
        return self in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BUDGET_STOPPED}


class StepAction(str, Enum):
    RETRIEVE = "retrieve"
    TOOL_CALL = "tool_call"
    REASON = "reason"          # bounded LLM reasoning over evidence (no side effects)
    WAIT_APPROVAL = "wait_approval"
    FINAL_ANSWER = "final_answer"


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    WAITING_APPROVAL = "WAITING_APPROVAL"


class ContentTrust(str, Enum):
    """Source-of-truth policy: external content is untrusted unless designated."""

    SYSTEM = "system"            # system instructions — highest trust
    APP_POLICY = "app_policy"    # developer policy
    USER = "user"                # user instructions (trusted as *requests*, not facts)
    APP_DATA = "app_data"        # trusted application data
    RETRIEVED = "retrieved"      # retrieved documents — UNTRUSTED data
    WEB = "web"                  # web content — UNTRUSTED data
    TOOL_OUTPUT = "tool_output"  # tool results — UNTRUSTED unless tool marked trusted
    MODEL = "model"              # model-generated — never evidence


class SourceAuthority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ApprovalDecision(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
