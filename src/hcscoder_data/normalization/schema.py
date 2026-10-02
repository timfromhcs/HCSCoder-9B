import hashlib
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: str  # system, user, assistant, tool
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ToolDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Outcome(BaseModel):
    success: bool = True
    tests_passed: int = 0
    tests_failed: int = 0
    exit_code: Optional[int] = 0
    details: str = ""


class ComputedMetadata(BaseModel):
    steps: int = 0
    tool_calls: int = 0
    unique_tools: int = 0
    tool_errors: int = 0
    retries: int = 0
    files_read: int = 0
    files_modified: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    has_planning: bool = False
    has_recovery: bool = False
    has_verification: bool = False
    long_horizon: bool = False
    capabilities: List[str] = Field(default_factory=list)
    quality_score: float = 0.0
    quality_bucket: str = "GOOD"


class Trajectory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message]
    tools: List[ToolDefinition] = Field(default_factory=list)
    outcome: Outcome = Field(default_factory=Outcome)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    computed: ComputedMetadata = Field(default_factory=ComputedMetadata)

    def compute_fields(self) -> None:
        """Compute statistical and behavioral metadata."""
        tool_names = set()
        total_tool_calls = 0
        tool_errors = 0
        has_plan = False
        has_rec = False
        has_verif = False

        for msg in self.messages:
            content = (msg.content or "").lower()
            if "plan" in content or "step 1" in content or "objective" in content:
                has_plan = True
            if "error" in content or "exception" in content or "traceback" in content:
                tool_errors += 1
            if "recover" in content or "retry" in content or "alternative" in content or "let me fix" in content:
                has_rec = True
            if "verify" in content or "test" in content or "assert" in content or "pytest" in content:
                has_verif = True

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_tool_calls += 1
                    tool_names.add(tc.name)

        steps = len([m for m in self.messages if m.role == "assistant"])
        self.computed.steps = steps
        self.computed.tool_calls = total_tool_calls
        self.computed.unique_tools = len(tool_names)
        self.computed.tool_errors = tool_errors
        self.computed.has_planning = has_plan
        self.computed.has_recovery = has_rec
        self.computed.has_verification = has_verif
        self.computed.long_horizon = steps >= 10
        self.computed.tests_passed = self.outcome.tests_passed
        self.computed.tests_failed = self.outcome.tests_failed

    def content_hash(self) -> str:
        text = "".join(f"{m.role}:{m.content}" for m in self.messages)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
