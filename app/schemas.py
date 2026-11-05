from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["answered", "not_found", "refused"]


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(max_length=1000)
    conversation: list[Turn] = Field(default_factory=list, max_length=50)

    @field_validator("message")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v


class ChatResponse(BaseModel):
    answer: str
    status: Status
    sources: list[str]
    confidence: float = Field(ge=0, le=1)
    latency_ms: int | None = None


class LLMOutput(BaseModel):
    """Schema the model must follow. Validated before anything reaches the client."""
    model_config = ConfigDict(extra="ignore")
    answer: str = Field(min_length=1, max_length=1500)
    status: Literal["answered", "not_found"]
    sources: list[str]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "answered" and not self.sources:
            raise ValueError("answered responses must cite at least one source")
        if self.status == "not_found" and self.sources:
            raise ValueError("not_found responses must have empty sources")
        return self
