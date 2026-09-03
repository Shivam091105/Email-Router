"""
Structured classification output.

This is the contract the LLM's response must satisfy. We never parse
free-form natural language from the model — we instruct it to return
JSON matching this shape, then validate with Pydantic. If validation
fails, that's treated as a real failure (retry or error), never silently
ignored.
"""

from pydantic import BaseModel, Field, field_validator


class ClassificationResult(BaseModel):
    department: str = Field(..., min_length=1)
    team: str = Field(..., min_length=1)
    team_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


class ClassificationError(Exception):
    """Raised when the LLM's output cannot be turned into a valid ClassificationResult."""
