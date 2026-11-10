from pydantic import BaseModel, Field


class OutputCheck(BaseModel):
    safe: bool
    deterministic_passed: bool
    referral_present: bool | None = None
    jev_called: bool = False
    jev_available: bool | None = None
    jev_scores: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
