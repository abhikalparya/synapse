from typing import Any

from pydantic import BaseModel, Field, ValidationError


class LintIssue(BaseModel):
    """Single lint finding; shape varies by ``type`` (``topic`` vs ``topics``)."""

    type: str = Field(
        ...,
        description="missing_summary | duplicate_title | self_dependency | orphan_dependency | cycle",
    )
    topic: str | None = Field(default=None, description="Affected topic id when a single topic")
    topics: list[str] | None = Field(default=None, description="Topic ids/titles involved (e.g. a cycle)")
    detail: str | None = Field(default=None, description="Extra context for the issue")


class LintResponse(BaseModel):
    issues: list[LintIssue] = Field(default_factory=list)

    @classmethod
    def from_issue_dicts(cls, rows: list[dict[str, Any]]) -> "LintResponse":
        out: list[LintIssue] = []
        for x in rows:
            try:
                out.append(LintIssue.model_validate(x))
            except ValidationError:
                out.append(
                    LintIssue(
                        type=str(x.get("type", "unknown")),
                        topic=x.get("topic"),
                        topics=x.get("topics") if isinstance(x.get("topics"), list) else None,
                        detail=str(x.get("detail", x))[:500],
                    ),
                )
        return cls(issues=out)
