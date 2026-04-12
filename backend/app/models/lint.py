from typing import Any

from pydantic import BaseModel, Field, ValidationError


class LintIssue(BaseModel):
    """Single lint finding; shape varies by ``type`` (``page`` vs ``pages``)."""

    type: str = Field(
        ...,
        description="duplicate | missing_tags | empty_summary | weak_key_points | inconsistent_formatting",
    )
    page: str | None = Field(default=None, description="Affected wiki title when a single page")
    pages: list[str] | None = Field(default=None, description="Titles in a duplicate cluster")
    detail: str | None = Field(default=None, description="Extra context for the issue")
    suggestion: str | None = Field(default=None, description="Optional LLM fix hint")


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
                        page=x.get("page"),
                        pages=x.get("pages") if isinstance(x.get("pages"), list) else None,
                        detail=str(x.get("detail", x))[:500],
                    ),
                )
        return cls(issues=out)
