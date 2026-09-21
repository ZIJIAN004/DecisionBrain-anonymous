"""Request models for the Run API."""

from pydantic import BaseModel, Field


class ContinueRunRequest(BaseModel):
    message: str = Field(min_length=1)


class StartBenchmarkRequest(BaseModel):
    feasibility_review_enabled: bool = True
    input_schema_enabled: bool = True
    algorithm_library_enabled: bool = True
    task_ids: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=1, gt=0)
    jobs: int = Field(default=1, gt=0, le=8)
