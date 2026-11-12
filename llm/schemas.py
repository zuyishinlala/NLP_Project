from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


BFCLCategory = Literal[
    "simple_python",
    "multiple",
    "parallel",
    "irrelevance",
]


class EvaluationOptions(BaseModel):
    model: str
    categories: list[BFCLCategory] = Field(
        default_factory=lambda: [
            "simple_python",
            "multiple",
            "parallel",
            "irrelevance",
        ]
    )
    workers: int = Field(default=5, ge=1, le=10)
    overwrite: bool = False
    first_only: bool = False
    dataset_dir: Path | None = None
