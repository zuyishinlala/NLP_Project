from pathlib import Path

from pydantic import BaseModel, Field


DATASET_ROOT = Path(__file__).resolve().parents[1] / "data"


def _discover_default_categories() -> list[str]:
    if not DATASET_ROOT.exists():
        return [
            "simple_python",
            "multiple",
            "parallel",
            "parallel_multiple",
            "irrelevance",
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
            "live_irrelevance",
            "live_relevance",
            "memory",
            "multi_turn_base",
            "multi_turn_long_context",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "simple_java",
            "simple_javascript",
            "web_search",
            "format_sensitivity",
        ]

    categories = []
    for path in sorted(DATASET_ROOT.glob("BFCL_v4_*.json")):
        category = path.name.removeprefix("BFCL_v4_").removesuffix(".json")
        if category:
            categories.append(category)
    return categories


DEFAULT_BFCL_CATEGORIES = tuple(_discover_default_categories())


class EvaluationOptions(BaseModel):
    model: str
    categories: list[str] = Field(default_factory=lambda: list(DEFAULT_BFCL_CATEGORIES))
    workers: int = Field(default=5, ge=1, le=10)
    overwrite: bool = False
    samples: int | None = Field(default=None, ge=1)
    dataset_dir: Path | None = None
