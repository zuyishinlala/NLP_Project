import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-oss-120b"


@dataclass(frozen=True)
class Settings:
    arc_api_key: str = field(repr=False)
    arc_base_url: str = "https://llm-api.arc.vt.edu/api/v1/"
    default_model: str = DEFAULT_MODEL
    results_root: Path = PROJECT_ROOT / "bfcl_official_results"


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("ARC_LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ARC_LLM_API_KEY was not found. Set it in the project .env file "
            "or in the environment."
        )
    return Settings(arc_api_key=api_key)
