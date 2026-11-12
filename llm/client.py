import os

from openai import OpenAI

from core.config import Settings


def create_arc_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.arc_api_key,
        base_url=settings.arc_base_url,
    )


def configure_bfcl_client(settings: Settings) -> None:
    os.environ["BFCL_PROJECT_ROOT"] = str(settings.results_root)
    # BFCL's OpenAI-compatible handler uses these names for ARC credentials and URL.
    os.environ["OPENAI_API_KEY"] = settings.arc_api_key
    os.environ["OPENAI_BASE_URL"] = settings.arc_base_url
