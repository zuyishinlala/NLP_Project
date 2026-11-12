import hashlib
import json
import re
import statistics
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

from core.config import Settings
from llm.client import configure_bfcl_client
from llm.schemas import EvaluationOptions


SCORING_MODEL = "gpt-4o-2024-11-20-FC"


def configure_dataset_dir(
    dataset_dir: Path,
    categories: list[str],
    bfcl_utils: ModuleType,
) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    prompt_files = [dataset_dir / f"BFCL_v4_{category}.json" for category in categories]
    answer_files = [
        dataset_dir / "possible_answer" / f"BFCL_v4_{category}.json"
        for category in categories
        if category != "irrelevance"
    ]
    missing_files = [path for path in prompt_files + answer_files if not path.is_file()]
    if missing_files:
        missing_list = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "The custom dataset must match the installed BFCL v4 format. "
            "Missing required files:\n" + missing_list
        )

    bfcl_utils.PROMPT_PATH = dataset_dir
    bfcl_utils.POSSIBLE_ANSWER_PATH = dataset_dir / "possible_answer"
    return dataset_dir


def run_bfcl_evaluation(options: EvaluationOptions, settings: Settings) -> None:
    settings.results_root.mkdir(parents=True, exist_ok=True)
    configure_bfcl_client(settings)

    try:
        from bfcl_eval._llm_response_generation import main as generate_main
        from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
        from bfcl_eval.eval_checker.eval_runner import main as evaluate_main
        from bfcl_eval.model_handler.api_inference.openai_completion import (
            OpenAICompletionsHandler,
        )
        import bfcl_eval.utils as bfcl_utils
        from bfcl_eval.utils import load_dataset_entry
    except ImportError as error:
        raise RuntimeError(
            "BFCL's official evaluator is not installed in this Python environment. "
            "Activate .venv and run `.venv/bin/python -m pip install -r "
            "requirements.txt`."
        ) from error

    base_config = MODEL_CONFIG_MAPPING.get(SCORING_MODEL)
    if base_config is None:
        raise RuntimeError(
            f"The installed BFCL evaluator has no {SCORING_MODEL!r} OpenAI "
            "function-calling configuration."
        )

    categories = ["simple_python"] if options.first_only else options.categories
    dataset_dir = None
    if options.dataset_dir is not None:
        dataset_dir = configure_dataset_dir(
            options.dataset_dir,
            categories,
            bfcl_utils,
        )

    registry_suffix = ""
    if dataset_dir is not None:
        dataset_identity = str(dataset_dir)
        dataset_hash = hashlib.sha256(dataset_identity.encode()).hexdigest()[:8]
        registry_suffix = f"-data-{dataset_hash}"
    registry_name = (
        f"arc-{re.sub(r'[^A-Za-z0-9.-]+', '-', options.model)}-FC{registry_suffix}"
    )
    MODEL_CONFIG_MAPPING[registry_name] = replace(
        base_config,
        model_name=options.model,
        display_name=f"ARC {options.model} (FC)",
        url="https://www.docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html",
        org="Virginia Tech ARC",
        license="See ARC model documentation",
        model_handler=OpenAICompletionsHandler,
    )

    if options.first_only:
        first_entry = load_dataset_entry("simple_python")[0]
        id_file = settings.results_root / "test_case_ids_to_generate.json"
        id_file.write_text(
            json.dumps({"simple_python": [first_entry["id"]]}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Testing first official simple_python example: {first_entry['id']}")

    generation_args = SimpleNamespace(
        model=[registry_name],
        test_category=categories,
        temperature=0.001,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=options.workers,
        gpu_memory_utilization=0.9,
        backend="vllm",
        skip_server_setup=False,
        local_model_path=None,
        result_dir=None,
        allow_overwrite=options.overwrite or options.first_only,
        run_ids=options.first_only,
        enable_lora=False,
        max_lora_rank=None,
        lora_modules=None,
    )

    print(f"Generating BFCL v4 responses with ARC model {options.model}...")
    generate_main(generation_args)

    print("Scoring results with BFCL's official evaluator...")
    try:
        evaluate_main(
            [registry_name],
            categories,
            None,
            None,
            partial_eval=options.first_only,
        )
    except statistics.StatisticsError:
        if not options.first_only:
            raise
        score_path = (
            SCORE_PATH
            / registry_name
            / "non_live"
            / "BFCL_v4_simple_python_score.json"
        )
        if not score_path.is_file():
            raise
        score = json.loads(score_path.read_text(encoding="utf-8"))
        print(
            "BFCL category scoring completed for the single example: "
            f"{score['correct_count']}/{score['total_count']} "
            f"({score['accuracy']:.1%}). BFCL skipped leaderboard CSV aggregation "
            "because standard deviation requires at least two examples."
        )

    print(f"Results: {RESULT_PATH / registry_name}")
    print(f"Scores: {SCORE_PATH / registry_name}")
