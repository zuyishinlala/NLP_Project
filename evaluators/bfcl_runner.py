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


def select_sample_ids(
    categories: list[str],
    sample_count: int,
    load_entries,
) -> dict[str, list[str]]:
    selected_ids = {}
    for category in categories:
        entries = load_entries(category)
        if not entries:
            raise ValueError(f"BFCL category {category!r} contains no examples.")
        selected = entries[:sample_count]
        if sample_count > len(entries):
            print(
                f"Requested {sample_count} samples for {category}, but only "
                f"{len(entries)} are available; using all available examples."
            )
        selected_ids[category] = [entry["id"] for entry in selected]
        print(f"Selected {len(selected)} of {len(entries)} examples for {category}.")
    return selected_ids


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


def register_existing_arc_models(model_mapping, base_config, *result_roots: Path) -> None:
    model_pattern = re.compile(r"^arc-(?P<model>.+)-FC(?:-data-[0-9a-f]{8})?$")
    for result_root in result_roots:
        if not result_root.is_dir():
            continue
        for model_dir in result_root.iterdir():
            if not model_dir.is_dir():
                continue
            match = model_pattern.fullmatch(model_dir.name)
            if match is None or model_dir.name in model_mapping:
                continue
            model_name = match.group("model")
            model_mapping[model_dir.name] = replace(
                base_config,
                model_name=model_name,
                display_name=f"ARC {model_name} (FC)",
                url="https://www.docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html",
                org="Virginia Tech ARC",
                license="See ARC model documentation",
                model_handler=base_config.model_handler,
            )


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

    categories = options.categories
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
    register_existing_arc_models(
        MODEL_CONFIG_MAPPING,
        replace(base_config, model_handler=OpenAICompletionsHandler),
        RESULT_PATH,
        SCORE_PATH,
    )

    run_ids = options.samples is not None
    if run_ids:
        sample_count = options.samples
        if sample_count is None:
            raise RuntimeError("Sample count was not provided for a partial run.")
        selected_ids = select_sample_ids(
            categories,
            sample_count,
            load_dataset_entry,
        )
        id_file = settings.results_root / "test_case_ids_to_generate.json"
        id_file.write_text(
            json.dumps(selected_ids, indent=2) + "\n",
            encoding="utf-8",
        )

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
        allow_overwrite=options.overwrite,
        run_ids=run_ids,
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
            partial_eval=run_ids,
        )
    except statistics.StatisticsError:
        if not run_ids:
            raise
        score_paths = [
            SCORE_PATH
            / registry_name
            / "non_live"
            / f"BFCL_v4_{category}_score.json"
            for category in categories
        ]
        available_scores = [path for path in score_paths if path.is_file()]
        if not available_scores:
            raise
        for score_path in available_scores:
            score = json.loads(score_path.read_text(encoding="utf-8"))
            print(
                f"BFCL category scoring completed for "
                f"{score['correct_count']}/{score['total_count']} "
                f"({score['accuracy']:.1%}). BFCL skipped leaderboard CSV "
                "aggregation because latency statistics require at least two "
                "samples."
            )

    print(f"Results: {RESULT_PATH / registry_name}")
    print(f"Scores: {SCORE_PATH / registry_name}")
