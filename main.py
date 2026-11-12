import argparse
from pathlib import Path

from core.config import DEFAULT_MODEL, load_settings
from evaluators.bfcl_runner import run_bfcl_evaluation
from llm.schemas import EvaluationOptions


def parse_args() -> EvaluationOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Run BFCL's official end-to-end evaluator on its packaged BFCL v4 "
            "dataset using an ARC-hosted model."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["simple_python", "multiple", "parallel", "irrelevance"],
        default=["simple_python", "multiple", "parallel", "irrelevance"],
        help="Official BFCL v4 categories to generate and evaluate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Concurrent ARC requests; the maximum is 10 per model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate result files even if they already exist.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help=(
            "Number of examples to run per selected category. If greater than "
            "a category's size, all available examples in that category are used."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "Optional BFCL v4 data directory containing BFCL_v4_<category>.json "
            "files and a possible_answer/ subdirectory."
        ),
    )
    args = parser.parse_args()
    return EvaluationOptions(
        model=args.model,
        categories=args.categories,
        workers=args.workers,
        overwrite=args.overwrite,
        samples=args.samples,
        dataset_dir=args.dataset_dir,
    )


def main() -> None:
    options = parse_args()
    settings = load_settings()
    run_bfcl_evaluation(options, settings)


if __name__ == "__main__":
    main()
