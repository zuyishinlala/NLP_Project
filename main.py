import argparse

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
        "--first-only",
        action="store_true",
        help="Evaluate only the first official BFCL v4 simple_python example.",
    )
    args = parser.parse_args()
    return EvaluationOptions(
        model=args.model,
        categories=args.categories,
        workers=args.workers,
        overwrite=args.overwrite,
        first_only=args.first_only,
    )


def main() -> None:
    options = parse_args()
    settings = load_settings()
    run_bfcl_evaluation(options, settings)


if __name__ == "__main__":
    main()
