from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash")
DEFAULT_CATEGORIES = tuple(
    path.name.removeprefix("BFCL_v4_").removesuffix(".json")
    for path in sorted((PROJECT_ROOT / "data").glob("BFCL_v4_*.json"))
)


def evaluate_models(categories: tuple[str, ...]) -> None:
    for model in MODELS:
        print(
            f"Evaluating {model} on {', '.join(categories)}...",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "--model",
                model,
                "--categories",
                *categories,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


def main() -> None:
    evaluate_models(DEFAULT_CATEGORIES)


if __name__ == "__main__":
    main()
