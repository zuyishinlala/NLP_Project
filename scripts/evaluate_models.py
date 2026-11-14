from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash")


def main() -> None:
    for model in MODELS:
        print(f"Evaluating {model} on the default categories...", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py"), "--model", model],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
