import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bfcl_eval._llm_response_generation as bfcl_generation
import bfcl_eval.constants.eval_config as eval_config
import bfcl_eval.eval_checker.eval_runner as bfcl_evaluator
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)

from core.config import Settings
from evaluators.bfcl_runner import run_bfcl_evaluation
from llm.schemas import EvaluationOptions


class BfclRunnerTests(unittest.TestCase):
    def test_arc_scoring_registers_existing_arc_results(self):
        arc_model = "arc-gpt-oss-120b-FC"
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_root = Path(temporary_directory)
            result_path = results_root / "result"
            score_path = results_root / "score"
            (result_path / arc_model).mkdir(parents=True)
            score_path.mkdir()

            with (
                patch.dict(os.environ),
                patch.dict(MODEL_CONFIG_MAPPING),
                patch.object(eval_config, "RESULT_PATH", result_path),
                patch.object(eval_config, "SCORE_PATH", score_path),
                patch.object(bfcl_generation, "main") as generate,
                patch.object(bfcl_evaluator, "main") as evaluate,
                patch("evaluators.bfcl_runner.configure_bfcl_client"),
            ):
                MODEL_CONFIG_MAPPING.pop(arc_model, None)

                def verify_registry(*args, **kwargs):
                    self.assertIn(arc_model, MODEL_CONFIG_MAPPING)
                    self.assertIs(
                        MODEL_CONFIG_MAPPING[arc_model].model_handler,
                        OpenAICompletionsHandler,
                    )

                evaluate.side_effect = verify_registry
                run_bfcl_evaluation(
                    options=EvaluationOptions(
                        model="gpt-oss-120b",
                        categories=["parallel"],
                    ),
                    settings=Settings(
                        arc_api_key="unused",
                        results_root=results_root,
                    ),
                )

                generate.assert_called_once()
                evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
