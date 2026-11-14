import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bfcl_eval._llm_response_generation as bfcl_generation
import bfcl_eval.constants.eval_config as eval_config
import bfcl_eval.eval_checker.eval_runner as bfcl_evaluator
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

from core.config import Settings
from evaluators.bfcl_runner import (
    _multi_threaded_inference_raise_on_403,
    run_bfcl_evaluation,
)
from llm.schemas import EvaluationOptions


class HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


class FailingHandler:
    def __init__(self, error):
        self.error = error

    def inference(self, *args):
        raise self.error


class BfclHttpErrorTests(unittest.TestCase):
    def test_403_is_raised_instead_of_becoming_a_result(self):
        error = HttpError(403)

        with self.assertRaises(HttpError) as raised:
            _multi_threaded_inference_raise_on_403(
                FailingHandler(error),
                {"id": "irrelevance_1", "function": []},
                False,
                False,
            )

        self.assertIs(raised.exception, error)

    def test_other_inference_errors_keep_bfcl_recording_behavior(self):
        error = HttpError(500)

        with patch("evaluators.bfcl_runner.tqdm.write"):
            result = _multi_threaded_inference_raise_on_403(
                FailingHandler(error),
                {"id": "irrelevance_1", "function": []},
                False,
                False,
            )

        self.assertEqual(
            result["result"],
            "Error during inference: Error code: 500",
        )
        self.assertIn("traceback", result)

    def test_runner_propagates_403_before_scoring(self):
        error = HttpError(403)
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_root = Path(temporary_directory)
            result_path = results_root / "result"
            score_path = results_root / "score"
            result_path.mkdir()
            score_path.mkdir()

            def generate(_args):
                return bfcl_generation.multi_threaded_inference(
                    FailingHandler(error),
                    {"id": "irrelevance_1", "function": []},
                    False,
                    False,
                )

            with (
                patch.dict(os.environ),
                patch.dict(MODEL_CONFIG_MAPPING),
                patch.object(eval_config, "RESULT_PATH", result_path),
                patch.object(eval_config, "SCORE_PATH", score_path),
                patch.object(
                    bfcl_generation,
                    "main",
                    side_effect=generate,
                ),
                patch.object(bfcl_evaluator, "main") as evaluate,
                patch("evaluators.bfcl_runner.configure_bfcl_client"),
            ):
                original_worker = bfcl_generation.multi_threaded_inference
                with self.assertRaises(HttpError) as raised:
                    run_bfcl_evaluation(
                        EvaluationOptions(
                            model="test-model",
                            categories=["irrelevance"],
                        ),
                        Settings(
                            arc_api_key="unused",
                            results_root=results_root,
                        ),
                    )

                self.assertIs(raised.exception, error)
                self.assertIs(
                    bfcl_generation.multi_threaded_inference,
                    original_worker,
                )
                evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
