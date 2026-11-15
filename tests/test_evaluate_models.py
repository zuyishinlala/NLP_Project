import unittest
from unittest.mock import patch

from scripts.evaluate_models import DEFAULT_CATEGORIES, MODELS, PROJECT_ROOT, evaluate_models


class EvaluateModelsTests(unittest.TestCase):
    def test_runs_each_model_for_only_the_selected_category(self):
        with patch("scripts.evaluate_models.subprocess.run") as run:
            evaluate_models(("parallel",))

        self.assertEqual(run.call_count, len(MODELS))
        for call, model in zip(run.call_args_list, MODELS):
            command = call.args[0]
            self.assertEqual(command[command.index("--model") + 1], model)
            category_index = command.index("--categories")
            self.assertEqual(command[category_index + 1 :], ["parallel"])
            self.assertEqual(call.kwargs["cwd"], PROJECT_ROOT)
            self.assertTrue(call.kwargs["check"])

    def test_default_categories_include_non_legacy_bfcl_sets(self):
        self.assertIn("live_simple", DEFAULT_CATEGORIES)
        self.assertIn("memory", DEFAULT_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
