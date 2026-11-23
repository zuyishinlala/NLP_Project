import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.gepa_full import BFCLGEPAAdapter, run_gepa
from scripts.gepa_optimize import BFCLSystemPromptAdapter


def fake_run_one(example, system_prompt):
    # "careful" fixes multi-turn, "polite" fixes irrelevance; only a prompt with both is perfect.
    word = "careful" if example["category"] == "multi_turn_base" else "polite"
    score = float(word in system_prompt)
    record = {"Inputs": "q", "Generated Outputs": "a", "Feedback": "right" if score else f"be {word}"}
    return {"id": example["id"], "category": example["category"], "score": score, "record": record}


@patch.dict(os.environ, {"OPENAI_API_KEY": "test"})
class GepaFullTests(unittest.TestCase):
    def test_reflective_dataset_feeds_each_component_every_record(self):
        bfcl = BFCLSystemPromptAdapter("m", 1)
        examples = [{"category": c, "id": i} for i, c in enumerate(["multi_turn_base", "irrelevance"])]
        with patch.object(bfcl, "run_one", side_effect=fake_run_one):
            batch = BFCLGEPAAdapter(bfcl).evaluate(examples, {"system_prompt": "Be careful."}, capture_traces=True)
        self.assertEqual(batch.scores, [1.0, 0.0])
        dataset = BFCLGEPAAdapter(bfcl).make_reflective_dataset({}, batch, ["system_prompt"])
        self.assertEqual([r["Feedback"] for r in dataset["system_prompt"]], ["right", "be polite"])

    def test_gepa_search_finds_better_prompt_and_reports_per_category_val(self):
        bfcl = BFCLSystemPromptAdapter("m", 1)
        examples = [{"category": c, "id": f"{c}_{i}"} for c in ("multi_turn_base", "irrelevance") for i in range(3)]
        proposals = iter(["```\nBe careful.\n```", "```\nBe polite.\n```", "```\nBe careful and polite.\n```"] * 20)
        with patch.object(bfcl, "run_one", side_effect=fake_run_one), tempfile.TemporaryDirectory() as run_dir:
            best, baseline_val, best_val, result = run_gepa(
                bfcl, lambda prompt: next(proposals), examples, examples,
                budget=80, minibatch_size=3, run_dir=run_dir,
            )
        self.assertEqual(baseline_val, {"multi_turn_base": 0.0, "irrelevance": 0.0})
        self.assertGreater(sum(best_val.values()), 0)
        self.assertEqual(best_val, {
            "multi_turn_base": float("careful" in best),
            "irrelevance": float("polite" in best),
        })
        self.assertLessEqual(result.total_metric_calls, 80 + 6)  # GEPA may finish its last full val pass


if __name__ == "__main__":
    unittest.main()
