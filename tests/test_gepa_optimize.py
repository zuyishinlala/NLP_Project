import ast
import json
import os
import unittest
from unittest.mock import patch

from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

import openai

from scripts.gepa_optimize import (
    BFCLSystemPromptAdapter,
    heuristic_score,
    load_examples,
    optimize,
    propose_candidates,
    split_ids,
)


def _as_fc_calls(call_strings):
    calls = []
    for call in call_strings:
        node = ast.parse(call, mode="eval").body
        args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        calls.append({node.func.id: json.dumps(args)})
    return calls


@patch.dict(os.environ, {"OPENAI_API_KEY": "test"})
class GepaOptimizeTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self):
        ids = [f"id_{i}" for i in range(200)]
        train, val, test = split_ids(ids)
        self.assertEqual((len(train), len(val), len(test)), (80, 40, 80))
        self.assertEqual(sorted(train + val + test), sorted(ids))
        self.assertEqual(split_ids(list(reversed(ids))), (train, val, test))

    def test_system_prompt_is_prepended(self):
        handler = BFCLSystemPromptAdapter("m", 1).handler_class(
            model_name="m", temperature=0, registry_name="arc-m-FC",
            is_fc_model=True, system_prompt="Be careful.",
        )
        data = handler.add_first_turn_message_FC({"message": []}, [{"role": "user", "content": "hi"}])
        self.assertEqual(data["message"][0], {"role": "system", "content": "Be careful."})

    def test_scores_use_bfcl_checkers_and_repeat_runs_do_not_share_state(self):
        train, _, _ = load_examples()
        adapter = BFCLSystemPromptAdapter("m", 1)
        multi_turn = next(e for e in train if e["category"] == "multi_turn_base")
        irrelevance = next(e for e in train if e["category"] == "irrelevance")
        perfect = [[_as_fc_calls(turn)] for turn in multi_turn["ground_truth"]]
        no_calls = [[] for _ in multi_turn["ground_truth"]]

        scores = []
        for response in (no_calls, perfect):
            with patch.object(adapter.handler_class, "inference", return_value=(response, {})):
                scores.append(adapter.run_one(multi_turn, "")["score"])
        self.assertEqual(scores, [0.0, 1.0])
        # BFCL caches environments in these globals; leftovers would leak state into reruns.
        self.assertFalse([key for key in vars(multi_turn_utils) if key.startswith("gepa_")])

        name = irrelevance["entry"]["function"][0]["name"]
        with patch.object(adapter.handler_class, "inference", return_value=([{name: "{}"}], {})):
            self.assertEqual(adapter.run_one(irrelevance, "")["score"], 0.0)
        with patch.object(adapter.handler_class, "inference", return_value=("I can't do that.", {})):
            self.assertEqual(adapter.run_one(irrelevance, "")["score"], 1.0)

    def test_inference_errors_propagate_instead_of_being_scored(self):
        train, _, _ = load_examples()
        adapter = BFCLSystemPromptAdapter("m", 1)
        irrelevance = next(e for e in train if e["category"] == "irrelevance")
        with patch.object(adapter.handler_class, "inference", side_effect=RuntimeError("timeout")):
            with self.assertRaises(RuntimeError):
                adapter.run_one(irrelevance, "")

    def test_heuristic_score_penalizes_regressions(self):
        baseline = {"multi_turn_base": 0.5, "irrelevance": 0.8}
        candidate = {"multi_turn_base": 0.7, "irrelevance": 0.6}
        self.assertAlmostEqual(heuristic_score(candidate, baseline, 0.0), 0.65)
        self.assertAlmostEqual(heuristic_score(candidate, baseline, 1.0), 0.45)

    def test_loop_promotes_only_better_candidates(self):
        adapter = BFCLSystemPromptAdapter("m", 1)
        examples = [{"category": c, "id": i} for c in ("multi_turn_base", "irrelevance") for i in range(3)]

        def fake_run_one(example, system_prompt):
            score = float("careful" in system_prompt)
            record = {"Inputs": "q", "Generated Outputs": "a", "Feedback": "be careful"}
            return {"id": example["id"], "category": example["category"], "score": score, "record": record}

        proposals = iter(["```\nBe sloppy.\n```", "```\nBe careful.\n```"] * 4)
        with patch.object(adapter, "run_one", side_effect=fake_run_one):
            best, baseline, history = optimize(
                adapter, lambda prompt: next(proposals), examples, examples,
                rounds=2, k=2, minibatch_size=4,
            )
        self.assertEqual(best["system_prompt"], "Be careful.")
        self.assertEqual(baseline, {"multi_turn_base": 0.0, "irrelevance": 0.0})
        self.assertTrue(history[0]["promoted"])
        self.assertEqual(history[1]["minibatch_failures"], 0)  # perfect prompt: nothing to reflect on

    def test_reflection_replays_rollouts_from_earlier_prompts(self):
        adapter = BFCLSystemPromptAdapter("m", 1)
        examples = [{"category": "irrelevance", "id": i} for i in range(4)]

        def fake_run_one(example, system_prompt):
            record = {"Inputs": "q", "Generated Outputs": "a", "Feedback": "f"}
            return {"id": example["id"], "category": "irrelevance",
                    "score": float("x" in system_prompt and example["id"] < 2), "record": record}

        prompts = []
        proposals = iter(["```\nx\n```", "```\nxy\n```"])

        def reflection_lm(prompt):
            prompts.append(prompt)
            return next(proposals)

        with patch.object(adapter, "run_one", side_effect=fake_run_one):
            _, _, history = optimize(adapter, reflection_lm, examples, examples,
                                     rounds=2, k=1, minibatch_size=4, replay_size=2,
                                     regression_penalty=0.0)
        # Round 0 has no replay data yet; round 1 (best = "x") replays rollouts made with no prompt.
        self.assertEqual([h["replayed"] for h in history], [0, 2])
        self.assertIn("## System Prompt Used\n(none)", prompts[1])

    def test_screening_skips_val_and_margin_blocks_small_gains(self):
        adapter = BFCLSystemPromptAdapter("m", 1)
        train = [{"category": "irrelevance", "id": i, "split": "train"} for i in range(4)]
        val = [{"category": "irrelevance", "id": i, "split": "val"} for i in range(100)]
        val_prompts = set()

        def fake_run_one(example, system_prompt):
            if example["split"] == "val":
                val_prompts.add(system_prompt)
                # "tiny" gets one extra validation example right: below the 0.02 margin.
                score = float(system_prompt == "tiny" and example["id"] == 0)
            else:
                score = float(system_prompt == "tiny")  # "worse" does no better than no prompt on train
            record = {"Inputs": "q", "Generated Outputs": "a", "Feedback": "f"}
            return {"id": example["id"], "category": "irrelevance", "score": score, "record": record}

        proposals = iter(["```\nworse\n```", "```\ntiny\n```"])
        with patch.object(adapter, "run_one", side_effect=fake_run_one):
            best, _, history = optimize(adapter, lambda prompt: next(proposals), train, val,
                                        rounds=1, k=2, minibatch_size=4)
        self.assertEqual(val_prompts, {"", "tiny"})  # "worse" never reached validation
        self.assertTrue(history[0]["candidates"][0]["screened_out"])
        self.assertFalse(history[0]["promoted"])
        self.assertEqual(best["system_prompt"], "")

    def test_failed_reflection_call_skips_candidate_instead_of_crashing(self):
        replies = iter([openai.APITimeoutError(request=None), "```\nBe careful.\n```"])

        def reflection_lm(prompt):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        record = {"Inputs": "q", "Generated Outputs": "a", "Feedback": "f"}
        failures = [{"record": record, "behavior_prompt": "", "score": 0.0}]
        self.assertEqual(propose_candidates(reflection_lm, "", failures, 2), ["Be careful."])


if __name__ == "__main__":
    unittest.main()
