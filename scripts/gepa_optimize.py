"""Optimize a system prompt on BFCL multi_turn_base + irrelevance by reflective hill climbing.

Each round: run the best prompt on a train minibatch, give its failure traces plus a
replay sample of rollouts from earlier prompts (off-policy data) to a reflection LLM
that proposes K new prompts, screen each on the same minibatch, score the
survivors on validation, and promote the best one if it beats the current best by a margin. The held-out test split is evaluated once at the end.

Run: .venv/bin/python -m scripts.gepa_optimize --model GLM-5.3
Writes output/<model>/log.txt (everything printed) and output/<model>/report.json.
"""

import argparse
import copy
import json
import random
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import openai
from gepa.strategies.instruction_proposal import InstructionProposalSignature

from core.config import PROJECT_ROOT, load_settings
from llm.client import configure_bfcl_client, create_arc_client


CATEGORIES = ("multi_turn_base", "irrelevance")
TRAIN_FRACTION, VAL_FRACTION = 0.4, 0.2  # the rest is the held-out test split
SEED = 0
# shortcut: traces are cut to this many characters per field, raise it if the reflection model needs more context.
MAX_TRACE_CHARS = 6000
REFLECTION_PROMPT = (
    InstructionProposalSignature.default_prompt_template
    + "\n\nWrite general rules about how and when to use tools. Do not mention specific "
    "tasks, file names, people, or values from these examples; the instruction will be "
    "tested on different tasks. Some examples were produced by earlier versions of the "
    "instruction (see 'System Prompt Used' and 'Score'); keep what made those versions "
    "succeed and avoid what made them fail."
)


def split_ids(ids, seed=SEED):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * TRAIN_FRACTION)
    n_val = round(len(ids) * VAL_FRACTION)
    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]


def load_examples():
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    ground_truth = {
        entry["id"]: entry["ground_truth"]
        for entry in load_ground_truth_entry("multi_turn_base")
    }
    splits = ([], [], [])
    for category in CATEGORIES:
        by_id = {entry["id"]: entry for entry in load_dataset_entry(category)}
        for split, ids in zip(splits, split_ids(by_id)):
            split.extend(
                {"category": category, "entry": by_id[i], "ground_truth": ground_truth.get(i)}
                for i in ids
            )
    return splits


def _clip(value):
    return json.dumps(value, default=str)[:MAX_TRACE_CHARS]


class BFCLSystemPromptAdapter:
    """Runs BFCL examples with a given system prompt, scored by BFCL's own checkers."""

    def __init__(self, model, workers):
        from bfcl_eval.model_handler.api_inference.openai_completion import (
            OpenAICompletionsHandler,
        )

        class SystemPromptHandler(OpenAICompletionsHandler):
            def add_first_turn_message_FC(self, inference_data, first_turn_message):
                if self.system_prompt:
                    first_turn_message = [
                        {"role": "system", "content": self.system_prompt},
                        *first_turn_message,
                    ]
                return super().add_first_turn_message_FC(inference_data, first_turn_message)

        self.handler_class = SystemPromptHandler
        self.model = model
        self.workers = workers

    def run_one(self, example, system_prompt):
        from bfcl_eval.eval_checker import eval_runner
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

        # BFCL caches environment instances in module globals keyed by this name;
        # a unique name per rollout keeps repeated runs of a task from sharing state.
        run_name = f"gepa_{uuid.uuid4().hex}"
        handler = self.handler_class(
            model_name=self.model,
            temperature=0.001,
            registry_name=f"arc-{self.model}-FC",
            is_fc_model=True,
            system_prompt=system_prompt,
        )
        handler.model_name_underline_replaced = run_name
        # A single ARC call can take over a minute; retry transient failures here.
        handler.client = handler.client.with_options(timeout=300, max_retries=5)
        entry, category = example["entry"], example["category"]
        try:
            # Errors must propagate: scored as text answers they pass every irrelevance check.
            result, metadata = handler.inference(copy.deepcopy(entry), False, True)
            if category == "irrelevance":
                check = eval_runner._evaluate_single_relevance_entry(
                    handler, entry["id"], result, entry, run_name, category
                )
            else:
                check = eval_runner._evaluate_single_multi_turn_entry(
                    handler,
                    entry["id"],
                    result,
                    example["ground_truth"],
                    copy.deepcopy(entry),
                    run_name,
                    category,
                )
        finally:
            for key in list(vars(multi_turn_utils)):
                if run_name in key:
                    vars(multi_turn_utils).pop(key, None)

        valid = bool(check["valid"])
        if category == "irrelevance":
            inputs = {"user_request": entry["question"], "available_functions": entry["function"]}
            outputs = result
            feedback = (
                "Correct: the assistant answered without calling a function."
                if valid
                else "Wrong: none of the available functions can fulfil this request, so "
                "the assistant should have replied in text without calling any function."
            )
        else:
            inputs = {"user_turns": entry["question"], "involved_classes": entry["involved_classes"]}
            outputs = metadata.get("inference_log", result)
            feedback = (
                "Correct: the final environment state and execution results match."
                if valid
                else f"Wrong. Checker result: {_clip(check.get('error'))}. "
                f"Expected function calls per turn: {_clip(example['ground_truth'])}"
            )
        return {
            "id": entry["id"],
            "category": category,
            "score": float(valid),
            "record": {
                "Inputs": _clip(inputs),
                "Generated Outputs": _clip(outputs),
                "Feedback": feedback,
            },
        }

    def evaluate(self, examples, system_prompt):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda example: self.run_one(example, system_prompt), examples))


def category_accuracy(rollouts):
    return {
        category: sum(r["score"] for r in rollouts if r["category"] == category)
        / max(1, sum(r["category"] == category for r in rollouts))
        for category in CATEGORIES
    }


def heuristic_score(accuracy, baseline, regression_penalty):
    """Mean per-category accuracy, minus a penalty for each category that drops below baseline."""
    regression = sum(max(0.0, baseline[c] - accuracy[c]) for c in CATEGORIES)
    return sum(accuracy.values()) / len(accuracy) - regression_penalty * regression


def propose_candidates(reflection_lm, system_prompt, failures, k):
    """`failures` are rollouts tagged with the `behavior_prompt` that produced them."""
    prompt = InstructionProposalSignature.prompt_renderer({
        "current_instruction_doc": system_prompt or "(no system prompt)",
        "dataset_with_feedback": [
            {**r["record"], "System Prompt Used": r["behavior_prompt"] or "(none)", "Score": r["score"]}
            for r in failures
        ],
        "prompt_template": REFLECTION_PROMPT,
    })
    texts = []
    for _ in range(k):
        try:
            texts.append(InstructionProposalSignature.output_extractor(reflection_lm(prompt))["new_instruction"])
        except openai.APIError as error:  # a lost proposal only shrinks this round; crashing loses the whole run
            print(f"  reflection call failed, skipping this candidate: {error!r}", flush=True)
    return [t for t in dict.fromkeys(texts) if t and t != system_prompt]  # drop empty and duplicate proposals


def optimize(adapter, reflection_lm, train, val, seed_prompt="", rounds=5, k=3,
             minibatch_size=16, regression_penalty=0.0, replay_size=8, promotion_margin=0.02, seed=SEED):
    rng = random.Random(seed)
    baseline = category_accuracy(adapter.evaluate(val, seed_prompt))
    best = {"system_prompt": seed_prompt, "val_accuracy": baseline,
            "score": heuristic_score(baseline, baseline, regression_penalty)}
    history = []
    # Replay buffer of train rollouts only; val rollouts never reach reflection, so selection stays unbiased.
    buffer = []
    for round_idx in range(rounds):
        minibatch = rng.sample(train, min(minibatch_size, len(train)))
        rollouts = adapter.evaluate(minibatch, best["system_prompt"])
        for r in rollouts:
            r["behavior_prompt"] = best["system_prompt"]
        failures = [r for r in rollouts if r["score"] < 1]
        past = [r for r in buffer if r["behavior_prompt"] != best["system_prompt"]]
        replay = rng.sample(past, min(replay_size, len(past)))
        buffer.extend(rollouts)
        entry = {"round": round_idx, "minibatch_failures": len(failures), "replayed": len(replay),
                 "candidates": [], "promoted": False}
        history.append(entry)
        print(f"Round {round_idx}: {len(failures)}/{len(minibatch)} minibatch failures, best score {best['score']:.4f}")
        if not failures:
            continue
        best_correct = sum(r["score"] for r in rollouts)
        print(f"  reflecting on {len(failures)} failures + {len(replay)} replayed rollouts", flush=True)
        for text in propose_candidates(reflection_lm, best["system_prompt"], failures + replay, k):
            # Screen on the same minibatch first; only candidates that beat the current prompt there get a full VAL run.
            print(f"  screening candidate on {len(minibatch)} train questions", flush=True)
            correct = sum(r["score"] for r in adapter.evaluate(minibatch, text))
            if correct <= best_correct:
                entry["candidates"].append({"system_prompt": text, "minibatch_correct": correct, "screened_out": True})
                print(f"  candidate screened out on minibatch ({correct:.0f} vs {best_correct:.0f} correct)")
                continue
            print(f"  passed screen ({correct:.0f} vs {best_correct:.0f}); running {len(val)} val questions", flush=True)
            accuracy = category_accuracy(adapter.evaluate(val, text))
            candidate = {"system_prompt": text, "minibatch_correct": correct, "val_accuracy": accuracy,
                         "score": heuristic_score(accuracy, baseline, regression_penalty)}
            entry["candidates"].append(candidate)
            print(f"  candidate score {candidate['score']:.4f} {accuracy}")
        top = max((c for c in entry["candidates"] if "score" in c), key=lambda c: c["score"], default=None)
        if top and top["score"] > best["score"] + promotion_margin:
            best, entry["promoted"] = top, True
    return best, baseline, history


class _Tee:
    """Writes to the terminal and the log file; flushes each write so a crash keeps the log."""

    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="GLM-5.3")
    parser.add_argument("--reflection-model", help="ARC model that rewrites the prompt; defaults to --model.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=3, help="Prompts proposed per round (K).")
    parser.add_argument("--regression-penalty", type=float, default=0.0,
                        help="Score penalty per point a category drops below the no-prompt baseline.")
    parser.add_argument("--replay-size", type=int, default=8,
                        help="Rollouts from earlier prompts added to each reflection (0 = on-policy only).")
    parser.add_argument("--promotion-margin", type=float, default=0.02,
                        help="Validation score gain needed to replace the best prompt; 0.02 is about 2 more "
                        "correct validation examples in either category.")
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=5, choices=range(1, 11))
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "output" / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "log.txt", "w", encoding="utf-8")
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)

    settings = load_settings()
    configure_bfcl_client(settings)  # must run before bfcl_eval is imported
    # Reflection prompts are long (~16 failure traces), so allow a slow reply but give up after one retry.
    client = create_arc_client(settings).with_options(timeout=600, max_retries=1)
    reflection_model = args.reflection_model or args.model

    def reflection_lm(prompt):
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        # ARC caps non-streaming replies at 8000 tokens; with thinking on, GLM spent them all on hidden reasoning
        # and returned empty content. shortcut: assumes a vLLM-served model, stream with a higher max_tokens if thinking is needed.
        response = client.chat.completions.create(
            model=reflection_model, messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        if not content:  # e.g. a reasoning model that spent its whole reply on hidden thinking
            print(f"  reflection reply was empty (finish_reason={response.choices[0].finish_reason})", flush=True)
        return content or ""

    train, val, test = load_examples()
    print(f"Split sizes: train={len(train)} val={len(val)} test={len(test)}")
    adapter = BFCLSystemPromptAdapter(args.model, args.workers)

    best, baseline_val, history = optimize(
        adapter, reflection_lm, train, val,
        rounds=args.rounds, k=args.candidates,
        minibatch_size=args.minibatch_size, regression_penalty=args.regression_penalty,
        replay_size=args.replay_size, promotion_margin=args.promotion_margin,
    )

    baseline_test = category_accuracy(adapter.evaluate(test, ""))
    optimized_test = category_accuracy(adapter.evaluate(test, best["system_prompt"])) if best["system_prompt"] else baseline_test
    report = {
        "model": args.model,
        "reflection_model": reflection_model,
        "best_system_prompt": best["system_prompt"],
        "val_accuracy": {"baseline": baseline_val, "optimized": best["val_accuracy"]},
        "test_accuracy": {"baseline": baseline_test, "optimized": optimized_test},
        "history": history,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
