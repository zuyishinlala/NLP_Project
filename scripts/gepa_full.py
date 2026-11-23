"""Optimize a system prompt on BFCL multi_turn_base + irrelevance with the full GEPA search.

Same data split, BFCL scoring, reflection call, and final test comparison as
scripts/gepa_optimize.py, but the search is gepa.optimize(): a Pareto set of
prompts (any prompt that is best on some validation question is kept), parents
sampled from that set, minibatch acceptance, and merging of two prompts. It runs
until --budget rollouts are used and can resume after a crash.

Run: .venv/bin/python -m scripts.gepa_full --model GLM-5.3
Writes output/<model>-gepa/report.json and gepa_state/ (for resuming).
"""

import argparse
import json

import gepa
from gepa.core.adapter import EvaluationBatch

from core.config import PROJECT_ROOT, load_settings
from llm.client import configure_bfcl_client, create_arc_client
from scripts.gepa_optimize import (
    GENERAL_RULES_PROMPT,
    SEED,
    BFCLSystemPromptAdapter,
    category_accuracy,
    load_examples,
    make_reflection_lm,
)

COMPONENT = "system_prompt"


class BFCLGEPAAdapter:
    """Plugs the BFCL rollouts into gepa.optimize()."""

    # None = use GEPA's own proposer; without this attribute GEPA logs an error every step and never proposes.
    propose_new_texts = None

    def __init__(self, bfcl):
        self.bfcl = bfcl

    def evaluate(self, batch, candidate, capture_traces=False):
        rollouts = self.bfcl.evaluate(batch, candidate[COMPONENT])
        return EvaluationBatch(
            outputs=rollouts,
            scores=[r["score"] for r in rollouts],
            trajectories=rollouts if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        # Like GEPA's default adapter, show successes too; each record's Feedback says right or wrong.
        records = [r["record"] for r in eval_batch.trajectories]
        return {name: records for name in components_to_update}


def run_gepa(bfcl, reflection_lm, train, val, budget, minibatch_size, use_merge=True, run_dir=None, seed=SEED):
    """Returns (best prompt, per-category val accuracy of the seed and of the best, gepa result)."""
    result = gepa.optimize(
        seed_candidate={COMPONENT: ""},
        trainset=train,
        valset=val,
        adapter=BFCLGEPAAdapter(bfcl),
        reflection_lm=reflection_lm,
        reflection_prompt_template=GENERAL_RULES_PROMPT,
        candidate_selection_strategy="pareto",
        use_merge=use_merge,
        reflection_minibatch_size=minibatch_size,
        max_metric_calls=budget,
        run_dir=run_dir,
        seed=seed,
    )

    def val_accuracy(index):
        # val_subscores maps each validation example's position to its 0/1 score.
        scores = result.val_subscores[index]
        return category_accuracy([{"category": val[i]["category"], "score": scores[i]} for i in scores])

    return result.best_candidate[COMPONENT], val_accuracy(0), val_accuracy(result.best_idx), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="GLM-5.3")
    parser.add_argument("--reflection-model", help="ARC model that rewrites the prompt; defaults to --model.")
    parser.add_argument("--budget", type=int, default=1300,
                        help="Total rollouts GEPA may use; ~1300 matches one default gepa_optimize run.")
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--no-merge", action="store_true", help="Turn off merging of two prompts.")
    parser.add_argument("--workers", type=int, default=5, choices=range(1, 11))
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "output" / f"{args.model}-gepa"
    run_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    configure_bfcl_client(settings)  # must run before bfcl_eval is imported
    client = create_arc_client(settings).with_options(timeout=600, max_retries=1)
    reflection_model = args.reflection_model or args.model

    train, val, test = load_examples()
    print(f"Split sizes: train={len(train)} val={len(val)} test={len(test)}")
    bfcl = BFCLSystemPromptAdapter(args.model, args.workers)

    best, baseline_val, best_val, result = run_gepa(
        bfcl, make_reflection_lm(client, reflection_model), train, val,
        budget=args.budget, minibatch_size=args.minibatch_size,
        use_merge=not args.no_merge, run_dir=str(run_dir / "gepa_state"),
    )

    baseline_test = category_accuracy(bfcl.evaluate(test, ""))
    optimized_test = category_accuracy(bfcl.evaluate(test, best)) if best else baseline_test
    report = {
        "model": args.model,
        "reflection_model": reflection_model,
        "search": "gepa.optimize (pareto, merge=%s)" % (not args.no_merge),
        "best_system_prompt": best,
        "val_accuracy": {"baseline": baseline_val, "optimized": best_val},
        "test_accuracy": {"baseline": baseline_test, "optimized": optimized_test},
        "rollouts_used": result.total_metric_calls,
        "full_val_evals": result.num_full_val_evals,
        "candidates": [
            {"system_prompt": c[COMPONENT], "val_score": score, "parents": parents}
            for c, score, parents in zip(result.candidates, result.val_aggregate_scores, result.parents)
        ],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
