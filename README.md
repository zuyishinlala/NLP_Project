# ARC BFCL Evaluation

Run the official BFCL evaluator with a model hosted by Virginia Tech ARC.

> **Dataset version:** The evaluator in this project uses the dataset packaged
> with `bfcl-eval` (BFCL v4). The `bfcl_dataset/` directory contains BFCL's
> upstream dataset files; it is not a project-authored dataset, and the v4
> evaluator does not read those local files.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a personal API key at [llm.arc.vt.edu](https://llm.arc.vt.edu) under
**User profile → Settings → Account → API keys**. Add it to a `.env` file in
the project root:

```dotenv
ARC_LLM_API_KEY=your-personal-arc-api-key
```

Keep `.env` private. It is ignored by Git; do not commit or share the key.

## Run the evaluation

Run a limited number of examples per selected category:

```bash
.venv/bin/python main.py --samples 1 --model gpt-oss-120b --workers 1
```

For example, `--samples 3` selects up to three examples from each selected
category. If a category contains fewer examples, the runner uses all available
examples; it never indexes beyond the category's data. Omit `--samples` to run
all examples in each selected category. Repeating the same partial run reuses
already generated responses; add `--overwrite` to generate them again.
ARC HTTP 403 errors stop generation and are raised instead of being recorded as
model results; other inference failures retain BFCL's per-example handling.

Run the default categories from the bundled BFCL v4 dataset files under
`data/` (all available categories unless you narrow them with `--categories`):

```bash
.venv/bin/python main.py --model gpt-oss-120b --workers 5
```

Choose a subset or another ARC model as needed:

```bash
.venv/bin/python main.py \
  --model DeepSeek-V4.1-Flash \
  --categories live_simple memory web_search \
  --workers 5
```

Evaluate all three example ARC models sequentially on the default categories:

```bash
.venv/bin/python scripts/evaluate_models.py
```

Run all three models on one category at a time:

```bash
.venv/bin/python scripts/evaluate_simple_python.py
.venv/bin/python scripts/evaluate_multiple.py
.venv/bin/python scripts/evaluate_parallel.py
.venv/bin/python scripts/evaluate_irrelevance.py
.venv/bin/python scripts/evaluate_multi_turn_base.py
.venv/bin/python scripts/evaluate_multi_turn_long_context.py
.venv/bin/python scripts/evaluate_multi_turn_miss_func.py
.venv/bin/python scripts/evaluate_multi_turn_miss_param.py
```

The supported model names can change; consult the
[ARC LLM API documentation](https://www.docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html).
The evaluator limits workers to 10 to stay within ARC's per-model concurrency
limit.

## Current results

The following BFCL v4 non-live scores are from the saved evaluation outputs.
Each category score covers the full category: 400 `simple_python`, 200
`multiple`, 200 `parallel`, and 240 `irrelevance` examples per model. The
scores are copied from BFCL's generated `data_non_live.csv`.

| Rank | Model | Simple Python | Multiple | Parallel | Irrelevance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | ARC GLM-5.3 (FC) | 95.00% | 95.00% | 92.00% | 69.58% |
| 2 | ARC DeepSeek-V4.1-Flash (FC) | 94.25% | 92.50% | 91.50% | 72.92% |
| 3 | ARC gpt-oss-120b (FC) | 88.00% | 84.50% | 0.00% | 87.08% |

These are a snapshot of the local results and may change after a new evaluation.

### Multi-turn results

The following BFCL v4 multi-turn scores are from
[data_multi_turn.csv](./bfcl_official_results/score/data_multi_turn.csv).

| Rank | Model | Base | Missing function | Missing parameter | Long context |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | ARC GLM-5.3 (FC) | 73.00% | 56.00% | 57.00% | 65.00% |
| 2 | ARC gpt-oss-120b (FC) | 64.00% | 52.50% | 51.00% | 54.50% |
| 3 | ARC DeepSeek-V4.1-Flash (FC) | 65.00% | 23.50% | 50.50% | 58.00% |

### Use a custom dataset

Pass `--dataset-dir` with a BFCL v4 data directory. It must contain prompt
files named `BFCL_v4_<category>.json` and, for scored categories other than
`irrelevance`, matching answer files under `possible_answer/`:

```text
/path/to/custom-bfcl-data/
├── BFCL_v4_simple_python.json
├── BFCL_v4_multiple.json
└── possible_answer/
    ├── BFCL_v4_simple_python.json
    └── BFCL_v4_multiple.json
```

For example:

```bash
.venv/bin/python main.py \
  --dataset-dir /path/to/custom-bfcl-data \
  --categories simple_python multiple \
  --model gpt-oss-120b
```

The custom data and answer files must follow the installed BFCL evaluator's
v4 schema. The local BFCL v3 files are not directly compatible with this v4
runner.

BFCL writes generated responses and scores to `bfcl_official_results/`. Small
sample runs can produce official per-category scores, but BFCL may be unable to
calculate leaderboard latency statistics when a category has only one sample.

## Optimize a system prompt by reflection

`scripts/gepa_optimize.py` improves a system prompt for `multi_turn_base` and
`irrelevance`. Each category is split with a fixed seed into 40% train, 20%
validation, and 40% held-out test. Each round runs the current best prompt on a
train minibatch, gives the failure traces and BFCL's checker messages to a
reflection model, which proposes K new prompts (using
[GEPA](https://github.com/gepa-ai/gepa)'s reflection template). Every train
rollout goes into a replay buffer tagged with the prompt that produced it, and
each reflection also sees `--replay-size` rollouts from earlier prompts, so the
proposer learns from off-policy data as well as the current prompt's failures
(`--replay-size 0` turns this off). Each candidate is first rerun on the same
train minibatch, and only candidates that get more examples right than the
current prompt are scored on validation, which skips some of the expensive
validation runs. The validation score is the mean per-category accuracy, minus
an optional penalty for each category that drops below the no-prompt baseline.
A candidate replaces the best prompt only if it beats it by more than
`--promotion-margin` (default 0.02, about 2 more correct validation examples in
either category), so small-sample noise is not promoted.

```bash
.venv/bin/python -m scripts.gepa_optimize --model GLM-5.3 --rounds 5 --candidates 3 --regression-penalty 0.5
```

The held-out test split is evaluated only once, at the end, for both the
best prompt and the no-prompt baseline. The results, including every round's
candidates, go to `bfcl_official_results/gepa/<model>/report.json`. A custom
system prompt is not the official BFCL setup, so report these scores separately
from the leaderboard numbers above.

### Optimization results

GLM-5.3 as both the target and the reflection model, with the command above (5
rounds, 3 candidates, replay size 8, regression penalty 0.5). Test is the
held-out split: 80 `multi_turn_base` and 96 `irrelevance` examples.

| Split | Prompt | Multi-turn base | Irrelevance | Mean |
| --- | --- | ---: | ---: | ---: |
| Validation | None | 75.00% | 72.92% | 73.96% |
| Validation | Optimized | 80.00% | 89.58% | 84.79% |
| Test | None | 75.00% | 75.00% | 75.00% |
| Test | Optimized | 78.75% | 90.62% | 84.69% |

The irrelevance gain (+15.6 points, 15 more correct test examples) is well
beyond sampling noise. The multi-turn gain (+3.75 points, 3 examples) is within
it. The best prompt was found in round 1. Later candidates grew longer and
traded multi-turn accuracy for irrelevance accuracy, and none beat it.

## Test a direct ARC chat request

The standalone API example uses the same `.env` configuration:

```bash
.venv/bin/python arc_llm_example.py
```

## Project files

```text
.
├── .env                       # Local ARC API key; never commit
├── main.py                    # BFCL CLI entry point
├── core/
│   └── config.py               # Loads and validates application settings
├── llm/
│   ├── client.py               # Creates ARC client; configures BFCL's API client
│   └── schemas.py              # Validated evaluation options and categories
├── evaluators/
│   └── bfcl_runner.py          # Adapts the official BFCL runner to ARC
├── arc_llm_example.py          # Minimal direct ARC API example
├── requirements.txt            # Python dependencies
├── bfcl_dataset/               # Upstream BFCL dataset files; not read by this v4 runner
└── bfcl_official_results/      # Generated BFCL v4 results and scores
```

BFCL's OpenAI-compatible handler reads credentials from variables named
`OPENAI_API_KEY` and `OPENAI_BASE_URL`. The project sets these at runtime using
the ARC key and ARC endpoint; no OpenAI account or key is used.
