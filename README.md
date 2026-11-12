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

Run the default categories (`simple_python`, `multiple`, `parallel`, and
`irrelevance`):

```bash
.venv/bin/python main.py --model gpt-oss-120b --workers 5
```

Choose categories or another ARC model as needed:

```bash
.venv/bin/python main.py \
  --model DeepSeek-V4.1-Flash \
  --categories simple_python multiple parallel irrelevance \
  --workers 5
```

The supported model names can change; consult the
[ARC LLM API documentation](https://www.docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html).
The evaluator limits workers to 10 to stay within ARC's per-model concurrency
limit.

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
