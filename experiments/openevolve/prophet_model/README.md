# OpenEvolve Prophet Model Optimization

This experiment evolves the Prophet model recipe itself rather than a
post-processing function. The evolved code can change hyperparameters,
engineered regressors, custom seasonalities, and target transformations.
The initial recipe explicitly uses `stan_backend="CMDSTANPY"` for reliable
Windows evaluation.
Candidate programs must use the original input columns `ds`, `y`,
`is_monsoon`, `typhoon_index`, `hour`, `day_of_week`, and `month`; any other
regressor columns must be created in `prepare_features()` before being listed
by `candidate_regressors()`.

The evaluator returns both:

- `penalty_score`: lower is better
- `combined_score`: higher is better, computed as `1 / (1 + penalty_score)`

OpenEvolve should maximize `combined_score`.

## vLLM endpoint

Run an OpenAI-compatible vLLM server first. Example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --port 8000
```

Then set any dummy API key expected by the OpenAI SDK:

```bash
export OPENAI_API_KEY=local-vllm
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "local-vllm"
```

The experiment config points OpenEvolve at:

```yaml
llm:
  models:
    - name: "YOUR_MODEL_NAME"
      api_base: "http://localhost:8000/v1"
      api_key: "dummy"
      weight: 1.0
  temperature: 0.4
  max_tokens: 4096
  timeout: 900
```

Set every `llm.models[].name` value in `config.yaml` to exactly the model name served by the OpenAI-compatible endpoint.
For private endpoints or real API keys, copy `config.yaml` to `config.local.yaml`
and edit that file. `config.local.yaml` is git-ignored.

## Recommended search profile

For the local H100 + Intel Ultra 7 setup, keep the holdout window at one full
year and reduce only the training window during exploratory search. This keeps
yearly seasonality and event coverage in the evaluation while making each
candidate faster to train.

PowerShell:

```powershell
$env:OPENEVOLVE_TRAIN_TAIL_ROWS = "17520"
$env:OPENEVOLVE_HOLDOUT_HEAD_ROWS = "8760"
```

Bash:

```bash
export OPENEVOLVE_TRAIN_TAIL_ROWS=17520
export OPENEVOLVE_HOLDOUT_HEAD_ROWS=8760
```

This profile means:

- train: recent 2 years
- holdout: full 1 year
- evaluator parallelism: 3 candidates

The default config uses:

```yaml
evaluator:
  timeout: 300
  parallel_evaluations: 3
```

For final validation, remove the row limits and re-run the best candidate on the
full training period and full holdout.

PowerShell:

```powershell
Remove-Item Env:OPENEVOLVE_TRAIN_TAIL_ROWS
Remove-Item Env:OPENEVOLVE_HOLDOUT_HEAD_ROWS
```

Bash:

```bash
unset OPENEVOLVE_TRAIN_TAIL_ROWS
unset OPENEVOLVE_HOLDOUT_HEAD_ROWS
```

## Quick row-limit reference

```bash
set OPENEVOLVE_TRAIN_TAIL_ROWS=8760
set OPENEVOLVE_HOLDOUT_HEAD_ROWS=720
```

Example command after installing OpenEvolve and configuring an LLM endpoint:

```bash
python openevolve-run.py experiments/openevolve/prophet_model/initial_program.py experiments/openevolve/prophet_model/evaluator.py --config experiments/openevolve/prophet_model/config.yaml --iterations 40
```

For a quick LLM/evaluator smoke run, start with fewer iterations:

```bash
python openevolve-run.py experiments/openevolve/prophet_model/initial_program.py experiments/openevolve/prophet_model/evaluator.py --config experiments/openevolve/prophet_model/config.yaml --iterations 5
```

This config currently uses:

```yaml
diff_based_evolution: false
```

This avoids repeated `No valid diffs found in response` failures from models
that do not reliably emit unified diffs.

If `/v1/chat/completions` times out, reduce `llm.max_tokens` before increasing
iteration count.
