Here's a clean README:

---

# ReconBench

ReconBench measures whether a frontier model can reconstruct a cardiac signaling network from a gene list alone — no pathway descriptions, no interaction hints. It replicates the bare-model baseline from Tewari et al. (2025) and extends it with precision and F1 scoring.

## Data

Two files are required, exported from `Ryall2012_cardiomyocyte_hypertrophy.xlsx` (available at [saucermanlab/Netflux2](https://github.com/saucermanlab/Netflux2/blob/main/models/Ryall2012_cardiomyocyte_hypertrophy.xlsx)):

- `reactions.csv` — ground truth reactions
- `species.csv` — gene symbols used as model input

## Running

Run 10 epochs per model:

```bash
uv run --project inspect_evals inspect eval reconbench/reconbench.py@reconbench --model openai/gpt-4.1 --epochs 10 --log-dir reconbench/logs/gpt-4.1
uv run --project inspect_evals inspect eval reconbench/reconbench.py@reconbench --model anthropic/claude-3-7-sonnet-latest --epochs 10 --log-dir reconbench/logs/claude
uv run --project inspect_evals inspect eval reconbench/reconbench.py@reconbench --model google/gemini-2.0-flash --epochs 10 --log-dir reconbench/logs/gemini-2.0
```

## Reporting

```bash
uv run --project inspect_evals python reconbench/report.py reconbench/logs
```

Prints per-model recall, precision, and F1 across all runs.

## Baseline

From Tewari et al. (2025), bare-model reconstruction recall on the hypertrophy network:

| Model | Recall |
|---|---|
| GPT-4.1 | ~45% |
| Claude 3.7 | ~58% |
| Gemini 2.0 | ~27% |

Precision and F1 are not reported in the paper; ReconBench adds both.

