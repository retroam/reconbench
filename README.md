Here's a clean README:

---

# ReconBench

ReconBench measures whether a frontier model can reconstruct a cardiac signaling network from a gene list alone — no pathway descriptions, no interaction hints. It replicates the bare-model baseline from Tewari et al. (2025) and extends it with precision and F1 scoring.

## Methodology

Faithful to Tewari et al. (2025) Methods, "Querying Large Language Models":

- **One sample per network.** The full gene list is provided once per run.
- **Three-step iterative prompt** in a single multi-turn conversation:
  1. `List of genes and other signaling nodes: {gene_list}`
  2. `For the first {batch_size} entries ... please provide more than 0 but fewer than {max_connections} direct interactions ...`
  3. `That looks great! Please do the same operation for the next {chunk_size} nodes! Thank you` — repeated for each remaining 20-node chunk.
- `max_connections` is the maximum out-degree of any source node in the manually-curated network.
- Reactions are extracted from the **entire transcript** (all assistant turns) and compared to the **full** ground-truth reaction list.

## Data

Two files are required, exported from `Ryall2012_cardiomyocyte_hypertrophy.xlsx` (available at [saucermanlab/Netflux2](https://github.com/saucermanlab/Netflux2/blob/main/models/Ryall2012_cardiomyocyte_hypertrophy.xlsx)):

- `reactions.csv` — ground truth reactions
- `species.csv` — gene symbols used as model input

## Running

Each Inspect epoch is one full multi-turn run; use `--epochs 10` for the paper's 10-runs-per-network protocol:

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

## Roadmap

**Phase 1 (current): Bare-model baseline**
Replicate and extend Tewari et al. (2025) with precision and F1 scoring across current model versions.

**Phase 2: Tool-augmented agents**
Give agents access to biological databases (KEGG, STRING, Reactome) and measure whether tool use closes the reconstruction gap.