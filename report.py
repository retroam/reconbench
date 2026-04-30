import argparse
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

try:
    from reconbench.reconbench import (
        Reaction,
        extract_reactions,
        load_ground_truth,
        score_reactions,
    )
except ModuleNotFoundError:
    from reconbench import (
        Reaction,
        extract_reactions,
        load_ground_truth,
        score_reactions,
    )

BASELINE_NOTE = (
    "Tewari et al. report 26.70-58.12% cardiomyocyte hypertrophy reaction "
    "reconstruction across GPT 4.1, Gemini 2.0, and Claude 3.7. The supplied "
    "baseline text does not include per-model precision or F1 baselines."
)


def _assistant_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages or []:
        if getattr(message, "role", None) != "assistant":
            continue
        text = getattr(message, "text", None) or ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _epoch_counts(log: EvalLog) -> list[dict[str, int]]:
    ground_truth = load_ground_truth()
    returned_by_run: dict[tuple[str, int], set[Reaction]] = {}
    samples = log.samples or []
    for sample in samples:
        epoch = getattr(sample, "epoch", 1) or 1
        sample_id = str(getattr(sample, "id", "sample"))
        metadata = sample.metadata or {}
        nodes = metadata.get("nodes")
        if not isinstance(nodes, list):
            continue
        text = _assistant_text(getattr(sample, "messages", []) or [])
        if not text:
            output = getattr(sample, "output", None)
            text = getattr(output, "completion", "") if output else ""
        if not text:
            continue
        key = (sample_id, epoch)
        returned = returned_by_run.setdefault(key, set())
        returned.update(extract_reactions(text, nodes))

    counts = []
    for returned in returned_by_run.values():
        scored = score_reactions(returned, ground_truth)
        counts.append(
            {
                "true_positive_count": int(scored["true_positive_count"]),
                "returned_count": int(scored["returned_count"]),
                "ground_truth_count": int(scored["ground_truth_count"]),
            }
        )
    return counts


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    recall = (
        counts["true_positive_count"] / counts["ground_truth_count"]
        if counts["ground_truth_count"]
        else 0.0
    )
    precision = (
        counts["true_positive_count"] / counts["returned_count"]
        if counts["returned_count"]
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {"recall": recall, "precision": precision, "f1": f1}


def _model_name(log: EvalLog) -> str:
    eval_config = getattr(log, "eval", None)
    model = getattr(eval_config, "model", None)
    return str(model) if model else "unknown"


def summarize(log_dir: Path) -> list[dict[str, Any]]:
    rows_by_model: dict[str, dict[str, Any]] = {}
    for log_ref in list_eval_logs(str(log_dir)):
        log = read_eval_log(log_ref)
        if getattr(log.eval, "task", None) != "reconbench":
            continue
        model = _model_name(log)
        model_row = rows_by_model.setdefault(
            model,
            {
                "model": model,
                "runs": 0,
                "true_positive_count": 0,
                "returned_count": 0,
                "ground_truth_count": 0,
            },
        )
        for counts in _epoch_counts(log):
            model_row["runs"] += 1
            for key, value in counts.items():
                model_row[key] += value

    rows = []
    for row in rows_by_model.values():
        metrics = _metrics(row)
        rows.append({**row, **metrics})
    return sorted(rows, key=lambda row: row["model"])


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| Model | Runs | Recall | Precision | F1 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['model']} | {row['runs']} | "
            f"{row['recall']:.2%} | {row['precision']:.2%} | {row['f1']:.2%} |"
        )
    print()
    print(f"Baseline comparison: {BASELINE_NOTE}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path, help="Inspect log directory")
    args = parser.parse_args()
    print_markdown(summarize(args.log_dir))


if __name__ == "__main__":
    main()
