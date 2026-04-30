import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

DATA_DIR = Path(__file__).parent
REACTIONS_CSV = DATA_DIR / "data" / "reactions.csv"
SPECIES_CSV = DATA_DIR / "data" / "species.csv"

BATCH_SIZE = 20
DEFAULT_PHENOTYPE = "cardiac hypertrophy"

# Per Tewari et al. 2025 (Methods, "Querying Large Language Models").
# Step 1 (gene list) + Step 2 (batch-1 instruction) are sent as the first
# user message; Step 3 (continuation) is a separate user message per chunk.
INITIAL_PROMPT_TEMPLATE = (
    "List of genes and other signaling nodes: {gene_list}\n\n"
    "For the first {batch_size} entries in this list of genes, proteins, and "
    "other signaling nodes from a {phenotype} network, please provide more "
    "than 0 but fewer than {max_connections} direct interactions with other "
    "nodes from the list supported by available literature. Simply list the "
    "input node, affected node, and if the affected node is stimulated / "
    "inhibited."
)
CONTINUATION_PROMPT_TEMPLATE = (
    "That looks great! Please do the same operation for the next "
    "{chunk_size} nodes! Thank you"
)

STIMULATES = {
    "stimulate",
    "stimulates",
    "stimulated",
    "activate",
    "activates",
    "activated",
    "upregulate",
    "upregulates",
    "upregulated",
    "up-regulate",
    "up-regulates",
    "up-regulated",
    "generate",
    "generates",
    "generated",
}
INHIBITS = {
    "inhibit",
    "inhibits",
    "inhibited",
    "suppress",
    "suppresses",
    "suppressed",
    "downregulate",
    "downregulates",
    "downregulated",
    "down-regulate",
    "down-regulates",
    "down-regulated",
}
VERB_PATTERN = "|".join(
    sorted((re.escape(v) for v in STIMULATES | INHIBITS), key=len, reverse=True)
)
INTERACTION_RE = re.compile(
    rf"(?P<src>[A-Za-z0-9_./+!-]+)\s+"
    rf"(?P<verb>{VERB_PATTERN})\s+"
    rf"(?P<dst>[A-Za-z0-9_./+!-]+)",
    flags=re.IGNORECASE,
)
ARROW_RE = re.compile(
    r"(?P<src>[A-Za-z0-9_./+!-]+)\s*(?P<op>->|=>|=\||\u2192)\s*"
    r"(?P<dst>[A-Za-z0-9_./+!-]+)"
    r"(?P<context>[^\n]{0,120})",
    flags=re.IGNORECASE,
)
FIELD_RE = re.compile(
    r"Input Node:\**\s*(?P<src>[A-Za-z0-9_./+!-]+).*?"
    r"Affected Node:\**\s*(?P<dst>[A-Za-z0-9_./+!-]+).*?"
    r"(?:Stimulation/Inhibition|Effect|Type):\**\s*(?P<effect>[A-Za-z-]+)",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, order=True)
class Reaction:
    source: str
    target: str
    effect: str

    def to_json(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target, "effect": self.effect}


def _read_csv_with_title(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        next(f)
        return list(csv.DictReader(f))


def load_species(path: Path = SPECIES_CSV) -> list[str]:
    rows = _read_csv_with_title(path)
    return [row["ID"].strip() for row in rows if row.get("ID", "").strip()]


def _split_rule_inputs(expression: str) -> Iterable[tuple[str, bool]]:
    for raw_node in re.split(r"\s*&\s*", expression):
        node = raw_node.strip()
        if not node:
            continue
        negated = node.startswith("!")
        yield node[1:].strip() if negated else node, negated


def load_ground_truth(path: Path = REACTIONS_CSV) -> set[Reaction]:
    rows = _read_csv_with_title(path)
    reactions: set[Reaction] = set()
    for row in rows:
        if row.get("module") == "inputs":
            continue
        rule = row.get("Rule", "").strip()
        match = re.fullmatch(r"(.+?)\s*(=>|=\|)\s*(.+)", rule)
        if not match:
            continue

        source_expr, operator, target = match.groups()
        target = target.strip()
        for source, negated in _split_rule_inputs(source_expr):
            if not source:
                continue
            stimulated = operator == "=>"
            effect = "inhibited" if stimulated == negated else "stimulated"
            reactions.add(Reaction(source=source, target=target, effect=effect))
    return reactions


def compute_max_connections(reactions: Iterable[Reaction]) -> int:
    """Maximum out-degree exhibited by any source node in the ground truth.

    Tewari et al. set ``max_connections`` to "the maximum number of connections
    exhibited in that parent network" (Methods); we interpret this as the
    largest number of outgoing signed edges from any single node.
    """
    counts: Counter[str] = Counter()
    for reaction in reactions:
        counts[reaction.source] += 1
    return max(counts.values()) if counts else 0


def make_batches(nodes: Sequence[str], batch_size: int = BATCH_SIZE) -> list[list[str]]:
    return [list(nodes[i : i + batch_size]) for i in range(0, len(nodes), batch_size)]


def make_initial_prompt(
    nodes: Sequence[str],
    batch_size: int,
    phenotype: str,
    max_connections: int,
) -> str:
    return INITIAL_PROMPT_TEMPLATE.format(
        gene_list=", ".join(nodes),
        batch_size=batch_size,
        phenotype=phenotype,
        max_connections=max_connections,
    )


def make_continuation_prompt(chunk_size: int) -> str:
    return CONTINUATION_PROMPT_TEMPLATE.format(chunk_size=chunk_size)


def extract_reactions(text: str, allowed_nodes: Iterable[str] | None = None) -> set[Reaction]:
    allowed = set(allowed_nodes) if allowed_nodes is not None else None
    reactions: set[Reaction] = set()
    for match in INTERACTION_RE.finditer(text):
        source = match.group("src").strip(".,;:()[]{}")
        target = match.group("dst").strip(".,;:()[]{}")
        if allowed is not None and (source not in allowed or target not in allowed):
            continue
        effect = _effect_from_text(match.group("verb"))
        if effect:
            reactions.add(Reaction(source=source, target=target, effect=effect))

    for match in ARROW_RE.finditer(text):
        source = _clean_node(match.group("src"))
        target = _clean_node(match.group("dst"))
        if allowed is not None and (source not in allowed or target not in allowed):
            continue
        effect = _effect_from_arrow(match.group("op"), match.group("context"))
        if effect:
            reactions.add(Reaction(source=source, target=target, effect=effect))

    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        source, target, effect_text = cells[:3]
        if source.lower() in {"input node", "input"}:
            continue
        if allowed is not None and (source not in allowed or target not in allowed):
            continue
        effect = _effect_from_text(effect_text)
        if effect:
            reactions.add(Reaction(source=source, target=target, effect=effect))

    for match in FIELD_RE.finditer(text):
        source = _clean_node(match.group("src"))
        target = _clean_node(match.group("dst"))
        if allowed is not None and (source not in allowed or target not in allowed):
            continue
        effect = _effect_from_text(match.group("effect"))
        if effect:
            reactions.add(Reaction(source=source, target=target, effect=effect))
    return reactions


def _clean_cell(text: str) -> str:
    return _clean_node(re.sub(r"<[^>]+>", "", text))


def _clean_node(text: str) -> str:
    return text.strip().strip("*` .,:;()[]{}")


def _effect_from_text(text: str) -> str | None:
    normalized = text.lower()
    if any(verb in normalized for verb in INHIBITS) or any(
        word in normalized for word in ["inhibition", "repression", "represses"]
    ):
        return "inhibited"
    if any(verb in normalized for verb in STIMULATES) or any(
        word in normalized for word in ["stimulation", "activation"]
    ):
        return "stimulated"
    return None


def _effect_from_arrow(operator: str, context: str) -> str | None:
    if operator == "=|":
        return "inhibited"
    contextual = _effect_from_text(context)
    if contextual:
        return contextual
    if operator in {"=>", "->", "\u2192"}:
        return "stimulated"
    return None


def score_reactions(
    returned: set[Reaction], ground_truth: set[Reaction]
) -> dict[str, float | int | list[dict[str, str]]]:
    true_positives = returned & ground_truth
    recall = len(true_positives) / len(ground_truth) if ground_truth else 0.0
    precision = len(true_positives) / len(returned) if returned else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "returned_count": len(returned),
        "ground_truth_count": len(ground_truth),
        "true_positive_count": len(true_positives),
        "matched": [reaction.to_json() for reaction in sorted(true_positives)],
        "false_positives": [
            reaction.to_json() for reaction in sorted(returned - ground_truth)
        ],
        "false_negatives": [
            reaction.to_json() for reaction in sorted(ground_truth - returned)
        ],
    }


@metric(name="reaction_recall")
def reaction_recall() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        counts = _aggregate_counts(scores)
        return (
            counts["true_positive_count"] / counts["ground_truth_count"]
            if counts["ground_truth_count"]
            else 0.0
        )

    return metric


@metric(name="precision")
def precision() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        counts = _aggregate_counts(scores)
        return (
            counts["true_positive_count"] / counts["returned_count"]
            if counts["returned_count"]
            else 0.0
        )

    return metric


@metric(name="f1")
def f1() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        counts = _aggregate_counts(scores)
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
        return (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

    return metric


def _aggregate_counts(scores: Sequence[SampleScore]) -> dict[str, int]:
    counts = {
        "true_positive_count": 0,
        "returned_count": 0,
        "ground_truth_count": 0,
    }
    for sample_score in scores:
        metadata = sample_score.score.metadata
        if not isinstance(metadata, dict):
            continue
        for key in counts:
            value = metadata.get(key)
            if isinstance(value, int):
                counts[key] += value
    return counts


def transcript_text(state: TaskState) -> str:
    """Concatenate every assistant turn from the conversation."""
    parts: list[str] = []
    for message in state.messages:
        if isinstance(message, ChatMessageAssistant):
            text = getattr(message, "text", None) or ""
            if text:
                parts.append(text)
    return "\n\n".join(parts)


@solver
def reconbench_solver() -> Solver:
    """Three-step iterative prompt from Tewari et al. (2025).

    Sends the gene list + first-batch instruction as one user message, then
    issues the continuation prompt once per remaining chunk, generating after
    each turn so the assistant accumulates outputs across the conversation.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        chunks = state.metadata.get("chunks") or []
        await generate(state)
        for chunk in chunks[1:]:
            state.messages.append(
                ChatMessageUser(content=make_continuation_prompt(len(chunk)))
            )
            await generate(state)
        return state

    return solve


@scorer(metrics=[reaction_recall(), precision(), f1()])
def reconbench_scorer() -> Scorer:
    ground_truth = load_ground_truth()

    async def score(state: TaskState, target: Target) -> Score:
        nodes = state.metadata["nodes"]
        text = transcript_text(state)
        returned = extract_reactions(text, nodes)
        value = score_reactions(returned, ground_truth)
        return Score(
            value={
                "recall": value["recall"],
                "precision": value["precision"],
                "f1": value["f1"],
            },
            answer=json.dumps([r.to_json() for r in sorted(returned)]),
            metadata=value,
        )

    return score


@task
def reconbench(
    phenotype: str = DEFAULT_PHENOTYPE,
    batch_size: int = BATCH_SIZE,
) -> Task:
    nodes = load_species()
    ground_truth = load_ground_truth()
    chunks = make_batches(nodes, batch_size=batch_size)
    max_connections = compute_max_connections(ground_truth)
    initial_prompt = make_initial_prompt(
        nodes=nodes,
        batch_size=len(chunks[0]) if chunks else batch_size,
        phenotype=phenotype,
        max_connections=max_connections,
    )
    sample = Sample(
        id=phenotype.replace(" ", "_"),
        input=initial_prompt,
        target="",
        metadata={
            "nodes": nodes,
            "chunks": chunks,
            "phenotype": phenotype,
            "max_connections": max_connections,
        },
    )
    return Task(
        dataset=MemoryDataset([sample]),
        solver=reconbench_solver(),
        scorer=reconbench_scorer(),
    )
