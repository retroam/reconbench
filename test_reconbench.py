from reconbench.reconbench import (
    BATCH_SIZE,
    DEFAULT_PHENOTYPE,
    Reaction,
    compute_max_connections,
    extract_reactions,
    extract_structured_reactions,
    load_ground_truth,
    load_species,
    make_batches,
    make_continuation_prompt,
    make_initial_prompt,
    reconbench,
    score_reactions,
)


def test_load_species() -> None:
    species = load_species()
    assert len(species) == 106
    assert species[:3] == ["aAR", "AC", "Akt"]


def test_load_ground_truth_expands_boolean_rules() -> None:
    ground_truth = load_ground_truth()
    assert Reaction("Akt", "foxo", "inhibited") in ground_truth
    assert Reaction("DAG", "PKC", "stimulated") in ground_truth
    assert Reaction("Calcium", "PKC", "stimulated") in ground_truth
    assert Reaction("IKK", "IkB", "inhibited") in ground_truth


def test_make_batches() -> None:
    batches = make_batches(load_species())
    assert len(batches) == 6
    assert len(batches[0]) == 20
    assert len(batches[-1]) == 6


def test_compute_max_connections() -> None:
    reactions = {
        Reaction("A", "B", "stimulated"),
        Reaction("A", "C", "stimulated"),
        Reaction("A", "D", "inhibited"),
        Reaction("B", "C", "stimulated"),
    }
    assert compute_max_connections(reactions) == 3
    assert compute_max_connections(set()) == 0


def test_compute_max_connections_on_real_data() -> None:
    assert compute_max_connections(load_ground_truth()) > 0


def test_make_initial_prompt_contains_paper_phrasing() -> None:
    prompt = make_initial_prompt(
        nodes=["A", "B", "C"],
        batch_size=2,
        phenotype="cardiac hypertrophy",
        max_connections=7,
    )
    assert "List of genes and other signaling nodes:" in prompt
    assert "A, B, C" in prompt
    assert "For the first 2 entries" in prompt
    assert "cardiac hypertrophy" in prompt
    assert "fewer than 7 direct interactions" in prompt
    assert "stimulated / inhibited" in prompt


def test_make_continuation_prompt_uses_paper_phrasing() -> None:
    prompt = make_continuation_prompt(6)
    assert prompt == (
        "That looks great! Please do the same operation for the next "
        "6 nodes! Thank you"
    )


def test_make_structured_continuation_prompt_repeats_json_instruction() -> None:
    prompt = make_continuation_prompt(6, structured_output=True)
    assert prompt.startswith(
        "That looks great! Please do the same operation for the next "
        "6 nodes! Thank you"
    )
    assert "Return only valid JSON" in prompt


def test_extract_reactions_with_synonyms_and_node_filter() -> None:
    text = """
    EGF activates EGFR.
    PI3K downregulates GSK3B.
    Outside activates EGFR.
    Akt up-regulates mTOR.
    BAR → AC: Stimulation
    | Calcium | CaM | Activation |
    **Input Node:** CaM
    *   **Affected Node:** CaN
    *   **Stimulation/Inhibition:** Stimulation
    """
    reactions = extract_reactions(
        text,
        allowed_nodes={
            "EGF",
            "EGFR",
            "PI3K",
            "GSK3B",
            "Akt",
            "mTOR",
            "BAR",
            "AC",
            "Calcium",
            "CaM",
            "CaN",
        },
    )
    assert reactions == {
        Reaction("EGF", "EGFR", "stimulated"),
        Reaction("PI3K", "GSK3B", "inhibited"),
        Reaction("Akt", "mTOR", "stimulated"),
        Reaction("BAR", "AC", "stimulated"),
        Reaction("Calcium", "CaM", "stimulated"),
        Reaction("CaM", "CaN", "stimulated"),
    }


def test_extract_reactions_handles_arrow_operators() -> None:
    reactions = extract_reactions(
        "A => B\nC =| D\nE -> F",
        allowed_nodes={"A", "B", "C", "D", "E", "F"},
    )
    assert Reaction("A", "B", "stimulated") in reactions
    assert Reaction("C", "D", "inhibited") in reactions
    assert Reaction("E", "F", "stimulated") in reactions


def test_extract_reactions_handles_csv_tuples() -> None:
    reactions = extract_reactions(
        """
        source, target, effect
        ADRB2, RAC1, Stimulated
        ADRB2, ADCY, Stimulated
        RAC1, MAP3K1, Inhibited
        Outside, RAC1, Stimulated
        """,
        allowed_nodes={"ADRB2", "RAC1", "ADCY", "MAP3K1"},
    )
    assert reactions == {
        Reaction("ADRB2", "RAC1", "stimulated"),
        Reaction("ADRB2", "ADCY", "stimulated"),
        Reaction("RAC1", "MAP3K1", "inhibited"),
    }


def test_extract_structured_reactions_handles_json_object_and_filters_nodes() -> None:
    reactions = extract_structured_reactions(
        '{"reactions":['
        '{"source":"ADRB2","target":"RAC1","effect":"stimulated"},'
        '{"source":"RAC1","target":"MAP3K1","effect":"inhibited"},'
        '{"source":"Outside","target":"MAP3K1","effect":"stimulated"}'
        "]}",
        allowed_nodes={"ADRB2", "RAC1", "MAP3K1"},
    )
    assert reactions == {
        Reaction("ADRB2", "RAC1", "stimulated"),
        Reaction("RAC1", "MAP3K1", "inhibited"),
    }


def test_extract_structured_reactions_handles_json_array_in_fence() -> None:
    reactions = extract_structured_reactions(
        """```json
        [{"source":"A","target":"B","effect":"activation"}]
        ```""",
        allowed_nodes={"A", "B"},
    )
    assert reactions == {Reaction("A", "B", "stimulated")}


def test_score_reactions() -> None:
    ground_truth = {
        Reaction("A", "B", "stimulated"),
        Reaction("C", "D", "inhibited"),
    }
    returned = {
        Reaction("A", "B", "stimulated"),
        Reaction("X", "Y", "stimulated"),
    }
    score = score_reactions(returned, ground_truth)
    assert score["recall"] == 0.5
    assert score["precision"] == 0.5
    assert score["f1"] == 0.5


def test_task_constructs_single_full_network_sample() -> None:
    task = reconbench()
    samples = list(task.dataset)
    assert len(samples) == 1
    sample = samples[0]
    metadata = sample.metadata or {}
    assert metadata["phenotype"] == DEFAULT_PHENOTYPE
    assert len(metadata["nodes"]) == 106
    chunks = metadata["chunks"]
    assert len(chunks) == 6
    assert len(chunks[0]) == BATCH_SIZE
    assert metadata["max_connections"] > 0
    assert "List of genes and other signaling nodes:" in sample.input
    assert f"fewer than {metadata['max_connections']}" in sample.input


def test_task_can_request_structured_output() -> None:
    task = reconbench(output_format="structured")
    sample = list(task.dataset)[0]
    metadata = sample.metadata or {}
    assert metadata["output_format"] == "structured"
    assert metadata["condition"] == "structured"
    assert "Return only valid JSON" in sample.input
    assert '"reactions"' in sample.input
