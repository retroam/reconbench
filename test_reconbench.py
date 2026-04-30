from reconbench.reconbench import (
    Reaction,
    extract_reactions,
    load_ground_truth,
    load_species,
    make_batches,
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
