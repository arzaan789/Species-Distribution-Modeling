from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from provenance_sdm.config import load_study_config


STUDY_CONFIG = Path(__file__).parents[1] / "config" / "study.yaml"


@pytest.fixture
def valid_payload() -> dict[str, object]:
    return yaml.safe_load(STUDY_CONFIG.read_text(encoding="utf-8"))


def write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_primary_simulation_design_is_frozen() -> None:
    config = load_study_config(STUDY_CONFIG)

    assert config.simulation.n_species == 200
    assert config.simulation.n_communities == 3
    assert config.simulation.n_taxonomic_groups == 10
    assert config.simulation.n_programmes == 6
    assert config.simulation.alignments == ("low", "partial", "high")
    assert config.simulation.bias_levels == ("moderate", "strong")
    assert (config.simulation.min_records, config.simulation.max_records) == (
        20,
        2_000,
    )
    assert config.simulation.background_cells == 10_000
    assert config.background_arms == (
        "uniform",
        "conventional_tgb",
        "pm_tgb",
        "oracle_effort",
    )


def test_empirical_scope_has_exact_non_bat_species_and_target_groups() -> None:
    config = load_study_config(STUDY_CONFIG)

    actual = {
        species.scientific_name: species.target_group
        for species in config.empirical_species
    }
    assert actual == {
        "Lepus europaeus": ("Lepus timidus", "Oryctolagus cuniculus"),
        "Muscardinus avellanarius": (
            "Apodemus flavicollis",
            "Apodemus sylvaticus",
            "Eliomys quercinus",
        ),
        "Erinaceus europaeus": (
            "Apodemus sylvaticus",
            "Sorex araneus",
            "Talpa europaea",
        ),
        "Sciurus vulgaris": (
            "Glis glis",
            "Muscardinus avellanarius",
            "Tamias sibiricus",
        ),
    }
    names = " ".join(
        name
        for species in config.empirical_species
        for name in (species.key, species.scientific_name, *species.target_group)
    ).lower()
    assert "bat" not in names
    assert "pipistrell" not in names


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("n_species", 0),
        ("n_communities", -1),
        ("n_taxonomic_groups", 0),
        ("n_programmes", 0),
        ("min_records", 0),
        ("max_records", -1),
        ("background_cells", 0),
    ],
)
def test_non_positive_simulation_counts_are_rejected(
    tmp_path: Path,
    valid_payload: dict[str, object],
    field: str,
    invalid_value: int,
) -> None:
    payload = deepcopy(valid_payload)
    payload["simulation"][field] = invalid_value

    with pytest.raises(ValueError, match=field):
        load_study_config(write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "invalid_value", "message"),
    [
        (("simulation", "alignments"), ["low", "unexpected"], "alignments"),
        (("simulation", "bias_levels"), ["moderate", "extreme"], "bias_levels"),
        (("background_arms",), ["uniform", "unknown"], "background_arms"),
        (("empirical_years",), [2021, 2025], "empirical_years"),
        (("empirical_country",), "UK", "empirical_country"),
    ],
)
def test_out_of_scope_design_values_are_rejected(
    tmp_path: Path,
    valid_payload: dict[str, object],
    path: tuple[str, ...],
    invalid_value: object,
    message: str,
) -> None:
    payload = deepcopy(valid_payload)
    target: dict[str, object] = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value

    with pytest.raises(ValueError, match=message):
        load_study_config(write_payload(tmp_path, payload))


def test_duplicate_focal_species_are_rejected(
    tmp_path: Path, valid_payload: dict[str, object]
) -> None:
    payload = deepcopy(valid_payload)
    payload["empirical_species"].append(deepcopy(payload["empirical_species"][0]))

    with pytest.raises(ValueError, match="duplicate"):
        load_study_config(write_payload(tmp_path, payload))


@pytest.mark.parametrize("excluded_name", ["Common bat", "Pipistrellus pygmaeus"])
def test_excluded_bat_names_are_rejected_anywhere_in_empirical_scope(
    tmp_path: Path,
    valid_payload: dict[str, object],
    excluded_name: str,
) -> None:
    payload = deepcopy(valid_payload)
    payload["empirical_species"][0]["target_group"].append(excluded_name)

    with pytest.raises(ValueError, match="excluded"):
        load_study_config(write_payload(tmp_path, payload))
