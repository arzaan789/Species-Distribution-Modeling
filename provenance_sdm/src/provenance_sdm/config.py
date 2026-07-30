"""Typed loading and validation of the frozen paper design."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


APPROVED_ALIGNMENTS = ("low", "partial", "high")
APPROVED_BIAS_LEVELS = ("moderate", "strong")
APPROVED_BACKGROUND_ARMS = (
    "uniform",
    "conventional_tgb",
    "pm_tgb",
    "oracle_effort",
)
EXCLUDED_TOKENS = ("bat", "pipistrell")


@dataclass(frozen=True)
class SimulationConfig:
    n_species: int
    n_communities: int
    n_taxonomic_groups: int
    n_programmes: int
    alignments: tuple[str, ...]
    bias_levels: tuple[str, ...]
    min_records: int
    max_records: int
    background_cells: int
    seed: int


@dataclass(frozen=True)
class EmpiricalSpecies:
    key: str
    scientific_name: str
    target_group: tuple[str, ...]


@dataclass(frozen=True)
class StudyConfig:
    simulation: SimulationConfig
    background_arms: tuple[str, ...]
    empirical_years: tuple[int, int]
    empirical_country: str
    empirical_species: tuple[EmpiricalSpecies, ...]
    output_dir: Path


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_sequence(value: object, field: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence")
    return tuple(value)


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _reject_excluded_name(value: str, field: str) -> None:
    lowered = value.casefold()
    if any(token in lowered for token in EXCLUDED_TOKENS):
        raise ValueError(f"{field} contains an excluded taxon")


def _validate_exact_tuple(
    value: object,
    field: str,
    approved: tuple[str, ...],
) -> tuple[str, ...]:
    items = tuple(_require_string(item, field) for item in _require_sequence(value, field))
    if items != approved:
        raise ValueError(f"{field} must equal {approved!r}")
    return items


def _load_simulation(payload: object) -> SimulationConfig:
    values = _require_mapping(payload, "simulation")
    positive_fields = (
        "n_species",
        "n_communities",
        "n_taxonomic_groups",
        "n_programmes",
        "min_records",
        "max_records",
        "background_cells",
    )
    counts = {
        field: _require_positive_int(values.get(field), field)
        for field in positive_fields
    }
    if counts["max_records"] < counts["min_records"]:
        raise ValueError("max_records must be at least min_records")
    seed = values.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return SimulationConfig(
        **counts,
        alignments=_validate_exact_tuple(
            values.get("alignments"),
            "alignments",
            APPROVED_ALIGNMENTS,
        ),
        bias_levels=_validate_exact_tuple(
            values.get("bias_levels"),
            "bias_levels",
            APPROVED_BIAS_LEVELS,
        ),
        seed=seed,
    )


def _load_empirical_species(payload: object) -> tuple[EmpiricalSpecies, ...]:
    raw_species = _require_sequence(payload, "empirical_species")
    species: list[EmpiricalSpecies] = []
    for position, raw in enumerate(raw_species):
        values = _require_mapping(raw, f"empirical_species[{position}]")
        key = _require_string(values.get("key"), f"empirical_species[{position}].key")
        scientific_name = _require_string(
            values.get("scientific_name"),
            f"empirical_species[{position}].scientific_name",
        )
        target_group = tuple(
            _require_string(name, f"empirical_species[{position}].target_group")
            for name in _require_sequence(
                values.get("target_group"),
                f"empirical_species[{position}].target_group",
            )
        )
        if not target_group:
            raise ValueError(f"empirical_species[{position}].target_group must not be empty")
        for field, value in (
            ("key", key),
            ("scientific_name", scientific_name),
            *(("target_group", name) for name in target_group),
        ):
            _reject_excluded_name(value, f"empirical_species[{position}].{field}")
        species.append(
            EmpiricalSpecies(
                key=key,
                scientific_name=scientific_name,
                target_group=target_group,
            )
        )
    identities = [(item.key.casefold(), item.scientific_name.casefold()) for item in species]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate empirical species are not allowed")
    return tuple(species)


def load_study_config(path: Path) -> StudyConfig:
    """Load and validate the frozen study configuration at *path*."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unable to load study configuration: {config_path}") from exc
    values = _require_mapping(payload, "study")

    years = _require_sequence(values.get("empirical_years"), "empirical_years")
    if years != (2022, 2025):
        raise ValueError("empirical_years must equal (2022, 2025)")
    country = _require_string(values.get("empirical_country"), "empirical_country")
    if country != "GB":
        raise ValueError("empirical_country must equal 'GB'")
    arms = _validate_exact_tuple(
        values.get("background_arms"),
        "background_arms",
        APPROVED_BACKGROUND_ARMS,
    )
    output_value = _require_string(values.get("output_dir"), "output_dir")
    output_dir = Path(output_value)
    if not output_dir.is_absolute():
        output_dir = (config_path.parent.parent / output_dir).resolve()

    return StudyConfig(
        simulation=_load_simulation(values.get("simulation")),
        background_arms=arms,
        empirical_years=(2022, 2025),
        empirical_country=country,
        empirical_species=_load_empirical_species(values.get("empirical_species")),
        output_dir=output_dir,
    )
