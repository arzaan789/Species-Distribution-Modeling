"""Heterogeneous recording programmes and biased presence-only observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from provenance_sdm.landscape import Landscape
from provenance_sdm.virtual_species import SpeciesTruth


APPROVED_ALIGNMENTS = ("low", "partial", "high")
BIAS_PARAMETERS = {
    "moderate": {"hotspot_scale": 0.35, "gradient_strength": 0.5},
    "strong": {"hotspot_scale": 0.13, "gradient_strength": 1.5},
}
ALIGNMENT_CONCENTRATION = {"partial": 8.0, "high": 80.0}


@dataclass(frozen=True)
class ProgrammeEffort:
    programme_id: str
    intensity: np.ndarray
    bias_level: str


@dataclass(frozen=True)
class ObservedCommunity:
    records: pd.DataFrame
    species_effort: Mapping[str, np.ndarray]
    programme_effort: tuple[ProgrammeEffort, ...]
    source_mixtures: pd.DataFrame
    truth: tuple[SpeciesTruth, ...]
    landscape: Landscape


def simulate_programmes(
    landscape: Landscape,
    n_programmes: int,
    bias_level: str,
    seed: int,
) -> tuple[ProgrammeEffort, ...]:
    """Create positive effort surfaces from hotspots and coordinate gradients."""

    if bias_level not in BIAS_PARAMETERS:
        raise ValueError(f"unknown bias level: {bias_level!r}")
    if (
        isinstance(n_programmes, bool)
        or not isinstance(n_programmes, int)
        or n_programmes <= 0
    ):
        raise ValueError("n_programmes must be a positive integer")

    generator = np.random.default_rng(seed)
    x = landscape.cells.x.to_numpy(dtype=float)
    y = landscape.cells.y.to_numpy(dtype=float)
    x_scaled = (x - x.min()) / max(float(np.ptp(x)), np.finfo(float).eps)
    y_scaled = (y - y.min()) / max(float(np.ptp(y)), np.finfo(float).eps)
    parameters = BIAS_PARAMETERS[bias_level]
    surfaces: list[ProgrammeEffort] = []

    for programme_index in range(n_programmes):
        centre = generator.uniform(0.1, 0.9, size=2)
        direction = generator.normal(size=2)
        direction /= np.linalg.norm(direction)
        squared_distance = (x_scaled - centre[0]) ** 2 + (
            y_scaled - centre[1]
        ) ** 2
        hotspot = np.exp(
            -squared_distance / (2 * parameters["hotspot_scale"] ** 2)
        )
        gradient = np.exp(
            parameters["gradient_strength"]
            * (
                direction[0] * (x_scaled - 0.5)
                + direction[1] * (y_scaled - 0.5)
            )
        )
        intensity = hotspot * gradient + 1e-9
        intensity /= intensity.sum()
        surfaces.append(
            ProgrammeEffort(
                programme_id=f"programme_{programme_index}",
                intensity=intensity,
                bias_level=bias_level,
            )
        )
    return tuple(surfaces)


def _programme_mixtures(
    truth: tuple[SpeciesTruth, ...],
    n_programmes: int,
    alignment: str,
    generator: np.random.Generator,
) -> np.ndarray:
    if alignment == "low":
        return generator.dirichlet(np.ones(n_programmes), size=len(truth))

    group_centres = generator.dirichlet(np.ones(n_programmes), size=10)
    concentration = ALIGNMENT_CONCENTRATION[alignment]
    mixtures = []
    for species in truth:
        alpha = group_centres[species.taxonomic_group] * concentration + 0.25
        mixtures.append(generator.dirichlet(alpha))
    return np.asarray(mixtures)


def _long_tail_counts(
    n_species: int,
    minimum: int,
    maximum: int,
    generator: np.random.Generator,
) -> np.ndarray:
    if minimum == maximum:
        return np.full(n_species, minimum, dtype=int)
    multipliers = np.exp(generator.normal(0.0, 1.25, size=n_species))
    counts = np.rint(minimum * multipliers).astype(int)
    return np.clip(counts, minimum, maximum)


def simulate_observations(
    truth: tuple[SpeciesTruth, ...],
    programmes: tuple[ProgrammeEffort, ...],
    alignment: str,
    bias_level: str,
    min_records: int,
    max_records: int,
    seed: int,
) -> ObservedCommunity:
    """Observe ecological truth through species-specific programme mixtures."""

    if not truth:
        raise ValueError("truth must contain at least one species")
    if not programmes:
        raise ValueError("programmes must contain at least one effort surface")
    if alignment not in APPROVED_ALIGNMENTS:
        raise ValueError(f"unknown alignment: {alignment!r}")
    if bias_level not in BIAS_PARAMETERS:
        raise ValueError(f"unknown bias level: {bias_level!r}")
    if any(programme.bias_level != bias_level for programme in programmes):
        raise ValueError("programme effort does not match requested bias level")
    if min_records <= 0 or max_records < min_records:
        raise ValueError("record bounds must be positive and ordered")

    n_cells = len(truth[0].suitability)
    if any(len(species.suitability) != n_cells for species in truth):
        raise ValueError("all species truth arrays must share one landscape")
    if any(len(programme.intensity) != n_cells for programme in programmes):
        raise ValueError("programme effort arrays must match the landscape")

    generator = np.random.default_rng(seed)
    mixtures = _programme_mixtures(
        truth,
        len(programmes),
        alignment,
        generator,
    )
    counts = _long_tail_counts(
        len(truth),
        min_records,
        max_records,
        generator,
    )
    effort_matrix = np.vstack([programme.intensity for programme in programmes])
    species_effort: dict[str, np.ndarray] = {}
    record_tables: list[pd.DataFrame] = []
    mixture_rows: list[dict[str, object]] = []
    next_record = 0

    for species_index, species in enumerate(truth):
        mixture = mixtures[species_index]
        mixed_effort = mixture @ effort_matrix
        mixed_effort /= mixed_effort.sum()
        species_effort[species.species_id] = np.asarray(
            mixed_effort,
            dtype=np.float32,
        )

        joint_probability = (
            mixture[:, None]
            * effort_matrix
            * species.suitability[None, :]
        )
        joint_probability /= joint_probability.sum()
        joint_index = generator.choice(
            joint_probability.size,
            size=int(counts[species_index]),
            replace=True,
            p=joint_probability.ravel(),
        )
        programme_index, cell_id = np.divmod(joint_index, n_cells)
        record_count = len(cell_id)
        record_tables.append(
            pd.DataFrame(
                {
                    "record_id": np.arange(
                        next_record,
                        next_record + record_count,
                        dtype=np.int64,
                    ),
                    "species_id": species.species_id,
                    "taxonomic_group": species.taxonomic_group,
                    "cell_id": cell_id.astype(np.int64),
                    "programme_id": [
                        programmes[index].programme_id for index in programme_index
                    ],
                }
            )
        )
        next_record += record_count
        mixture_rows.extend(
            {
                "species_id": species.species_id,
                "taxonomic_group": species.taxonomic_group,
                "programme_id": programme.programme_id,
                "weight": float(mixture[programme_index]),
            }
            for programme_index, programme in enumerate(programmes)
        )

    return ObservedCommunity(
        records=pd.concat(record_tables, ignore_index=True),
        species_effort=species_effort,
        programme_effort=programmes,
        source_mixtures=pd.DataFrame(mixture_rows),
        truth=truth,
        landscape=_landscape_from_truth_context(truth),
    )


def _landscape_from_truth_context(
    truth: tuple[SpeciesTruth, ...],
) -> Landscape:
    landscape = truth[0].landscape
    if any(species.landscape is not landscape for species in truth):
        raise ValueError("all SpeciesTruth values must share one Landscape")
    return landscape
