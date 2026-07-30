"""Virtual ecological suitability truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from provenance_sdm.landscape import Landscape


N_TAXONOMIC_GROUPS = 10
NICHE_BREADTH_RANGE = (0.5, 2.0)


@dataclass(frozen=True)
class SpeciesTruth:
    species_id: str
    taxonomic_group: int
    coefficients: np.ndarray
    niche_breadth: float
    suitability: np.ndarray
    landscape: Landscape


def _environmental_design(landscape: Landscape) -> np.ndarray:
    linear = landscape.cells.loc[:, landscape.feature_names].to_numpy(dtype=float)
    quadratic = linear**2
    interactions = [
        linear[:, left] * linear[:, right]
        for left in range(linear.shape[1])
        for right in range(left + 1, linear.shape[1])
    ]
    return np.column_stack([linear, quadratic, *interactions])


def simulate_species_truth(
    landscape: Landscape,
    n_species: int,
    seed: int,
) -> tuple[SpeciesTruth, ...]:
    """Generate deterministic linear, quadratic, and interacting niches."""

    if isinstance(n_species, bool) or not isinstance(n_species, int) or n_species <= 0:
        raise ValueError("n_species must be a positive integer")

    generator = np.random.default_rng(seed)
    environmental = _environmental_design(landscape)
    n_features = len(landscape.feature_names)
    n_interactions = n_features * (n_features - 1) // 2
    area_weight = landscape.cells.area_weight.to_numpy(dtype=float)
    truths: list[SpeciesTruth] = []

    for species_index in range(n_species):
        niche_breadth = float(generator.uniform(*NICHE_BREADTH_RANGE))
        linear = generator.normal(0.0, 0.9, size=n_features)
        quadratic = -np.abs(generator.normal(0.45, 0.2, size=n_features))
        interaction = generator.normal(0.0, 0.25, size=n_interactions)
        coefficients = np.concatenate([linear, quadratic, interaction])

        log_intensity = environmental @ coefficients / niche_breadth
        log_intensity -= float(log_intensity.max())
        intensity = np.exp(np.clip(log_intensity, -50.0, 0.0)) * area_weight
        suitability = np.asarray(
            intensity / intensity.sum(),
            dtype=np.float32,
        )
        truths.append(
            SpeciesTruth(
                species_id=f"sp_{species_index:03d}",
                taxonomic_group=species_index % N_TAXONOMIC_GROUPS,
                coefficients=coefficients,
                niche_breadth=niche_breadth,
                suitability=suitability,
                landscape=landscape,
            )
        )

    return tuple(truths)
