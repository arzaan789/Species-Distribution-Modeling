"""Equal-budget background construction for all simulation arms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from provenance_sdm.observation import ObservedCommunity
from provenance_sdm.provenance import pm_tgb_weights


def _sample_cells(
    cell_weights: pd.Series,
    n_cells: int,
    generator: np.random.Generator,
    diagnostic: str,
) -> np.ndarray:
    weights = cell_weights.groupby(level=0).sum().astype(float)
    weights = weights[weights > 0]
    if len(weights) < n_cells:
        raise ValueError(
            f"fewer than {n_cells} unique {diagnostic} are available "
            f"({len(weights)} found)"
        )
    probability = weights.to_numpy(copy=True)
    probability /= probability.sum()
    return generator.choice(
        weights.index.to_numpy(dtype=np.int64),
        size=n_cells,
        replace=False,
        p=probability,
    )


def _background_frame(
    observed: ObservedCommunity,
    selected: np.ndarray,
    arm: str,
) -> pd.DataFrame:
    frame = (
        observed.landscape.cells.set_index("cell_id")
        .loc[selected]
        .reset_index()
    )
    frame["background_arm"] = arm
    return frame


def make_backgrounds(
    observed: ObservedCommunity,
    focal_species: str,
    n_cells: int,
    seed: int,
    minimum_cells: int = 1,
) -> dict[str, pd.DataFrame]:
    """Construct four unique-cell samples using one common supported budget."""

    if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells <= 0:
        raise ValueError("n_cells must be a positive integer")
    if (
        isinstance(minimum_cells, bool)
        or not isinstance(minimum_cells, int)
        or minimum_cells <= 0
        or minimum_cells > n_cells
    ):
        raise ValueError("minimum_cells must be positive and no greater than n_cells")
    truth_by_id = {species.species_id: species for species in observed.truth}
    if focal_species not in truth_by_id:
        raise ValueError(f"unknown focal species: {focal_species!r}")
    focal_records = observed.records.query("species_id == @focal_species")
    if focal_records.empty:
        raise ValueError("focal species has no observed records")

    focal_truth = truth_by_id[focal_species]
    candidates = observed.records.query(
        "taxonomic_group == @focal_truth.taxonomic_group "
        "and species_id != @focal_species"
    )
    generator = np.random.default_rng(seed)
    landscape = observed.landscape.cells
    uniform_weights = pd.Series(
        landscape.area_weight.to_numpy(dtype=float),
        index=landscape.cell_id,
    )
    conventional_weights = pd.Series(
        1.0,
        index=candidates.cell_id.to_numpy(dtype=np.int64),
    )
    provenance = pm_tgb_weights(
        focal_records.programme_id,
        candidates.set_index("record_id").programme_id,
    )
    pm_cell_weights = provenance.weights.groupby(
        candidates.set_index("record_id").cell_id
    ).sum()
    oracle_weights = pd.Series(
        observed.species_effort[focal_species],
        index=landscape.cell_id,
    )
    available_cells = min(
        int(uniform_weights.gt(0).sum()),
        int(conventional_weights.groupby(level=0).sum().gt(0).sum()),
        int(pm_cell_weights.gt(0).sum()),
        int(oracle_weights.gt(0).sum()),
    )
    common_budget = min(n_cells, available_cells)
    if common_budget < minimum_cells:
        raise ValueError(
            "insufficient unique target-group cells for the common paired "
            f"budget: {common_budget} available, {minimum_cells} required"
        )

    selected = {
        "uniform": _sample_cells(
            uniform_weights,
            common_budget,
            generator,
            "landscape cells",
        ),
        "conventional_tgb": _sample_cells(
            conventional_weights,
            common_budget,
            generator,
            "target-group cells",
        ),
        "pm_tgb": _sample_cells(
            pm_cell_weights,
            common_budget,
            generator,
            "provenance-supported target-group cells",
        ),
        "oracle_effort": _sample_cells(
            oracle_weights,
            common_budget,
            generator,
            "oracle-effort cells",
        ),
    }
    arms = {
        arm: _background_frame(observed, cells, arm)
        for arm, cells in selected.items()
    }
    arms["pm_tgb"]["unsupported_mass"] = provenance.unsupported_mass
    oracle_lookup = pd.Series(
        observed.species_effort[focal_species],
        index=landscape.cell_id,
    )
    arms["oracle_effort"]["true_effort"] = arms["oracle_effort"].cell_id.map(
        oracle_lookup
    )
    return arms
