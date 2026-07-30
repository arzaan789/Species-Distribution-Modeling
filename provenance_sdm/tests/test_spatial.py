from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.spatial import projected_block_folds


@pytest.fixture
def projected_occurrences() -> pd.DataFrame:
    x, y = np.meshgrid(np.arange(10) * 25_000, np.arange(10) * 25_000)
    return pd.DataFrame(
        {
            "record_id": np.arange(x.size),
            "x": x.ravel(),
            "y": y.ravel(),
            "label": np.tile([0, 1], x.size // 2),
        }
    )


def test_projected_blocks_never_overlap_train_and_test(
    projected_occurrences,
) -> None:
    folds = projected_block_folds(
        projected_occurrences,
        width_m=50_000,
        n_folds=5,
        seed=8,
    )

    assert len(folds) == 5
    for fold in folds:
        assert set(fold.train_block_ids).isdisjoint(fold.test_block_ids)
        assert set(fold.train_row_indices).isdisjoint(fold.test_row_indices)
        assert len(fold.test_row_indices) > 0


def test_block_assignment_is_deterministic(projected_occurrences) -> None:
    first = projected_block_folds(projected_occurrences, 50_000, 5, seed=8)
    second = projected_block_folds(projected_occurrences, 50_000, 5, seed=8)

    assert first == second


@pytest.mark.parametrize("width_m", [25_000, 50_000, 100_000])
def test_all_predeclared_block_widths_are_supported(
    projected_occurrences,
    width_m: int,
) -> None:
    folds = projected_block_folds(projected_occurrences, width_m, 5, seed=1)

    assert len(folds) == 5


def test_geographic_or_nonfinite_coordinates_are_rejected(
    projected_occurrences,
) -> None:
    invalid = projected_occurrences.copy()
    invalid.loc[0, "x"] = np.nan

    with pytest.raises(ValueError, match="finite"):
        projected_block_folds(invalid, 50_000, 5, seed=1)
