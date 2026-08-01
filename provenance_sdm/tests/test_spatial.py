from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_sdm.spatial import (
    projected_block_folds,
    spatial_assignment_frames,
)


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


def test_assignment_frames_cover_rows_and_keep_blocks_whole(
    projected_occurrences,
) -> None:
    folds = projected_block_folds(projected_occurrences, 50_000, 5, seed=8)

    assignments, block_audit = spatial_assignment_frames(
        projected_occurrences,
        50_000,
        folds,
    )

    assert len(assignments) == len(projected_occurrences)
    assert assignments.row_index.is_unique
    assert assignments.fold_id.nunique() == 5
    assert assignments.groupby("block_id").fold_id.nunique().eq(1).all()
    fold_counts = block_audit.groupby("fold_id")[
        ["positive_rows", "negative_rows"]
    ].sum()
    assert fold_counts.positive_rows.gt(0).all()
    assert fold_counts.negative_rows.gt(0).all()


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


def test_sparse_positive_blocks_are_distributed_across_folds(
    projected_occurrences,
) -> None:
    records = projected_occurrences.copy()
    records["label"] = 0
    positive_blocks = [(0, 0), (0, 2), (2, 0), (2, 2), (4, 4)]
    for x_index, y_index in positive_blocks:
        records.loc[
            records.x.eq(x_index * 25_000)
            & records.y.eq(y_index * 25_000),
            "label",
        ] = 1

    folds = projected_block_folds(records, 50_000, 5, seed=8)

    assert all(records.loc[list(fold.test_row_indices), "label"].sum() >= 1 for fold in folds)
