"""Projected contiguous-block cross-validation folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


APPROVED_BLOCK_WIDTHS = (25_000, 50_000, 100_000)


@dataclass(frozen=True)
class SpatialFold:
    fold_id: int
    train_block_ids: tuple[str, ...]
    test_block_ids: tuple[str, ...]
    train_row_indices: tuple[int, ...]
    test_row_indices: tuple[int, ...]


def _block_ids(records: pd.DataFrame, width_m: int) -> np.ndarray:
    coordinates = records.loc[:, ["x", "y"]].to_numpy(dtype=float)
    block_x = np.floor(
        (coordinates[:, 0] - coordinates[:, 0].min()) / width_m
    )
    block_y = np.floor(
        (coordinates[:, 1] - coordinates[:, 1].min()) / width_m
    )
    return np.array(
        [
            f"{int(left)}:{int(right)}"
            for left, right in zip(block_x, block_y, strict=True)
        ]
    )


def spatial_assignment_frames(
    records: pd.DataFrame,
    width_m: int,
    folds: tuple[SpatialFold, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-level fold assignments and block/class count audits."""

    if "label" not in records:
        raise ValueError("spatial assignment audit requires a label column")
    block_ids = _block_ids(records, width_m)
    fold_by_index = {
        int(row_index): fold.fold_id
        for fold in folds
        for row_index in fold.test_row_indices
    }
    if set(fold_by_index) != set(records.index.astype(int)):
        raise ValueError("test folds do not cover every spatial row exactly")
    assignments = pd.DataFrame(
        {
            "row_index": records.index.to_numpy(dtype=int),
            "block_id": block_ids,
            "fold_id": [
                fold_by_index[int(index)] for index in records.index
            ],
            "label": records.label.to_numpy(dtype=np.int8),
        }
    )
    if assignments.groupby("block_id").fold_id.nunique().ne(1).any():
        raise ValueError("a projected block spans multiple test folds")
    block_audit = (
        assignments.groupby(["fold_id", "block_id"], as_index=False)
        .agg(
            rows=("row_index", "size"),
            positive_rows=("label", lambda value: int((value == 1).sum())),
            negative_rows=("label", lambda value: int((value == 0).sum())),
        )
    )
    return assignments, block_audit


def projected_block_folds(
    records: pd.DataFrame,
    width_m: int,
    n_folds: int,
    seed: int,
) -> tuple[SpatialFold, ...]:
    """Assign whole metric-coordinate blocks to deterministic test folds."""

    missing = {"x", "y"}.difference(records.columns)
    if missing:
        raise ValueError(f"spatial records are missing columns: {sorted(missing)}")
    if width_m not in APPROVED_BLOCK_WIDTHS:
        raise ValueError(f"width_m must be one of {APPROVED_BLOCK_WIDTHS}")
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    coordinates = records.loc[:, ["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("projected x/y coordinates must be finite")

    block_ids = _block_ids(records, width_m)
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < n_folds:
        raise ValueError("fewer projected blocks than requested folds")
    generator = np.random.default_rng(seed)
    assignments: dict[str, int] = {}
    fold_block_counts = np.zeros(n_folds, dtype=int)
    if "label" in records:
        labels = records.label.to_numpy()
        positive_blocks = np.array(
            [
                block_id
                for block_id in unique_blocks
                if np.any(labels[block_ids == block_id] == 1)
            ]
        )
        if len(positive_blocks) < n_folds:
            raise ValueError("fewer positive projected blocks than requested folds")
        for position, block_id in enumerate(generator.permutation(positive_blocks)):
            fold_id = int(position % n_folds)
            assignments[str(block_id)] = fold_id
            fold_block_counts[fold_id] += 1
        remaining = np.setdiff1d(unique_blocks, positive_blocks)
    else:
        remaining = unique_blocks
    for block_id in generator.permutation(remaining):
        fold_id = int(np.argmin(fold_block_counts))
        assignments[str(block_id)] = fold_id
        fold_block_counts[fold_id] += 1
    row_fold = np.array([assignments[block_id] for block_id in block_ids])

    folds = []
    index = records.index.to_numpy(dtype=int)
    for fold_id in range(n_folds):
        test_mask = row_fold == fold_id
        if not test_mask.any() or test_mask.all():
            raise ValueError(f"fold {fold_id} lacks train or test support")
        if "label" in records:
            labels = set(records.loc[test_mask, "label"])
            if labels != {0, 1}:
                raise ValueError(f"fold {fold_id} lacks both evaluation classes")
        test_blocks = tuple(sorted(set(block_ids[test_mask])))
        train_blocks = tuple(sorted(set(block_ids[~test_mask])))
        folds.append(
            SpatialFold(
                fold_id=fold_id,
                train_block_ids=train_blocks,
                test_block_ids=test_blocks,
                train_row_indices=tuple(index[~test_mask]),
                test_row_indices=tuple(index[test_mask]),
            )
        )
    return tuple(folds)
