# Linear MaxEnt Stability Amendment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unstable polynomial MaxEnt-equivalent feature basis with one global linear basis, export model-stability and spatial-fold audits, and regenerate every affected paper artifact from clean paths.

**Architecture:** `maxent.py` will own the fixed linear transform and one-pass prediction diagnostics. The simulation and empirical runners will persist those diagnostics and their audits will fail on numerical map collapse; `spatial.py` will convert deterministic folds to auditable row-level assignments. Existing nonlinear outputs will be preserved in a dated ignored directory before complete simulation and empirical reruns.

**Tech Stack:** Python 3.12, NumPy, pandas, scikit-learn, PyArrow, pytest, Parquet/CSV/JSON artifacts, existing `provenance_sdm` CLI.

## Global Constraints

- Work only on `paper/ecomodelling-provenance-backgrounds`; never merge into `master`.
- Use the linear environmental feature basis in both simulations and GBIF analyses.
- Keep L2 regularization, balanced classes, background budgets, seed derivation, evaluation samples, and PM-TGB weighting unchanged.
- Keep nonlinear virtual-species truth; fitted-model misspecification must be identical across background arms.
- Preserve old nonlinear outputs and never combine them with corrected outputs.
- Label every corrected result row `feature_basis=linear`.
- Reject fitted maps with non-finite diagnostics, fewer than 50 inverse-Simpson effective cells, or maximum normalized cell mass above `0.10`.
- Retain Boyce undefined flags and reasons; never impute undefined Boyce values.
- Export row-level fold assignments and block/class counts for all four species and all three block widths.
- Exclude bats from every acquisition, output, test, figure, and claim.
- Implement each behavior test-first and commit only after the complete suite passes.

---

## File structure

- Modify `provenance_sdm/src/provenance_sdm/maxent.py`: linear design, convergence capture, and prediction diagnostics.
- Modify `provenance_sdm/src/provenance_sdm/simulation_runner.py`: persist and audit feature-basis/stability fields.
- Modify `provenance_sdm/src/provenance_sdm/empirical.py`: persist diagnostics and write fold artifacts.
- Modify `provenance_sdm/src/provenance_sdm/spatial.py`: turn `SpatialFold` values into row-level and block-level audit frames.
- Modify `provenance_sdm/src/provenance_sdm/reproducibility.py`: require corrected basis and stability fields in final artifacts.
- Modify `provenance_sdm/README.md`: document corrected model and regeneration commands.
- Modify `provenance_sdm/tests/test_maxent.py`: linear-basis and outlier-stability regressions.
- Modify `provenance_sdm/tests/test_simulation_runner.py`: corrected schema and collapse rejection.
- Modify `provenance_sdm/tests/test_empirical.py`: diagnostics, assignments, and block-count exports.
- Modify `provenance_sdm/tests/test_spatial.py`: assignment integrity.
- Modify `provenance_sdm/tests/test_reproducibility.py`: reject stale nonlinear artifacts.

### Task 1: Implement the fixed linear model and one-pass diagnostics

**Files:**
- Modify: `provenance_sdm/src/provenance_sdm/maxent.py`
- Test: `provenance_sdm/tests/test_maxent.py`

**Interfaces:**
- Consumes: training presence/background frames, declared feature names, regularization, seed, and a full landscape.
- Produces: `PredictionResult`, `MaxentModel.predict_with_diagnostics(landscape, batch_size=50_000) -> PredictionResult`, and the existing `predict_suitability(...) -> np.ndarray` compatibility method.

- [ ] **Step 1: Write linear-transform and outlier-stability tests**

Add imports for `pandas as pd` and `landscape_from_arrays` if absent, then add:

```python
def test_transform_returns_only_standardized_linear_features(toy_gradient_data):
    model = fit_maxent(
        toy_gradient_data.presence,
        toy_gradient_data.background,
        feature_names=("env_1",),
        regularization=1.0,
        seed=5,
    )
    design = model.transform(toy_gradient_data.landscape.cells)
    assert design.shape == (len(toy_gradient_data.landscape.cells), 1)
    np.testing.assert_allclose(
        design[:, 0],
        (
            toy_gradient_data.landscape.cells.env_1.to_numpy()
            - model.feature_means[0]
        )
        / model.feature_scales[0],
    )


def test_linear_basis_does_not_collapse_on_an_extreme_predictor_cell():
    training = pd.DataFrame(
        {
            "env_1": [-1.0, -0.5, 0.5, 1.0],
            "env_2": [0.0, 1.0, 0.0, 1.0],
            "env_3": [1.0, 0.0, 1.0, 0.0],
        }
    )
    presence = training.iloc[[2, 3]].copy()
    background = training.iloc[[0, 1]].copy()
    grid = pd.concat(
        [training, pd.DataFrame({"env_1": [12.0], "env_2": [0.5], "env_3": [0.5]})],
        ignore_index=True,
    )
    grid.insert(0, "cell_id", np.arange(len(grid)))
    grid["x"] = np.arange(len(grid), dtype=float) * 1_000
    grid["y"] = 0.0
    grid["area_weight"] = 1.0
    model = fit_maxent(
        presence,
        background,
        feature_names=("env_1", "env_2", "env_3"),
        regularization=1.0,
        seed=5,
    )
    result = model.predict_with_diagnostics(grid)

    assert result.feature_basis == "linear"
    assert result.max_cell_mass < 0.90
    assert result.effective_cell_count > 1.0
    assert np.isfinite(result.log_intensity_range)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd provenance_sdm
.venv/bin/python -m pytest \
  tests/test_maxent.py::test_transform_returns_only_standardized_linear_features \
  tests/test_maxent.py::test_linear_basis_does_not_collapse_on_an_extreme_predictor_cell -q
```

Expected: the transform-shape assertion fails because quadratic columns remain, and `predict_with_diagnostics` is absent.

- [ ] **Step 3: Implement the linear transform and diagnostic result**

In `maxent.py`, import `warnings` and `ConvergenceWarning`, then add:

```python
FEATURE_BASIS = "linear"
LOWER_LOG_INTENSITY_CLIP = -50.0


@dataclass(frozen=True)
class PredictionResult:
    suitability: np.ndarray
    feature_basis: str
    max_cell_mass: float
    effective_cell_count: float
    log_intensity_range: float
    lower_clip_cells: int
    lower_clip_fraction: float
    solver_converged: bool
```

Add `solver_converged: bool` to `MaxentModel`. Replace `transform` after standardization with:

```python
return linear
```

Replace the prediction body with these two methods:

```python
def predict_with_diagnostics(
    self,
    landscape: Landscape | pd.DataFrame,
    batch_size: int = 50_000,
) -> PredictionResult:
    frame = landscape.cells if isinstance(landscape, Landscape) else landscape
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    log_intensity = np.empty(len(frame), dtype=float)
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        log_intensity[start:stop] = self.predict_log_intensity(frame.iloc[start:stop])
    raw_range = float(np.ptp(log_intensity))
    log_intensity -= float(log_intensity.max())
    lower_clip = log_intensity < LOWER_LOG_INTENSITY_CLIP
    intensity = np.exp(np.clip(log_intensity, LOWER_LOG_INTENSITY_CLIP, 0.0))
    if "area_weight" in frame:
        intensity *= frame.area_weight.to_numpy(dtype=float)
    total = float(intensity.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("predicted landscape intensity must be finite and positive")
    suitability = intensity / total
    effective = float(1.0 / np.square(suitability).sum())
    return PredictionResult(
        suitability=suitability,
        feature_basis=FEATURE_BASIS,
        max_cell_mass=float(suitability.max()),
        effective_cell_count=effective,
        log_intensity_range=raw_range,
        lower_clip_cells=int(lower_clip.sum()),
        lower_clip_fraction=float(lower_clip.mean()),
        solver_converged=self.solver_converged,
    )

def predict_suitability(
    self,
    landscape: Landscape | pd.DataFrame,
    batch_size: int = 50_000,
) -> np.ndarray:
    return self.predict_with_diagnostics(landscape, batch_size).suitability
```

Construct the temporary transform model with `solver_converged=False`. Capture solver convergence in `fit_maxent`:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConvergenceWarning)
    estimator.fit(design, labels)
converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
```

Return the fitted `MaxentModel` with `solver_converged=converged`.

- [ ] **Step 4: Run focused and complete tests**

Run:

```bash
cd provenance_sdm
.venv/bin/python -m pytest tests/test_maxent.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the model correction**

```bash
git add provenance_sdm/src/provenance_sdm/maxent.py provenance_sdm/tests/test_maxent.py
git commit -m "fix: use stable linear MaxEnt feature basis"
```

### Task 2: Persist and audit simulation stability

**Files:**
- Modify: `provenance_sdm/src/provenance_sdm/simulation_runner.py`
- Test: `provenance_sdm/tests/test_simulation_runner.py`

**Interfaces:**
- Consumes: `PredictionResult` from Task 1.
- Produces: simulation rows containing `feature_basis`, `max_cell_mass`, `effective_cell_count`, `log_intensity_range`, `lower_clip_cells`, `lower_clip_fraction`, and `solver_converged`; `audit_simulation` validates these fields.

- [ ] **Step 1: Write corrected-schema and collapse-audit tests**

Extend the tiny-run test after reading its result:

```python
required = {
    "feature_basis",
    "max_cell_mass",
    "effective_cell_count",
    "log_intensity_range",
    "lower_clip_cells",
    "lower_clip_fraction",
    "solver_converged",
}
assert required <= set(first)
assert set(first.feature_basis) == {"linear"}
assert np.isfinite(first[list(required - {"feature_basis", "solver_converged"})]).all().all()
```

Add:

```python
def test_audit_rejects_a_numerically_collapsed_map(tiny_config, toy_landscape, tmp_path):
    path = run_simulation(tiny_config, toy_landscape, tmp_path)
    rows = pd.read_parquet(path)
    rows.loc[0, "max_cell_mass"] = 0.50
    rows.loc[0, "effective_cell_count"] = 4.0
    rows.to_parquet(path, index=False)

    report = audit_simulation(path, tiny_config)

    assert report["status"] == "failed"
    assert report["stable_predictions"] is False
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd provenance_sdm
.venv/bin/python -m pytest tests/test_simulation_runner.py -q
```

Expected: missing-column assertions fail and the audit lacks `stable_predictions`.

- [ ] **Step 3: Persist diagnostics without a second prediction pass**

In `run_simulation`, replace:

```python
prediction = model.predict_suitability(landscape)
```

with:

```python
prediction_result = model.predict_with_diagnostics(landscape)
prediction = prediction_result.suitability
```

Add these fields to each output row:

```python
"feature_basis": prediction_result.feature_basis,
"max_cell_mass": prediction_result.max_cell_mass,
"effective_cell_count": prediction_result.effective_cell_count,
"log_intensity_range": prediction_result.log_intensity_range,
"lower_clip_cells": prediction_result.lower_clip_cells,
"lower_clip_fraction": prediction_result.lower_clip_fraction,
"solver_converged": prediction_result.solver_converged,
```

In `audit_simulation`, define:

```python
STABILITY_COLUMNS = (
    "max_cell_mass",
    "effective_cell_count",
    "log_intensity_range",
    "lower_clip_cells",
    "lower_clip_fraction",
)
```

Calculate:

```python
correct_basis = "feature_basis" in actual and actual.feature_basis.eq("linear").all()
stable_predictions = (
    all(column in actual for column in STABILITY_COLUMNS)
    and np.isfinite(actual.loc[:, STABILITY_COLUMNS].to_numpy(dtype=float)).all()
    and actual.max_cell_mass.le(0.10).all()
    and actual.effective_cell_count.ge(50.0).all()
    and actual.lower_clip_fraction.between(0.0, 1.0, inclusive="both").all()
    and "solver_converged" in actual
    and actual.solver_converged.eq(True).all()
)
```

Require both Booleans for passed status and export them in the audit JSON.

- [ ] **Step 4: Run focused and complete tests**

```bash
cd provenance_sdm
.venv/bin/python -m pytest tests/test_simulation_runner.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit simulation stability outputs**

```bash
git add provenance_sdm/src/provenance_sdm/simulation_runner.py provenance_sdm/tests/test_simulation_runner.py
git commit -m "feat: audit simulation prediction stability"
```

### Task 3: Export empirical stability and auditable spatial assignments

**Files:**
- Modify: `provenance_sdm/src/provenance_sdm/spatial.py`
- Modify: `provenance_sdm/src/provenance_sdm/empirical.py`
- Test: `provenance_sdm/tests/test_spatial.py`
- Test: `provenance_sdm/tests/test_empirical.py`

**Interfaces:**
- Consumes: evaluation-grid rows and `tuple[SpatialFold, ...]`.
- Produces: `spatial_assignment_frames(records, width_m, folds) -> tuple[pd.DataFrame, pd.DataFrame]`, `spatial_fold_assignments.parquet`, and `spatial_block_class_audit.csv`.

- [ ] **Step 1: Write assignment-integrity test**

In `test_spatial.py`, import `spatial_assignment_frames` and add:

```python
def test_assignment_frames_cover_rows_and_keep_blocks_whole(projected_occurrences):
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
    assert block_audit.groupby("fold_id").positive_rows.sum().gt(0).all()
    assert block_audit.groupby("fold_id").negative_rows.sum().gt(0).all()
```

- [ ] **Step 2: Write empirical artifact and diagnostic tests**

Extend the tiny empirical run test:

```python
required = {
    "feature_basis",
    "max_cell_mass",
    "effective_cell_count",
    "log_intensity_range",
    "lower_clip_cells",
    "lower_clip_fraction",
    "solver_converged",
}
assert required <= set(rows)
assert rows.feature_basis.eq("linear").all()
assignments = pd.read_parquet(tmp_path / "spatial_fold_assignments.parquet")
block_audit = pd.read_csv(tmp_path / "spatial_block_class_audit.csv")
assert set(assignments.species) == {item.key for item in tiny_empirical_inputs.config.empirical_species}
assert set(assignments.block_width_m) == set(tiny_empirical_inputs.block_widths)
assert block_audit.positive_rows.gt(0).all()
assert block_audit.negative_rows.gt(0).all()
```

- [ ] **Step 3: Run tests and verify RED**

```bash
cd provenance_sdm
.venv/bin/python -m pytest tests/test_spatial.py tests/test_empirical.py -q
```

Expected: import failure for `spatial_assignment_frames` and missing empirical diagnostic/artifact assertions.

- [ ] **Step 4: Implement assignment frames**

In `spatial.py`, factor block IDs into a private helper shared by both fold construction and assignment export:

```python
def _block_ids(records: pd.DataFrame, width_m: int) -> np.ndarray:
    coordinates = records.loc[:, ["x", "y"]].to_numpy(dtype=float)
    block_x = np.floor((coordinates[:, 0] - coordinates[:, 0].min()) / width_m)
    block_y = np.floor((coordinates[:, 1] - coordinates[:, 1].min()) / width_m)
    return np.array(
        [f"{int(left)}:{int(right)}" for left, right in zip(block_x, block_y, strict=True)]
    )
```

Add:

```python
def spatial_assignment_frames(
    records: pd.DataFrame,
    width_m: int,
    folds: tuple[SpatialFold, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_ids = _block_ids(records, width_m)
    fold_by_index = {
        int(row_index): fold.fold_id
        for fold in folds
        for row_index in fold.test_row_indices
    }
    assignments = pd.DataFrame(
        {
            "row_index": records.index.to_numpy(dtype=int),
            "block_id": block_ids,
            "fold_id": [fold_by_index[int(index)] for index in records.index],
        }
    )
    if "label" in records:
        assignments["label"] = records.label.to_numpy(dtype=np.int8)
    if assignments.groupby("block_id").fold_id.nunique().ne(1).any():
        raise ValueError("a projected block spans multiple test folds")
    grouped = assignments.groupby(["fold_id", "block_id"], as_index=False)
    block_audit = grouped.agg(
        rows=("row_index", "size"),
        positive_rows=("label", lambda value: int((value == 1).sum())),
        negative_rows=("label", lambda value: int((value == 0).sum())),
    )
    return assignments, block_audit
```

Use `_block_ids` inside `projected_block_folds` to remove duplicate formulas.

- [ ] **Step 5: Persist empirical diagnostics and assignments**

In `empirical.py`, import `spatial_assignment_frames`. Initialize `assignment_tables` and `block_audit_tables`. Immediately after each species/width fold tuple is created:

```python
assignments, block_audit = spatial_assignment_frames(evaluation_grid, width, folds)
assignments.insert(0, "species", species.key)
assignments.insert(1, "block_width_m", width)
assignments["cell_id"] = evaluation_grid.cell_id.to_numpy(dtype=np.int64)
block_audit.insert(0, "species", species.key)
block_audit.insert(1, "block_width_m", width)
assignment_tables.append(assignments)
block_audit_tables.append(block_audit)
```

Replace each empirical prediction call with `predict_with_diagnostics`; retain both `prediction_result` and its suitability by arm. Add the seven Task 2 diagnostic fields to every empirical row.

Before returning, write:

```python
pd.concat(assignment_tables, ignore_index=True).to_parquet(
    destination / "spatial_fold_assignments.parquet",
    index=False,
)
pd.concat(block_audit_tables, ignore_index=True).to_csv(
    destination / "spatial_block_class_audit.csv",
    index=False,
)
```

- [ ] **Step 6: Run focused and complete tests**

```bash
cd provenance_sdm
.venv/bin/python -m pytest tests/test_spatial.py tests/test_empirical.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit empirical audit outputs**

```bash
git add provenance_sdm/src/provenance_sdm/spatial.py provenance_sdm/src/provenance_sdm/empirical.py provenance_sdm/tests/test_spatial.py provenance_sdm/tests/test_empirical.py
git commit -m "feat: export empirical model and fold audits"
```

### Task 4: Require corrected artifacts in reproducibility checks

**Files:**
- Modify: `provenance_sdm/src/provenance_sdm/reproducibility.py`
- Modify: `provenance_sdm/README.md`
- Test: `provenance_sdm/tests/test_reproducibility.py`

**Interfaces:**
- Consumes: corrected simulation/empirical artifacts and fold audit files.
- Produces: final audit fields `linear_feature_basis`, `stable_predictions`, and `spatial_fold_artifacts`.

- [ ] **Step 1: Write stale-artifact rejection test**

In the complete tiny-run fixture, ensure corrected fields and fold files exist. Add:

```python
def test_reproducibility_audit_rejects_stale_nonlinear_results(complete_tiny_run):
    path = complete_tiny_run / "outputs" / "simulation_metrics.parquet"
    rows = pd.read_parquet(path)
    rows["feature_basis"] = "linear_quadratic_interactions"
    rows.to_parquet(path, index=False)

    audit = build_reproducibility_audit(complete_tiny_run, complete_tiny_run.config)

    assert audit["core_status"] == "failed"
    assert audit["linear_feature_basis"] is False
```

- [ ] **Step 2: Run test and verify RED**

```bash
cd provenance_sdm
.venv/bin/python -m pytest tests/test_reproducibility.py::test_reproducibility_audit_rejects_stale_nonlinear_results -q
```

Expected: failure because the audit lacks `linear_feature_basis` or still passes stale results.

- [ ] **Step 3: Add final corrected-artifact requirements**

In `build_reproducibility_audit`, load both metric tables and calculate:

```python
linear_feature_basis = (
    simulation.feature_basis.eq("linear").all()
    and empirical.feature_basis.eq("linear").all()
)
stable_predictions = (
    simulation.max_cell_mass.le(0.10).all()
    and empirical.max_cell_mass.le(0.10).all()
    and simulation.effective_cell_count.ge(50.0).all()
    and empirical.effective_cell_count.ge(50.0).all()
)
spatial_fold_artifacts = all(
    (root / "outputs" / name).is_file()
    for name in (
        "spatial_fold_assignments.parquet",
        "spatial_block_class_audit.csv",
    )
)
```

Export all three fields and require them for `core_status="passed"`.

Update README’s model description to state that the fitted basis is linear, virtual truth remains nonlinear, numerical diagnostics are exported, and nonlinear diagnostic outputs are preserved outside primary paths.

- [ ] **Step 4: Run focused and complete tests**

```bash
cd provenance_sdm
.venv/bin/python -m pytest tests/test_reproducibility.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit final audit requirements**

```bash
git add provenance_sdm/src/provenance_sdm/reproducibility.py provenance_sdm/tests/test_reproducibility.py provenance_sdm/README.md
git commit -m "feat: require stable linear submission artifacts"
```

### Task 5: Preserve diagnostics and regenerate production results

**Files:**
- Move generated ignored artifacts under `provenance_sdm/outputs/diagnostic-nonlinear-20260801/`.
- Regenerate ignored artifacts under `provenance_sdm/outputs/` and `provenance_sdm/manuscript/`.

**Interfaces:**
- Consumes: fixed configuration, predictor grid, cleaned citable GBIF records, taxon manifest, and corrected runners.
- Produces: complete audited corrected simulation/empirical results, figures, tables, and reproducibility report.

- [ ] **Step 1: Verify branch and clean tracked state**

```bash
git branch --show-current
git status --short
```

Expected: `paper/ecomodelling-provenance-backgrounds` and no tracked changes.

- [ ] **Step 2: Preserve the nonlinear outputs recoverably**

Create `outputs/diagnostic-nonlinear-20260801/`, then move these exact files when present:

```text
outputs/simulation_metrics.parquet
outputs/simulation_audit.json
outputs/empirical_metrics.parquet
outputs/empirical_maps.parquet
```

Verify all four exist in the diagnostic directory and none remains at its primary path. Do not move the GBIF archive, DOI manifests, cleaned occurrences, predictor grid, taxon manifest, or DeepMaxEnt gate.

- [ ] **Step 3: Run the complete corrected simulation**

```bash
cd provenance_sdm
PYTHONPATH=src .venv/bin/python -c \
  "from provenance_sdm.cli import main; raise SystemExit(main(['simulate','--config','config/study.yaml','--landscape','outputs/gb_grid.parquet']))"
```

Expected: `outputs/simulation_metrics.parquet` contains exactly 14,400 unique corrected rows.

- [ ] **Step 4: Run the formal simulation audit**

```bash
PYTHONPATH=src .venv/bin/python -c \
  "from provenance_sdm.cli import main; raise SystemExit(main(['audit-simulation','--config','config/study.yaml','--results','outputs/simulation_metrics.parquet']))"
```

Expected: exit `0`; audit reports 14,400 completed, zero failed/missing/unexpected/duplicates, linear basis, stable predictions, finite primary metrics, and complete arms.

- [ ] **Step 5: Rerun the complete empirical evaluation**

```bash
PYTHONPATH=src .venv/bin/python -c \
  "from provenance_sdm.cli import main; raise SystemExit(main(['run-empirical','--config','config/study.yaml','--records','outputs/clean_occurrences.parquet','--grid','outputs/gb_grid.parquet','--taxa','outputs/taxa.json','--output','outputs']))"
```

Expected: 360 unique rows across four species, three widths, five folds, two provenance levels, and three arms; corrected map and spatial-fold artifacts exist.

- [ ] **Step 6: Validate empirical contracts directly**

Read corrected Parquet/CSV outputs and require:

```python
assert len(empirical) == 360
assert empirical.feature_basis.eq("linear").all()
assert empirical.max_cell_mass.le(0.10).all()
assert empirical.effective_cell_count.ge(50.0).all()
assert empirical.solver_converged.all()
assert empirical.duplicated(
    ["species", "block_width_m", "fold_id", "provenance_level", "background_arm"]
).sum() == 0
assert empirical.groupby(
    ["species", "block_width_m", "fold_id", "provenance_level"]
).evaluation_hash.nunique().eq(1).all()
assert assignments.groupby(["species", "block_width_m", "block_id"]).fold_id.nunique().eq(1).all()
```

Record Boyce defined/undefined counts and reasons without treating undefined rows as failed fits.

- [ ] **Step 7: Regenerate summaries and figures**

```bash
PYTHONPATH=src .venv/bin/python -c \
  "from provenance_sdm.cli import main; raise SystemExit(main(['summarize-simulation','--results','outputs/simulation_metrics.parquet','--output','outputs']))"
PYTHONPATH=src .venv/bin/python -c \
  "from provenance_sdm.cli import main; raise SystemExit(main(['figures-simulation','--results','outputs/simulation_metrics.parquet','--output','manuscript/figures']))"
PYTHONPATH=src .venv/bin/python -c \
  "from provenance_sdm.cli import main; raise SystemExit(main(['figures-empirical','--results','outputs/empirical_metrics.parquet','--maps','outputs/empirical_maps.parquet','--output','manuscript/figures']))"
```

Expected: corrected paired effects, bootstrap intervals, and all planned figure files are regenerated from corrected artifacts only.

- [ ] **Step 8: Export manuscript tables and run final audit**

```bash
PYTHONPATH=src .venv/bin/python -c \
  "from provenance_sdm.cli import main; raise SystemExit(main(['export-manuscript','--config','config/study.yaml','--root','.','--output','manuscript']))"
PYTHONPATH=src .venv/bin/python -c \
  "from provenance_sdm.cli import main; raise SystemExit(main(['audit-all','--config','config/study.yaml','--root','.','--output','outputs/reproducibility_audit.json']))"
```

Expected: export succeeds and final audit exits `0` with `core_status=passed`.

- [ ] **Step 9: Run final code and repository verification**

```bash
.venv/bin/python -m pytest -q
git diff --check
git status --short --branch
git log -6 --oneline
```

Expected: complete suite passes; no whitespace errors; only intentionally tracked manuscript work remains; `master` has not changed.

- [ ] **Step 10: Review and commit corrected manuscript artifacts**

Visually inspect the five PNG files and inspect the schemas/content of the four
CSV files. When they contain only corrected results, commit these exact files:

```bash
git add \
  provenance_sdm/manuscript/figures/simulation_workflow.png \
  provenance_sdm/manuscript/figures/paired_truth_contrasts.png \
  provenance_sdm/manuscript/figures/contrast_conditions.png \
  provenance_sdm/manuscript/figures/empirical_source_contrasts.png \
  provenance_sdm/manuscript/figures/empirical_map_contrast.png \
  provenance_sdm/manuscript/tables/table_1_simulation_design.csv \
  provenance_sdm/manuscript/tables/table_2_primary_effects.csv \
  provenance_sdm/manuscript/tables/table_3_empirical_composition_metrics.csv \
  provenance_sdm/manuscript/tables/table_4_reproducibility_manifest.csv
git commit -m "docs: regenerate corrected paper artifacts"
```

Do not merge into `master`. Push only `paper/ecomodelling-provenance-backgrounds` after verification.
