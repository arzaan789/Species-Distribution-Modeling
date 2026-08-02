# Provenance-Matching Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, and report the preregistered diagnostics that test why observed provenance matching is not generally equivalent to sampling-effort correction.

**Architecture:** Preserve the frozen primary artifacts and recreate their deterministic virtual communities in separate extension runners. A mechanism module computes source-composition estimands, a generalized provenance weighting function supports the latent-mixture diagnostic, and a separate clamped polynomial model supports the flexible sensitivity. Extension audits and manuscript exporters consume tidy files without rerunning models.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, scikit-learn, PyArrow, Matplotlib, pytest, Markdown, Word/PDF export tooling.

## Global Constraints

- Target journal: *Ecological Modelling* through its subscription publication route.
- Deadline: a submission-ready package within roughly three months.
- Primary files `outputs/simulation_metrics.parquet` and `outputs/empirical_metrics.parquet` are read-only evidence.
- Keep the frozen linear basis, normalized balanced loss, L2 value `2.0`, common budgets, semantic seeds, evaluation samples, and stability limits unchanged.
- All bats are excluded from acquisition, configurations, tests, results, figures, and manuscript claims.
- Work only on `paper/ecomodelling-provenance-backgrounds`; never merge this workflow into `master`.
- Never commit `.env`, GBIF credentials, the raw GBIF archive, or other secrets.
- Retain null, negative, failed, and heterogeneous results; do not tune against truth-recovery outcomes.
- DeepMaxEnt is not a core dependency, and a passed pilot gate alone is not a reported DeepMaxEnt result.

---

## File map

- Create `src/provenance_sdm/mechanism.py`: source-composition formulas and one-row diagnostic construction.
- Modify `src/provenance_sdm/provenance.py`: generalized source-target weighting used by observed and latent mixtures.
- Modify `src/provenance_sdm/backgrounds.py`: construct one latent-mixture TGB sample at an externally supplied common budget.
- Create `src/provenance_sdm/mechanism_runner.py`: deterministic 3,600-row diagnostics, 3,600 latent-arm fits, checkpoints, and audits.
- Create `src/provenance_sdm/flexible_maxent.py`: clamped linear/quadratic/interaction presence-background model.
- Create `src/provenance_sdm/flexible_runner.py`: 600-fit result-blind pilot, gate, 1,800-fit full sensitivity, and audits.
- Modify `src/provenance_sdm/summaries.py`: oriented effects, diagnostic contrasts, and hierarchical mechanism correlations.
- Modify `src/provenance_sdm/figures.py`: mechanism figures and interpretable empirical map scaling.
- Modify `src/provenance_sdm/reproducibility.py`: extension tables, hashes, figure checks, and complete-run DeepMaxEnt status.
- Modify `src/provenance_sdm/cli.py`: explicit commands for extension runs and audits.
- Add focused test files matching each new module and extend existing tests only for changed public behavior.
- Create manuscript source files under `manuscript/` and render verified submission `.docx` and `.pdf` copies under `submission/`.

---

### Task 1: Source-composition mechanism mathematics

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/mechanism.py`
- Create: `provenance_sdm/tests/test_mechanism.py`

**Interfaces:**
- Consumes: `ObservedCommunity`, focal `SpeciesTruth`, and focal occurrence records.
- Produces: `total_variation(left: pd.Series, right: pd.Series) -> float`, `expected_source_composition(observed: ObservedCommunity, focal_species: str) -> pd.Series`, and `mechanism_row(observed: ObservedCommunity, focal_species: str) -> dict[str, object]`.

- [ ] **Step 1: Write failing formula and validation tests**

```python
def test_expected_composition_includes_ecological_overlap(manual_observed):
    expected = expected_source_composition(manual_observed, "sp_000")
    assert expected.sum() == pytest.approx(1.0)
    assert expected.to_dict() == pytest.approx({"programme_0": 0.8, "programme_1": 0.2})

def test_total_variation_is_symmetric_and_bounded():
    left = pd.Series({"a": 0.75, "b": 0.25})
    right = pd.Series({"a": 0.25, "c": 0.75})
    assert total_variation(left, right) == total_variation(right, left)
    assert total_variation(left, right) == pytest.approx(0.75)

def test_mechanism_row_reports_three_distortions(observed_community):
    row = mechanism_row(observed_community, "sp_000")
    assert {"ecological_overlap_tv", "finite_record_tv", "total_composition_tv"} <= set(row)
    assert all(0 <= row[name] <= 1 for name in ("ecological_overlap_tv", "finite_record_tv", "total_composition_tv"))
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_mechanism.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'provenance_sdm.mechanism'`.

- [ ] **Step 3: Implement normalized programme overlap and TV calculations**

```python
def total_variation(left: pd.Series, right: pd.Series) -> float:
    labels = left.index.union(right.index, sort=False)
    left_mass = left.reindex(labels, fill_value=0.0).astype(float)
    right_mass = right.reindex(labels, fill_value=0.0).astype(float)
    if not np.isfinite(left_mass).all() or not np.isfinite(right_mass).all():
        raise ValueError("source masses must be finite")
    if left_mass.lt(0).any() or right_mass.lt(0).any():
        raise ValueError("source masses must be non-negative")
    if not np.isclose(left_mass.sum(), 1.0) or not np.isclose(right_mass.sum(), 1.0):
        raise ValueError("source masses must each sum to one")
    return float(0.5 * np.abs(left_mass - right_mass).sum())

def expected_source_composition(observed, focal_species):
    latent = _latent_mass(observed, focal_species)
    suitability = _truth_by_id(observed)[focal_species].suitability
    overlap = pd.Series(
        {programme.programme_id: float(programme.intensity @ suitability)
         for programme in observed.programme_effort}
    )
    expected = latent * overlap.reindex(latent.index)
    return expected / expected.sum()
```

`mechanism_row` must compare latent `m`, expected `q`, and realized `p`, retain focal record count and niche breadth, and reject unknown species, incomplete programme labels, and non-finite arrays.

- [ ] **Step 4: Run focused and full tests**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_mechanism.py -v && .venv/bin/pytest -q`

Expected: focused tests pass and the full suite passes.

- [ ] **Step 5: Commit mechanism mathematics**

```bash
git add provenance_sdm/src/provenance_sdm/mechanism.py provenance_sdm/tests/test_mechanism.py
git commit -m "feat: quantify provenance composition distortion"
```

---

### Task 2: Generalized source weighting and latent-mixture background

**Files:**
- Modify: `provenance_sdm/src/provenance_sdm/provenance.py`
- Modify: `provenance_sdm/src/provenance_sdm/backgrounds.py`
- Modify: `provenance_sdm/tests/test_provenance.py`
- Modify: `provenance_sdm/tests/test_backgrounds.py`

**Interfaces:**
- Consumes: normalized target source mass, uniquely indexed candidate source labels, `ObservedCommunity`, common budget, and seed.
- Produces: `source_target_weights(target_mass: pd.Series, candidate_sources: pd.Series) -> ProvenanceWeights` and `make_latent_mixture_background(observed: ObservedCommunity, focal_species: str, n_cells: int, seed: int) -> pd.DataFrame`.

- [ ] **Step 1: Write failing generalized-weight tests**

```python
def test_source_target_weights_match_supported_mass_and_fallback():
    target = pd.Series({"source_a": 0.6, "source_b": 0.3, "missing": 0.1})
    candidates = pd.Series(["source_a", "source_a", "source_b"], index=[10, 11, 12])
    result = source_target_weights(target, candidates)
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.unsupported_mass == pytest.approx(0.1)
    assert result.weights.groupby(candidates).sum()["source_a"] == pytest.approx(0.6 + 0.1 * 2 / 3)

def test_observed_pm_wrapper_preserves_existing_weights():
    focal = pd.Series(["a", "a", "b"])
    candidates = pd.Series(["a", "b", "b"], index=[4, 5, 6])
    assert pm_tgb_weights(focal, candidates).weights.tolist() == pytest.approx(
        source_target_weights(focal.value_counts(normalize=True), candidates).weights.tolist()
    )
```

- [ ] **Step 2: Write failing latent-background tests**

```python
def test_latent_background_uses_candidate_cells_and_exact_budget(observed_community):
    frame = make_latent_mixture_background(observed_community, "sp_000", 20, seed=17)
    candidate = observed_community.records.query(
        "taxonomic_group == 0 and species_id != 'sp_000'"
    )
    assert len(frame) == 20
    assert frame.cell_id.is_unique
    assert set(frame.cell_id) <= set(candidate.cell_id)
    assert set(frame.background_arm) == {"latent_mixture_tgb"}
    assert "true_effort" not in frame
```

- [ ] **Step 3: Run tests and verify missing-symbol failures**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_provenance.py tests/test_backgrounds.py -v`

Expected: collection fails because `source_target_weights` and `make_latent_mixture_background` do not exist.

- [ ] **Step 4: Implement generalized weighting and preserve the observed wrapper**

```python
def pm_tgb_weights(focal_sources, candidate_sources):
    if focal_sources.empty or focal_sources.isna().any():
        raise ValueError("source series must be non-empty and complete")
    return source_target_weights(
        focal_sources.value_counts(normalize=True, sort=False),
        candidate_sources,
    )
```

`source_target_weights` must require finite non-negative target mass summing to one, allocate each supported source total equally among its candidate records, distribute unsupported mass uniformly over every candidate record, renormalize once, and return the existing `ProvenanceWeights` structure.

- [ ] **Step 5: Implement latent-mixture sampling at the supplied budget**

```python
def make_latent_mixture_background(observed, focal_species, n_cells, seed):
    focal_truth = _validated_focal_truth(observed, focal_species)
    candidates = observed.records.query(
        "taxonomic_group == @focal_truth.taxonomic_group and species_id != @focal_species"
    )
    latent = observed.source_mixtures.query("species_id == @focal_species").set_index("programme_id").weight
    weighted = source_target_weights(latent, candidates.set_index("record_id").programme_id)
    cell_weights = weighted.weights.groupby(candidates.set_index("record_id").cell_id).sum()
    selected = _sample_cells(cell_weights, n_cells, np.random.default_rng(seed), "latent-mixture target-group cells")
    frame = _background_frame(observed, selected, "latent_mixture_tgb")
    frame["unsupported_mass"] = weighted.unsupported_mass
    return frame
```

- [ ] **Step 6: Run focused and full tests**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_provenance.py tests/test_backgrounds.py -v && .venv/bin/pytest -q`

Expected: all tests pass and existing four-arm behavior is unchanged.

- [ ] **Step 7: Commit weighting and background construction**

```bash
git add provenance_sdm/src/provenance_sdm/provenance.py provenance_sdm/src/provenance_sdm/backgrounds.py provenance_sdm/tests/test_provenance.py provenance_sdm/tests/test_backgrounds.py
git commit -m "feat: add latent-mixture target-group background"
```

---

### Task 3: Restartable mechanism and latent-arm runner

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/mechanism_runner.py`
- Create: `provenance_sdm/tests/test_mechanism_runner.py`
- Modify: `provenance_sdm/src/provenance_sdm/cli.py`
- Modify: `provenance_sdm/tests/test_cli.py`

**Interfaces:**
- Consumes: frozen `StudyConfig`, `Landscape`, semantic seed helpers, mechanism functions, and latent background constructor.
- Produces: `expected_mechanism_keys(config) -> pd.DataFrame`, `run_mechanism(config, landscape, output_dir) -> tuple[Path, Path]`, and `audit_mechanism(output_dir, config) -> dict[str, object]`.

- [ ] **Step 1: Write failing key, resume, and audit tests**

```python
def test_expected_mechanism_design_has_3600_unique_pairs(study_config):
    keys = expected_mechanism_keys(study_config)
    assert len(keys) == 3_600
    assert not keys.duplicated().any()

def test_mechanism_runner_resumes_two_complete_artifacts(tiny_config, runner_landscape, tmp_path):
    first_paths = run_mechanism(tiny_config, runner_landscape, tmp_path)
    first = tuple(pd.read_parquet(path) for path in first_paths)
    second_paths = run_mechanism(tiny_config, runner_landscape, tmp_path)
    second = tuple(pd.read_parquet(path) for path in second_paths)
    for left, right in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(left, right)
    assert audit_mechanism(tmp_path, tiny_config)["status"] == "passed"

def test_mechanism_audit_rejects_missing_and_unstable_rows(tiny_config, runner_landscape, tmp_path):
    diagnostics_path, latent_path = run_mechanism(tiny_config, runner_landscape, tmp_path)
    diagnostics = pd.read_parquet(diagnostics_path).iloc[:-1]
    diagnostics.to_parquet(diagnostics_path, index=False)
    latent = pd.read_parquet(latent_path)
    latent.loc[0, "max_cell_mass"] = 0.5
    latent.to_parquet(latent_path, index=False)
    audit = audit_mechanism(tmp_path, tiny_config)
    assert audit["missing_diagnostics"] == 1
    assert audit["stable_latent_predictions"] is False
    assert audit["status"] == "failed"
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_mechanism_runner.py -v`

Expected: collection fails with `ModuleNotFoundError` for `mechanism_runner`.

- [ ] **Step 3: Implement exact keys and atomic keyed checkpoints**

Mechanism keys are `(community_seed, alignment, bias_level, species_id)`.
Latent metric keys add `background_arm`, whose only allowed value is
`latent_mixture_tgb`. Before resuming, reject wrong landscape hashes, feature
bases, model regularization, and duplicate or unexpected keys.

- [ ] **Step 4: Implement deterministic reconstruction and paired evaluation**

```python
primary_backgrounds = make_backgrounds(
    observed, species.species_id, simulation.background_cells,
    seed_for(simulation.seed, "backgrounds", community_seed, alignment, bias_level, species.species_id),
    minimum_cells=simulation.minimum_background_cells,
)
latent = make_latent_mixture_background(
    observed, species.species_id, len(primary_backgrounds["pm_tgb"]),
    seed_for(simulation.seed, "latent-background", community_seed, alignment, bias_level, species.species_id),
)
evaluation = generate_unbiased_evaluation(
    species, 500, 500,
    seed_for(simulation.seed, "evaluation", community_seed, alignment, bias_level, species.species_id),
)
```

Write `mechanism_diagnostics.parquet` before the model fit and
`latent_mixture_metrics.parquet` after each successful fit. Include all five
truth metrics, stability diagnostics, common budget, basis, regularization,
landscape hash, and a hash of evaluation cell IDs and labels.

- [ ] **Step 5: Implement strict audit output**

The audit must compare exact expected keys, reject failures, duplicates,
unexpected rows, non-finite values, unequal landscape hashes, incorrect linear
basis or L2 value, maximum mass above 0.10, effective cells below 50, and
unconverged fits. Return JSON-serializable Python booleans and lists.

- [ ] **Step 6: Add CLI commands and parser tests**

```text
provenance-sdm run-mechanism --config config/study.yaml --landscape outputs/gb_grid.parquet --crs EPSG:27700 --output outputs
provenance-sdm audit-mechanism --config config/study.yaml --output outputs --report outputs/mechanism_audit.json
```

`audit-mechanism` writes the report with
`write_manifest(report, arguments.report, allow_replace=True)` and returns zero
only for passed status.

- [ ] **Step 7: Run focused and full tests**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_mechanism_runner.py tests/test_cli.py -v && .venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 8: Commit the extension runner**

```bash
git add provenance_sdm/src/provenance_sdm/mechanism_runner.py provenance_sdm/src/provenance_sdm/cli.py provenance_sdm/tests/test_mechanism_runner.py provenance_sdm/tests/test_cli.py
git commit -m "feat: run and audit provenance mechanism experiment"
```

---

### Task 4: Stable clamped flexible MaxEnt model

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/flexible_maxent.py`
- Create: `provenance_sdm/tests/test_flexible_maxent.py`

**Interfaces:**
- Consumes: presence/background frames, declared feature names, L2 value, seed, and landscape frame.
- Produces: `fit_flexible_maxent(presence: pd.DataFrame, background: pd.DataFrame, feature_names: Sequence[str], regularization: float, seed: int) -> FlexibleMaxentModel` with `transform`, `predict_log_intensity`, and `predict_with_diagnostics` methods matching the primary model's output fields.

- [ ] **Step 1: Write failing basis and extrapolation tests**

```python
def test_flexible_design_has_linear_square_and_pairwise_columns(training_frames):
    model = fit_flexible_maxent(*training_frames, ("a", "b", "c"), 2.0, seed=1)
    assert model.transform(training_frames[0]).shape[1] == 3 + 3 + 3
    assert model.feature_basis == "clamped_linear_quadratic_interactions"

def test_prediction_clamps_raw_values_before_polynomial_expansion(training_frames):
    model = fit_flexible_maxent(*training_frames, ("a", "b"), 2.0, seed=1)
    extreme = pd.DataFrame({"a": [1e12], "b": [-1e12]})
    transformed = model.transform(extreme)
    boundary = model.transform(pd.DataFrame({"a": [model.raw_maxs[0]], "b": [model.raw_mins[1]]}))
    np.testing.assert_allclose(transformed, boundary)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_flexible_maxent.py -v`

Expected: collection fails for missing `flexible_maxent`.

- [ ] **Step 3: Implement train-only raw bounds and deterministic expansion**

```python
def _expand(standardized):
    columns = [standardized, np.square(standardized)]
    columns.extend(
        (standardized[:, left] * standardized[:, right])[:, None]
        for left in range(standardized.shape[1])
        for right in range(left + 1, standardized.shape[1])
    )
    return np.column_stack(columns)
```

Fit means, scales, raw minimums, and raw maximums from combined training rows.
Clamp raw prediction rows to those minimums and maximums, standardize, then
expand in fixed linear/square/pairwise order. Reuse balanced normalized class
weights, `lbfgs`, 1,000 iterations, lower log-intensity clipping, area weights,
and the primary stability calculations.

- [ ] **Step 4: Add invalid-input and finite-prediction tests**

Test empty classes, duplicate/missing features, non-finite data, constant
features, non-positive regularization, invalid batch sizes, and a hand-built
outlier landscape. Assert finite normalized mass, maximum mass at most one,
and an effective cell count at least one.

- [ ] **Step 5: Run focused and full tests**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_flexible_maxent.py -v && .venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the flexible model**

```bash
git add provenance_sdm/src/provenance_sdm/flexible_maxent.py provenance_sdm/tests/test_flexible_maxent.py
git commit -m "feat: add clamped flexible MaxEnt sensitivity model"
```

---

### Task 5: Result-blind flexible pilot, gate, runner, and audit

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/flexible_runner.py`
- Create: `provenance_sdm/tests/test_flexible_runner.py`
- Modify: `provenance_sdm/src/provenance_sdm/cli.py`
- Modify: `provenance_sdm/tests/test_cli.py`

**Interfaces:**
- Consumes: exact every-fourth-species subset, conventional/PM primary backgrounds, flexible model, and candidate L2 values `(2.0, 5.0, 10.0)`.
- Produces: `expected_flexible_keys(config: StudyConfig, community_indices: Sequence[int] = (0, 1, 2)) -> pd.DataFrame`, `run_flexible_pilot(config: StudyConfig, landscape: Landscape, output_dir: Path, regularizations: Sequence[float]) -> tuple[Path, Path]`, `run_flexible_sensitivity(config: StudyConfig, landscape: Landscape, gate_path: Path, output_dir: Path) -> Path`, and `audit_flexible_sensitivity(output_dir: Path, config: StudyConfig) -> dict[str, object]`.

- [ ] **Step 1: Write failing exact-subset and gate tests**

```python
def test_flexible_design_has_1800_exact_keys(study_config):
    keys = expected_flexible_keys(study_config)
    assert len(keys) == 1_800
    assert set(keys.species_id) == {f"sp_{index:03d}" for index in range(0, 200, 4)}
    assert set(keys.background_arm) == {"conventional_tgb", "pm_tgb"}

def test_gate_selects_smallest_fully_stable_regularization():
    rows = fake_pilot_rows({2.0: False, 5.0: True, 10.0: True})
    gate = select_flexible_regularization(rows, (2.0, 5.0, 10.0))
    assert gate == {"include": True, "regularization": 5.0, "reason": "smallest fully stable candidate"}

def test_gate_does_not_read_truth_metrics():
    rows = fake_pilot_rows({2.0: True}).drop(columns=PRIMARY_METRICS, errors="ignore")
    assert select_flexible_regularization(rows, (2.0,))["include"] is True
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_flexible_runner.py -v`

Expected: collection fails for missing `flexible_runner`.

- [ ] **Step 3: Implement the 600-row-per-candidate pilot**

Recreate community index zero, all six scenarios, the 50-species subset, and
the two TGB arms. Pilot rows contain keys, candidate regularization, stability
diagnostics, convergence, common budget, basis, and landscape hash; they must
not contain any truth metric. Stop testing candidates after the first fully
stable 600-row set and write `flexible_pilot.parquet` plus
`flexible_gate.json`.

- [ ] **Step 4: Implement the gated 1,800-row run**

Refuse a gate without `include: true` and an allowed frozen regularization.
Recreate all three communities and use the same primary background and
evaluation seeds. Write all five truth metrics, evaluation hashes, stability
fields, exact model label, selected L2 value, and source-condition metadata to
`flexible_sensitivity_metrics.parquet`. Resume only missing exact keys and
reject stale checkpoints.

- [ ] **Step 5: Implement pilot and full-run audits**

For an included gate, require exactly 600 stable pilot rows for the selected
candidate and exactly 1,800 complete stable full rows. For an excluded gate,
require a complete tested-candidate pilot history and no claim that a full
artifact passed. Ensure every return value can be serialized by `json.dumps`.

- [ ] **Step 6: Add CLI commands and parser tests**

```text
provenance-sdm flexible-pilot --config config/study.yaml --landscape outputs/gb_grid.parquet --output outputs --regularizations 2 5 10
provenance-sdm run-flexible --config config/study.yaml --landscape outputs/gb_grid.parquet --gate outputs/flexible_gate.json --output outputs
provenance-sdm audit-flexible --config config/study.yaml --output outputs --report outputs/flexible_audit.json
```

- [ ] **Step 7: Run focused and full tests**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_flexible_runner.py tests/test_cli.py -v && .venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 8: Commit flexible orchestration**

```bash
git add provenance_sdm/src/provenance_sdm/flexible_runner.py provenance_sdm/src/provenance_sdm/cli.py provenance_sdm/tests/test_flexible_runner.py provenance_sdm/tests/test_cli.py
git commit -m "feat: gate and run flexible model sensitivity"
```

---

### Task 6: Oriented effects and hierarchy-preserving mechanism summaries

**Files:**
- Modify: `provenance_sdm/src/provenance_sdm/summaries.py`
- Modify: `provenance_sdm/tests/test_summaries.py`

**Interfaces:**
- Consumes: frozen primary metrics, mechanism diagnostics, latent metrics, and optional flexible metrics.
- Produces: `oriented_paired_effects(metrics)`, `diagnostic_arm_effects(primary, latent)`, and `mechanism_correlations(primary, diagnostics, n_boot=2_000, seed=20260730)`.

- [ ] **Step 1: Write failing orientation and merge tests**

```python
def test_oriented_effects_make_lower_errors_positive(fake_metrics):
    effects = oriented_paired_effects(fake_metrics)
    assert effects.query("metric == 'integrated_error'").oriented_effect.tolist() == pytest.approx([0.05] * 4)
    assert effects.query("metric == 'suitability_spearman'").oriented_effect.tolist() == pytest.approx([0.1] * 4)

def test_mechanism_correlations_require_one_diagnostic_per_pair(fake_metrics, fake_diagnostics):
    duplicate = pd.concat([fake_diagnostics, fake_diagnostics.iloc[[0]]])
    with pytest.raises(ValueError, match="unique"):
        mechanism_correlations(fake_metrics, duplicate, n_boot=20)
```

- [ ] **Step 2: Run tests and verify missing-symbol failures**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_summaries.py -v`

Expected: imports fail for the three new functions.

- [ ] **Step 3: Implement oriented primary and diagnostic contrasts**

Use `paired_effects` as the validated primary pair builder. Add
`orientation = -1` only for `integrated_error` and `response_curve_error`, and
`orientation = 1` for all other primary metrics. Diagnostic contrasts must
align all four semantic pair keys exactly and retain an explicit contrast
label.

- [ ] **Step 4: Implement deterministic nested bootstrap correlations**

For each metric/alignment/bias group, calculate Spearman correlation between
`ecological_overlap_tv` and `oriented_effect`. In each of 2,000 draws, sample
community labels with replacement, then species labels within each selected
community, retaining paired scenario rows. Return estimate, 2.5% and 97.5%
percentiles, pair count, draw count, and seed.

- [ ] **Step 5: Run focused and full tests**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_summaries.py -v && .venv/bin/pytest -q`

Expected: all tests pass and existing primary bootstrap output is unchanged.

- [ ] **Step 6: Commit the summaries**

```bash
git add provenance_sdm/src/provenance_sdm/summaries.py provenance_sdm/tests/test_summaries.py
git commit -m "feat: summarize provenance-matching mechanism"
```

---

### Task 7: Extension figures, tables, and fail-closed reproducibility audit

**Files:**
- Modify: `provenance_sdm/src/provenance_sdm/figures.py`
- Modify: `provenance_sdm/src/provenance_sdm/reproducibility.py`
- Modify: `provenance_sdm/src/provenance_sdm/cli.py`
- Modify: `provenance_sdm/tests/test_figures.py`
- Modify: `provenance_sdm/tests/test_reproducibility.py`
- Modify: `provenance_sdm/tests/test_cli.py`

**Interfaces:**
- Consumes: all frozen primary and audited extension artifacts.
- Produces: updated simulation/mechanism figures, deterministic empirical map, tables 5-6, and a final audit requiring complete extension evidence.

- [ ] **Step 1: Write failing mechanism figure tests**

```python
def test_mechanism_figures_replace_constant_unsupported_panel(tmp_path, figure_metrics, fake_diagnostics, fake_latent):
    paths = write_mechanism_figures(figure_metrics, fake_diagnostics, fake_latent, tmp_path)
    assert {path.name for path in paths} == {"source_composition_mechanism.png", "latent_mixture_contrasts.png"}
    assert all(path.stat().st_size > 1_000 for path in paths)
```

Update the empirical map fixture to contain species in non-alphabetic input
order, assert the alphabetically first configured species is selected, and
assert its colorbar label contains `per million cells`.

- [ ] **Step 2: Write failing extension export and audit tests**

```python
def test_export_includes_mechanism_and_flexible_tables(complete_extension_run, tmp_path):
    paths = export_manuscript(tmp_path, complete_extension_run, tmp_path / "submission", n_boot=20)
    assert {path.name for path in paths} >= {"table_5_mechanism.csv", "table_6_flexible_sensitivity.csv"}

def test_deepmaxent_gate_without_complete_metrics_is_not_included(tmp_path, study_config):
    config = complete_tiny_run(tmp_path, study_config)
    audit = build_reproducibility_audit(tmp_path, config)
    assert audit["deepmaxent"]["status"] == "gate_passed_no_complete_run"
```

- [ ] **Step 3: Run focused tests and verify failures**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_figures.py tests/test_reproducibility.py -v`

Expected: tests fail because extension builders and complete-run DeepMaxEnt status do not exist.

- [ ] **Step 4: Implement neutral figures and map scaling**

The mechanism panel plots ecological-overlap TV against oriented Spearman
effect with scenario-specific smooth-free point summaries. The diagnostic-arm
panel displays hierarchical mean contrasts with intervals and a zero line.
Multiply empirical normalized-mass differences by `1_000_000`, label the
colorbar accordingly, and select the alphabetically first configured focal
species available in the map artifact.

- [ ] **Step 5: Export auditable extension tables**

`table_5_mechanism.csv` contains the 30 scenario/metric correlations and both
latent diagnostic contrasts with sample counts, units, method, configuration
hash, and input hashes. `table_6_flexible_sensitivity.csv` contains paired
scenario/metric effects and the selected L2 value when included; for a failed
gate it contains one explicit exclusion row backed by the pilot and gate
hashes.

- [ ] **Step 6: Extend the reproducibility audit**

Require passing mechanism and latent audits. Require either a passing flexible
full audit or a valid failed-gate pilot report. Scan extension species labels,
tables, captions, and manuscript-source filenames for excluded taxa without
scanning the sentence that documents the exclusion itself. A DeepMaxEnt gate
with no exact complete metric artifact reports
`gate_passed_no_complete_run`, never `included`.

- [ ] **Step 7: Add a tested mechanism-figure command**

```text
provenance-sdm figures-mechanism --primary outputs/simulation_metrics.parquet --diagnostics outputs/mechanism_diagnostics.parquet --latent outputs/latent_mixture_metrics.parquet --output manuscript/figures
```

The command must read only the three supplied tidy artifacts and call
`write_mechanism_figures`; add a parser test asserting every required path and
an integration test asserting both expected PNG files.

- [ ] **Step 8: Run focused and full tests**

Run: `cd provenance_sdm && .venv/bin/pytest tests/test_figures.py tests/test_reproducibility.py tests/test_cli.py -v && .venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 9: Commit reporting and audit changes**

```bash
git add provenance_sdm/src/provenance_sdm/figures.py provenance_sdm/src/provenance_sdm/reproducibility.py provenance_sdm/src/provenance_sdm/cli.py provenance_sdm/tests/test_figures.py provenance_sdm/tests/test_reproducibility.py provenance_sdm/tests/test_cli.py
git commit -m "feat: export and audit mechanism evidence"
```

---

### Task 8: Execute and freeze extension production artifacts

**Files:**
- Generate: `provenance_sdm/outputs/mechanism_diagnostics.parquet`
- Generate: `provenance_sdm/outputs/latent_mixture_metrics.parquet`
- Generate: `provenance_sdm/outputs/mechanism_audit.json`
- Generate: `provenance_sdm/outputs/flexible_pilot.parquet`
- Generate: `provenance_sdm/outputs/flexible_gate.json`
- Generate conditionally: `provenance_sdm/outputs/flexible_sensitivity_metrics.parquet`
- Generate: `provenance_sdm/outputs/flexible_audit.json`

**Interfaces:**
- Consumes: tested runners, frozen study configuration, and frozen GB grid.
- Produces: complete audited extension evidence used by every subsequent claim.

- [ ] **Step 1: Verify branch, free space, primary hashes, and clean tracked state**

Run: `git branch --show-current && df -h . && git status --short && cd provenance_sdm && shasum -a 256 outputs/simulation_metrics.parquet outputs/empirical_metrics.parquet`

Expected: branch is `paper/ecomodelling-provenance-backgrounds`, available space exceeds 20 GiB, and no tracked primary file is modified. Record both hashes in the run log.

- [ ] **Step 2: Run and audit the complete mechanism experiment**

Run: `cd provenance_sdm && .venv/bin/provenance-sdm run-mechanism --config config/study.yaml --landscape outputs/gb_grid.parquet --crs EPSG:27700 --output outputs`

Then run: `cd provenance_sdm && .venv/bin/provenance-sdm audit-mechanism --config config/study.yaml --output outputs --report outputs/mechanism_audit.json`

Expected: 3,600 unique diagnostics, 3,600 unique latent fits, zero missing/unexpected/duplicate/failure rows, and passed finite/stability checks.

- [ ] **Step 3: Run the result-blind flexible pilot**

Run: `cd provenance_sdm && .venv/bin/provenance-sdm flexible-pilot --config config/study.yaml --landscape outputs/gb_grid.parquet --output outputs --regularizations 2 5 10`

Expected: the gate selects the smallest fully stable candidate or records a
complete transparent exclusion. Do not inspect truth-recovery metrics because
the pilot artifact does not contain them.

- [ ] **Step 4: Run the full flexible sensitivity only if included**

Read only `include`, `regularization`, and stability summary fields from
`outputs/flexible_gate.json`. If `include` is true, run:

`cd provenance_sdm && .venv/bin/provenance-sdm run-flexible --config config/study.yaml --landscape outputs/gb_grid.parquet --gate outputs/flexible_gate.json --output outputs`

If false, skip this command and retain the failed gate as the preregistered result.

- [ ] **Step 5: Audit flexible evidence and rerun the final complete audit**

Run: `cd provenance_sdm && .venv/bin/provenance-sdm audit-flexible --config config/study.yaml --output outputs --report outputs/flexible_audit.json`

Then run: `cd provenance_sdm && .venv/bin/provenance-sdm audit-all --config config/study.yaml --root . --output outputs/reproducibility_audit.json`

Expected: both commands return zero and `core_status` is `passed`.

- [ ] **Step 6: Regenerate figures and tables from audited artifacts**

Run:

```bash
cd provenance_sdm
.venv/bin/provenance-sdm figures-simulation --results outputs/simulation_metrics.parquet --output manuscript/figures
.venv/bin/provenance-sdm figures-mechanism --primary outputs/simulation_metrics.parquet --diagnostics outputs/mechanism_diagnostics.parquet --latent outputs/latent_mixture_metrics.parquet --output manuscript/figures
.venv/bin/provenance-sdm figures-empirical --results outputs/empirical_metrics.parquet --maps outputs/empirical_maps.parquet --output manuscript/figures
.venv/bin/provenance-sdm export-manuscript --config config/study.yaml --root . --output manuscript --bootstrap-draws 2000
```

Expected: all declared PNG and CSV files are regenerated from tidy artifacts.
Inspect every PNG with the local image viewer and every CSV with pandas for row
counts, finite values, hashes, units, and non-promotional labels. Remove the
superseded `contrast_conditions.png` only after confirming its replacement is
valid; report that removal as recoverable from the preceding commit or
regeneration command.

- [ ] **Step 7: Commit selected frozen artifacts**

```bash
git add provenance_sdm/outputs/mechanism_audit.json provenance_sdm/outputs/flexible_gate.json provenance_sdm/outputs/flexible_audit.json provenance_sdm/manuscript/figures provenance_sdm/manuscript/tables
git commit -m "results: freeze provenance mechanism evidence"
```

Large Parquet artifacts remain outside ordinary Git unless repository size
checks show each is appropriate; their hashes and reproduction commands must
be committed in the manifest regardless.

---

### Task 9: Draft and verify the journal submission package

**Files:**
- Create: `provenance_sdm/manuscript/manuscript.md`
- Create: `provenance_sdm/manuscript/supplement.md`
- Create: `provenance_sdm/manuscript/references.bib`
- Create: `provenance_sdm/manuscript/highlights.txt`
- Create: `provenance_sdm/manuscript/cover_letter.md`
- Create: `provenance_sdm/manuscript/declarations.md`
- Create: `provenance_sdm/manuscript/author_metadata.yaml`
- Generate: `provenance_sdm/submission/manuscript.docx`
- Generate: `provenance_sdm/submission/manuscript.pdf`
- Generate: `provenance_sdm/submission/supplement.docx`

**Interfaces:**
- Consumes: only passing audited tables, figures, literature sources, journal instructions, and frozen configuration.
- Produces: a self-contained *Ecological Modelling* submission package.

- [ ] **Step 1: Recheck current official journal requirements**

Browse the official *Ecological Modelling* guide for authors and record the
current article type, abstract and highlight limits, figure/table rules,
required declarations, data statement requirements, and subscription/OA
choice in `manuscript/submission_checklist.md`. Cite the official URLs and
retrieval date `2026-08-02`.

- [ ] **Step 2: Draft the manuscript around one bounded claim**

Use this exact section order:

```markdown
# Provenance matching is not effort correction in presence-only species distribution models
## Abstract
## 1. Introduction
## 2. Methods
### 2.1 Source-composition identifiability
### 2.2 Virtual communities and observation programmes
### 2.3 Background arms and models
### 2.4 Truth-based evaluation and uncertainty
### 2.5 GBIF demonstration
## 3. Results
### 3.1 Frozen primary experiment
### 3.2 Source-composition mechanism
### 3.3 Latent-mixture diagnostic and flexible sensitivity
### 3.4 Empirical model sensitivity
## 4. Discussion
## 5. Conclusions
## Data and code availability
## Declarations
## References
```

State numerical estimates with intervals and sample counts from exported
tables. Describe empirical outputs as map sensitivity, not ecological truth.
Discuss DeepMaxEnt as related work unless a complete audited comparator exists.

- [ ] **Step 3: Draft supplement and ancillary files**

The supplement must include complete scenario effects, all model-stability
diagnostics, publisher and spatial-width robustness, the flexible pilot gate,
full cleaning audit, taxon/target-group manifests, and reproducibility
commands. `author_metadata.yaml` uses author `Arzaan Ul Mairaj`, corresponding
email `arzaaan789@gmail.com`, and affiliation `Independent researcher, United
Kingdom` unless the user supplies an authorized institutional affiliation
before submission.

- [ ] **Step 4: Verify every citation and originality claim**

Check each DOI/title against a primary publisher page or official repository.
The literature set must include the original DeepMaxEnt paper, target-group
background theory and empirical tests, virtual-species bias-correction
benchmarks, GBIF survey/effort metadata guidance, and robust ML importance
weighting under covariate shift. Phrase novelty as a documented search result,
not proof that no related method exists.

- [ ] **Step 5: Human-edit the prose and run consistency checks**

Use the available human-writing cleanup skills, then search for unsupported
superiority language, occupancy/probability claims, inconsistent sample
counts, excluded taxa in result-bearing artifacts, missing figure/table
citations, unexpanded abbreviations, and references absent from the
bibliography. Preserve technical precision over stylistic variety.

- [ ] **Step 6: Render Word and PDF outputs and inspect every page**

Use the document skill's required render-and-verify workflow. Check title page,
headings, equations, captions, references, figure resolution, table overflow,
page breaks, line numbering if required, and supplement cross-references.
Iterate until the rendered pages have no clipping, overlap, blank accidental
pages, or unreadable graphics.

- [ ] **Step 7: Run final code, artifact, and repository verification**

Run: `cd provenance_sdm && .venv/bin/pytest -q && .venv/bin/provenance-sdm audit-all --config config/study.yaml --root . --output outputs/reproducibility_audit.json`

Run from repository root: `git diff --check && git status --short --branch && git rev-parse master && git rev-parse paper/ecomodelling-provenance-backgrounds`

Expected: full tests and audit pass, no whitespace errors or secrets are
tracked, `master` remains at its original commit, and all submission changes
are on the paper branch.

- [ ] **Step 8: Commit the verified submission package**

```bash
git add provenance_sdm/manuscript provenance_sdm/submission provenance_sdm/README.md
git commit -m "docs: prepare Ecological Modelling submission"
```

- [ ] **Step 9: Push only the separate paper branch**

Run: `git push -u origin paper/ecomodelling-provenance-backgrounds`

Expected: the remote paper branch updates successfully; do not open or execute any merge into `master`.
