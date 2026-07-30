# Provenance-Aware SDM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible simulation-led evaluation of provenance-aware target-group backgrounds, followed by a four-species GBIF demonstration and an optional faithful DeepMaxEnt comparison.

**Architecture:** A new `provenance_sdm/` Python project will keep simulation truth, observation processes, background construction, models, GBIF acquisition, empirical evaluation, and reporting in separate modules. Configuration and immutable manifests will drive restartable runners that emit tidy Parquet/CSV artifacts; figures and manuscript tables will consume only those artifacts.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, scikit-learn, PyYAML, requests, pyarrow, geopandas, shapely, pyproj, rasterio, matplotlib, seaborn, pytest, responses, and optionally the official `RYCKEWAERT/deepmaxent` dependencies.

## Global Constraints

- Work only on `paper/ecomodelling-provenance-backgrounds`; never merge into `master`.
- Target *Ecological Modelling* through its subscription publication route.
- The publishable core must not depend on DeepMaxEnt.
- Include brown hare, hazel dormouse, European hedgehog, and red squirrel; exclude every bat from acquisition, configuration, tests, outputs, figures, and claims.
- Treat background records as background, not confirmed absences or occupancy outcomes.
- Use the GBIF asynchronous occurrence-download API for bulk acquisition and preserve its DOI.
- Use `datasetKey` as primary empirical provenance and `publishingOrgKey` as a supplementary sensitivity level.
- Freeze complete empirical years to 2022–2025 and country to `GB`.
- Never commit GBIF credentials, notification addresses, raw archives, or generated caches.
- Give all four simulation background arms the same cell budget and independent evaluation samples.
- Prevent suitability and true effort from reaching non-oracle methods.
- Include DeepMaxEnt only through the predeclared end-of-week-6 correctness and runtime gate.
- Preserve existing dissertation files and the detached spatial-benchmark worktree; migrate only reviewed source/tests.
- Use test-first development and commit after each independently verified task.

---

## Planned file structure

```text
provenance_sdm/
├── pyproject.toml                    # Package metadata, dependencies, test settings
├── README.md                         # Installation and exact workflow
├── .gitignore                       # Credentials, raw data, caches, generated outputs
├── config/
│   └── study.yaml                   # Frozen primary design and empirical species
├── queries/
│   └── occurrence_download.template.json
├── src/provenance_sdm/
│   ├── __init__.py
│   ├── config.py                    # Typed configuration loading and validation
│   ├── manifests.py                 # Hashes, input/output manifests, audit records
│   ├── provenance.py                # Conventional and PM-TGB source weighting
│   ├── landscape.py                 # Projected environmental analysis grid
│   ├── virtual_species.py           # Ecological suitability truth
│   ├── observation.py               # Programme effort and biased PO records
│   ├── backgrounds.py               # Uniform, TGB, PM-TGB, oracle cell sampling
│   ├── maxent.py                    # Regularized MaxEnt-equivalent baseline
│   ├── metrics.py                   # Truth and empirical metrics
│   ├── simulation_runner.py         # Restartable paired simulation experiment
│   ├── summaries.py                 # Paired effects, bootstrap intervals, audits
│   ├── gbif.py                      # Species, occurrence-download, dataset APIs
│   ├── empirical.py                 # Cleaning, raster extraction, source analysis
│   ├── spatial.py                   # Projected block folds and validation
│   ├── deepmaxent_adapter.py        # Optional official implementation boundary
│   ├── figures.py                   # Frozen-result figures
│   └── cli.py                       # User-facing commands
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_manifests.py
│   ├── test_provenance.py
│   ├── test_landscape.py
│   ├── test_virtual_species.py
│   ├── test_observation.py
│   ├── test_backgrounds.py
│   ├── test_maxent.py
│   ├── test_metrics.py
│   ├── test_simulation_runner.py
│   ├── test_summaries.py
│   ├── test_figures.py
│   ├── test_gbif.py
│   ├── test_empirical.py
│   ├── test_spatial.py
│   ├── test_deepmaxent_adapter.py
│   └── test_cli.py
└── manuscript/
    ├── tables/                      # Generated submission tables
    └── figures/                     # Generated submission figures
```

### Task 1: Create the validated study package and frozen configuration

**Files:**
- Create: `provenance_sdm/pyproject.toml`
- Create: `provenance_sdm/.gitignore`
- Create: `provenance_sdm/README.md`
- Create: `provenance_sdm/config/study.yaml`
- Create: `provenance_sdm/src/provenance_sdm/__init__.py`
- Create: `provenance_sdm/src/provenance_sdm/config.py`
- Create: `provenance_sdm/tests/test_config.py`
- Create: `provenance_sdm/tests/conftest.py`

**Interfaces:**
- Consumes: YAML path supplied by CLI or test.
- Produces: `StudyConfig`, `SimulationConfig`, `EmpiricalSpecies`, and `load_study_config(path: Path) -> StudyConfig`.

- [ ] **Step 1: Write configuration tests**

```python
def test_primary_design_is_frozen(study_config):
    assert study_config.simulation.n_species == 200
    assert study_config.simulation.n_communities == 3
    assert study_config.simulation.alignments == ("low", "partial", "high")
    assert study_config.simulation.bias_levels == ("moderate", "strong")
    assert study_config.background_arms == (
        "uniform", "conventional_tgb", "pm_tgb", "oracle_effort"
    )

def test_empirical_scope_contains_no_bats(study_config):
    names = " ".join(
        [s.key + " " + s.scientific_name for s in study_config.empirical_species]
    ).lower()
    assert "bat" not in names
    assert "pipistrell" not in names
    assert len(study_config.empirical_species) == 4
```

- [ ] **Step 2: Run the tests and confirm the package is absent**

Run: `cd provenance_sdm && python -m pytest tests/test_config.py -v`

Expected: FAIL because `provenance_sdm.config` does not exist.

- [ ] **Step 3: Create package metadata and exact dependencies**

Use `requires-python = ">=3.11"` and declare the packages from the Tech Stack.
Configure pytest with `pythonpath = ["src"]` and `testpaths = ["tests"]`.
Ignore `.venv/`, `.env`, `raw/`, `outputs/`, `*.zip`, `__pycache__/`,
`.pytest_cache/`, and credentials files.

- [ ] **Step 4: Implement typed configuration**

```python
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
    minimum_background_cells: int
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
```

`load_study_config` must reject unknown alignment/bias/arm names, non-positive
counts, years other than `(2022, 2025)`, country other than `GB`, duplicate
species, and any key or scientific name containing `bat` or `pipistrell`
case-insensitively.

- [ ] **Step 5: Write `study.yaml` with the approved exact design**

Set 200 species, 3 communities, 10 taxonomic groups, 6 programmes, alignments
`low/partial/high`, bias levels `moderate/strong`, 20–2,000 records, a
500-cell requested background cap, a 50-cell minimum common budget, seed
`20260730`, all four arms, and the approved four focal species/target groups.

- [ ] **Step 6: Run the configuration tests**

Run: `cd provenance_sdm && python -m pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the foundation**

```bash
git add provenance_sdm
git commit -m "build: initialize provenance SDM study package"
```

### Task 2: Implement immutable manifests and provenance weighting

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/manifests.py`
- Create: `provenance_sdm/src/provenance_sdm/provenance.py`
- Create: `provenance_sdm/tests/test_manifests.py`
- Create: `provenance_sdm/tests/test_provenance.py`

**Interfaces:**
- Consumes: file paths and focal/candidate source series.
- Produces: `sha256_file(path: Path) -> str`,
  `write_manifest(payload: Mapping[str, object], path: Path) -> Path`,
  `ProvenanceWeights`, and
  `pm_tgb_weights(focal_sources: pd.Series, candidate_sources: pd.Series) -> ProvenanceWeights`.

- [ ] **Step 1: Write manifest and weighting tests**

```python
def test_manifest_json_is_stable(tmp_path):
    path = write_manifest({"b": 2, "a": 1}, tmp_path / "manifest.json")
    assert path.read_text().startswith('{\n  "a": 1,')

def test_pm_tgb_matches_supported_focal_mass():
    focal = pd.Series(["A"] * 8 + ["B"] * 2)
    candidates = pd.Series(["A", "A", "B", "B"], index=[10, 11, 12, 13])
    result = pm_tgb_weights(focal, candidates)
    assert result.weights.groupby(candidates).sum().to_dict() == pytest.approx(
        {"A": 0.8, "B": 0.2}
    )
    assert result.unsupported_mass == 0.0

def test_unsupported_mass_uses_conventional_pool_fallback():
    focal = pd.Series(["A"] * 6 + ["C"] * 4)
    candidates = pd.Series(["A", "A", "B", "B"], index=[10, 11, 12, 13])
    result = pm_tgb_weights(focal, candidates)
    assert result.unsupported_mass == pytest.approx(0.4)
    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights > 0).all()
```

- [ ] **Step 2: Confirm the tests fail**

Run: `cd provenance_sdm && python -m pytest tests/test_manifests.py tests/test_provenance.py -v`

Expected: FAIL because both modules are absent.

- [ ] **Step 3: Implement deterministic manifests**

Use SHA-256 streaming in 1 MiB chunks. Serialize JSON with sorted keys,
two-space indentation, UTF-8, and a trailing newline. Refuse to overwrite a
manifest when its existing content differs unless `allow_replace=True` is
explicitly supplied.

- [ ] **Step 4: Implement the approved PM-TGB formula**

```python
@dataclass(frozen=True)
class ProvenanceWeights:
    weights: pd.Series
    focal_mass: pd.Series
    supported_mass: float
    unsupported_mass: float

def pm_tgb_weights(
    focal_sources: pd.Series, candidate_sources: pd.Series
) -> ProvenanceWeights:
    if focal_sources.empty or candidate_sources.empty:
        raise ValueError("source series must be non-empty")
    if focal_sources.isna().any() or candidate_sources.isna().any():
        raise ValueError("source labels must be complete")
    if not candidate_sources.index.is_unique:
        raise ValueError("candidate indices must be unique")

    focal_mass = focal_sources.value_counts(normalize=True)
    candidate_count = candidate_sources.value_counts()
    supported = focal_mass.index.intersection(candidate_count.index)
    weights = pd.Series(0.0, index=candidate_sources.index)
    for source in supported:
        source_rows = candidate_sources.eq(source)
        weights.loc[source_rows] = focal_mass.loc[source] / source_rows.sum()

    unsupported_mass = float(focal_mass.drop(index=supported).sum())
    weights += unsupported_mass / len(candidate_sources)
    weights /= weights.sum()
    return ProvenanceWeights(
        weights=weights,
        focal_mass=focal_mass,
        supported_mass=1.0 - unsupported_mass,
        unsupported_mass=unsupported_mass,
    )
```

Raise `ValueError` for empty inputs, missing labels, non-unique candidate
indices, or non-finite weights.

- [ ] **Step 5: Run focused and complete tests**

Run: `cd provenance_sdm && python -m pytest tests/test_manifests.py tests/test_provenance.py -v`

Expected: PASS.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 6: Commit the independently tested method**

```bash
git add provenance_sdm/src/provenance_sdm/manifests.py provenance_sdm/src/provenance_sdm/provenance.py provenance_sdm/tests
git commit -m "feat: add provenance-matched target-group weights"
```

### Task 3: Build projected landscapes and virtual ecological truth

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/landscape.py`
- Create: `provenance_sdm/src/provenance_sdm/virtual_species.py`
- Create: `provenance_sdm/tests/test_landscape.py`
- Create: `provenance_sdm/tests/test_virtual_species.py`

**Interfaces:**
- Consumes: aligned environmental rasters or deterministic toy arrays.
- Produces: `Landscape`, `SpeciesTruth`,
  `landscape_from_arrays(predictors: Mapping[str, np.ndarray], x: np.ndarray, y: np.ndarray, area: np.ndarray, crs: str) -> Landscape`, and
  `simulate_species_truth(landscape: Landscape, n_species: int, seed: int) -> tuple[SpeciesTruth, ...]`.

- [ ] **Step 1: Write projected-grid and deterministic-truth tests**

```python
def test_landscape_rejects_geographic_coordinates():
    with pytest.raises(ValueError, match="projected"):
        landscape_from_arrays(features, x, y, crs="EPSG:4326")

def test_species_truth_is_deterministic(toy_landscape):
    first = simulate_species_truth(toy_landscape, n_species=4, seed=7)
    second = simulate_species_truth(toy_landscape, n_species=4, seed=7)
    np.testing.assert_allclose(first[0].suitability, second[0].suitability)

def test_species_truth_is_normalized(toy_landscape):
    truth = simulate_species_truth(toy_landscape, n_species=4, seed=7)
    assert all(np.isclose(s.suitability.sum(), 1.0) for s in truth)
    assert all(np.ptp(s.suitability) > 0 for s in truth)
```

- [ ] **Step 2: Run and observe missing modules**

Run: `cd provenance_sdm && python -m pytest tests/test_landscape.py tests/test_virtual_species.py -v`

Expected: FAIL because landscape and virtual-species APIs do not exist.

- [ ] **Step 3: Implement `Landscape` validation**

```python
@dataclass(frozen=True)
class Landscape:
    cells: pd.DataFrame
    feature_names: tuple[str, ...]
    crs: str

    # cells columns: cell_id, x, y, area_weight, and feature_names
```

Require unique `cell_id`, projected CRS, finite coordinates/features,
strictly-positive area weights, and at least three environmental features.
Standardize only environmental columns and record their means/scales in the
landscape manifest.

- [ ] **Step 4: Implement virtual niches**

```python
@dataclass(frozen=True)
class SpeciesTruth:
    species_id: str
    taxonomic_group: int
    coefficients: np.ndarray
    niche_breadth: float
    suitability: np.ndarray
```

Generate fixed-seed linear, quadratic, and pairwise-interaction coefficients.
Assign species evenly across ten taxonomic groups, draw niche breadth over a
bounded specialist–generalist range, exponentiate stabilized scores, multiply
by area weight, and normalize suitability over valid cells.

- [ ] **Step 5: Run focused and complete tests**

Run: `cd provenance_sdm && python -m pytest tests/test_landscape.py tests/test_virtual_species.py -v`

Expected: PASS.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 6: Commit landscape and truth generation**

```bash
git add provenance_sdm/src/provenance_sdm/landscape.py provenance_sdm/src/provenance_sdm/virtual_species.py provenance_sdm/tests
git commit -m "feat: generate projected virtual-species truth"
```

### Task 4: Simulate recording programmes and all background arms

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/observation.py`
- Create: `provenance_sdm/src/provenance_sdm/backgrounds.py`
- Create: `provenance_sdm/tests/test_observation.py`
- Create: `provenance_sdm/tests/test_backgrounds.py`

**Interfaces:**
- Consumes: `Landscape`, `SpeciesTruth`, alignment/bias labels, and background budget.
- Produces: `ProgrammeEffort`, `ObservedCommunity`,
  `simulate_observations(truth: tuple[SpeciesTruth, ...], programmes: tuple[ProgrammeEffort, ...], alignment: str, bias_level: str, min_records: int, max_records: int, seed: int) -> ObservedCommunity`, and
  `make_backgrounds(observed: ObservedCommunity, focal_species: str, n_cells: int, seed: int) -> dict[str, pd.DataFrame]`.

- [ ] **Step 1: Write effort, alignment, and arm tests**

```python
def test_observed_records_follow_suitability_times_effort(toy_truth, toy_programmes):
    observed = simulate_observations(
        toy_truth, toy_programmes, alignment="partial",
        bias_level="strong", min_records=2000, max_records=2000, seed=9
    )
    expected = toy_truth[0].suitability * observed.species_effort[0]
    expected = expected / expected.sum()
    actual = observed.records.query("species_id == 'sp_000'").cell_id.value_counts(
        normalize=True
    ).sort_index()
    assert np.corrcoef(actual.reindex(range(len(expected)), fill_value=0), expected)[0, 1] > 0.9

def test_all_arms_have_same_unique_cell_budget(observed_community):
    arms = make_backgrounds(observed_community, focal_species="sp_000", n_cells=20, seed=3)
    assert set(arms) == {"uniform", "conventional_tgb", "pm_tgb", "oracle_effort"}
    assert {len(frame) for frame in arms.values()} == {20}
    assert all(frame.cell_id.is_unique for frame in arms.values())

def test_non_oracle_arms_do_not_expose_truth(observed_community):
    arms = make_backgrounds(observed_community, focal_species="sp_000", n_cells=20, seed=3)
    assert all("true_effort" not in arms[name] for name in arms if name != "oracle_effort")
```

- [ ] **Step 2: Confirm tests fail**

Run: `cd provenance_sdm && python -m pytest tests/test_observation.py tests/test_backgrounds.py -v`

Expected: FAIL because observation/background modules are absent.

- [ ] **Step 3: Implement six programme effort surfaces**

```python
@dataclass(frozen=True)
class ProgrammeEffort:
    programme_id: str
    intensity: np.ndarray

@dataclass(frozen=True)
class ObservedCommunity:
    records: pd.DataFrame
    species_effort: Mapping[str, np.ndarray]
    programme_effort: tuple[ProgrammeEffort, ...]
    source_mixtures: pd.DataFrame
    truth: tuple[SpeciesTruth, ...]
    landscape: Landscape
```

Create positive, normalized effort surfaces from Gaussian spatial hotspots and
an accessibility-like coordinate gradient. Moderate/strong bias must map to
fixed concentration parameters declared in code and written to manifests.

- [ ] **Step 4: Implement taxonomy–programme alignment and long-tail counts**

High alignment gives species in one taxonomic group closely related Dirichlet
programme mixtures; partial alignment weakens that grouping; low alignment
draws mixtures independently of taxonomy. Draw counts from a clipped lognormal
distribution and deterministically map them to 20–2,000.

- [ ] **Step 5: Implement paired arm sampling**

Use `numpy.random.Generator.choice(..., replace=False, p=weights)` over unique
valid cells. Conventional TGB pools other species in the focal taxonomic group.
PM-TGB calls `pm_tgb_weights` using focal programme labels and candidate
programme labels. Oracle effort samples from the known focal mixed effort.
Uniform samples by area weight. Before sampling, set the common paired budget
to the minimum of the 500-cell cap, conventional unique support, and positive
PM-TGB unique support. Apply that exact budget to all four arms, record it, and
raise a diagnostic only when common support is below 50 cells.

- [ ] **Step 6: Run focused and complete tests**

Run: `cd provenance_sdm && python -m pytest tests/test_observation.py tests/test_backgrounds.py -v`

Expected: PASS.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit observation and background generation**

```bash
git add provenance_sdm/src/provenance_sdm/observation.py provenance_sdm/src/provenance_sdm/backgrounds.py provenance_sdm/tests
git commit -m "feat: simulate observation programmes and background arms"
```

### Task 5: Add the primary MaxEnt-equivalent model and truth metrics

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/maxent.py`
- Create: `provenance_sdm/src/provenance_sdm/metrics.py`
- Create: `provenance_sdm/tests/test_maxent.py`
- Create: `provenance_sdm/tests/test_metrics.py`

**Interfaces:**
- Consumes: presence/background feature matrices, full landscape features, and `SpeciesTruth`.
- Produces: `MaxentModel`,
  `fit_maxent(presence: pd.DataFrame, background: pd.DataFrame, feature_names: Sequence[str], regularization: float, seed: int) -> MaxentModel`,
  and `truth_metrics(predicted: np.ndarray, truth: SpeciesTruth, unbiased_y: np.ndarray, unbiased_score: np.ndarray) -> dict[str, float]`.

- [ ] **Step 1: Write model and metric tests**

```python
def test_maxent_recovers_simple_environmental_gradient(toy_gradient_data):
    model = fit_maxent(
        toy_gradient_data.presence, toy_gradient_data.background,
        feature_names=("env_1",), regularization=1.0, seed=5
    )
    score = model.predict_suitability(toy_gradient_data.landscape)
    assert scipy.stats.spearmanr(score, toy_gradient_data.truth).statistic > 0.8

def test_truth_metrics_identical_prediction_is_optimal(toy_truth):
    values = truth_metrics(
        toy_truth.suitability, toy_truth,
        unbiased_y=np.array([1, 1, 0, 0]),
        unbiased_score=np.array([0.9, 0.8, 0.2, 0.1]),
    )
    assert values["suitability_spearman"] == pytest.approx(1.0)
    assert values["integrated_error"] == pytest.approx(0.0)
    assert values["top10_overlap"] == pytest.approx(1.0)
    assert values["unbiased_auc"] == pytest.approx(1.0)
```

- [ ] **Step 2: Confirm missing implementation**

Run: `cd provenance_sdm && python -m pytest tests/test_maxent.py tests/test_metrics.py -v`

Expected: FAIL because model and truth metrics are absent.

- [ ] **Step 3: Implement the fixed feature basis**

Construct linear and squared standardized environmental features plus the
predeclared first-order interactions. Fit an L2-regularized
`sklearn.linear_model.LogisticRegression` with deterministic solver settings.
Normalize full-landscape exponential scores to a relative suitability
distribution and document the presence-background/point-process equivalence
boundary.

- [ ] **Step 4: Implement independent evaluation generation and metrics**

Generate unbiased evaluation presences from true suitability and evaluation
background from area-weighted landscape cells with a seed not used for model
training. Implement `integrated_normalized_error` by independently normalizing
predicted and true non-negative cell values under area weights and summing the
area-weighted absolute difference. Implement `top_quantile_overlap` as the
area-weighted Jaccard overlap of cells at or above each surface's 90th
percentile. Implement `response_curve_rmse` by evaluating predicted and true
partial responses on the same 100-point grid for each predictor, scaling each
curve to `[0, 1]`, and taking the root mean square error over every predictor
and grid point. Implement continuous Boyce with ten equal-frequency
landscape-score bins, observed-to-expected ratios, and Spearman correlation
between bin rank and that ratio; return an explicit undefined-status flag
rather than a silent numeric value when fewer than three populated bins exist.

All metrics must reject non-finite values, wrong lengths, constant predictions
where undefined, and accidental use of training labels as unbiased outcomes.

- [ ] **Step 5: Run focused and complete tests**

Run: `cd provenance_sdm && python -m pytest tests/test_maxent.py tests/test_metrics.py -v`

Expected: PASS.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 6: Commit the primary modelling layer**

```bash
git add provenance_sdm/src/provenance_sdm/maxent.py provenance_sdm/src/provenance_sdm/metrics.py provenance_sdm/tests
git commit -m "feat: evaluate MaxEnt-equivalent models against truth"
```

### Task 6: Create the restartable simulation runner and completeness audit

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/simulation_runner.py`
- Create: `provenance_sdm/src/provenance_sdm/cli.py`
- Create: `provenance_sdm/tests/test_simulation_runner.py`
- Create: `provenance_sdm/tests/test_cli.py`
- Modify: `provenance_sdm/pyproject.toml`

**Interfaces:**
- Consumes: `StudyConfig`, a `Landscape`, and output directory.
- Produces: `expected_simulation_keys(config) -> pd.DataFrame`,
  `run_simulation(config, landscape, output_dir) -> Path`,
  `audit_simulation(path, config) -> dict[str, object]`, and CLI commands
  `simulate`, `audit-simulation`.

- [ ] **Step 1: Write key-count, resume, and CLI tests**

```python
def test_full_design_has_14400_fit_keys(study_config):
    keys = expected_simulation_keys(study_config)
    assert len(keys) == 14_400
    assert not keys.duplicated().any()

def test_runner_resumes_without_duplicate_rows(tiny_config, toy_landscape, tmp_path):
    path = run_simulation(tiny_config, toy_landscape, tmp_path)
    first = pd.read_parquet(path)
    path = run_simulation(tiny_config, toy_landscape, tmp_path)
    second = pd.read_parquet(path)
    pd.testing.assert_frame_equal(first, second)
    assert not second.duplicated(tiny_config.result_key_columns).any()
```

- [ ] **Step 2: Confirm runner tests fail**

Run: `cd provenance_sdm && python -m pytest tests/test_simulation_runner.py tests/test_cli.py -v`

Expected: FAIL because runner and CLI do not exist.

- [ ] **Step 3: Implement deterministic fit keys**

Use `(community_seed, alignment, bias_level, species_id, background_arm)` as
the composite key. Derive child RNG seeds by hashing the composite key with
the study seed so scheduling order cannot alter results.

- [ ] **Step 4: Implement atomic incremental outputs**

Write each completed batch to a temporary Parquet file, validate schema and
unique keys, then replace the result artifact atomically. On resume, calculate
the exact missing-key anti-join and execute only missing fits. Write failed
fits to `simulation_failures.csv` with key, exception type, and message; never
convert failures into missing numeric values silently.

- [ ] **Step 5: Implement the simulation audit**

The audit must compare actual and expected composite keys, reject duplicates,
check finite primary metrics, verify four equal-budget arms per
species-scenario, and write `simulation_audit.json` with expected/completed/
failed/missing counts and input/configuration hashes.

- [ ] **Step 6: Add CLI entry point**

Configure:

```toml
[project.scripts]
provenance-sdm = "provenance_sdm.cli:main"
```

Support:

```text
provenance-sdm simulate --config config/study.yaml --landscape path.parquet
provenance-sdm audit-simulation --config config/study.yaml --results outputs/simulation_metrics.parquet
```

- [ ] **Step 7: Run focused and complete tests**

Run: `cd provenance_sdm && python -m pytest tests/test_simulation_runner.py tests/test_cli.py -v`

Expected: PASS.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 8: Commit the restartable experiment**

```bash
git add provenance_sdm
git commit -m "feat: run and audit paired SDM simulations"
```

### Task 7: Produce paired statistical summaries and simulation figures

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/summaries.py`
- Create: `provenance_sdm/src/provenance_sdm/figures.py`
- Create: `provenance_sdm/tests/test_summaries.py`
- Create: `provenance_sdm/tests/test_figures.py`
- Modify: `provenance_sdm/src/provenance_sdm/cli.py`

**Interfaces:**
- Consumes: audited `simulation_metrics.parquet`.
- Produces: `paired_effects(metrics) -> pd.DataFrame`,
  `hierarchical_bootstrap(metrics, n_boot=2000, seed=20260730) -> pd.DataFrame`,
  and `write_simulation_figures(metrics, output_dir) -> tuple[Path, ...]`.

- [ ] **Step 1: Write paired-summary tests**

```python
def test_primary_effect_is_pm_minus_conventional(fake_metrics):
    effects = paired_effects(fake_metrics)
    row = effects.query("metric == 'suitability_spearman'").iloc[0]
    assert row.effect == pytest.approx(
        fake_metrics.query("background_arm == 'pm_tgb'").value.iloc[0]
        - fake_metrics.query("background_arm == 'conventional_tgb'").value.iloc[0]
    )

def test_bootstrap_is_deterministic(fake_metrics):
    first = hierarchical_bootstrap(fake_metrics, n_boot=100, seed=4)
    second = hierarchical_bootstrap(fake_metrics, n_boot=100, seed=4)
    pd.testing.assert_frame_equal(first, second)
```

- [ ] **Step 2: Confirm summary tests fail**

Run: `cd provenance_sdm && python -m pytest tests/test_summaries.py tests/test_figures.py -v`

Expected: FAIL because summary/figure APIs do not exist.

- [ ] **Step 3: Implement paired and hierarchical summaries**

Require complete PM-TGB/conventional pairs. Resample community first and
species second, keeping all scenarios/arms for a sampled species together.
Return estimate, 2.5%/97.5% percentile interval, sample size, metric,
alignment, bias level, and contrast. Produce stratified summaries over record
abundance, niche breadth, source-distribution distance, and unsupported mass.

- [ ] **Step 4: Implement neutral figures**

Create:

1. scenario/arm workflow panel;
2. PM-minus-conventional truth-recovery effects by alignment and bias;
3. source mismatch/record abundance/niche breadth failure-condition panels.

Titles and legends must describe effects neutrally and show uncertainty. No
figure may label non-significant effects as improvement or failure.

- [ ] **Step 5: Add summary CLI and tests**

```text
provenance-sdm summarize-simulation --results outputs/simulation_metrics.parquet --output outputs
provenance-sdm figures-simulation --results outputs/simulation_metrics.parquet --output manuscript/figures
```

Run: `cd provenance_sdm && python -m pytest tests/test_summaries.py tests/test_figures.py tests/test_cli.py -v`

Expected: PASS.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 6: Commit summaries and figures**

```bash
git add provenance_sdm
git commit -m "feat: summarize provenance background effects"
```

### Task 8: Implement authenticated, citable GBIF API acquisition

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/gbif.py`
- Create: `provenance_sdm/queries/occurrence_download.template.json`
- Create: `provenance_sdm/tests/test_gbif.py`
- Modify: `provenance_sdm/src/provenance_sdm/cli.py`
- Modify: `provenance_sdm/README.md`

**Interfaces:**
- Consumes: scientific names, frozen empirical filters, and credentials from
  `GBIF_USERNAME`, `GBIF_PASSWORD`, `GBIF_NOTIFICATION_EMAIL`.
- Produces: `GBIFClient.resolve_taxon`,
  `GBIFClient.submit_download`, `GBIFClient.download_status`,
  `GBIFClient.dataset_metadata`, `GBIFClient.organization_metadata`,
  and sanitized acquisition manifest.

- [ ] **Step 1: Write mocked API and secret-safety tests**

```python
@responses.activate
def test_resolve_taxon_requires_accepted_species(client):
    responses.get(
        "https://api.gbif.org/v2/species/match",
        json={"usage": {"key": "7952072", "status": "ACCEPTED",
                        "rank": "SPECIES",
                        "name": "Lepus europaeus Pallas, 1778"},
              "diagnostics": {"confidence": 99}},
    )
    taxon = client.resolve_taxon("Lepus europaeus")
    assert taxon.taxon_key == 7952072

def test_sanitized_manifest_contains_no_credentials(acquisition_manifest):
    serialized = json.dumps(acquisition_manifest)
    assert "GBIF_PASSWORD" not in serialized
    assert "@" not in serialized
```

- [ ] **Step 2: Confirm GBIF tests fail**

Run: `cd provenance_sdm && python -m pytest tests/test_gbif.py -v`

Expected: FAIL because the client does not exist.

- [ ] **Step 3: Implement species and metadata GET operations**

Use GBIF's current `/v2/species/match` response contract. Use a
`requests.Session` with timeout `(10, 60)`, package user-agent, and
bounded exponential retries for 429/5xx responses. Require accepted
species-rank matches; write requested/accepted names and taxon keys to the
sanitized manifest. Implement Registry API retrieval by dataset/publisher key.

- [ ] **Step 4: Build the exact download predicate**

Build the request from accepted taxon records returned by `resolve_taxon`:

```python
predicates = [
    {"type": "equals", "key": "COUNTRY", "value": "GB"},
    {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
    {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
    {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
    {"type": "greaterThanOrEquals", "key": "YEAR", "value": "2022"},
    {"type": "lessThanOrEquals", "key": "YEAR", "value": "2025"},
    {
        "type": "in",
        "key": "TAXON_KEY",
        "values": [str(taxon.taxon_key) for taxon in accepted_taxa],
    },
]
```

Use `SIMPLE_CSV`, authenticate with username/password, and pass the
notification address only in the in-memory submitted payload.

- [ ] **Step 5: Implement submit, poll, DOI, and archive verification**

Require status `SUCCEEDED`, non-empty download URL, and DOI before treating a
download as complete. Stream the archive, calculate SHA-256, verify it opens as
ZIP, and write a sanitized manifest containing query predicates, taxon
manifest, download key, DOI, URL, retrieval timestamp, archive hash, and
record-count audit.

- [ ] **Step 6: Add API CLI commands**

```text
provenance-sdm gbif-resolve --config config/study.yaml --output outputs/taxa.json
provenance-sdm gbif-request --config config/study.yaml --taxa outputs/taxa.json
provenance-sdm gbif-status --download-key 0000000-260730120000000 --output outputs/gbif_download.json
provenance-sdm gbif-metadata --archive raw/0000000-260730120000000.zip --output outputs/source_metadata.parquet
```

- [ ] **Step 7: Run focused and complete tests**

Run: `cd provenance_sdm && python -m pytest tests/test_gbif.py tests/test_cli.py -v`

Expected: PASS without making a live network request.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 8: Commit the API workflow**

```bash
git add provenance_sdm
git commit -m "feat: acquire citable GBIF occurrence data"
```

### Task 9: Add empirical cleaning, projected spatial evaluation, and map comparisons

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/empirical.py`
- Create: `provenance_sdm/src/provenance_sdm/spatial.py`
- Create: `provenance_sdm/tests/test_empirical.py`
- Create: `provenance_sdm/tests/test_spatial.py`
- Modify: `provenance_sdm/src/provenance_sdm/figures.py`
- Modify: `provenance_sdm/src/provenance_sdm/cli.py`

**Interfaces:**
- Consumes: verified GBIF archive, source metadata, projected predictor grid,
  and `StudyConfig`.
- Produces:
  `clean_occurrences(records: pd.DataFrame, valid_cell_ids: Collection[int], allowed_taxa: Collection[int]) -> CleanedOccurrences`,
  `projected_block_folds(records: pd.DataFrame, width_m: int, n_folds: int, seed: int) -> tuple[SpatialFold, ...]`,
  `run_empirical(inputs: EmpiricalInputs, output_dir: Path) -> Path`, and
  empirical figures/tables.

- [ ] **Step 1: Write staged-cleaning and bat-exclusion tests**

```python
def test_cleaning_retains_provenance_and_reports_each_stage(raw_occurrences):
    cleaned = clean_occurrences(raw_occurrences, valid_cell_ids, allowed_taxa)
    assert {"datasetKey", "publishingOrgKey", "cell_id"} <= set(cleaned.records)
    assert cleaned.audit.stage.is_unique
    assert cleaned.audit.iloc[-1].records == len(cleaned.records)

def test_cleaning_rejects_any_bat_taxon(raw_occurrences_with_bat):
    with pytest.raises(ValueError, match="excluded taxon"):
        clean_occurrences(raw_occurrences_with_bat, valid_cell_ids, allowed_taxa)
```

- [ ] **Step 2: Write projected-fold and common-evaluation tests**

```python
def test_projected_blocks_do_not_overlap(projected_occurrences):
    folds = projected_block_folds(projected_occurrences, width_m=50_000, n_folds=5, seed=8)
    for fold in folds:
        assert set(fold.train_block_ids).isdisjoint(fold.test_block_ids)

def test_empirical_arms_share_evaluation_rows(tiny_empirical_inputs, tmp_path):
    path = run_empirical(tiny_empirical_inputs, tmp_path)
    rows = pd.read_parquet(path)
    counts = rows.groupby(["species", "fold_id"]).evaluation_hash.nunique()
    assert counts.eq(1).all()
```

- [ ] **Step 3: Confirm empirical tests fail**

Run: `cd provenance_sdm && python -m pytest tests/test_empirical.py tests/test_spatial.py -v`

Expected: FAIL because empirical and projected spatial APIs are absent.

- [ ] **Step 4: Implement audited occurrence cleaning**

Apply, in order: allowed-taxon check; GB/year/status check; finite-coordinate
check; geospatial-issue check; duplicate
`(taxonKey, cell_id, event_date, datasetKey)` removal; valid-predictor-cell
join; required-provenance check. Return record counts after every stage and
fail if any focal species or target group becomes empty.

- [ ] **Step 5: Implement projected contiguous block folds**

Use metric `x/y` coordinates and block widths 25,000, 50,000, and 100,000
metres. Allocate whole blocks to five folds while requiring focal presence and
common landscape evaluation support in every test fold. Save row-level
assignments and block/class counts. Use 50 km as primary and 25/100 km as
sensitivity analyses.

- [ ] **Step 6: Implement empirical background and evaluation protocol**

For each training fold/species, create uniform, conventional TGB, and PM-TGB
training backgrounds with equal budgets. Fit preprocessing on training rows
only. Evaluate every arm against the same held-out focal presence cells and
same common area-weighted landscape cells. Produce AUC, Boyce where defined,
source-distribution distance, unsupported mass, map Spearman correlation,
upper-area overlap, area shift, and centroid shift.

- [ ] **Step 7: Add publisher-level sensitivity**

Run the same PM-TGB formula with `publishingOrgKey`. Mark rows
`provenance_level=dataset` or `publisher`; keep dataset results primary in
figures/tables and always export publisher results to supplementary tables.

- [ ] **Step 8: Add empirical CLI and figures**

```text
provenance-sdm clean-gbif --config config/study.yaml --archive raw/0000000-260730120000000.zip --grid data/gb_grid.parquet
provenance-sdm run-empirical --config config/study.yaml --records outputs/clean_occurrences.parquet
provenance-sdm figures-empirical --results outputs/empirical_metrics.parquet --maps outputs/empirical_maps.parquet
```

Generate source-overlap and selected neutral map-disagreement panels.

- [ ] **Step 9: Run focused and complete tests**

Run: `cd provenance_sdm && python -m pytest tests/test_empirical.py tests/test_spatial.py tests/test_figures.py tests/test_cli.py -v`

Expected: PASS.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 10: Commit the empirical demonstration**

```bash
git add provenance_sdm
git commit -m "feat: evaluate provenance backgrounds on GBIF mammals"
```

### Task 10: Evaluate the official DeepMaxEnt inclusion gate

**Files:**
- Create: `provenance_sdm/src/provenance_sdm/deepmaxent_adapter.py`
- Create: `provenance_sdm/tests/test_deepmaxent_adapter.py`
- Modify: `provenance_sdm/src/provenance_sdm/cli.py`
- Modify: `provenance_sdm/README.md`

**Interfaces:**
- Consumes: official DeepMaxEnt checkout/version, pilot virtual community, and
  runtime measurements.
- Produces: `normalized_poisson_reference_loss`,
  `DeepMaxentAdapter`, and
  `evaluate_deepmaxent_gate(checkout: Path, pilot: Path, config: StudyConfig) -> GateReport`.

- [ ] **Step 1: Write a framework-independent loss oracle**

```python
def test_reference_loss_matches_manual_normalized_poisson():
    counts = np.array([[2.0, 0.0], [0.0, 1.0]])
    logits = np.log(np.array([[0.8, 0.2], [0.25, 0.75]]))
    actual = normalized_poisson_reference_loss(counts, logits)
    expected = -(2.0 * np.log(0.8) + 1.0 * np.log(0.75)) / 3.0
    assert actual == pytest.approx(expected)
```

- [ ] **Step 2: Confirm adapter tests fail**

Run: `cd provenance_sdm && python -m pytest tests/test_deepmaxent_adapter.py -v`

Expected: FAIL because the adapter is absent.

- [ ] **Step 3: Implement the independent reference loss and adapter boundary**

The adapter must import the official package rather than reproduce its network
internals. Record repository URL, commit hash, environment, site/species
tensor shapes, settings, seeds, and runtime. Compare official loss output or
gradients on a toy tensor with the independent NumPy reference.

- [ ] **Step 4: Implement the gate report**

```python
@dataclass(frozen=True)
class GateReport:
    official_commit: str
    formula_check_passed: bool
    repository_example_passed: bool
    multi_seed_pilot_passed: bool
    projected_calendar_days: float
    comparable_predictions_passed: bool
    include: bool
    reasons: tuple[str, ...]
```

Set `include=True` only when every Boolean is true and projected full runtime
is at most seven calendar days. Write `deepmaxent_gate.json` before any full
DeepMaxEnt run. The CLI must refuse a full run when `include=False`.

- [ ] **Step 5: Add gate and optional-run CLI**

```text
provenance-sdm deepmaxent-gate --config config/study.yaml --official-checkout vendor/deepmaxent --pilot outputs/pilot.parquet
provenance-sdm run-deepmaxent --config config/study.yaml --gate outputs/deepmaxent_gate.json
```

- [ ] **Step 6: Verify exclusion behavior without official dependency**

Run: `cd provenance_sdm && python -m pytest tests/test_deepmaxent_adapter.py tests/test_cli.py -v`

Expected: PASS, including a test that a failed gate exits non-zero without
starting training.

Run: `cd provenance_sdm && python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit the gate independently of its outcome**

```bash
git add provenance_sdm
git commit -m "feat: enforce faithful DeepMaxEnt inclusion gate"
```

### Task 11: Add final reproducibility audit and manuscript exports

**Files:**
- Modify: `provenance_sdm/src/provenance_sdm/manifests.py`
- Modify: `provenance_sdm/src/provenance_sdm/summaries.py`
- Modify: `provenance_sdm/src/provenance_sdm/figures.py`
- Modify: `provenance_sdm/src/provenance_sdm/cli.py`
- Modify: `provenance_sdm/README.md`
- Create: `provenance_sdm/tests/test_reproducibility.py`
- Create: `provenance_sdm/manuscript/tables/.gitkeep`
- Create: `provenance_sdm/manuscript/figures/.gitkeep`

**Interfaces:**
- Consumes: audited simulation and empirical artifacts, GBIF DOI manifest, and
  optional DeepMaxEnt gate/results.
- Produces: four manuscript tables, four figure files, and
  `reproducibility_audit.json` through
  `build_reproducibility_audit(root: Path, config: StudyConfig) -> dict[str, object]`.

- [ ] **Step 1: Write end-to-end artifact contract tests**

```python
def test_reproducibility_audit_requires_core_artifacts(tmp_path, study_config):
    with pytest.raises(FileNotFoundError, match="simulation_metrics"):
        build_reproducibility_audit(tmp_path, study_config)

def test_bat_tokens_are_absent_from_submission_artifacts(complete_tiny_run):
    audit = build_reproducibility_audit(complete_tiny_run, complete_tiny_run.config)
    assert audit["excluded_taxon_scan"]["matches"] == []
```

- [ ] **Step 2: Confirm audit tests fail**

Run: `cd provenance_sdm && python -m pytest tests/test_reproducibility.py -v`

Expected: FAIL because final audit/export functions do not exist.

- [ ] **Step 3: Implement submission tables**

Write:

1. `table_1_simulation_design.csv`;
2. `table_2_primary_effects.csv`;
3. `table_3_empirical_composition_metrics.csv`;
4. `table_4_reproducibility_manifest.csv`.

Every table must include units, sample counts, method labels, primary/
supplementary status, and configuration/input hashes where applicable.

- [ ] **Step 4: Implement final artifact audit**

Require successful simulation completeness audit, finite primary effects,
GBIF download key/DOI/hash, empirical record-stage audit, three block-size
results, exact expected species names, no bat tokens, valid figure files, and
clean regeneration timestamps. Record DeepMaxEnt as `included` or `excluded`
with gate reasons; never require it for core success.

- [ ] **Step 5: Document the exact clean workflow**

README commands must cover environment creation, tests, GBIF resolution/
request/status, source metadata, landscape build, simulation pilot/full run,
audits, empirical run, optional DeepMaxEnt gate, summaries, figures, tables,
and final reproducibility audit.

- [ ] **Step 6: Add final CLI**

```text
provenance-sdm export-manuscript --config config/study.yaml --output manuscript
provenance-sdm audit-all --config config/study.yaml --output outputs/reproducibility_audit.json
```

- [ ] **Step 7: Run all verification**

Run: `cd provenance_sdm && python -m pytest -q`

Expected: all tests PASS.

Run: `cd provenance_sdm && provenance-sdm audit-simulation --config config/study.yaml --results outputs/simulation_metrics.parquet`

Expected: exit 0 with no missing/duplicate keys.

Run: `cd provenance_sdm && provenance-sdm audit-all --config config/study.yaml --output outputs/reproducibility_audit.json`

Expected: exit 0 with core status `passed`; DeepMaxEnt may be either
`included` or `excluded`.

Run: `git status --short`

Expected: only intentionally selected generated manuscript artifacts, if they
are configured for version control.

- [ ] **Step 8: Commit reproducibility and submission exports**

```bash
git add provenance_sdm
git commit -m "feat: audit and export Ecological Modelling artifacts"
```

## Execution checkpoints

1. **Core-method checkpoint:** Tasks 1–2 establish the independently testable
   PM-TGB algorithm.
2. **Publishable simulation checkpoint:** Tasks 3–7 produce the paper's
   truth-based core and must finish before empirical or DeepMaxEnt work can
   threaten the deadline.
3. **Empirical checkpoint:** Tasks 8–9 produce a citable, provenance-preserving
   four-species demonstration.
4. **Optional-method checkpoint:** Task 10 records a result-neutral DeepMaxEnt
   inclusion decision.
5. **Submission checkpoint:** Task 11 verifies and exports every core artifact.

At each checkpoint, review the diff, run the complete test suite, inspect
generated tabular schemas, and verify that `master` and the detached benchmark
worktree remain unchanged.

## Twelve-week delivery map

- **Weeks 1–2:** Tasks 1–4; freeze configuration, implement PM-TGB, and
  validate landscape, truth, effort, and background sampling.
- **Weeks 3–5:** Tasks 5–7; complete the simulation pilot, full 14,400-fit
  design, paired summaries, and truth-based figures.
- **Week 6:** Run Task 10's predeclared DeepMaxEnt gate without delaying the
  simulation core.
- **Weeks 6–8:** Tasks 8–9; submit the citable GBIF request, preserve source
  metadata, and complete the four-species empirical demonstration.
- **Weeks 9–10:** Stress-test sensitivities, lock final tables/figures, and
  draft Methods and Results from audited artifacts.
- **Weeks 11–12:** Task 11; complete reproducibility audit, internal
  manuscript review, formatting, and *Ecological Modelling* submission files.
