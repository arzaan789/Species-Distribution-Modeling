# Linear MaxEnt stability amendment

## Decision

The primary MaxEnt-equivalent model will use a linear environmental feature
basis in both the virtual-species experiment and the GBIF demonstration. The
completed nonlinear simulation and empirical artifacts are diagnostic outputs,
not paper results, and will be preserved separately. Every affected analysis
will be rerun from clean result paths before manuscript export.

This amendment does not change the paper's question, PM-TGB algorithm,
background budgets, random-seed derivation, evaluation samples, species,
provenance levels, spatial blocks, or primary contrast.

## Reason for the correction

The original basis contained linear, squared, and all pairwise interaction
terms. On the empirical grid, several remote-sensing indices are exact or
near-exact transformations of one another, and a few cells lie outside the
training predictor cloud. In a representative European hedgehog fold, the
nonlinear model produced a maximum logit of 181.9 while the 99.9th percentile
was 5.8. One cell received all normalized suitability mass after numerical
clipping, leaving an effective map size of one cell.

Removing redundant predictors alone retained the isolated maximum. Holding
the data, class balancing, and regularization fixed while replacing the
polynomial basis with a linear basis reduced maximum cell mass to 0.019%,
increased inverse-Simpson effective map size to approximately 159,000 cells,
and kept all 243,541 grid cells within the retained log-intensity range. The
root cause is therefore uncontrolled polynomial extrapolation, not storage,
cross-validation, GBIF data, or the provenance weighting method.

The correction is based on numerical validity rather than favourable PM-TGB
results. Old outputs will remain available for audit and will not be mixed
with corrected outputs.

## Model boundary

`MaxentModel.transform` will standardize the declared environmental predictors
using means and scales estimated from the combined training presence and
background rows, then return only those standardized linear columns. No
squared terms, interactions, predictor clamping, or empirical feature selection
will be introduced. The existing lower log-intensity clip will remain solely
as a documented numerical safeguard before exponentiation.

`fit_maxent` will retain the frozen L2 regularization value, balanced classes,
deterministic solver, and presence/background interpretation. Predictions will
remain normalized relative suitability masses, not occurrence or occupancy
probabilities.

Virtual ecological truth will continue to include nonlinear niches. This
intentional model misspecification applies equally to every background arm and
tests whether background construction improves recovery under a simpler fitted
model; it does not allow any method access to truth.

## Stability diagnostics

Each fitted result will export:

- maximum normalized cell mass;
- inverse-Simpson effective cell count;
- full-grid log-intensity range;
- number and proportion of cells at the numerical lower clip;
- logistic solver convergence status.

The simulation and empirical audits will require finite diagnostics and reject
maps with fewer than 50 effective cells or maximum cell mass above 10%. These
thresholds detect numerical collapse rather than select favourable PM-TGB
effects, and will be applied identically to every arm. The diagnostics will
otherwise be reported. Boyce will remain secondary and will retain its explicit
defined flag and reason; undefined values will never be imputed.

## Artifact handling and reruns

The following generated artifacts will be moved intact to a dated diagnostic
directory before reruns:

- `simulation_metrics.parquet` and `simulation_audit.json`;
- `empirical_metrics.parquet` and `empirical_maps.parquet`.

The corrected simulation will run all 14,400 frozen fit keys from a clean
primary result path. The empirical analysis will rerun all 360 combinations:
four species, three block widths, five folds, two provenance levels, and three
background arms. Corrected outputs will carry a feature-basis label so stale
and current results cannot be combined silently.

The empirical runner will also export row-level fold assignments and a
block/class count audit for every species and block width. Paired arms will
continue to share exact evaluation hashes.

## Tests and verification

Implementation will be test-first. Regression tests will establish that:

1. the transformed design contains exactly one column per declared predictor;
2. a hand-built outlier grid cannot reproduce the polynomial single-cell
   collapse under the linear basis;
3. simulation and empirical outputs identify the linear basis and contain
   finite stability diagnostics;
4. row-level spatial assignments preserve whole blocks and cover every planned
   species, width, and fold;
5. corrected runners retain unique complete design keys and shared paired
   evaluation hashes.

Focused tests and the complete suite must pass before either production rerun.
After reruns, formal simulation, empirical, and reproducibility audits must pass
before figures, tables, or manuscript claims are regenerated.

## Claim boundary

The paper's single message remains that source provenance can improve or alter
target-group background approximation under heterogeneous recording effort.
The linear model is a controlled comparison instrument, not a claim that
linear responses are universally optimal. Empirical evidence will describe
model sensitivity and map disagreement, not ecological truth, and negative or
null PM-TGB effects will be retained.
