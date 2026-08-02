# Provenance matching is not effort correction: mechanism-paper amendment

## Decision summary

The paper will retain the frozen 14,400-fit simulation and 360-fit GBIF
experiment, but its central claim will change from introducing an expected
improvement to testing the limits of a plausible correction. The paper will
show that matching a focal species' observed recording-source composition is
not generally equivalent to matching its latent sampling effort. Observed
source proportions can contain ecological signal because a species is more
likely to be recorded by programmes whose spatial effort overlaps its niche.

The manuscript will target *Ecological Modelling*. It will remain a
simulation-led methods paper with a secondary four-species British mammal
demonstration. All bats remain excluded. Work remains on
`paper/ecomodelling-provenance-backgrounds`; this workflow will not merge the
branch into `master`.

## Contribution and claim boundary

The paper's contribution will be a controlled counterexample and an
identifiability result, not a claim that provenance metadata are useless. It
will:

1. formalize why observed source proportions combine sampling allocation,
   spatial programme effort, and focal-species suitability;
2. quantify that distortion in virtual communities where all three processes
   are known;
3. test conventional TGB, observed-composition PM-TGB, a latent-mixture
   diagnostic TGB, and true oracle effort under identical conditions;
4. show whether conclusions about PM-TGB persist under a stable flexible
   feature basis; and
5. demonstrate that provenance matching can materially alter empirical maps
   even when empirical ecological truth is unavailable.

The manuscript may conclude that PM-TGB improves, has no effect, or worsens
truth recovery in any condition. It will not describe association as causal,
treat `datasetKey` as a survey protocol, infer occupancy from
presence-background scores, or claim universal failure of target-group
backgrounds. Its practical conclusion will distinguish coarse source labels
from event-level protocol and effort metadata.

## Frozen evidence

The following outputs are primary evidence and will not be redefined or tuned:

- 14,400 simulation fits: 200 species, three communities, three
  taxonomy-programme alignments, two bias levels, and four background arms;
- 360 empirical fits: four non-bat mammals, three spatial block widths, five
  folds, two provenance levels, and three background arms;
- the linear MaxEnt-equivalent basis, normalized balanced loss, L2 value 2.0,
  common background budgets, seeds, evaluation samples, and stability rules;
- hierarchical bootstrap settings and the PM-TGB minus conventional-TGB
  primary contrast.

The completed primary experiment found small, heterogeneous, and often
negative PM-TGB effects, while true oracle effort performed best on average.
Those results motivated this amendment but will not be used to choose new
seeds, species, scenarios, metrics, or favourable subsets.

## Mechanistic estimands

For focal species \(f\), programme \(s\), and landscape cell \(x\), define:

- \(m_f(s)\): the latent allocation weight of species \(f\) to programme
  \(s\);
- \(e_s(x)\): programme \(s\)'s normalized spatial effort;
- \(\lambda_f(x)\): focal-species suitability; and
- \(p_f(s)\): the realized proportion of focal occurrence records carrying
  source label \(s\).

The expected observed source composition is

\[
q_f(s) =
\frac{m_f(s)\sum_x e_s(x)\lambda_f(x)}
     {\sum_r m_f(r)\sum_x e_r(x)\lambda_f(x)}.
\]

Thus \(p_f(s)\) estimates \(q_f(s)\), not \(m_f(s)\). The analysis will export
one row for every one of the 3,600 species-scenario pairs with three
predeclared total-variation distances:

- ecological-overlap distortion: \(TV(q_f,m_f)\);
- finite-record distortion: \(TV(p_f,q_f)\); and
- total source-composition distortion: \(TV(p_f,m_f)\).

It will also retain alignment, bias level, niche breadth, record count, and the
existing focal-versus-target source distance. Formula calculations must use
the deterministic simulation objects and must not estimate latent quantities
from model outcomes.

The primary mechanistic summary will relate ecological-overlap distortion to
oriented PM-TGB effects, where positive always means better truth recovery.
Spearman, AUC, and upper-decile overlap retain their sign; integrated and
response-curve errors are multiplied by minus one. For each outcome and
alignment-bias scenario, the summary will report Spearman rank correlation.
Two thousand bootstrap draws will resample communities and then species while
retaining all paired scenario rows, matching the primary uncertainty
hierarchy. Estimates and 95% percentile intervals will be retained regardless
of direction or statistical significance.

## Latent-mixture diagnostic arm

A new simulation-only arm, `latent_mixture_tgb`, will use the same
source-stratified target-group candidate records as PM-TGB, but source totals
will match known latent weights \(m_f(s)\) rather than realized focal source
proportions \(p_f(s)\). Unsupported source mass will use the same explicit
conventional-TGB fallback. It is an oracle diagnostic and will never be
presented as an empirically available method.

The arm will run once for all 3,600 species-scenario pairs using the frozen
primary linear model, common background budget, evaluation samples, and
stability thresholds. Its predeclared contrasts are latent-mixture TGB minus
observed PM-TGB and latent-mixture TGB minus conventional TGB. These contrasts
test whether access to allocation weights resolves the problem; they do not
assume that target-group occurrence locations within a programme equal the
programme's effort surface. The existing `oracle_effort` arm remains the only
arm sampled directly from known focal effort.

## Flexible-feature sensitivity

The primary linear basis deliberately misspecifies nonlinear virtual niches.
To test whether that underfitting masks a background effect, a simulation-only
sensitivity will compare conventional TGB and observed PM-TGB for exactly 50
species per community: `sp_000`, `sp_004`, ..., `sp_196`. This deterministic
subset gives 1,800 paired fits across all communities and observation
scenarios.

The sensitivity model will use standardized linear, squared, and pairwise
interaction features. Standardization parameters and each feature's raw
training range will be learned only from the combined training presence and
background rows. Prediction inputs will be clamped to those ranges before
feature expansion, preventing the uncontrolled polynomial extrapolation found
in the discarded empirical diagnostic run.

Regularization will be selected without viewing any effect or truth-recovery
metric. A 600-fit one-community pilot will test L2 values 2.0, 5.0, and 10.0 in
ascending order; the smallest value with finite outputs, converged solvers,
maximum cell mass at most 10%, and at least 50 effective cells in every fit
will be frozen. If no value passes, the flexible analysis will be reported as
a failed sensitivity rather than altered until favourable. Once frozen, all
1,800 planned fits will be audited for exact keys, paired evaluation hashes,
and the same stability limits.

## DeepMaxEnt decision

Ryckewaert et al. (2026) will be used to position learned multispecies
features and normalized Poisson training in the literature. DeepMaxEnt is not
required to establish the source-composition mechanism and will not delay the
submission package. It may appear as a separately labelled secondary
comparator only if the already declared implementation gate passes and a
complete preregistered run is produced. A pilot or gate label alone is not a
DeepMaxEnt result. If no complete comparable run is available, the manuscript
will cite and discuss DeepMaxEnt but exclude it from result tables and claims.

## Empirical role

The empirical analysis remains a sensitivity demonstration rather than proof
of ecological accuracy. It will report source overlap, spatially held-out
discrimination, upper-suitability-area overlap, centroid shift, and map
disagreement for brown hare, European hedgehog, hazel dormouse, and red
squirrel. The primary view remains `datasetKey` at 50-km blocks; publisher
provenance and 25-km/100-km blocks remain robustness analyses.

The map-difference legend will be rescaled to interpretable mass per million
grid cells. Species selection for the displayed map will follow a deterministic
rule stated in the caption rather than the largest or most favourable effect.
No empirical result will be described as improved ecological truth.

## Artifacts and figures

New auditable artifacts will be kept separate from the frozen primary files:

- `outputs/mechanism_diagnostics.parquet`: exactly 3,600 distortion rows;
- `outputs/latent_mixture_metrics.parquet`: exactly 3,600 diagnostic-arm rows;
- `outputs/flexible_sensitivity_metrics.parquet`: exactly 1,800 rows if the
  result-blind stability gate passes; otherwise a pilot failure JSON replaces
  this artifact and no full flexible run is claimed;
- JSON audits for each artifact, including exact expected keys and stability;
- a mechanism table with distortion summaries and diagnostic contrasts; and
- a flexible-sensitivity table with the frozen regularization and paired
  contrasts.

The current condition figure's constant unsupported-mass panel will be
replaced by ecological-overlap distortion. The planned main figures are:

1. a process workflow separating suitability, programme effort, latent
   allocation, and observed source composition;
2. frozen PM-TGB versus conventional-TGB truth-recovery contrasts;
3. source-composition distortion and diagnostic-arm contrasts;
4. empirical source-level held-out contrasts; and
5. one deterministic empirical map-disagreement example.

All figure and table builders will consume frozen tidy artifacts, never rerun
models, and will be tested for expected filenames and required columns.

## Failure handling and reproducibility

Every added runner will derive seeds from semantic keys, checkpoint atomically,
refuse stale feature-basis or regularization labels, and write explicit failure
records. Audits will reject missing, duplicate, unexpected, non-finite, or
unstable rows. Existing primary files will be read-only inputs to summaries.

Implementation will be test-first. Focused tests will cover the composition
formula, total-variation bounds, latent-mixture weighting and fallback,
deterministic subset keys, clamped feature expansion, result-blind
regularization selection, checkpoint rejection, oriented effects, and complete
audits. The full suite and final reproducibility audit must pass before claims,
figures, tables, or manuscript prose are committed.

## Submission success criteria

The paper is submission-ready when:

1. the mechanism and latent-mixture artifacts and their exact-key audits pass;
2. the flexible sensitivity artifact passes its exact-key audit, or a complete
   result-blind pilot report documents why the full run was excluded;
3. all primary, mechanism, diagnostic, empirical, and reproducibility tests
   pass with no bat tokens in submission artifacts;
4. the manuscript states one bounded conclusion supported even by null or
   negative effects; and
5. the cover letter, highlights, declarations, data/code availability text,
   figures, tables, supplement, and manuscript satisfy current
   *Ecological Modelling* submission requirements.
