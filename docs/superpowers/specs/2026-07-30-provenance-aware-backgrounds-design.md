# Provenance-aware backgrounds for presence-only species distribution models

## Decision summary

The manuscript will target *Ecological Modelling* through its subscription
publication route. It will be a simulation-led methodological paper with a
secondary empirical demonstration using four non-bat British mammals. Its
publishable core will not depend on DeepMaxEnt.

All work will remain on the branch
`paper/ecomodelling-provenance-backgrounds`. Nothing will be merged into
`master` by this workflow.

## Purpose and contribution

Presence-only species distribution models commonly use occurrences of
taxonomically related species as target-group background (TGB) records. This
assumes that related species share an observation process. Aggregated
biodiversity repositories such as GBIF combine records from many datasets,
publishers, citizen-science platforms, and recording programmes, so taxonomy
may be a poor proxy for shared sampling effort.

The paper will introduce a provenance-aware target-group background (PM-TGB)
strategy. PM-TGB will stratify candidate target-group records by their source
and reweight those records to reproduce the focal species' observed source
composition. The method is intended to approximate the focal observation
process more closely than an unweighted taxonomy-based TGB.

The methodological contribution has three parts:

1. a source-stratified background construction algorithm;
2. a virtual-species framework in which ecological suitability and observation
   effort are separately known;
3. an analysis of the conditions under which provenance matching succeeds,
   has no material effect, or fails.

The empirical demonstration will quantify source overlap and model sensitivity
for brown hare, hazel dormouse, European hedgehog, and red squirrel in Great
Britain. It will not be used as proof of ecological accuracy because unbiased
empirical presence-absence truth is unavailable.

## Research questions and hypotheses

The study will answer:

1. Does PM-TGB recover true suitability better than conventional TGB when
   taxonomic grouping and recording-programme membership are misaligned?
2. How do observation-bias strength, taxonomy–programme alignment, niche
   breadth, record abundance, and unsupported provenance affect its
   performance?
3. Does empirical source overlap explain the magnitude of map disagreement
   between conventional and provenance-aware backgrounds?
4. If the optional DeepMaxEnt arm passes its inclusion gate, does multispecies
   feature sharing reduce or amplify sensitivity to heterogeneous observation
   processes?

The primary hypothesis is that PM-TGB will improve truth recovery as the focal
species' source composition becomes less representative of its taxonomic
target group. It should converge towards conventional TGB when their source
distributions already match. Its gains are expected to diminish when focal
sources contain too few target-group records to estimate effort.

## Scope and claim boundaries

Included empirical focal species:

- brown hare (*Lepus europaeus*);
- hazel dormouse (*Muscardinus avellanarius*);
- European hedgehog (*Erinaceus europaeus*);
- red squirrel (*Sciurus vulgaris*).

All bats are excluded from acquisition, simulation labels, configurations,
tests, results, figures, and manuscript claims.

The paper will not:

- treat background records as confirmed absences;
- interpret presence-background scores as occupancy probabilities;
- claim that `datasetKey` perfectly identifies survey protocol;
- claim universal superiority for PM-TGB, MaxEnt, neural networks, or
  DeepMaxEnt;
- use random cross-validation as evidence of spatial transferability;
- decide whether to report DeepMaxEnt based on favourable or statistically
  significant results.

## Provenance-aware background algorithm

For a focal species, let \(p_f(s)\) be the proportion of its occurrence records
from source \(s\). Let \(n_T(s)\) be the number of candidate target-group
records from source \(s\).

For every supported source with \(n_T(s) > 0\), each candidate record from that
source receives an initial weight proportional to:

\[
\frac{p_f(s)}{n_T(s)}.
\]

Consequently, the total background weight assigned to a supported source
equals the focal species' observed mass for that source, rather than the
source's abundance in the pooled target group.

Unsupported focal-source mass is
\(u = \sum_{s:n_T(s)=0}p_f(s)\). That mass will be assigned to the conventional
pooled TGB distribution as an explicit fallback. The value of \(u\) will be
reported for every species and simulation. This makes PM-TGB degrade
continuously towards conventional TGB instead of silently dropping focal
sources or failing to produce a model.

The primary empirical source identifier will be GBIF `datasetKey`. A
predeclared robustness analysis will repeat provenance matching at the
`publishingOrgKey` level to test sensitivity to source granularity. The
publisher-level analysis will always be reported as supplementary evidence and
will be discussed explicitly if it changes the primary interpretation.

Background points will be sampled as unique landscape cells after source
weights are calculated. All methods will use the same background-cell budget
per focal species.

## Simulation design

### Landscape and ecological truth

The primary simulation landscape will be a projected Great Britain
environmental grid derived from the existing climate and land-cover
predictors. Coordinates will use a projected metric coordinate reference
system for spatial operations. Environmental predictors will be standardized,
and a bounded set of continuous environmental components will drive virtual
niches.

Each virtual species will have a known suitability intensity generated from
linear, quadratic, and interacting environmental responses. Species parameters
will span specialist-to-generalist niche breadths and small-to-large suitable
areas. Suitability truth will be retained separately from every model input.

### Virtual communities and observation programmes

Each simulation replicate will contain 200 virtual species, ten synthetic
taxonomic groups, and six recording programmes. Three independently seeded
communities will be generated.

Every ecological community will be observed under a paired grid of:

- taxonomy–programme alignment: low, partial, and high;
- observation-bias intensities: moderate and strong.

This produces six observation scenarios per community while retaining the
same ecological truth for paired comparisons. Species occurrence counts will
follow a bounded long-tail distribution from 20 to 2,000 records. Niche
breadth and record abundance will be retained as continuous explanatory
variables.

Recording programmes will have distinct, known spatial effort surfaces built
from smooth hotspots and accessibility-like gradients. Every species will
have a mixture over those programmes. The alignment factor will control how
similar programme mixtures are among species in the same synthetic taxonomic
group. High alignment makes conventional TGB a good proxy for focal effort;
low alignment makes taxonomy and provenance disagree. Observed presence-only
records will be sampled from the product of true suitability and the
species-specific mixture of programme effort surfaces. Every record will
retain its programme identifier, which acts as the simulated analogue of
`datasetKey`.

### Background arms

Every species-scenario pair will use:

1. **Uniform:** cells sampled uniformly across the accessible landscape.
2. **Conventional TGB:** pooled occurrences of other species in the same
   synthetic taxonomic group.
3. **PM-TGB:** the same candidate taxonomic pool, source-stratified and
   reweighted to match the focal source composition.
4. **Oracle effort:** cells sampled from the focal species' known observation
   effort surface.

The oracle arm defines a best-available effort correction and is prohibited
from exposing effort or suitability truth to any other arm.

### Primary model

The primary model will be a regularized MaxEnt-equivalent
presence-background logistic or Poisson point-process model using a fixed,
documented environmental feature basis. The same feature representation,
regularization, background budget, seeds, and evaluation samples will be used
for all four background arms. Model complexity will be frozen after pilot
verification and before the full experiment.

The full primary experiment comprises:

- 200 species;
- three independently generated communities;
- six paired observation scenarios;
- four background arms.

This yields 14,400 primary model fits. The models are computationally light,
and outputs will be written incrementally so interrupted runs can resume
without duplicating completed fits.

### Truth-based evaluation

Primary outcomes:

- rank correlation between predicted and true suitability;
- integrated normalized suitability error;
- AUC using an independently generated unbiased evaluation sample;
- environmental response-curve error;
- overlap of the upper 10% true and predicted suitability cells.

The primary contrast is PM-TGB minus conventional TGB for each
species-scenario pair. Secondary contrasts compare PM-TGB with uniform and
oracle effort arms. Hierarchical bootstrap intervals will resample communities
and species while preserving paired background-arm results. Regression
summaries will estimate interactions with source mismatch, bias intensity,
taxonomy–programme alignment, niche breadth, record abundance, and unsupported
source mass. Effect sizes and uncertainty will be reported regardless of
statistical significance.

## DeepMaxEnt inclusion gate

DeepMaxEnt is a secondary robustness comparator and will be included only if
all conditions are satisfied by the end of week 6:

1. it uses the official public multispecies implementation and normalized
   Poisson formulation from Ryckewaert et al. (2026);
2. mathematical tests and the official repository example confirm the
   implementation is being called correctly;
3. pilot scenarios complete successfully across multiple seeds;
4. projected full execution can finish within one additional calendar week;
5. it produces truth-based predictions comparable with the primary arms.

If all conditions pass, DeepMaxEnt will run on the same virtual communities
under at least the six primary observation scenarios. Valid null or negative
results will still be included. If any condition fails, DeepMaxEnt will be
excluded from the results and identified as future work. The current
dissertation notebook's binary-cross-entropy function must never be labelled
or used as DeepMaxEnt.

## GBIF acquisition and provenance

### Authoritative API workflow

GBIF APIs will be used where possible:

1. validate accepted taxa and resolve taxon identifiers through the species
   API;
2. submit a reproducible, authenticated, asynchronous multi-taxon occurrence
   download request;
3. poll the download status and retrieve the resulting archive and DOI;
4. fetch source dataset metadata with
   `GET /v1/dataset/{datasetKey}`;
5. fetch publisher metadata with
   `GET /v1/organization/{publishingOrgKey}`.

The occurrence search API may be used for previews, facets, and record-count
checks, but not as the principal bulk acquisition route because paginated
search results do not automatically receive a download DOI.

The frozen occurrence query will include:

- country `GB`;
- occurrence status `PRESENT`;
- valid coordinates;
- no GBIF-flagged geospatial issue;
- complete calendar years 2022 through 2025;
- the four focal species and their documented non-bat target-group taxa.

The retained occurrence schema will include `gbifID`, `occurrenceID`,
coordinates, accepted taxon identifiers, event date/year, `datasetKey`,
`publishingOrgKey`, `basisOfRecord`, and available sampling metadata.

The repository will store sanitized query JSON, taxon manifest, download key,
download DOI, retrieval timestamp, record counts, software version, and
cryptographic hashes. GBIF username, password, notification email, and other
credentials will be supplied through ignored environment configuration and
must never be committed.

Raw GBIF archives will remain immutable and outside ordinary Git tracking.
Processed tables will retain the source keys required for provenance analysis.
The final data citation will use the GBIF download DOI; a derived-dataset DOI
will be registered if the final filtered record set requires one.

### Empirical target groups

The conventional target-group definitions documented in the dissertation will
be frozen before modelling:

- brown hare: *Lepus timidus* and *Oryctolagus cuniculus*;
- hazel dormouse: *Apodemus flavicollis*, *Apodemus sylvaticus*, and
  *Eliomys quercinus*;
- European hedgehog: *Apodemus sylvaticus*, *Sorex araneus*, and
  *Talpa europaea*;
- red squirrel: *Glis glis*, *Muscardinus avellanarius*, and
  *Tamias sibiricus*.

Taxonomic-name resolution, introduced species, and records outside Great
Britain will be audited before modelling. Any change to this frozen list must
be justified as a data-quality correction and recorded in the provenance
manifest.

## Empirical analysis

The empirical analysis will compare uniform, conventional TGB, and PM-TGB
backgrounds. The oracle arm is unavailable because true recording effort is
unknown.

Occurrence cleaning and raster extraction will be performed from the new
citable download rather than relying on processed files that dropped source
fields. Duplicate species-cell-date-source records, impossible dates,
coordinate problems, and environmental extraction failures will be reported
through a staged record-count audit.

Spatial evaluation will use projected, contiguous blocks with primary block
width 50 km and sensitivity widths 25 km and 100 km. Blocks will remain wholly
within training or test partitions, transformations will be fit only on
training folds, and folds with inadequate class support will fail with a clear
diagnostic. Training backgrounds will differ by method, but every method will
be evaluated against the same held-out focal presences and common landscape
evaluation cells within each fold.

Empirical outcomes:

- spatially blocked presence-background AUC;
- continuous Boyce index where mathematically valid;
- map rank correlation;
- overlap of upper-suitability areas;
- area and centroid shifts;
- source overlap, source-distribution distance, and unsupported provenance
  mass.

These outcomes describe practical sensitivity and transfer within the
presence-background design. They will not be described as independent
ecological truth or prevalence-calibrated accuracy.

## Software structure and data flow

New analysis code will live in a dedicated, documented package directory with
bounded modules for:

- configuration and schema validation;
- GBIF taxon resolution, download requests, status polling, and metadata;
- immutable input manifests and hashes;
- landscape construction and raster extraction;
- virtual species and observation-programme simulation;
- background construction;
- primary and optional models;
- spatial splitting;
- metrics and statistical summaries;
- figures and manuscript tables;
- command-line orchestration.

The data flow is:

1. API query or simulation configuration;
2. immutable raw records and truth;
3. validated analysis tables;
4. matched background arms;
5. fitted models;
6. tidy per-fit predictions and metrics;
7. paired summaries, figures, and manuscript tables.

Generated artifacts will record configuration and input hashes. Existing
dissertation notebooks, datasets, maps, and saved results will never be
overwritten. Useful tested code from the detached spatial-benchmark worktree
may be migrated selectively after review; caches, obsolete outputs, and its
incorrect paper framing will not be imported.

## Error handling and automated verification

The workflow will fail early with actionable diagnostics for:

- missing or changed schemas;
- invalid taxon matches;
- unsuccessful or incomplete GBIF downloads;
- missing source identifiers;
- insufficient target-group support;
- invalid or overlapping spatial folds;
- non-finite model inputs or outputs;
- incomplete scenario result matrices;
- configuration or input-hash mismatches.

Automated tests will verify:

- deterministic simulation under fixed seeds;
- expected suitability and effort behavior in controlled toy landscapes;
- PM-TGB source weights sum to one and reproduce intended source mass;
- unsupported source mass uses only the documented fallback;
- oracle truth is inaccessible to non-oracle arms;
- equal background budgets and shared evaluation samples across arms;
- no train-test leakage in spatial folds or transformations;
- metric behavior on edge cases;
- restartable, non-duplicating result generation;
- complete expected output rows;
- total exclusion of bats;
- official normalized-Poisson behavior for an included DeepMaxEnt arm.

Before claims of completion, the complete test suite, provenance audit,
scenario-count audit, and a clean rerun of the main figure/table generation
must pass.

If a local source file appears corrupt, it will first be compared by path,
size, and hash with the working copy in
`arzaan789/Species-Distribution-Modeling`. Replacement will occur only on this
paper branch and will be documented.

## Manuscript framing and outputs

The manuscript will emphasize the observation-process model and the
source-stratified background algorithm rather than presenting a four-species
case study.

Proposed working title:

> Provenance-aware target-group backgrounds under heterogeneous observation
> processes: A virtual-species and GBIF evaluation

Core figures:

1. ecological truth, recording programmes, and background-arm workflow;
2. paired truth-recovery effects across taxonomy–programme alignment and bias
   intensity;
3. failure-condition response surfaces for source mismatch, rarity, niche
   breadth, and unsupported source mass;
4. empirical source overlap and selected map-disagreement panels.

Core tables:

1. simulation factors and model settings;
2. primary paired effect sizes and uncertainty;
3. empirical record/source composition and spatial-evaluation summaries;
4. reproducibility, API, DOI, and compute manifest.

Manuscript outline:

1. taxonomy is an imperfect proxy for observation process in aggregated
   presence-only repositories;
2. formulation of conventional, provenance-aware, uniform, and oracle
   backgrounds;
3. virtual-species truth-recovery experiment;
4. four-species GBIF demonstration;
5. conditions of applicability, metadata limitations, and implications for
   reproducible ecological modelling.

## Twelve-week delivery schedule

- **Weeks 1–2:** API acquisition, provenance manifests, and simulation
  foundations.
- **Weeks 3–5:** background methods, primary model, metrics, and pilot
  experiment.
- **Week 6:** DeepMaxEnt inclusion decision.
- **Weeks 7–8:** full simulation and robustness runs.
- **Week 9:** empirical spatial analysis and maps.
- **Weeks 10–11:** statistical synthesis, figures, and manuscript.
- **Week 12:** clean rerun, reproducibility audit, repository release, and
  submission package.

The primary simulation and empirical analysis take precedence over
DeepMaxEnt, publisher-level sensitivity, and supplementary visualizations if
schedule pressure arises.

## Acceptance criteria

The design is complete when:

- PM-TGB is precisely implemented and independently tested;
- all four simulation arms use paired ecological truth and evaluation samples;
- primary simulation outputs cover the frozen design;
- claims follow truth-based simulation results rather than internal
  cross-validation alone;
- the empirical workflow is reproducible from sanitized API query files and
  citable GBIF DOI records;
- source mismatch and unsupported mass are reported, not hidden;
- DeepMaxEnt is included or excluded solely through the week-6 gate;
- no bat enters any stage;
- all manuscript figures and tables regenerate from tidy result artifacts;
- master remains unmodified and unmerged.

## Principal prior work and technical sources

- Barber et al. (2022), target-group background evaluation:
  <https://doi.org/10.1111/ddi.13442>
- Baker et al. (2024), bias correction without independent test data:
  <https://doi.org/10.1111/ddi.13802>
- Zbinden et al. (2024), pseudo-absence selection for deep-learning SDMs:
  <https://doi.org/10.1016/j.ecoinf.2024.102623>
- Ryckewaert et al. (2026), official DeepMaxEnt formulation:
  <https://doi.org/10.1111/2041-210x.70262>
- GBIF occurrence download API:
  <https://techdocs.gbif.org/en/data-use/api-downloads>
- GBIF Occurrence API:
  <https://techdocs.gbif.org/en/openapi/v1/occurrence>
- GBIF Registry API:
  <https://techdocs.gbif.org/en/openapi/v1/registry-principal-methods>
- GBIF citation guidelines:
  <https://www.gbif.org/citation-guidelines>
