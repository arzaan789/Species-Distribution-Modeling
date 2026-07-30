# Provenance-aware SDM experiment

This package implements the simulation-led evaluation described in the paper
design. Its core compares uniform, conventional target-group,
provenance-matched target-group, and oracle-effort backgrounds under known
ecological and observation-process truth.

The empirical component is restricted to brown hare, hazel dormouse, European
hedgehog, and red squirrel in Great Britain. Bats are excluded throughout.

## Development

Create an isolated environment, install the package with its test dependencies,
and run:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Citable GBIF acquisition

Taxa are resolved with the current GBIF Species Match API. Bulk occurrences
are requested through GBIF's authenticated asynchronous download API so the
paper can cite a download DOI rather than an uncitable paginated search.

Provide credentials only through ignored environment variables:

```text
GBIF_USERNAME
GBIF_PASSWORD
GBIF_NOTIFICATION_EMAIL
```

Then run:

```bash
provenance-sdm gbif-resolve \
  --config config/study.yaml \
  --output outputs/taxa.json
provenance-sdm gbif-request \
  --taxa outputs/taxa.json \
  --output outputs/gbif_request.json
provenance-sdm gbif-status \
  --download-key 0000000-260730120000000 \
  --output outputs/gbif_download.json
provenance-sdm gbif-retrieve \
  --status outputs/gbif_download.json \
  --archive raw/0000000-260730120000000.zip \
  --output outputs/gbif_archive.json
```

The committed query template excludes usernames, passwords, and notification
addresses. Generated manifests are also sanitized. Official references:
[API downloads](https://techdocs.gbif.org/en/data-use/api-downloads),
[Species API](https://techdocs.gbif.org/en/openapi/v1/species), and
[Registry API](https://techdocs.gbif.org/en/openapi/v1/registry-principal-methods).

## Empirical workflow

Build the shared projected analysis grid from the dissertation's existing
remote-sensing table:

```bash
provenance-sdm build-grid \
  --predictors ../inference_data_indices.csv \
  --output outputs/gb_grid.parquet
```

The command uses the nine continuous remote-sensing predictors, removes rows
with incomplete inputs, projects cell centres to EPSG:27700, assigns equal
1-km-cell area weights, and writes a source hash and count audit beside the
Parquet file.

After the DOI download succeeds and the archive is retrieved, run:

```bash
provenance-sdm clean-gbif \
  --config config/study.yaml \
  --archive raw/0018113-260721160103020.zip \
  --grid outputs/gb_grid.parquet \
  --taxa outputs/taxa.json \
  --output outputs
provenance-sdm run-empirical \
  --config config/study.yaml \
  --records outputs/clean_occurrences.parquet \
  --grid outputs/gb_grid.parquet \
  --taxa outputs/taxa.json \
  --output outputs
provenance-sdm figures-empirical \
  --results outputs/empirical_metrics.parquet \
  --maps outputs/empirical_maps.parquet \
  --output manuscript/figures
```

The primary empirical comparison uses 50-km projected spatial blocks.
Twenty-five- and 100-km blocks and publisher-level provenance are exported as
sensitivity analyses. Uniform, conventional target-group, and
provenance-matched backgrounds share one feasible cell budget within every
fold and are evaluated on identical held-out rows.

## DeepMaxent gate

DeepMaxent is an optional multispecies reference, loaded from the official
repository rather than reimplemented here. Install the isolated optional
dependencies, run the representative timing pilot, and evaluate the gate:

```bash
python -m pip install -e ".[deepmaxent]"
provenance-sdm deepmaxent-pilot \
  --official-checkout ../official-deepmaxent \
  --output outputs/deepmaxent_pilot.parquet
provenance-sdm deepmaxent-gate \
  --config config/study.yaml \
  --official-checkout ../official-deepmaxent \
  --pilot outputs/deepmaxent_pilot.parquet \
  --output outputs/deepmaxent_gate.json
```

The tested official version is commit
`3587ad743b3c1898f61ac1c1c5f8b2884b750db4`, associated with the
[published DeepMaxent paper](https://doi.org/10.1111/2041-210x.70262) and
[archived software](https://doi.org/10.5281/zenodo.18377697). The gate checks
the normalized Poisson formula against an independent NumPy oracle, exercises
official model/loss gradients, executes the reviewed core cells of the
repository tutorial, requires three successful pilot seeds, verifies
site-normalized prediction surfaces, and enforces a seven-day projected
runtime ceiling.

The distributed tutorial's raw pre-cropping raster folder is absent, so the
gate starts from its 19 supplied cropped rasters and reduces the tutorial to
two preflight epochs without changing model or loss code. The official
multispecies TGB mode uses the pooled set of recorded sites. It cannot express
species-specific PM-TGB supports without changing the official loss.
Accordingly, DeepMaxent may be reported only as uniform/conventional pooled
TGB context, never as an implementation of PM-TGB.

## Clean regeneration and submission checks

From a fresh checkout on the paper branch:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,deepmaxent]"
python -m pytest -q
set -a
source .env
set +a
```

Run the GBIF resolve/request/status/retrieve commands above until the DOI
archive is complete, then build the grid, clean the archive, and run the
empirical workflow. Run the simulation and its completeness audit:

```bash
provenance-sdm simulate \
  --config config/study.yaml \
  --landscape outputs/gb_grid.parquet
provenance-sdm audit-simulation \
  --config config/study.yaml \
  --results outputs/simulation_metrics.parquet
provenance-sdm summarize-simulation \
  --results outputs/simulation_metrics.parquet \
  --output outputs
provenance-sdm figures-simulation \
  --results outputs/simulation_metrics.parquet \
  --output manuscript/figures
```

Run the optional DeepMaxent pilot and gate without delaying the core
simulation. Finally export the four manuscript tables and audit the complete
submission bundle:

```bash
provenance-sdm export-manuscript \
  --config config/study.yaml \
  --output manuscript
provenance-sdm audit-all \
  --config config/study.yaml \
  --output outputs/reproducibility_audit.json
```

`audit-all` fails unless the 14,400 primary fits are complete and finite,
background budgets are paired, all four empirical species have five folds at
25/50/100 km and both provenance levels, the GBIF DOI/hash and grid hash are
present, all cleaning stages are retained, four valid submission PNGs exist,
and no excluded taxon occurs in retained submission data. DeepMaxent status is
recorded but never determines core success.
