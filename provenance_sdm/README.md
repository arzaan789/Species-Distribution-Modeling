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
