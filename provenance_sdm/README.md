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

The complete data acquisition and analysis workflow will be documented as each
audited command is implemented.

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
