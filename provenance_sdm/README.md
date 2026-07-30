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
