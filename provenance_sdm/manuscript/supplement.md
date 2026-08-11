# Supplementary material

## Observed provenance composition is not sampling effort in presence-only species distribution models

Arzaan Ul Mairaj

## S1. Purpose and contents

This supplement records the prespecified design, complete primary contrasts, mechanistic diagnostics, flexible-feature gate, empirical sensitivity specifications, and reproducibility manifest. Machine-readable versions of all tables accompany the submission.

## S2. Simulation design

The primary experiment contained 200 species, three replicated communities, 10 taxonomic groups, six observation programmes, three taxon-programme alignment levels, two sampling-bias strengths, and four background arms. This yielded 14,400 planned and completed fits. Every scenario contained 600 paired community-species units. A further 3,600 fits evaluated the latent-mixture diagnostic.

### Table S1. Simulation design and audit identity

| Component | Value |
|---|---:|
| Virtual species | 200 |
| Replicated communities | 3 |
| Taxonomic groups | 10 |
| Observation programmes | 6 |
| Alignment levels | 3 |
| Bias levels | 2 |
| Background arms | 4 |
| Primary fits | 14,400 |
| Latent-mixture diagnostic fits | 3,600 |
| Requested background cells | 500 |
| Minimum background cells | 50 |
| Bootstrap draws | 2,000 |
| Seed | 20260730 |

## S3. Primary suitability-rank contrasts

Positive values favour PM-TGB. Values are mean paired differences with 95% community-species hierarchical bootstrap intervals.

| Alignment | Bias | PM-TGB minus conventional TGB | Pairs |
|---|---|---:|---:|
| High | Moderate | -0.0025 [-0.0052, 0.0007] | 600 |
| High | Strong | -0.0051 [-0.0104, -0.0007] | 600 |
| Low | Moderate | -0.0001 [-0.0043, 0.0043] | 600 |
| Low | Strong | -0.0024 [-0.0085, 0.0043] | 600 |
| Partial | Moderate | 0.0011 [-0.0048, 0.0076] | 600 |
| Partial | Strong | -0.0056 [-0.0110, -0.0008] | 600 |

The complete 30-row table for all five metrics is supplied as `table_2_primary_effects.csv`.

## S4. Mechanistic estimands

For focal species f, programme s, and cell x, latent programme allocation was m_f(s), effort was e_s(x), and ecological intensity was λ_f(x). Expected source composition was

\[
q_f(s) = [m_f(s) Σ_x e_s(x) λ_f(x)] / [Σ_r m_f(r) Σ_x e_r(x) λ_f(x)].
\]

Realised empirical composition p_f(s) was calculated from finite simulated records. The diagnostics were:

* ecological-overlap distortion: TV(q_f,m_f);
* finite-record distortion: TV(p_f,q_f);
* total distortion: TV(p_f,m_f).

For suitability-rank effects, within-scenario Spearman correlations between ecological-overlap distortion and oriented PM-TGB performance ranged from -0.060 to 0.079. Every hierarchical-bootstrap interval included zero.

## S5. Latent-mixture diagnostic

### Table S2. Suitability-rank contrasts for latent programme allocation

| Alignment | Bias | Versus conventional TGB | Versus observed PM-TGB |
|---|---|---:|---:|
| High | Moderate | -0.0004 [-0.0029, 0.0021] | 0.0021 [-0.0009, 0.0050] |
| High | Strong | 0.0033 [-0.0012, 0.0079] | 0.0084 [0.0029, 0.0148] |
| Low | Moderate | 0.0005 [-0.0035, 0.0050] | 0.0006 [-0.0019, 0.0032] |
| Low | Strong | 0.0053 [-0.0001, 0.0115] | 0.0077 [0.0043, 0.0111] |
| Partial | Moderate | 0.0015 [-0.0032, 0.0062] | 0.0004 [-0.0026, 0.0030] |
| Partial | Strong | 0.0041 [0.0012, 0.0069] | 0.0097 [0.0054, 0.0146] |

The full mechanism table contains 90 rows covering five metrics, six scenarios, one distortion association, and two diagnostic contrasts. It is supplied as `table_5_mechanism.csv`.

## S6. Flexible-feature gate

The result-blind pilot tested L2 regularisation values 2, 5, and 10 in 1,800 model fits, 600 per candidate. The inclusion rule required a candidate to pass every convergence and prediction-stability check. No candidate passed every check. Under the prespecified rule, the full flexible-feature sensitivity run was excluded and no effect estimate was produced. The gate result is supplied in `flexible_gate.json`, and Table S3 is supplied as `table_6_flexible_sensitivity.csv`.

## S7. GBIF occurrence cleaning

The GBIF archive contained 205,532 records. Taxon filtering removed 160 records; deduplication removed 28,636; and intersection with valid predictor cells removed 483. The final dataset contained 176,253 records, all with dataset and publisher identifiers. Duplicates shared taxon key, 1-km cell, event date, and dataset key.

### Table S3. Primary empirical results at dataset-key provenance and 50-km blocks

| Focal species | Background | Mean AUC | Mean Boyce | Centroid shift from conventional TGB (km) |
|---|---|---:|---:|---:|
| Brown hare | Conventional TGB | 0.534 | 0.273 | 0.0 |
| Brown hare | PM-TGB | 0.536 | 0.297 | 16.1 |
| Brown hare | Uniform | 0.614 | 0.688 | 193.5 |
| European hedgehog | Conventional TGB | 0.658 | 0.901 | 0.0 |
| European hedgehog | PM-TGB | 0.661 | 0.884 | 64.9 |
| European hedgehog | Uniform | 0.714 | 0.915 | 133.6 |
| Hazel dormouse | Conventional TGB | 0.656 | 0.874 | 0.0 |
| Hazel dormouse | PM-TGB | 0.659 | 0.903 | 9.5 |
| Hazel dormouse | Uniform | 0.695 | 0.897 | 94.8 |
| Red squirrel | Conventional TGB | 0.648 | 0.656 | 0.0 |
| Red squirrel | PM-TGB | 0.649 | 0.690 | 7.9 |
| Red squirrel | Uniform | 0.706 | 0.932 | 263.1 |

Results for both provenance levels and all block widths are supplied as `table_3_empirical_composition_metrics.csv`.

## S8. Reproducibility manifest

The configuration hash is `6231cb729d98622e7127d0fa7165fe6da91f36b188bcad8ecebf601f7ffaaad3`. The primary simulation-metrics SHA-256 hash is `6eeaa114817aaf723b1dd0ce21f21fa1f694d4aa47eefb4d4dea16e87b01d1fb`; the empirical-metrics hash is `94128f5fee5bf645296c5a95e149d05fb453c96256353a8dd9b5b1508b3c9438`. The complete manifest is supplied as `table_4_reproducibility_manifest.csv`. The fail-closed audit checks expected sample counts, duplicated keys, missing scenarios, prediction stability, convergence, spatial-fold artefacts, and the prespecified feature basis.
