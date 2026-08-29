# Observed provenance composition is not sampling effort in presence-only species distribution models

**Arzaan Ul Mairaj**

Independent researcher, Birmingham, B1 1BA, United Kingdom

Corresponding author: Arzaan Ul Mairaj, arzaaan789@gmail.com

Target journal: *Ecological Modelling*

## Abstract

Presence-only species distribution models often use records from related taxa to represent uneven sampling effort. Source metadata suggests a refinement: match the target-group background to the focal species' observed source composition. This assumes that observed source proportions measure survey allocation. We tested that assumption with 200 virtual species, three replicated communities, six observation programmes, three levels of taxon-programme alignment, and two strengths of spatial bias. Four background strategies were compared across 14,400 models. Provenance-matched target-group backgrounds produced negligible or slightly adverse changes relative to conventional target-group backgrounds. Suitability-rank effects ranged from -0.0056 to 0.0011, with intervals excluding zero in two strongly biased scenarios. A programme's observed contribution depended jointly on its latent allocation, spatial effort, and overlap with the focal species' ecological intensity. Correlations between this ecological-overlap distortion and provenance-matching effects were weak and uncertain. A diagnostic background matched to latent programme allocation instead improved suitability-rank correlation over observed provenance matching by 0.0077 to 0.0097 in all strongly biased scenarios. A planned flexible-feature sensitivity was excluded because no regularisation candidate passed a result-blind stability gate. In an empirical demonstration with 176,253 cleaned GBIF records for four British mammals, provenance matching shifted fitted map centroids by 7.9 to 64.9 km despite small discrimination changes. Provenance labels can identify heterogeneous data-generating processes, but their observed proportions are not direct measurements of effort without information about survey allocation or detection.

**Keywords:** GBIF; MaxEnt; presence-only data; provenance; sampling bias; target-group background; virtual species

## 1. Introduction

Presence-only occurrence data support broad-scale species distribution models for many taxa, but recorded locations rarely form a random sample of the landscape. Accessibility, observer preference, institutional priorities, and survey design all affect where records are collected. When these processes correlate with environmental predictors, a model can confuse sampling effort with ecological response (Phillips et al., 2009; Fithian et al., 2015; Baker et al., 2022). This is a particular concern when models combine opportunistic records from several organisations or recording schemes.

The target-group background (TGB) is a widely used correction for this problem. Rather than drawing background locations uniformly, it samples them from occurrences of taxa believed to share the focal species' observation process. If the focal species and its target group have the same spatial sampling bias, their common bias can cancel in the presence-background contrast (Phillips et al., 2009). Empirical, theoretical, and virtual-species studies show that TGB can improve models, but its performance depends on the choice of target taxa, the amount of data, and the relation between sampling bias and ecological niches (Botella et al., 2020; Barber et al., 2021; Inman et al., 2021; Baker et al., 2022, 2024). Recent work continues to show that background selection can materially change current and projected distributions (Rausell-Moreno et al., 2025).

Aggregated biodiversity databases often retain the provenance of each record, such as its dataset, publisher, institution, or programme. This suggests a simple extension to TGB: reweight target-group records so that the background has the same mix of sources as the focal species. We call this provenance-matched TGB (PM-TGB). It uses metadata that a conventional TGB ignores. Previous theory has shown that the ecological preferences of target-group species can bias a pooled target-group background (Botella et al., 2020). We address a separate identification problem within that pool: whether the focal species' observed shares across provenance-labelled programmes recover how observation effort was allocated among them.

That assumption generally fails even when provenance labels are complete. Let m_f(s) be the latent allocation of observation programme s to focal species f, e_s(x) the programme's spatial effort at location x, and λ_f(x) the species' ecological intensity. Ignoring finite sampling, the expected observed share from programme s is

\[
q_f(s) = [m_f(s) Σ_x e_s(x) λ_f(x)] / [Σ_r m_f(r) Σ_x e_r(x) λ_f(x)].
\]

Thus q_f(s) equals the latent allocation only when the programme-specific overlap between effort and ecological intensity is constant across sources, or when another restrictive cancellation occurs. The empirical composition p_f(s) adds finite-record noise to q_f(s). Matching a background to p_f may reproduce the observed mixture of sources without reproducing the focal species' total sampling effort surface.

This distinction concerns the background sample, not the model architecture. Maximum entropy models have a point-process interpretation (Renner and Warton, 2013), and recent multi-species neural formulations such as DeepMaxEnt add flexible response functions and information sharing (Ryckewaert et al., 2026). That flexibility cannot identify effort from source proportions when ecology and observation jointly determine those proportions.

We tested whether matching a TGB to the focal species' observed provenance composition improves recovery of known ecological truth. We also measured how ecological overlap and finite records separate observed composition from latent programme allocation, then applied the method to a real multi-source occurrence archive. The study combined paired virtual-species experiments, a diagnostic that replaces observed source proportions with latent allocation, and a descriptive GBIF application to four British mammals. We expected no consistent advantage for PM-TGB because observed provenance composition is an outcome of both observation and ecology.

## 2. Materials and methods

### 2.1. Study design and reproducibility controls

We created 200 virtual species in each of three replicated communities and assigned them to 10 taxonomic groups. Six observation programmes generated occurrences. We crossed three levels of alignment between taxonomic groups and programme allocation (low, partial, and high) with two strengths of programme-specific spatial bias (moderate and strong). Each species-community-scenario combination was fitted with four background strategies, giving 200 x 3 x 3 x 2 x 4 = 14,400 primary model fits.

All background strategies for a focal species used the same simulated ecological truth and observed presences. They also used a common requested budget of 500 unique background cells, subject to a minimum of 50 available cells. This paired design made differences attributable to the background strategy rather than a new draw of the focal species or its presences. The random seed was 20260730. Before inspecting flexible-model results, we specified a stability gate and the rule for including or excluding that sensitivity analysis. Configuration hashes, input hashes, sample counts, convergence checks, and spatial-fold artefacts were audited by code.

### 2.2. Virtual landscape and species

The landscape consisted of environmental predictors standardised over valid cells. For each virtual species, ecological intensity was generated from a response surface containing linear, quadratic, and pairwise interaction terms. Linear coefficients were drawn from a normal distribution with mean 0 and standard deviation 0.9. Quadratic coefficients were negative absolute draws from a normal distribution with mean 0.45 and standard deviation 0.2, producing bounded responses. Pairwise interaction coefficients were drawn from a normal distribution with mean 0 and standard deviation 0.25. A niche-breadth multiplier was drawn uniformly from 0.5 to 2.0. Intensities were normalised over the landscape to create the known relative suitability distribution.

### 2.3. Observation programmes and occurrence records

Each programme had a spatial effort surface comprising a randomly located hotspot and a coordinate gradient. Moderate bias used a hotspot scale of 0.35 and gradient strength of 0.5; strong bias used a hotspot scale of 0.13 and gradient strength of 1.5. The smaller hotspot and steeper gradient concentrated strong-bias effort more sharply.

Programme allocations varied among species. Under low alignment, species-specific allocations were independent draws from a symmetric Dirichlet distribution. Under partial alignment, taxonomic groups shared allocation centres with concentration 8. Under high alignment, the group signal was much tighter, with concentration 80 and a small common offset. The number of occurrences followed a long-tailed distribution: a log-normal multiplier with standard deviation 1.25 was clipped to 20 to 2,000 records. An occurrence at a programme-location pair was sampled in proportion to the product of that species' programme allocation, the programme's spatial effort, and the species' ecological intensity. This joint process generated a known distinction between latent programme allocation and observed source composition.

### 2.4. Background strategies

We compared four strategies.

1. **Uniform background** sampled valid landscape cells without reference to occurrence data.
2. **Conventional TGB** used unique cells containing occurrences of other species in the focal species' taxonomic group. Every candidate occurrence received equal weight.
3. **PM-TGB** used the same target-group candidates but reweighted their source totals to match the focal species' observed programme proportions. The focal species itself was excluded from its target group.
4. **Oracle-effort background** sampled from the focal species' known total effort surface. This arm served as a simulation benchmark, not an implementable empirical method.

Sampling was without replacement at the cell level. PM-TGB and conventional TGB therefore differed only in source reweighting, not their taxonomic pool or requested background budget.

### 2.5. Model fitting and evaluation

The primary model was a regularised linear presence-background logistic approximation to a maximum entropy or Poisson point-process model (Phillips et al., 2006; Renner and Warton, 2013). Predictors were standardised within the modelling pipeline. Presence and background sample weights were separately normalised to 0.5 so that sample-size imbalance did not set the fitted intercept. We used L2 regularisation of 2, the limited-memory Broyden-Fletcher-Goldfarb-Shanno optimiser, and a maximum of 1,000 iterations. Predictions were converted to relative suitability mass over the landscape. They should not be read as occurrence probabilities.

We evaluated five complementary properties against known truth: Spearman correlation between predicted and true suitability, total-variation integrated error, area-weighted upper-decile Jaccard overlap, response-curve root mean square error across 20 environmental bins, and AUC using independently sampled truth presences and area background. Fits were required to converge, place no more than 10% of predicted mass in one cell, and have an effective predicted cell count of at least 50.

The primary contrast was PM-TGB minus conventional TGB. For metrics where a smaller value was better, signs were retained in result tables but oriented before mechanism-correlation analyses so that positive always meant an improvement. Uncertainty intervals came from 2,000 hierarchical bootstrap draws that sampled communities and then species within communities. This preserved the replicated community structure and repeated measurements of species.

### 2.6. Mechanistic decomposition

For every simulated focal species, we retained three programme compositions: latent allocation m_f, expected observed composition q_f, and realised composition p_f. We measured ecological-overlap distortion as the total-variation distance TV(q_f,m_f), finite-record distortion as TV(p_f,q_f), and total distortion as TV(p_f,m_f). Within each alignment-bias scenario, we estimated Spearman correlations between ecological-overlap distortion and the oriented PM-TGB effect. Community-species hierarchical bootstraps provided 95% intervals.

We then ran a diagnostic background strategy identical to PM-TGB except that it matched target-group candidates to the known latent allocation m_f, rather than the observed composition p_f. This latent-mixture TGB is not available for ordinary presence-only records. It tests whether the information PM-TGB tries to recover could be useful if it were observed without ecological-overlap and finite-record distortion. We compared this diagnostic with both conventional TGB and PM-TGB in 3,600 additional fits.

### 2.7. Flexible-feature sensitivity gate

The primary linear model deliberately isolated the effect of background construction. To test whether model flexibility changed the conclusion, we planned a polynomial-feature sensitivity analysis. Before examining contrast estimates, a pilot tested L2 values of 2, 5, and 10 in 1,800 fits. A candidate had to pass every predefined convergence and prediction-stability check. The full sensitivity run would proceed only if at least one candidate passed. No candidate passed all checks, so the analysis was excluded under the predefined rule and no flexible-effect estimate was calculated.

### 2.8. Empirical GBIF demonstration

We downloaded GBIF occurrences from Great Britain for 2022 to 2025 for four focal mammals and their prespecified target taxa: brown hare (*Lepus europaeus*), European hedgehog (*Erinaceus europaeus*), hazel dormouse (*Muscardinus avellanarius*), and red squirrel (*Sciurus vulgaris*). The archived download contained 205,532 records (GBIF.org, 2026). We retained records with allowed taxa, present occurrence status, finite coordinates, no geospatial issue, valid predictor cells, and complete provenance. We removed duplicates sharing taxon, 1-km cell, event date, and dataset key. The resulting archive contained 176,253 records across focal and target taxa.

The environmental grid covered Great Britain in EPSG:27700 at 1-km resolution. Nine remotely sensed predictors were retained after preprocessing: bare soil index, land-surface temperature, modified normalised difference water index, normalised difference built-up index, normalised difference snow index, normalised difference vegetation index, normalised difference water index, soil-adjusted vegetation index, and urban index. The final landscape contained 243,541 valid cells.

We compared uniform, conventional TGB, and PM-TGB backgrounds. Dataset key was the primary provenance level; publisher key was a sensitivity level. Projected spatial blocks with widths of 25, 50, and 100 km defined five folds, with 50-km blocks prespecified as primary. We recorded fold-wise AUC, continuous Boyce correlation, source-composition distance, and the distance between each fitted map's suitability-weighted centroid and the conventional-TGB centroid. Because true suitability and true effort were unavailable, this application was descriptive. Cross-validation scores indicate transfer among held-out spatial blocks under the same observation system; they do not establish which map is ecologically correct.

## 3. Results

### 3.1. Primary virtual-species experiment

All 14,400 planned primary fits were present, converged, and passed the prediction-stability checks. PM-TGB did not consistently improve truth recovery over conventional TGB (Figure 2). For suitability rank correlation, mean paired effects ranged from -0.0056 to 0.0011 across the six observation scenarios. The estimates were -0.0025 (95% interval -0.0052 to 0.0007) under high alignment and moderate bias, and -0.0051 (-0.0104 to -0.0007) under high alignment and strong bias. Under partial alignment they were 0.0011 (-0.0048 to 0.0076) for moderate bias and -0.0056 (-0.0110 to -0.0008) for strong bias. Both low-alignment intervals included zero.

The remaining metrics also showed no uniform advantage. Mean differences in integrated error ranged from -0.0001 to 0.0015; response-curve error from -0.0007 to 0.0021; upper-decile overlap from -0.0032 to less than 0.0001; and unbiased AUC from -0.0019 to 0.0002. Signs and interval coverage varied among scenarios, but effect sizes were small. In this controlled design, adding observed source proportions to a correctly specified target group supplied no general improvement.

### 3.2. Why observed composition differed from allocation

Observed source composition frequently differed from latent programme allocation because programmes sampled different parts of each species' niche (Figure 3). Nevertheless, ecological-overlap distortion alone did not have a clear monotonic association with the performance of PM-TGB. For suitability rank correlation, within-scenario Spearman estimates ranged from -0.060 to 0.079, and every 95% interval included zero. This result does not imply that the decomposition is irrelevant. It shows that the final model contrast also depends on the spatial arrangement of target-group records, finite sampling, background-cell availability, and the fitted response surface.

The latent-mixture diagnostic separated the value of programme-allocation information from the use of observed proportions (Figure 4). Under strong bias, matching the known latent allocation improved suitability rank correlation over PM-TGB in all alignment scenarios: 0.0084 (0.0029 to 0.0148) for high alignment, 0.0077 (0.0043 to 0.0111) for low alignment, and 0.0097 (0.0054 to 0.0146) for partial alignment. Compared with conventional TGB, the latent-mixture effect was 0.0041 (0.0012 to 0.0069) under partial alignment and strong bias; the corresponding high- and low-alignment intervals included or nearly included zero. Under moderate bias, latent-mixture contrasts were small and uncertain.

Programme-allocation information can therefore improve a background under concentrated sampling, but the observed mixture of provenance labels is not that allocation. The diagnostics explain why PM-TGB can underperform, although they do not establish a simple failure threshold.

### 3.3. Flexible-feature sensitivity

The polynomial-feature pilot completed all 1,800 planned fits. None of the L2 candidates passed every result-blind stability requirement. The predefined gate therefore excluded the full run. We report no flexible-model contrast and do not use the pilot to select a favourable regularisation value.

### 3.4. Empirical demonstration

At the primary dataset-key and 50-km specification, PM-TGB changed discrimination only slightly relative to conventional TGB (Figure 5). Mean AUC changed from 0.534 to 0.536 for brown hare, 0.658 to 0.661 for European hedgehog, 0.656 to 0.659 for hazel dormouse, and 0.648 to 0.649 for red squirrel. Mean Boyce correlations changed in both directions, from 0.273 to 0.297, 0.901 to 0.884, 0.874 to 0.903, and 0.656 to 0.690, respectively.

Small changes in these scores did not mean that the maps were identical. Relative to conventional TGB, the PM-TGB suitability centroid shifted by 16.1 km for brown hare, 64.9 km for European hedgehog, 9.5 km for hazel dormouse, and 7.9 km for red squirrel (Figure 6). Uniform backgrounds had higher mean AUC for all four species, but also produced much larger centroid shifts. Neither pattern identifies ecological truth because evaluation presences came from the same heterogeneous occurrence system. The empirical result instead demonstrates that provenance reweighting can alter spatial inference even when a familiar discrimination statistic changes little.

## 4. Discussion

Matching a target-group background to a focal species' observed provenance composition did not provide a consistent correction for sampling effort. Across 14,400 paired virtual-species fits, its effects were near zero or slightly adverse. Under strong spatial bias, the diagnostic using latent programme allocation performed better than observed matching. Programme heterogeneity therefore mattered, but correcting for it required information that aggregate source proportions did not contain.

The distinction follows directly from the observation process. A programme contributes many records to a species when it is allocated to that species, when it searches places where the species is ecologically likely, or both. An observed source share cannot separate these causes. Complete dataset keys or publisher identifiers solve a traceability problem, but they do not by themselves identify sampling effort. PM-TGB estimates an observed mixture p_f, which is a noisy version of q_f; the quantity needed for this form of effort matching is closer to m_f. Calling both quantities "effort" hides the central identification problem.

The result does not invalidate the TGB principle. Conventional TGB can be effective when focal and target taxa share an observation process (Phillips et al., 2009; Barber et al., 2021), and its performance depends on how bias and niches align (Baker et al., 2022). PM-TGB adds another assumption: after restricting the pool to a suitable target group, the focal species' source proportions must describe source allocation rather than ecological overlap. Stronger taxon-programme alignment did not guarantee that this assumption held. More metadata did not automatically produce a less biased background.

Recently released taxon-stratified GBIF effort rasters use the spatial distribution of observation counts and species richness, rather than a focal species' source composition (El-Gabbas, 2026). Our experiment does not evaluate those products. It instead cautions against treating provenance shares as an additional species-specific allocation signal when the underlying survey allocation is unknown.

The weak within-scenario correlations between ecological-overlap distortion and PM-TGB effects deserve a cautious reading. Total-variation distance compresses a multivariate composition difference into one number. Two focal species can have the same distance but distort different programmes, and those programmes can occupy different environmental regions. Model effects also depend on which target-group cells remain after deduplication and on the fitted environmental coefficients. The mechanistic equation establishes non-identifiability, while the correlation analysis shows that its downstream magnitude cannot be predicted from a scalar composition distance alone.

Bias correction and model flexibility address different problems. DeepMaxEnt applies the maximum entropy principle to flexible multi-species neural models and can improve ecological representation through shared learning (Ryckewaert et al., 2026). A flexible response surface may reduce misspecification or exploit cross-species information, but it cannot determine whether a programme's large observed share arose from greater allocation or stronger niche overlap without additional information. DeepMaxEnt therefore provides modelling context for this study, not a comparator. The planned polynomial sensitivity did not pass its predefined stability gate, so the present empirical claim is limited to the audited linear feature basis.

The British mammal demonstration shows why a single validation score is insufficient. PM-TGB changed AUC by only 0.001 to 0.003 for three species and by 0.002 for the fourth, yet shifted one map centroid by nearly 65 km. Uniform backgrounds scored higher under the chosen spatial cross-validation design, but the held-out records inherited the same heterogeneous observation system. Higher discrimination against those records does not prove better recovery of ecological suitability. When ecological truth is unavailable, studies should compare maps, test sensitivity to the provenance definition, and state the estimand explicitly.

The inference has four main limitations. The primary model used linear standardised features. This improved interpretability and stability but did not represent every ecological response used in contemporary SDMs. The failed flexible-feature gate describes the audited pipeline; it is not evidence that all flexible models are unstable. The simulation included programme hotspots and gradients, taxon-programme alignment, long-tailed sample sizes, and nonlinear ecological truth, but it cannot cover every citizen-science or monitoring workflow. In the empirical data, one GBIF dataset key may combine several field protocols, while different keys may share an observation process; publisher key is coarser still. Finally, no independent standardised survey was available to assess ecological accuracy.

Finer reweighting of aggregate provenance counts will not resolve the identification problem. Data providers and aggregators should preserve sampling-event identifiers, lists of taxa sought, routes or visited sites, observation duration, absences or non-detections, protocol, and changes in programme scope through time. The Humboldt Extension to Darwin Core now provides structured terms for survey scope, sampling design, protocol, and effort (Sica et al., 2026). These fields can separate where a programme looked from what it found. Without them, provenance labels remain useful for stratified sensitivity analyses, source holdouts, hierarchical effects, and transparent reporting, but they describe data origin rather than effort.

## 5. Conclusions

Observed provenance composition is not, in general, sampling effort. Reweighting a target-group background to reproduce that composition gave no general improvement in known-truth recovery and sometimes reduced it slightly. A diagnostic based on latent programme allocation recovered some performance under strong bias, confirming that the missing quantity can matter. Presence-only studies can use provenance to expose heterogeneity and test sensitivity, but effort correction requires metadata about allocation, visits, protocols, or detection, not only the proportions of records supplied by each source.

## Declarations

### Funding

This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

### Competing interests

The author declares no competing interests.

### Ethics statement

No human participants, personal data, or animal handling were involved. The empirical analysis used publicly archived biodiversity occurrence records.

### Data and code availability

The GBIF occurrence archive is available at https://doi.org/10.15468/dl.nb4kse. Code, configuration, audit manifests, manuscript tables, and figure-generation workflows are available from a versioned Zenodo archive (Ul Mairaj, 2026) and at https://github.com/arzaan789/Species-Distribution-Modeling/tree/paper/ecomodelling-provenance-backgrounds. Remotely sensed predictor rasters that cannot be redistributed are documented through source and preprocessing manifests; the processed modelling grid is covered by the repository audit manifest.

### CRediT author statement

Arzaan Ul Mairaj: Conceptualisation, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualisation, Writing - original draft, Writing - review and editing, Project administration.

### Declaration of generative AI and AI-assisted technologies in the writing process

During preparation of this work, the author used OpenAI Codex for literature discovery, code review, analysis verification, document preparation, and language editing. The author verified the cited sources, analytical outputs, and manuscript text, revised the generated material, and takes full responsibility for the content of the published article.

## References

Baker, D.J., Maclean, I.M.D., Goodall, M., Gaston, K.J., 2022. Correlations between spatial sampling biases and environmental niches affect species distribution models. Global Ecology and Biogeography 31, 1038-1050. https://doi.org/10.1111/geb.13491.

Baker, D.J., Maclean, I.M.D., Gaston, K.J., 2024. Effective strategies for correcting spatial sampling bias in species distribution models without independent test data. Diversity and Distributions 30, e13802. https://doi.org/10.1111/ddi.13802.

Barber, R.A., Ball, S.G., Morris, R.K.A., Gilbert, F., 2021. Target-group backgrounds prove effective at correcting sampling bias in Maxent models. Diversity and Distributions 28, 128-141. https://doi.org/10.1111/ddi.13442.

Botella, C., Joly, A., Monestiez, P., Bonnet, P., Munoz, F., 2020. Bias in presence-only niche models related to sampling effort and species niches: Lessons for background point selection. PLOS ONE 15, e0232078. https://doi.org/10.1371/journal.pone.0232078.

El-Gabbas, A., 2026. A global, taxon-stratified, high-resolution sampling-effort dataset from GBIF for bias-aware ecological modelling. Diversity and Distributions 32, e70205. https://doi.org/10.1111/ddi.70205.

Fithian, W., Elith, J., Hastie, T., Keith, D.A., 2015. Bias correction in species distribution models: pooling survey and collection data for multiple species. Methods in Ecology and Evolution 6, 424-438. https://doi.org/10.1111/2041-210X.12242.

GBIF.org, 2026. GBIF occurrence download. https://doi.org/10.15468/dl.nb4kse.

Inman, R., Franklin, J., Esque, T., Nussear, K., 2021. Comparing sample bias correction methods for species distribution modeling using virtual species. Ecosphere 12, e03422. https://doi.org/10.1002/ecs2.3422.

Phillips, S.J., Anderson, R.P., Schapire, R.E., 2006. Maximum entropy modeling of species geographic distributions. Ecological Modelling 190, 231-259. https://doi.org/10.1016/j.ecolmodel.2005.03.026.

Phillips, S.J., Dudik, M., Elith, J., Graham, C.H., Lehmann, A., Leathwick, J., Ferrier, S., 2009. Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data. Ecological Applications 19, 181-197. https://doi.org/10.1890/07-2153.1.

Rausell-Moreno, A., Galiana, N., Naimi, B., Araujo, M.B., 2025. Improving species distribution models by optimising background points: impacts on current and future climate projections. Ecological Modelling 507, 111177. https://doi.org/10.1016/j.ecolmodel.2025.111177.

Renner, I.W., Warton, D.I., 2013. Equivalence of MAXENT and Poisson point process models for species distribution modeling in ecology. Biometrics 69, 274-281. https://doi.org/10.1111/j.1541-0420.2012.01824.x.

Ryckewaert, M., Marcos, D., Botella, C., Servajean, M., Bonnet, P., Joly, A., 2026. Applying the maximum entropy principle to neural networks enhances multi-species distribution models. Methods in Ecology and Evolution 17, 1655-1670. https://doi.org/10.1111/2041-210X.70262.

Sica, Y.V., Hochachka, W.M., Stevenson, R.D., Ingenloff, K., Zermoglio, P.F., Wieczorek, J., Gan, Y.M., Schigel, D., Kachian, Z.R., Baskauf, S., Brenton, P., Kazem, A.J.N., Jetz, W., Guralnick, R., 2026. Enabling ecological survey data integration with the Humboldt Extension to Darwin Core. Ecography 2026, e08223. https://doi.org/10.1002/ecog.08223.

Ul Mairaj, A., 2026. Observed provenance composition is not sampling effort in presence-only species distribution models: code and reproducibility materials, version 1.0.0. Zenodo. https://doi.org/10.5281/zenodo.22164506.

## Figure captions

**Figure 1.** Paired simulation workflow. The same virtual species and presence sample were evaluated under four background strategies across six combinations of taxon-programme alignment and spatial sampling bias.

**Figure 2.** Paired effects of provenance-matched target-group background (PM-TGB) relative to conventional target-group background across five truth-based metrics. Points are mean paired differences; lines are 95% community-species hierarchical bootstrap intervals from 2,000 draws. Metric direction is labelled in each panel.

**Figure 3.** Mechanism linking latent programme allocation, ecological-overlap distortion, and PM-TGB performance. Each point represents a community-species combination. Lines summarise within-scenario associations and are descriptive; bootstrap intervals for all suitability-rank associations included zero.

**Figure 4.** Diagnostic contrasts for target-group backgrounds matched to latent programme allocation. Points are paired mean differences relative to conventional TGB or observed PM-TGB; lines are 95% community-species hierarchical bootstrap intervals.

**Figure 5.** Empirical source and model contrasts for four British mammals under the primary dataset-key and 50-km spatial-block specification. Cross-validation scores are descriptive because held-out records share the heterogeneous observation system.

**Figure 6.** Empirical suitability-map contrast for the primary specification. Maps compare conventional TGB and PM-TGB predictions and their difference. Values are relative suitability mass, not occurrence probability.
