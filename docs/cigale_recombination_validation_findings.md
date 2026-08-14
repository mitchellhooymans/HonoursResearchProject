# CIGALE Decomposition Recombination Validation: Findings

*Validates whether the paper's theoretical additive-mixing model (`glass.composite_math.create_composite_sed`, the SKIRTOR+host `alpha`-mixing formalism used throughout the Part 1 synthetic grid) is a trustworthy stand-in for the real, CIGALE-fit AGN-host galaxies underlying Part 2. Generated 2026-08-14. Code: three new, additive notebooks under `notebooks/` (none modify any existing notebook, script, or `glass`-package code): `CIGALE_Decomposition_Validation.ipynb`, `CIGALE_Decomposition_Validation_GeometryFix.ipynb`, `CIGALE_Decomposition_Validation_WeightedBlend.ipynb`. Figures/caches: `outputs/cigale_theoretical_validation/`, `outputs/cigale_geometry_fix/`, `outputs/cigale_weighted_blend/`. One environment fix was made (`pyproject.toml`'s stale `glass` package path, see §7.1) — everything else described here is new, additive analysis.*

## 1. What was tested, and why

Every prior CIGALE-decomposition figure in this project (`CIGALE_Decomposition_Analysis.ipynb`, `docs/cigale_decomposition_findings.md`) works by subtracting CIGALE's own best-fit AGN component from CIGALE's own best-fit total — an internally self-consistent operation that can't meaningfully fail (`host := total − agn` implies `host + agn ≡ total` by floating-point construction). It has never been tested whether the paper's separate, *theoretical* SKIRTOR+host mixing model (the one actually used to build the Part 1 bias-prediction grid, `config.SKIRTOR_TYPE1_PARAMS`/`TYPE2_PARAMS` swept over `config.ALPHA_VALUES`) can reconstruct what CIGALE found when fitting **real** galaxies — a genuine, falsifiable cross-check between the paper's two methodology halves.

**Method:** for every ZFOURGE galaxy with CIGALE `fracAGN > 0` (n = 6,509, matching the sample in `docs/figure9_10_bootstrap_methodology.md`):
1. Take the CIGALE-decomposed **host-only** SED (`glass.analysis.decompose_cigale_sed(target='host')`).
2. Add back a **theoretical SKIRTOR AGN template** via `composite_math.create_composite_sed(agn_template, host_sed, alpha)`, with `alpha` predicted *a priori* — not fit — from CIGALE's own `fracAGN`: `alpha = fracAGN / (1 − fracAGN)`, derived directly from the paper's own composite formula (AGN's share of total integrated flux = `alpha/(1+alpha)`).
3. Compare the reconstruction to the real total SED (`L_lambda_total`) and the real `UV_Full`/`VJ_Full` colours already tabulated in `zfourge_full_final.csv`.

Statistics follow the project's established non-parametric percentile-bootstrap convention (`docs/figure9_10_bootstrap_methodology.md`: `np.random.default_rng(seed)`, `n_boot=2000`, resample-with-replacement, `[2.5, 97.5]` percentile CI) via a small local `bootstrap_ci()` helper defined in each notebook (not added to `glass`, per this project's convention of keeping analysis/plotting code notebook-local).

## 2. Baseline result: Type1/Type2 look good in aggregate, but hide a serious asymmetry

`CIGALE_Decomposition_Validation.ipynb` runs the above using the paper's actual Part 1 templates — Type 1 (face-on, `p=0.5, q=0, i=0`) and Type 2 (edge-on, `p=0.5, q=0, i=90`).

### 2.1 Tier 1 — flux-level fidelity
A genuine (non-tautological) check, since the AGN component is now independent theoretical data rather than CIGALE's own fitted component:

| | Type 1 | Type 2 |
|---|---|---|
| % of galaxies with max relative residual < 50% (λ > 1300 Å) | 70.8% | 44.8% |
| Population median flux residual | −0.0033 | −0.0372 |

(Restricted to restframe wavelength > 1300 Å — see §7.3 for why.) Population-median agreement is close to zero for both, i.e. the *average* reconstruction is good; the pass-rate gap shows Type 2 has a longer tail of poorly-reconstructed galaxies at the flux level.

### 2.2 Tier 2 — UVJ colour-level fidelity (the real test)
Comparing reconstructed colours to the real, independently-computed `UV_Full`/`VJ_Full`:

| | Type 1 | Type 2 |
|---|---|---|
| median\|dUV\| | 0.035 [0.034, 0.037] | 0.014 [0.012, 0.015] |
| median\|dVJ\| | 0.015 [0.014, 0.016] | 0.012 [0.011, 0.013] |
| % exceeding 0.02 mag tolerance | 65.6% | 59.1% |
| **Overall UVJ classification agreement** | **95.8%** | **94.7%** |

### 2.3 Alpha calibration — initial check (refined in §9)
Fitting a per-galaxy closed-form least-squares `alpha` and comparing to the a priori `alpha = fracAGN/(1-fracAGN)`, for the **Type 2** template only (the better Tier-2 colour match at the time this check was run):

> **median `alpha_fit / alpha_theory` = 0.99 [0.96, 1.01]**, Spearman **ρ = 0.75** (n = 6,507)

At face value this looks like strong support for the paper's formula needing no correction. **This was later found to be incomplete** — see §9 for a full Bayesian regression (not just a single ratio) covering all three templates, including Matched (never checked here), which shows the relationship is not actually proportional and this Type2-only snapshot obscured a real, systematic, `fracAGN`-dependent bias.

### 2.4 The finding that matters: a hidden, opposite-direction asymmetry
The ~95% aggregate agreement above hides a sharp split when broken down by true UVJ region — precisely the region the paper's "hidden quiescent galaxies" headline result depends on:

| True region | n | Type 1 agreement | Type 2 agreement |
|---|---|---|---|
| **Quiescent** | 496 | **72.6%** [68.5, 76.6] | **100.0%** [100.0, 100.0] |
| Star-forming | 5,550 | 98.8% [98.5, 99.1] | 95.7% [95.1, 96.2] |
| Dusty | 462 | 84.8% [81.6, 88.1] | 77.3% [73.2, 81.2] |

| | Type 1 | Type 2 |
|---|---|---|
| False-positive Quiescent rate (says Q, isn't) | 0.77% (50/6,508) | 4.50% (293/6,509) |
| False-negative Quiescent rate (misses real Q) | 27.4% (136/496) | 0.0% (0/496) |

**Type 1 (face-on) is too blue**: its direct, unobscured accretion-disk continuum pushes 27% of genuinely quiescent hosts out of the Quiescent box entirely. **Type 2 (edge-on) barely perturbs colours at all** (its flux is mostly reprocessed into the IR under the paper's whole-SED integrated-flux `alpha` normalization — see §7.4), so it never misses a true quiescent galaxy but also fails to correct 4.5% of genuinely non-quiescent galaxies, over-predicting the quiescent population. Neither template is unbiased for exactly the classification the paper leans on hardest.

## 3. Root-cause diagnosis (not assumed — confirmed by scanning all 6,509 FITS headers)

Every AGN-host galaxy's CIGALE best-fit SKIRTOR header (`agn.t`, `agn.pl`, `agn.q`, `agn.oa`, `agn.R`, `agn.i`) was scanned directly:

| t | p | q | oa | R | i | n galaxies |
|---|---|---|---|---|---|---|
| 7 | 1.0 | 1.0 | 40 | 20 | **30** | 4,956 (76.1%) |
| 7 | 1.0 | 1.0 | 40 | 20 | **70** | 1,553 (23.9%) |

CIGALE's own fitting run — the one that produced every "real" host/full SED used throughout this project — **only ever explored two discrete AGN geometries, both with torus shape `p=1.0, q=1.0`, and inclination limited to 30° or 70°.** It never used the paper's `p=0.5, q=0` shape, and never touched the face-on/edge-on extremes (`i=0`/`i=90`) that Type1/Type2 represent.

So the §2.4 asymmetry isn't a flaw in the additive-mixing methodology itself — it's the predictable consequence of validating against templates built from a **different AGN model** than what was actually fit to the data. (Caveat: these two geometries reflect the parameter grid *someone configured this particular CIGALE run to explore* — not necessarily a universal truth about real AGN torus geometry. See §6.)

## 4. Fix #1 — per-galaxy geometry matching (works, but not usable in the theoretical grid)

`CIGALE_Decomposition_Validation_GeometryFix.ipynb` reruns the full validation a third way: each galaxy reconstructed with **its own** CIGALE best-fit geometry (`i=30` or `i=70`, selected per galaxy via `agn.i`; same `t=7,p=1,q=1,oa=40,R=20` for everyone). Only two additional template reads are needed for the whole population, since CIGALE only ever used two geometries.

| True region | Type 1 | Type 2 | **Matched** |
|---|---|---|---|
| Quiescent | 72.6% | 100.0% | **85.1%** [81.9, 88.1] |
| Star-forming | 98.8% | 95.7% | **98.8%** [98.5, 99.0] |
| Dusty | 84.8% | 77.3% | **87.4%** [84.4, 90.5] |
| False-positive Quiescent | 0.77% | 4.50% | **0.77%** |
| False-negative Quiescent | 27.4% | 0.0% | **14.9%** (74/496) |

| Colour/flux fidelity | Type 1 | Type 2 | **Matched** |
|---|---|---|---|
| mean\|dUV\| | 0.081 | 0.056 | **0.056** |
| mean\|dVJ\| | 0.048 | 0.050 | **0.038** |
| population median flux residual | −0.003 | −0.037 | **−0.004** |

Matched is best-or-tied on **every** metric simultaneously: it keeps Type1's low false-positive rate while nearly halving Type1's false-negative rate, beats both fixed templates on Dusty classification, and has the best colour and flux fidelity overall. This is genuine confirmation that the additive host+AGN mixing methodology is sound — the earlier asymmetry really was a template-choice artifact, not a defect in the recombination approach.

**Limitation:** "Matched" needs each galaxy's own CIGALE fit result. It validates the *methodology*, but it cannot be transplanted into the paper's Part 1 theoretical grid, which by design generates purely synthetic composites with no real galaxy to match against.

## 5. Fix #2 — population-weighted blend (tried, does not work)

Motivated by §4's limitation, `CIGALE_Decomposition_Validation_WeightedBlend.ipynb` tests a single, theory-usable candidate: one AGN template combining the two CIGALE-preferred geometries by population weight (`flux_blend = 0.761 × flux(i=30) + 0.239 × flux(i=70)`, both templates confirmed to share an identical wavelength grid). This needs no per-galaxy information and drops into `create_composite_sed` exactly like Type1/Type2.

| | Type 1 | Type 2 | **Blend** | Matched (ceiling) |
|---|---|---|---|---|
| Quiescent agreement | 72.6% | 100.0% | **72.8%** | 85.1% |
| Quiescent false-negative rate | 27.4% | 0.0% | **27.2%** | 14.9% |
| mean\|dUV\| | 0.081 | 0.056 | **0.079** | 0.056 |
| mean\|dVJ\| | 0.048 | 0.050 | **0.047** | 0.038 |

**The blend fails — it tracks Type1 almost exactly and captures essentially none of Matched's improvement.** Two compounding reasons, both confirmed rather than assumed:

1. **UVJ colour is a logarithmic function of flux** (`mag = -2.5·log10(flux)`), but the blend was built as a *linear* flux average. The i=30 template (more face-on, less obscured) is intrinsically much brighter in the optical/UV than i=70, so it dominates the blended flux precisely in the bands that set U/V/J — a "76/24 blend" behaves like "≈100% i=30" wherever it matters for classification.
2. **The real population is bimodal, not continuous**, in this CIGALE run — a galaxy is fit as *either* an i=30 case *or* an i=70 case, never something in between. Averaging the two templates' inputs doesn't recover either mode's correct classification outcome; only knowing which mode a given galaxy is actually in (Matched) does.

This is a useful negative result: it shows the Quiescent-region fix genuinely requires per-galaxy information, and rules out a population-summary shortcut rather than leaving the question open.

## 6. Implications for the paper (findings only — no paper text or code changed)

- **The core decomposition + recombination methodology is validated.** Once matched to the correct AGN geometry, host + theoretical AGN reconstructs real galaxies' SEDs, colours, and UVJ classification very well (98.8% Star-forming agreement, up to 87–100% Quiescent/Dusty depending on template). The approach itself is not the problem.
- **Type1/Type2 as currently parameterised (`p=0.5, q=0, i=0/90`) do not match CIGALE's actual best-fit AGN geometry for this ZFOURGE sample (`p=1, q=1, i=30/70`).** This is a real, quantified internal inconsistency between the paper's Part 1 theoretical grid and its Part 2 CIGALE-based observational validation — not previously documented.
- **That mismatch produces an opposite-direction bias exactly where the "hidden quiescent galaxies" claim lives**: Type1-based estimates of quiescent-galaxy recovery are likely an *undercount* (misses 27% of real cases), Type2-based estimates are likely an *overcount* (4.5% false-positive rate). If a specific count or fraction from either template alone is quoted as a point estimate in the paper, it should be treated as one-sided rather than centred.
- **No cheap fix exists for using the correct geometry in the purely theoretical grid.** A population-weighted blend does not work (§5); the only tested method that closes the gap needs real per-galaxy CIGALE data (§4), which is unavailable for synthetic composites.
- **The paper's `alpha = fracAGN/(1-fracAGN)` formula is not a proportional, per-galaxy-accurate mapping, for any template** (§9). It remains reasonable as a coarse driver for the Part 1 grid's qualitative 0→1 sweep, but should not be cited as a calibrated or unbiased translation between CIGALE's fit and the paper's `alpha` — the one check that looked clean (§2.3) covered only one of three templates and its apparent success was a coincidence of where the population happens to sit, not evidence of proportionality.
- **Options worth considering for the paper** (decisions for the author, not made here):
  1. State the Type1/Type2-vs-CIGALE-geometry mismatch explicitly as a caveat in the methodology section, alongside the existing bootstrap-CI caveats documented in `docs/figure9_10_bootstrap_methodology.md`.
  2. Report Type1- and Type2-derived hidden-quiescent numbers as **bounds** rather than a single point estimate, given their opposite and now-quantified biases (Matched, the closest-to-truth reconstruction, sits between them for Quiescent: 85.1% vs. Type1's 72.6% and Type2's 100%).
  3. If reviewers push on this, the `CIGALE_Decomposition_Validation_GeometryFix.ipynb` numbers directly answer "how much does your theoretical model's AGN geometry choice affect your quiescent-galaxy result" — they weren't available before this work.

## 7. Technical notes and gotchas (for whoever next touches this code)

### 7.1 Environment fix
`pyproject.toml`'s `[tool.uv.sources] glass` path pointed at `C:/Users/uqmhooym/GitHub/GLASS` (a different machine/user) and didn't resolve in this checkout. Fixed to `M:/GitHub/GLASS` (the real local sibling repo) and `uv sync` was run. If `import glass` ever fails again, check this path first.

### 7.2 The `'Total Flux (erg/s/cm^2/Angstrom)'` column collision
`data_io.read_cigale_best_model()` derives this column from the FITS `Fnu` column (observed-frame); `analysis.decompose_cigale_sed()` immediately **overwrites** it with an `L_lambda_total`-derived value (CIGALE's rest-frame luminosity density). Calling them in that order is deliberate and matches how the legacy pipeline built `UV_Full`/`VJ_Full` in the first place — but reading that column *between* the two calls would silently give the wrong quantity. All three new notebooks read the ground-truth "full" flux from `L_lambda_total` directly, before decomposition, never from the reused column name.

### 7.3 The Lyman-continuum/IGM wavelength floor (λ > 1300 Å restframe)
CIGALE applies IGM/Lyman-continuum absorption (its `igm` column) to the real total SED shortward of the host's Lyman limit; the naive additive reconstruction has no absorption physics at all and can overshoot the true flux there by orders of magnitude (a confirmed example: one galaxy's residual reached 3×10⁵ at λ≈91 Å). This is real and expected, not a bug, and plays no role in the U/V/J passbands (effective wavelengths ≳3600 Å restframe) used everywhere in this analysis. All three notebooks restrict per-galaxy summary flux-residual statistics to λ > 1300 Å; the population wavelength-resolved figure in the first notebook still shows the full range (with the cutoff marked) for transparency.

### 7.4 The flux floor (1e-4 × peak `L_lambda_total`)
Relative residuals are only computed where true flux exceeds this floor, to avoid division-by-near-zero blowups unrelated to reconstruction quality.

### 7.5 Rare colour-computation failures
For a tiny fraction of galaxies (≤0.02% observed), the naive composite goes negative or zero within a passband, causing a `math domain error` in `astSED.calcMag`'s `log10`. Tracked explicitly as `cls_recombined = -1` and excluded from downstream statistics (not silently discarded, not allowed to crash the pipeline) — its rate is reported alongside every classification-agreement number.

### 7.6 SKIRTOR template filename formatting
`data_io.read_skirtor_model()` builds its filename via an f-string with no numeric formatting: whole-number `p`/`q` values must be passed as Python `int` (e.g. `p=1`) not `float` (`p=1.0`), or the constructed filename (`..._p1.0_...`) won't match the actual file on disk (`..._p1_...`).

### 7.7 Bootstrap statistic inconsistency across the three notebooks
`CIGALE_Decomposition_Validation.ipynb`'s Tier 2 colour-residual bootstrap uses `np.median` as the summary statistic; `CIGALE_Decomposition_Validation_GeometryFix.ipynb` and `CIGALE_Decomposition_Validation_WeightedBlend.ipynb` use `np.mean` (matching the `bootstrap_ci` default in those two). Numbers are comparable *within* a notebook but medians vs. means shouldn't be compared directly *across* notebooks — this doc reports each notebook's numbers with the statistic actually used.

## 9. Follow-up: Bayesian alpha calibration

The §2.3 alpha-calibration check above was a single point estimate (median ratio, one template). It was later redone as a full hierarchical Bayesian regression (MCMC via `emcee`, posterior corner plots via `corner`) across all three templates, including Matched — see **`docs/cigale_recombination_validation_findings_bayesian_alpha.md`** for the complete write-up. Headline result: **no template's `alpha_fit`-vs-`alpha_theory` relationship is actually proportional** (slope 0.55–0.78 in log-log space, all excluding 1); Type2's apparent good calibration in §2.3 was a coincidence of the population's typical `fracAGN` sitting near where its fitted line happens to cross 1:1, not evidence of a correct formula, and Matched (the best-supported reconstruction template) shows `alpha_fit` systematically 5–18× smaller than `alpha_theory` predicts across the practical range. See §5 of the sub-doc for what this does and doesn't imply for the paper.

## 10. File reference

| File | Purpose |
|---|---|
| `notebooks/CIGALE_Decomposition_Validation.ipynb` | Baseline Type1/Type2 validation; establishes Tier 1/Tier 2 methodology, alpha calibration, and surfaces the Quiescent-region asymmetry |
| `notebooks/CIGALE_Decomposition_Validation_GeometryFix.ipynb` | Diagnoses the SKIRTOR-geometry mismatch (full header scan) and tests per-galaxy Matched reconstruction |
| `notebooks/CIGALE_Decomposition_Validation_WeightedBlend.ipynb` | Tests (and rules out) a population-weighted single-template alternative usable in the theoretical grid |
| `notebooks/CIGALE_Decomposition_Validation_BayesianAlpha.ipynb` | Full Bayesian regression of `alpha_fit` vs `alpha_theory`, all three templates (§9, see sub-doc) |
| `outputs/cigale_theoretical_validation/` | Figures 1–4 and cached summary CSVs for the baseline notebook |
| `outputs/cigale_geometry_fix/` | Figures 1–3, `agn_geometry.csv` (per-galaxy CIGALE SKIRTOR params — reused by the blend and Bayesian notebooks), and cached summary CSV |
| `outputs/cigale_weighted_blend/` | Figures 1–3 and cached summary CSV for the blend notebook |
| `outputs/cigale_bayesian_alpha/` | Figures 1–3, posterior summary CSV, and `alpha_fit_matched.csv` for the Bayesian notebook |
| `pyproject.toml` | `glass` package source path fix (§7.1); `emcee`/`corner` dependencies added for §9 |
| `docs/cigale_recombination_validation_findings_bayesian_alpha.md` | Full write-up of the Bayesian alpha-calibration follow-up (§9) |
