# Sub-doc: Bayesian Alpha Calibration

*Extends `docs/cigale_recombination_validation_findings.md` §2.3 and §4 with a full Bayesian regression of `alpha_fit` against `alpha_theory`, replacing the earlier single-number "median ratio ≈ 0.99" check with posterior distributions over the population-level relationship. Generated 2026-08-14. Code: `notebooks/CIGALE_Decomposition_Validation_BayesianAlpha.ipynb` (additive — does not modify any prior notebook). Figures/cache: `outputs/cigale_bayesian_alpha/`. New project dependencies added for this notebook: `emcee` (MCMC ensemble sampler), `corner` (posterior corner plots) — this project previously did all statistics by hand in `scipy`/`numpy` with no probabilistic-programming library.*

## 0. TL;DR

The earlier "alpha calibration is essentially correct" conclusion (`docs/cigale_recombination_validation_findings.md` §2.3) was based on checking **only the Type2 template's** median `alpha_fit/alpha_theory` ratio (0.99 [0.96, 1.01]) — Type1 and Matched were never checked this way. Fitting the full population relationship with MCMC, for all three templates, shows:

- **No template's slope is consistent with 1:1** (`beta1` = 0.55–0.78 for all three, all excluding 1 at high confidence) — the relationship is *compressed*, not proportional.
- **For Type1 and Matched, `alpha_fit` is systematically far below `alpha_theory` across the entire practical range**, not just at the tails. At the population-typical point (`fracAGN=0.5`, `alpha_theory=1`), the best-fit `alpha` under Matched is **~18× smaller** than the paper's formula predicts; under Type1, **~130× smaller**.
- **Type2 does cross close to 1:1 near `fracAGN≈0.44`**, which is why its earlier point-estimate looked clean — but even Type2 over-predicts by ~1.9× at `fracAGN=0.9` and under-predicts by ~0.6× at `fracAGN=0.1`.
- **Intrinsic scatter is large for every template (0.45–1.0 dex, i.e. roughly 3–10× per galaxy)** — even where the population-level trend is well behaved, `alpha_theory` is a poor *per-galaxy* predictor.

This does not overturn the earlier finding that the paper's `alpha` formula is a reasonable driver for the Part 1 theoretical grid's coarse 0→1 sweep — it refines it into a much more precise, and more cautionary, statement (see §5).

## 1. The model

```
y_i = log10(alpha_fit_i),   x_i = log10(alpha_theory_i)
y_i ~ Normal(beta0 + beta1 * x_i, sigma^2)
```
fit independently for `alpha_fit_Type1`, `alpha_fit_Type2` (both reused from `CIGALE_Decomposition_Validation.ipynb`'s cache) and `alpha_fit_Matched` (computed fresh for this notebook — the first time Matched's closed-form best-fit alpha has been checked against `alpha_theory` at all). Weakly-informative flat priors: `beta0, beta1 ~ Uniform(-3,3)`, `log10(sigma) ~ Uniform(-5,3)`. The null hypothesis "`alpha_theory` unbiasedly predicts `alpha_fit`" is exactly `beta0=0, beta1=1`. Sampled with `emcee` (32 walkers, 4000 steps, 1000-step burn-in discarded).

**Convergence:** mean acceptance fraction 0.64–0.65 (healthy for an affine-invariant ensemble sampler), autocorrelation time 34–39 steps against a 3000-step post-burn-in chain (~80+ independent samples per walker, ~2,700+ effective samples total per parameter) — all three fits are well-converged. See `Figure1_trace_matched.png` for the representative chain trace.

**Exclusions:** only 2 of 6,509 galaxies per template were excluded (`alpha_fit ≤ 0`, un-log-transformable) — negligible, and does not materially affect any of the following.

## 2. Posterior summary

| Template | beta0 | beta1 | sigma (dex) | beta1 excludes 1? |
|---|---|---|---|---|
| Type1 | −2.113 [−2.130, −2.096] | 0.547 [0.531, 0.563] | 1.012 [1.003, 1.020] | yes |
| Type2 | −0.026 [−0.034, −0.018] | 0.738 [0.731, 0.746] | 0.446 [0.442, 0.450] | yes |
| Matched | −1.257 [−1.269, −1.245] | 0.780 [0.769, 0.791] | 0.699 [0.693, 0.705] | yes |

(All intervals are 16th–84th percentile, i.e. ≈1σ; all comfortably exclude `beta1=1` given their narrow width relative to the gap — see `Figure2_corner_*.png` for the full 2D posterior contours around the `(beta0=0, beta1=1)` reference point.)

## 3. What the slope actually means, per template

Because `beta1 < 1` for every template, the ratio `alpha_fit/alpha_theory` is not constant — it depends systematically on `fracAGN`. Evaluated at three representative points:

| fracAGN | alpha_theory | Type1 ratio | Type2 ratio | Matched ratio |
|---|---|---|---|---|
| 0.1 | 0.111 | 0.021 | 1.67 | 0.090 |
| 0.5 | 1.0 | 0.0077 | 0.94 | 0.055 |
| 0.9 | 9.0 | 0.0029 | 0.53 | 0.034 |

**Type1**: `alpha_fit` is 2–3 orders of magnitude below `alpha_theory` everywhere in the observed range. Given Type1 is already known to be the worst-reconstructing template (`docs/cigale_recombination_validation_findings.md` §2), this may partly reflect the least-squares fit finding a small, damped scaling simply because Type1's spectral *shape* doesn't match real galaxies well — `alpha_fit` for a poorly-shaped template describes "the best linear scaling given a wrong shape," not necessarily a physically meaningful AGN strength. Treat Type1's numbers here as a diagnostic of shape mismatch, not as evidence about the true AGN contribution.

**Type2**: crosses 1:1 near `fracAGN≈0.44` (close to where most of the population sits), which is exactly why the earlier point-estimate check — restricted to Type2 — looked well-calibrated. But the full relation shows real curvature: `alpha_theory` under-predicts by ~40% at `fracAGN=0.1` and over-predicts by ~90% at `fracAGN=0.9`. A single ratio number, computed once across the whole sample, averages this curvature away.

**Matched**: this is the new result — Matched is the best-supported reconstruction template (`docs/cigale_recombination_validation_findings.md` §4: best-or-tied on every flux/colour/classification metric), and its `alpha_fit` is *still* systematically far below `alpha_theory` (5–18× smaller across the practical range, worse at low `fracAGN`, better but still off by ~30× at high `fracAGN`). Because Matched's spectral shape is the one independently validated to reconstruct real galaxies best, this gap is harder to dismiss as a shape-mismatch artefact the way Type1's can be — it looks like a genuine, population-level miscalibration of the `alpha = fracAGN/(1-fracAGN)` formula itself, not previously visible because Matched's alpha was never checked against theory before this notebook.

## 4. Intrinsic scatter

Even setting the mean-trend question aside, the fitted `sigma` (0.45–1.0 dex) means that for a *fixed* `fracAGN`, individual galaxies' best-fit `alpha` still varies by roughly a factor of 3–10× around whatever the population trend predicts. `Figure3_posterior_predictive.png` shows this directly: the raw scatter of points around each template's median regression line is wide relative to the credible band, which is narrow (reflecting the population-level trend being well-constrained by ~6,500 points, not the same thing as individual galaxies being well-predicted). Type2 has the tightest scatter (0.45 dex) of the three, but this likely reflects that Type2 barely perturbs colours at all (established in the main findings doc), making its closed-form `alpha_fit` a numerically less sensitive — not necessarily more physically meaningful — quantity. Tightest scatter and best physical model are not the same claim here.

## 5. Implications

- **For the Part 1 theoretical grid's actual use** (a coarse 11-step sweep of `alpha` from 0 to 1 across `config.ALPHA_VALUES`, used to bracket a qualitative range of AGN contamination): this finding doesn't invalidate that use. The grid isn't claiming to predict any specific real galaxy's exact AGN fraction — it's exploring a plausible range, and the formula remains a reasonable order-of-magnitude way to parameterise that range.
- **For any claim that treats `alpha_theory = fracAGN/(1-fracAGN)` as a precise, unbiased per-galaxy translation between CIGALE's fit and the paper's mixing model**: this analysis says that claim doesn't hold, for any of the three templates tested, and the earlier "0.99 ratio, well-calibrated" result should not be generalised beyond the one template (Type2) it was actually computed for.
- **If a quantitative alpha-fracAGN mapping is needed going forward**, this notebook's posterior (`beta0`, `beta1` for whichever template is used) is directly usable as a *calibrated* correction — e.g. `alpha_corrected = 10^(beta0 + beta1 * log10(fracAGN/(1-fracAGN)))` — rather than the raw formula, though the ~0.5–1 dex scatter means even a corrected mapping is a population-level statement, not a precise per-galaxy one.

## 6. File reference

| File | Purpose |
|---|---|
| `notebooks/CIGALE_Decomposition_Validation_BayesianAlpha.ipynb` | Full analysis: data prep, MCMC fitting, all figures below |
| `outputs/cigale_bayesian_alpha/alpha_fit_matched.csv` | Newly computed `alpha_fit_Matched` per galaxy (not previously cached anywhere) |
| `outputs/cigale_bayesian_alpha/Figure1_trace_matched.png` | Chain trace / convergence diagnostic |
| `outputs/cigale_bayesian_alpha/Figure2_corner_{Type1,Type2,Matched}.png` | Posterior corner plots, one per template |
| `outputs/cigale_bayesian_alpha/Figure3_posterior_predictive.png` | Posterior predictive regression lines + raw data, all three templates overlaid |
| `outputs/cigale_bayesian_alpha/posterior_summary.csv` | Numeric posterior summary (median, 16th/84th percentile) per template |
