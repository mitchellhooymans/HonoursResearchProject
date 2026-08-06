# Figures 9 & 10: Bootstrap Methodology and Interpretation

*How the error bars/confidence intervals in `UVJ_CIGALE_fracAGN_distribution_and_offset.pdf` (Fig. 9) and `UVJ_CIGALE_fracAGN_redshift_confound.pdf` (Fig. 10) were computed, and what they do and don't tell us. Source: `notebooks/CIGALE_Decomposition_Analysis.ipynb`, cells 34, 36 and 38 (0-indexed). Generated 2026-08-06.*

## 0. TL;DR

Every error bar in both figures is a **non-parametric percentile bootstrap 95% CI**: resample the relevant per-galaxy values *with replacement* within a fixed CIGALE `fracAGN` bin, 2000 times, recompute the summary statistic (mean, median, or proportion) each time, and take the 2.5th/97.5th percentiles of the resulting distribution of 2000 statistic values as the interval. Four separate bootstrap loops do this (different statistic, different bin, different RNG seed each time) — there is no shared helper function; each is written inline. No parametric assumption (normality, etc.) is used anywhere.

The single most important interpretive point: **sample size collapses at high `fracAGN`** (from ~1700 galaxies down to 12), so the bootstrap CIs balloon in width there, and in the migration-rate panels, bins with zero observed "successes" (nobody migrated) produce a **degenerate `[0, 0]` CI** that looks like strong evidence of "no effect" but is actually just an artefact of the bootstrap method at a hard boundary (see §4).

---

## 1. Where the numbers come from

Both figures sit downstream of the same per-galaxy CIGALE decomposition table `d`, which — for every ZFOURGE AGN-host galaxy (`fracAGN > 0`, n = 6509) — carries:

- `fracAGN`: CIGALE's best-fit AGN fraction, on its **discrete fitting grid**: `{0.01, 0.1, 0.2, 0.3, ..., 0.9, 0.99}` (11 values; this is a CIGALE grid artefact, not a continuous quantity — see the `% todo` note in `paper.tex` §3.4).
- `VJ_Full`, `UV_Full`: rest-frame UVJ colours from the observed (AGN+host) photometry.
- `VJ_Decomposed`, `UV_Decomposed`: rest-frame UVJ colours after CIGALE's best-fit AGN component is subtracted, leaving host-only photometry.
- `cls_full`, `cls_dec`: UVJ classification (0 = quiescent, else star-forming/dusty) from `photometry.classify_uvj(...)` applied to the full and decomposed colours respectively.

Both figures bin this table by the discrete `fracAGN` grid value and compute a summary + bootstrap CI **per grid point** (11 points), not by the coarser 4-bin grouping (`FRAC_BINS`) used elsewhere in the paper — those coarser bin edges are only overlaid as dotted vertical guide lines for visual reference.

---

## 2. Figure 9 — `fracagn-distribution-offset`

### Top panel
Just a bar count of `n` galaxies per `fracAGN` grid value. No bootstrapping — it's the literal `len(sub)` per bin, and it's what makes the bottom panel's uncertainty pattern make sense (compare bar heights to CI widths below).

### Bottom panel: mean UVJ vector offset, with 95% CI

**What's being bootstrapped:** the *per-galaxy* UVJ shift magnitude,
```python
per_gal = np.hypot(dvj, duv)   # dvj = VJ_Decomposed - VJ_Full, duv = UV_Decomposed - UV_Full
```
i.e. the Euclidean distance each galaxy moves in (V−J, U−V) space between its full and AGN-decomposed colours. This is always ≥ 0 (it's a distance, not a signed offset).

**Procedure**, for each of the 11 `fracAGN` grid values independently:
```python
rng = np.random.default_rng(42)
n_boot = 2000
boot_means = rng.choice(per_gal, size=(n_boot, len(per_gal)), replace=True).mean(axis=1)
lo, hi = np.percentile(boot_means, [2.5, 97.5])
```
1. Draw a resample of the **same size** as the original bin (`len(per_gal)`), **with replacement**, from that bin's per-galaxy offsets.
2. Take the mean of that resample → one bootstrap replicate.
3. Repeat 2000 times → a distribution of 2000 candidate bin-means.
4. The 95% CI is the [2.5th, 97.5th] percentile of that distribution (the "percentile method" — no normal-approximation, no bootstrap-SE-times-1.96 shortcut).

The plotted central point itself is *not* one of the bootstrap replicates — it's the actual observed bin mean, computed via the separate helper `analysis.calculate_mean_vector_offset(...)` (same underlying quantity, computed through a redundant code path — harmless, but worth knowing the point estimate and the CI aren't literally from the same line of code).

**Reproducibility:** `np.random.default_rng(42)`, a single RNG instance reused across all 11 bins sequentially (so the bins are *not* independent random draws in an absolute sense, but this doesn't matter for validity — each bin's resampling pool is disjoint by construction).

### The actual numbers produced

| fracAGN | N | mean offset [mag] | 95% CI |
|---|---|---|---|
| 0.01 | 1270 | 0.0158 | [0.0138, 0.0179] |
| 0.10 | 1672 | 0.0608 | [0.0559, 0.0662] |
| 0.20 | 1110 | 0.0876 | [0.0801, 0.0958] |
| 0.30 | 716 | 0.0871 | [0.0810, 0.0939] |
| 0.40 | 516 | 0.1165 | [0.1067, 0.1266] |
| 0.50 | 409 | 0.1425 | [0.1285, 0.1586] |
| 0.60 | 315 | 0.1577 | [0.1428, 0.1732] |
| 0.70 | 205 | 0.2397 | [0.2124, 0.2671] |
| 0.80 | 154 | 0.2426 | [0.1995, 0.2879] |
| 0.90 | 130 | 0.3116 | [0.2348, 0.3897] |
| 0.99 | 12 | 0.7362 | **[0.3144, 1.3222]** |

Plus a **fully independent robustness check**, run once on all 6509 galaxies without any binning: a per-galaxy Spearman rank correlation between `fracAGN` and offset magnitude, giving **ρ = 0.488, p < 10⁻³⁰⁰** — this is the number quoted in the paper caption and is *not* a bootstrap result at all, it's an exact rank-correlation test.

### Interpretation

- **CI width tracks 1/√N almost exactly**, as expected for a bootstrap of a mean: the CI half-width goes from ~0.002 mag at N=1672 to ~0.5 mag at N=12. The last bin's CI is enormous and strongly **right-skewed** (0.42 mag below the point estimate, 0.59 mag above) — a signature of bootstrapping a small, right-skewed sample of positive distances, not something a symmetric error bar would capture correctly. This is exactly why the percentile method (which doesn't assume symmetry) is the appropriate choice here, and why a naive `mean ± 1.96·SE` bar would have misrepresented this bin.
- **The rising trend in the binned means is visually convincing but is being carried almost entirely by bins with N ≳ 200–700.** The last 2–3 bins (N = 130, 12) have CIs wide enough that you can't claim high statistical confidence in their *individual* bin means — but that's exactly why the paper leans on the **per-galaxy Spearman ρ=0.49 (n=6509)** as the real evidence for the trend, rather than the bin-mean plot alone. The bootstrap CIs on the plot are honest about where the binned-mean picture gets weak; the Spearman test is what actually carries the "rises with fracAGN" claim, since it uses every individual galaxy rather than 11 aggregated points and isn't sensitive to how the bins happen to fall.
- **Practical reading:** trust the shape/trend (strongly monotonic, backed by the whole-sample rank test), but treat the exact bin-mean value and its CI at `fracAGN=0.99` (n=12) as illustrative rather than a precise estimate — a single unusual galaxy in that bin could shift it substantially.

---

## 3. Figure 10 — `fracagn-redshift-confound`

This figure's code is split across two cells: **cell 36** computes the migration-rate arrays (and an AIC curvature test) that get *re-plotted* here; **cell 38** computes everything else and does the actual plotting/saving.

### Top panel: median redshift vs. fracAGN

**What's bootstrapped:** individual galaxies' redshifts within each `fracAGN` bin.
```python
rng3 = np.random.default_rng(11)
zvals = d.loc[d['fracAGN'] == fv, z_col].values
boot = rng3.choice(zvals, size=(n_boot, len(zvals)), replace=True)
lo, hi = np.percentile(np.median(boot, axis=1), [2.5, 97.5])
```
Same resample-with-replacement logic as Fig. 9, but the per-replicate statistic is the **median** (not the mean) — appropriate since redshift distributions within a bin are typically skewed, and the point estimate plotted is also the median. `n_boot = 2000` (inherited from cell 36, not redefined).

Also on this panel: **not a bootstrap result** — `stats.spearmanr(d['fracAGN'], d[z_col])` → **ρ = 0.322, p = 1.25×10⁻¹⁵⁶** (n = full AGN-host sample), the exact-test number quoted in the caption and in the paper's marginal note.

### Bottom panel: migration rate to quiescent, two overlaid curves

**Migration rate definition** (shared by both curves, from cell 36):
```python
at_risk_all = d[d['cls_full'] != 0].copy()                       # not already quiescent pre-decomposition
at_risk_all['moved_to_q'] = (at_risk_all['cls_dec'] == 0).astype(int)  # quiescent after decomposition?
```
i.e., among AGN hosts that were *not* quiescent under the full (AGN-included) photometry, what fraction *become* quiescent once the AGN component is subtracted. This is a **binary indicator per galaxy** (0 or 1), so its bin mean is a proportion (a migration *rate*), not a magnitude.

**"All redshifts" curve** — bootstrapped in cell 36 (seed `rng2 = np.random.default_rng(7)`), reused unchanged in Fig. 10:
```python
sub = at_risk_all[at_risk_all['fracAGN'] == fv]
moved = sub['moved_to_q'].values
boot = rng2.choice(moved, size=(n_boot, len(sub)), replace=True).mean(axis=1)
lo, hi = np.percentile(boot, [2.5, 97.5])
```

**"z < 1.5 only" curve** — a *fresh* bootstrap in cell 38 (seed `rng4 = np.random.default_rng(13)`), over a **redshift-restricted subset** of the same `at_risk_all` table:
```python
at_risk_lowz = at_risk_all[at_risk_all[z_col] < 1.5]
sub = at_risk_lowz[at_risk_lowz['fracAGN'] == fv]
moved = sub['moved_to_q'].values
boot = rng4.choice(moved, size=(n_boot, n), replace=True).mean(axis=1)
```

### The actual numbers produced

| fracAGN | n at risk (all z) | migration rate | 95% CI (all z) | n (z<1.5) | rate (z<1.5) |
|---|---|---|---|---|---|
| 0.01 | 1135 | 0.0009 | [0.0000, 0.0026] | 746 | 0.0013 |
| 0.10 | 1512 | 0.0582 | [0.0456, 0.0708] | 1033 | 0.0852 |
| 0.20 | 1006 | 0.0954 | [0.0775, 0.1153] | 605 | 0.1587 |
| 0.30 | 668 | 0.0554 | [0.0389, 0.0734] | 349 | 0.1060 |
| 0.40 | 488 | 0.0840 | [0.0594, 0.1066] | 229 | 0.1790 |
| 0.50 | 400 | 0.0375 | [0.0200, 0.0575] | 141 | 0.1064 |
| 0.60 | 308 | 0.0357 | [0.0162, 0.0584] | 99 | 0.1111 |
| 0.70 | 202 | 0.0198 | [0.0050, 0.0396] | 66 | 0.0606 |
| 0.80 | 153 | 0.0000 | **[0.0000, 0.0000]** | 33 | 0.0000 |
| 0.90 | 129 | 0.0000 | **[0.0000, 0.0000]** | 9 | 0.0000 |
| 0.99 | 12 | 0.0000 | **[0.0000, 0.0000]** | 0 | **n/a (no data)** |

At `fracAGN=0.99`, `z<1.5` has **zero galaxies at risk** — the `if n:` guard in the loop catches this and sets `lo = hi = np.nan`, so that point/error bar simply doesn't appear on the red curve.

**Curvature test (is the "hump" real or noise?):** rather than trusting the bin shape by eye, a per-galaxy logistic regression of `moved_to_q` on `fracAGN` is fit two ways — linear (`b0 + b1·f`) and quadratic (`b0 + b1·f + b2·f²`) — by direct MLE (`scipy.optimize.minimize` on the negative log-likelihood), and compared by AIC:
```
Linear fracAGN-only model:    AIC=2345.7
Quadratic fracAGN-only model: AIC=2225.6  (b1=+11.63, b2=-18.49)
Delta AIC (linear - quadratic) = 120.1
```
and restricted to `z<1.5` (n=3310):
```
Delta AIC = 99.1  (b1=+12.78, b2=-17.86)
```
ΔAIC ≳ 2 conventionally favours the more complex model; 99–120 is decisive. The **negative** quadratic coefficient (`b2 ≈ -18`) is what encodes "rises then falls" (a concave-down term), i.e. this is the actual statistical backing for calling it a hump rather than a monotonic trend — the bootstrap CIs on the binned plot illustrate it, but the AIC test is what establishes it isn't sampling noise in the bin means.

### Interpretation

- **This is the figure's most important caveat, and it isn't stated in the current paper text:** the `[0.0000, 0.0000]` CIs at `fracAGN = 0.80, 0.90, 0.99` do **not** mean "we're 95% confident the true migration rate is exactly zero." They mean **zero migrations were observed in those bins**, so *every possible bootstrap resample of an all-zero array is also all-zero* — the percentile bootstrap is mechanically incapable of producing a nonzero upper bound when the observed count of successes is 0, regardless of the true underlying rate or sample size. A standard remedy for this exact situation is a binomial/Wilson/Jeffreys interval instead of a naive bootstrap (e.g. the "rule of three" gives an approximate 95% upper bound of ~3/n for zero observed events: ~2% at n=153, ~2.3% at n=129, ~25% at n=12). In other words, the plot slightly *overstates* certainty that the migration effect vanishes at high fracAGN — what the data actually support is "we saw no migrations in a moderate-to-small sample," which is compatible with a true rate as high as a few percent (or, at n=12, uncomfortably high). Worth a caveat sentence in the paper if this claim is going to be leaned on.
- **The two curves ("All redshifts" vs. "z<1.5 only") are *not* statistically independent** — the z<1.5 sample is a strict subset of the all-redshift sample in every bin. This is fine for the qualitative purpose here (showing the hump survives when you restrict to the regime where the effect is even physically able to occur — see Fig. 11's z>1.5 null result), but it means you cannot treat "the two CIs overlap/don't overlap" as a valid two-sample independence test the way you could for two genuinely separate samples. The AIC comparison (run separately on the full sample and the z<1.5 subsample) is the more rigorous version of "does the hump survive" — the overlaid bootstrap bands are the visual/intuitive companion to that test, not a substitute for it.
- **Sample-size floor:** the z<1.5 curve is built on genuinely small bins at the high-fracAGN end (n=33, 9, 0) — the paper should treat any z<1.5-only claim above `fracAGN≈0.6-0.7` (n≥66) as qualitative at best; below n~30 a single galaxy changes the rate by several percentage points.
- **Top vs. bottom panel together:** the reason the migration hump *needs* the z<1.5 check at all is exactly what the top panel shows — median redshift rises steeply with fracAGN (ρ=0.32), so high-fracAGN galaxies are disproportionately high-z, where quiescent-region migration is structurally rarer (established elsewhere via Fig. 11's "null above z≈1.5" result). The AIC-preserved curvature within z<1.5 is what lets the paper argue the hump isn't *purely* that redshift confound — but per the point above, the highest-fracAGN bins of the z<1.5 curve are themselves too sparse to bear much of that argument's weight; it's really the mid-fracAGN bins (0.1-0.5, n=141-1033) doing the work.

---

## 4. Cross-cutting notes

- **Consistent method, four independent instantiations.** All four bootstrap loops (Fig 9 offset; Fig 10 median-z; Fig 10 migration-rate ×2) use the identical pattern — `rng.choice(values, size=(n_boot, n), replace=True)` then apply a statistic along `axis=1`, then `np.percentile(..., [2.5, 97.5])` — just with a different underlying array, statistic, and seed each time (42, 11, 7, 13). There's no shared helper function in `src/`; each is written inline in the notebook. This isn't a correctness problem, but it does mean a future edit to "the bootstrap method" has to be made in four places, not one — worth refactoring into a shared `bootstrap_ci(values, stat_fn, n_boot, seed)` utility if these plots get revisited.
- **`n_boot = 2000` throughout** — large enough that Monte Carlo noise in the percentile estimates themselves is negligible (standard rule of thumb wants ≥1000 for 95% percentile bootstrap; 2000 is comfortably above that) — this isn't the limiting factor anywhere; **bin sample size is**.
- **The paper's methods text doesn't currently describe the bootstrap procedure anywhere** — `paper.tex` §3.4 (`subsec:cigale_method`) is still a `% todo` placeholder, and the only place bootstrapping is mentioned at all is the Fig. 9 caption ("with bootstrap 95% confidence intervals"). If reviewers will see these figures, it's worth adding 1-2 sentences to Methodology stating: percentile bootstrap, 2000 resamples per fracAGN grid bin, resampling individual galaxies with replacement — plus the zero-count caveat from §3 above if the high-fracAGN "no migration" claim is going to be stated as a finding rather than just illustrated.
- **Package provenance caveat:** the point-estimate helper `analysis.calculate_mean_vector_offset` used in Fig. 9 (but *not* the CI bootstrap itself, which recomputes the same distance formula inline) lives in the external `glass` package (`git show c5d38cd5^:src/glass/analysis.py` in this repo's history), which is now an editable pip dependency pointing outside this checkout. Its logic was verified against the historical in-repo copy and matches exactly what the notebook calls, but byte-identical current-state confirmation wasn't possible from this checkout alone.
