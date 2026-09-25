# Supervisor Feedback Investigation: Findings

*Investigates three open supervisor/committee comments on `Context/AGNPaper/paper.tex`, ahead of any paper-text changes. Generated 2026-09-16. Code: `notebooks/Supervisor_Feedback_Investigation.ipynb` (additive — does not modify any existing notebook, script, or the `glass` package). Figures/caches: `outputs/supervisor_feedback/`.*

Two other comments from the same feedback pass are not covered here: looping in **Vanessa** on the CIGALE methodology is a logistics action item (see §4 below), and the `dex`/`mag` unit inconsistency was fixed directly in `paper.tex` (lines 217, 256, 292, 361, 384 changed `dex`→`mag`; line 177 and Table 1 correctly stay `dex`, since that's IRAC log-flux-ratio space, not UVJ magnitude space).

## 1. Section A — `alpha` vs `f_AGN` (Jamie's comment)

**The claim.** `glass.composite_math.create_composite_sed` (`C:\Users\uqmhooym\GitHub\GLASS\src\glass\composite_math.py:56-81`) scales the AGN template so its integrated flux equals the galaxy's (`SF = integral_flux(gal)/integral_flux(agn)`) before mixing: `combined_flux = gal_flux + alpha * SF * agn_flux`. At `alpha=1` this is a 50/50 integrated-flux mix, not "100% AGN" — the true AGN luminosity fraction is `f_AGN = alpha/(1+alpha)`, so the paper's `config.ALPHA_VALUES` grid (`np.linspace(0,1,11)`) only ever explores `f_AGN` from 0 to 0.5, while every "AGN contribution 0% to 100%" sentence in `paper.tex` (lines 166-168, 217, 256, 292, 361, 384; Table 1's caption) claims the full 0–1 range.

**Numeric verification.** Confirmed directly against the actual composite-generation code path (not just algebra): for `alpha` in `{0, 0.25, 0.5, 1, 2, 9, 99}`, the numerically-integrated AGN flux fraction of a real composite SED matches `alpha/(1+alpha)` to floating-point precision in every case.

**This correction already exists, unintegrated.** `docs/cigale_recombination_validation_findings.md` and its sub-doc `..._bayesian_alpha.md` (2026-08-14) already derived and used `alpha_theory = fracAGN/(1-fracAGN)` (the exact inverse) to compare CIGALE fits against the theoretical model, in three additive notebooks. None of that made it into `paper.tex`. Their headline finding matters here: a Bayesian log-log regression of per-galaxy fitted `alpha` against `alpha_theory` gives a slope of 0.55–0.78 (not 1) for every AGN geometry template tested — the relationship is directionally correct but **not a proportional, per-galaxy-accurate calibration**. Treat it as a reasonable driver for a coarse qualitative grid, not a precise translation.

**Extended grid.** Rather than keep sweeping `alpha` and mislabelling the result, the notebook drives the grid directly by `f_AGN`, matching CIGALE's own discrete grid (`{0, 0.01, 0.1, 0.2, ..., 0.9, 0.99}`, `docs/cigale_decomposition_findings.md`), via `alpha = f_AGN/(1-f_AGN)`. Reran the full theoretical grid (129 Brown templates × Type 1/Type 2) on this grid for both IRAC and UVJ colour spaces (9.4s total).

**Sanity check (passed):** the extended grid's `f_AGN=0.5` point is the *same composite* as the legacy grid's `alpha=1.0` ("100%") point, and reproduces it: IRAC offset Type1=0.356 (paper: 0.36), Type2=0.612 (paper: 0.61); UVJ offset Type1=0.541 (paper: 0.54 mag). Small residual differences are pre-existing template/pipeline-version drift between the paper text and a fresh rerun of the *unmodified* legacy code — not something this correction introduces (also visible independently: rerunning `Model_Validation_via_IRAC.ipynb`'s existing Table 2 cell unmodified today gives Type2 values ~0.24/0.35/... vs. the paper's printed 0.23/0.34/... — a small, unrelated drift worth a separate look someday, out of scope here).

**The corrected numbers (full table in `outputs/supervisor_feedback/corrected_table1_extended_fAGN.csv`):**

| f_AGN | IRAC Type1 [dex] | IRAC Type2 [dex] | UVJ Type1 [mag] | UVJ Type2 [mag] | Lacy compl. Type1 | Lacy compl. Type2 |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.217 | 0.217 |
| 0.5 (legacy "100%") | 0.356 | 0.612 | 0.541 | 0.002 | 0.054 | 1.000 |
| 0.9 | 0.566 | 0.772 | 1.206 | 0.019 | 0.000 | 1.000 |
| 0.99 | 0.615 | 0.799 | **1.460** | 0.173 | 0.000 | 1.000 |

**The correction makes the paper's headline finding *stronger*, not weaker.** Past the legacy grid's endpoint, Type 1's UVJ offset keeps climbing to 1.46 mag at the true `f_AGN=0.99` — the paper's current numbers describe less than half of the AGN-fraction range they claim to. Type 1's Lacy-wedge completeness reaches exactly 0% by `f_AGN≈0.9` (not the ~6% the legacy "100%" point implies). A previously invisible tail effect also appears: Type 2's UVJ offset, exactly negligible everywhere the legacy grid could see, rises to 0.17 mag by `f_AGN=0.99`.

**Recommendation for `paper.tex`:** replace "AGN contribution 0–100%" framing with `f_AGN` throughout; regenerate Table 1/2 and Figures 1–3 from the extended grid (swap `config.ALPHA_VALUES` for the `FRAC_AGN_GRID`/`ALPHA_EXT` pattern in `Model_Validation_via_IRAC.ipynb` and `Paper_Results_Master.ipynb`); add one sentence on the `alpha_theory` calibration caveat. **Decision for Mitchell** — the main pipeline notebooks were not touched by this investigation.

### 1.1 Is `f_AGN = alpha/(1+alpha)` actually a 1:1 match to CIGALE's `fracAGN`?

Checked directly, since the label fix above only addressed what the paper's grid *claims*, not whether that quantity is the same thing CIGALE reports. `paper.tex` §3.4 already states CIGALE's own definition precisely: `fracAGN` is **the ratio of AGN infrared luminosity (1–1000 µm) to total infrared luminosity** — an IR-specific quantity that only counts the AGN's thermally-reprocessed dust emission, not its direct/scattered accretion-disk continuum at any other wavelength. `f_AGN = alpha/(1+alpha)` is the AGN's share of the **entire composite SED's integrated flux** (X-ray through ~35 µm). These coincide only if the AGN's own SED is close to 100% IR emission.

**It isn't, for one of the two AGN types.** Decomposing each native SKIRTOR template by its own output columns:

| | Dust emission (CIGALE's L_IR,AGN) | Direct AGN continuum | Scattered |
|---|---|---|---|
| Type 1 (face-on) | 18.6% | 80.2% | 1.1% |
| Type 2 (edge-on) | 99.9% | 0.0% | 0.1% |

Type 1's own SED is 80% light CIGALE's `fracAGN` doesn't count at all; Type 2's is essentially all dust emission. Quantifying what this does to the actual composite — comparing `f_AGN=alpha/(1+alpha)` against a CIGALE-equivalent quantity built from the AGN's dust-emission component only, integrated 1–1000 µm, over the same window's host flux:

| f_AGN (whole-SED) | CIGALE-style fracAGN Type1 | CIGALE-style fracAGN Type2 |
|---|---|---|
| 0.10 | 0.039 | 0.166 |
| 0.30 | 0.135 | 0.435 |
| 0.50 | 0.267 | 0.642 |
| 0.70 | 0.459 | 0.807 |
| 0.90 | 0.766 | 0.942 |

**Answer: no, and it's wrong in opposite directions for the two types.** For Type 1, the CIGALE-equivalent fraction runs at roughly *half* the whole-SED `f_AGN` value throughout. For Type 2, it runs *higher* than `f_AGN`. No single correction factor can map `alpha/(1+alpha)` onto a CIGALE-equivalent quantity for both types at once — the direction of the error flips depending on obscuration. This gives a first-principles mechanism for the existing `docs/cigale_recombination_validation_findings_bayesian_alpha.md` finding that the empirical `alpha_fit`-vs-`alpha_theory` relationship is non-proportional and differs by template (slope 0.55–0.78): that finding was an empirical per-galaxy fit result; this is *why* it comes out that way.

**Caveat on this specific calculation**: the host side of the comparison isn't perfectly CIGALE-equivalent either — Brown/GALSEDATLAS templates give total observed galaxy flux with no separate "dust emission only" component the way CIGALE's own stellar+dust energy-balance module provides, so some of what's counted here as "host flux in the IR window" is really stellar photospheric Rayleigh-Jeans tail, not genuine host dust emission. The qualitative conclusion (large, oppositely-signed divergence between the two AGN types) is robust to this; the exact percentages are a proxy, not an exact CIGALE reproduction.

**Recommendation**: don't describe `f_AGN = alpha/(1+alpha)` in the paper as "CIGALE's fracAGN, just relabelled." It's a legitimate, well-defined quantity (AGN's share of the composite's total bolometric flux) and a genuine improvement over the current "0–100%" mislabelling — but name it as what it is. Anywhere the paper compares the theoretical grid to CIGALE's fracAGN axis, flag explicitly that the two are analogous in spirit (both rise from 0 as AGN dominance grows) but not numerically interchangeable, especially for Type 1.

## 2. Section B — ZFOURGE sample-cut definition

**The problem is worse than "undocumented".** A background investigation this session traced the paper's 10,876-galaxy parent sample (`datasets/full_zfourge_decomposed/zfourge_full_final.csv`) back through several now-deleted notebooks (recovered only via `git log -S` archaeology) to a file (`zfourge_full.csv`) that was overwritten from 22,276 rows to 10,876 rows in commit `ac2ed311` with **no accompanying code change** — an untracked, non-diffable edit. `full_{CDFS,COSMOS,UDS}_ids.csv`, `*_RecalculatedUVJids_full.csv`, and everything in `datasets/zfourge/GalaxySelectionOutputs/` are dead artifacts nothing current reads.

**Fresh, reproducible characterization (from the raw FITS catalogs):**

| Field | Raw N | `Use==1` N | Legacy parent N | Retention |
|---|---|---|---|---|
| CDFS | 30,911 | 13,299 | 3,769 | 28.3% |
| COSMOS | 20,786 | 12,901 | 3,811 | 29.5% |
| UDS | 22,093 | 11,447 | 3,296 | 28.8% |
| **TOTAL** | **73,790** | **37,647** | **10,876** | **28.9%** |

Every legacy ID is a strict subset of `Use==1` in every field (0 exceptions), and 100% of `Use==1` galaxies already have valid rest-frame U/V/J flux — so `Use==1` isn't the bottleneck. Candidate simple cuts tried and ruled out: redshift range (z<3/3.5/4 retains 80–89% of `Use==1`, nowhere near 29%); a `KsR` (Kron radius) cut (distributions are essentially identical between the legacy sample and the rest of `Use==1`). **CIGALE (external SED-fitting software, not part of this repo's Python pipeline) was evidently run on only a ~29% subsample of `Use==1` for a reason that predates this repo's tracked history and cannot be recovered or re-run from here.**

**Given that, the honest, reproducible characterization is:** *"the subset of ZFOURGE `Use==1` galaxies with an available CIGALE best-fit SED."* What matters for the paper is whether that historical subsample is a fair draw from the full quality-flagged population:

**Selection-bias check (previously undocumented):** the legacy 10,876-galaxy sample is significantly biased toward **lower redshift and lower stellar mass** than the `Use==1` galaxies CIGALE was never run on (median z=1.01 vs 1.56; median log M\*=8.87 vs 9.12; KS test D=0.223/0.133, p≈0 and p=1.5×10⁻¹²⁰ respectively — figure: `ZFOURGE_sample_selection_bias.pdf`).

**Implication:** not fatal to the paper's core claims — the analysis sample is *enriched* in exactly the low-z regime (z≲1.5) where the headline AGN-contamination effects are found — but it means the z≳1.5 "null recovery" result generalises less certainly to the full ZFOURGE population than to this specific historical CIGALE subsample. Worth a caveat sentence.

**Recommended `paper.tex` text for §3.2 (`subsec:obsdata`):**

> *"Our analysis is restricted to the subset of ZFOURGE `Use==1` quality-flagged sources (N=37,647 across CDFS, COSMOS, and UDS; combining Straatman et al. 2016's K-band signal-to-noise, star/artifact, and near-neighbour flags) for which a CIGALE multi-wavelength SED fit is available (N=10,876, ≈29%). This CIGALE-fit subsample is systematically biased toward lower redshift and lower stellar mass than the full quality-flagged population (median z=1.0 vs 1.6; median log M\*=8.9 vs 9.1); we discuss the implications of this for the redshift-dependent results in [Limitations]."*

## 3. Section C — Type 1 vs Type 2 IRAC composite SEDs

Figure 4 (`CompositeSEDs_UVJ.pdf`) only ever shows Type 1 composites, in UVJ wavelength space. There was no figure showing *why* Type 1 and Type 2 diverge so strongly in IRAC/Lacy-wedge space (Figure 1) — despite the paper's own Limitations section (`paper.tex` ~line 373) already citing the right literature (Donley et al. 2007, 2012; Messias et al. 2012, 2014) for exactly this.

**Built:** `IRAC_CompositeSEDs_Type1vType2.pdf` — a 2×3 panel grid (Type 1/Type 2 × Star-forming/Quiescent/Dusty representative galaxies, same three galaxies as Figure 4: `NGC_0337`, `NGC_4552`, `IC_4553`), zoomed to the 1–10 µm window with the four IRAC bands overlaid instead of U/V/J. Full SEDs were regenerated live (`outputs/composite_seds/*.csv` only ever cached photometric colours, not flux arrays, so there was nothing to reuse).

**Quantified mechanism:** at `f_AGN=0.5` (`alpha=1`), the AGN's fractional contribution to each IRAC band:

| IRAC band | Type 1 | Type 2 |
|---|---|---|
| 3.6 µm | 49.9% | 49.3% |
| 4.5 µm | 64.8% | 80.1% |
| 5.8 µm | 77.2% | 93.2% |
| 8.0 µm | 87.9% | 96.4% |

Both types start equal at 3.6 µm (by construction — that's the shortest band, closest to where the integral-flux normalisation anchors), but Type 2's fractional contribution rises far more steeply with wavelength. This is why Type 2 composites redden fast enough to enter the Lacy wedge (designed around a red thermal-dust power law) while Type 1 composites — whose face-on, less-obscured SED reddens much more gently — stay too blue to enter it. Directly consistent with the already-cited literature's explanation (host-diluted, flatter non-thermal continuum under-selected by a red-power-law-based wedge).

**Recommendation:** add the new figure to Results, directly after Figure 1 (§4.1 `subsec:irac_results`), with a sentence tying it explicitly to the Donley/Messias citations currently isolated in Limitations (§5.5). The supervisor's "perhaps I missed the discussion" comment is a *placement* problem, not a missing-citation one.

### 3.1 Stress test: is this a methodology artifact rather than real physics?

Mitchell raised a fair concern after seeing this: is Type 1's IRAC failure a genuine result, or an artifact of how the composite pipeline mixes AGN and host light? Worth taking seriously rather than assuming the literature match settles it.

**The concern, precisely stated:** `create_composite_sed`'s `SF` normalisation matches AGN and galaxy on *total bolometric integrated flux* (the entire overlapping wavelength range, ~100 Å to ~35 µm), not on flux in any band a telescope measures. Type 1 (face-on) and Type 2 (edge-on) are the *same* SKIRTOR AGN engine, but obscuration redistributes where their light emerges — so "50/50 by total flux" is not obviously the same thing as "50/50 where it's observable" for the two types.

**Checked directly — where does each AGN template's own native flux actually live?**

| Wavelength window | Type 1 | Type 2 |
|---|---|---|
| X-ray/EUV (<912 Å) | 48.2% | 0.0% |
| UV (912–3000 Å) | 14.8% | 0.0% |
| Optical (3000–10000 Å) | 8.0% | 0.0% |
| NIR (1–3 µm) | 6.1% | 0.3% |
| MIR/IRAC (3–9 µm) | 8.3% | 21.8% |
| FIR (9–1000 µm) | 9.5% | 77.1% |

**The concern has real teeth: Type 1 spends 63% of its normalised "budget" on flux shortward of 3000 Å — wavelengths no filter in this paper (bluest is U, ~3600 Å) ever samples.** Type 2 has essentially 0% of its flux there (it's absorbed and reprocessed). So matching total integrated flux between the two types is not the same as matching observable AGN light.

**Does correcting this change the Lacy-wedge conclusion?** Tested an alternative normalisation restricting the `SF` integral to λ>3000 Å (excluding the unobservable EUV/X-ray), rerun across the full 129-template grid:

| f_AGN | Type1 completeness (default) | Type1 completeness (>3000 Å window) | Type2 completeness (default) | Type2 completeness (>3000 Å window) |
|---|---|---|---|---|
| 0.5 | 0.054 | **0.008** | 1.000 | 1.000 |
| 0.7 | 0.008 | 0.000 | 1.000 | 1.000 |
| 0.9 | 0.000 | 0.000 | 1.000 | 1.000 |

**Result: the opposite of the naive expectation.** Excluding the unobservable EUV/X-ray makes Type 1's IRAC-band AGN flux *fraction* go up substantially (e.g. 3.6 µm: 49.9%→71.9% at f_AGN=0.5 for a test galaxy) — but its Lacy-wedge completeness gets *worse*, dropping from 5.4% to 0.8% at f_AGN=0.5, and it still hits exactly 0% by f_AGN≈0.7–0.9 either way. Type 2 is essentially unaffected (it has ~0% flux below 3000 Å to begin with, so there's nothing for the window change to alter).

**Why:** the Lacy wedge selects on *colour* — the slope of rising flux from 3.6→8.0 µm — not on total AGN flux fraction. Removing the EUV/X-ray adds proportionally more flux to Type 1's *bluest* IRAC band (3.6 µm, nearest the excluded region) than its reddest (8.0 µm), flattening its already-too-flat colour slope further and pushing it *further* from the wedge.

**Conclusion (as far as it goes): the >3000Å normalisation asymmetry is real, but doesn't explain Type 1's IRAC failure away — correcting it makes Type 1's failure stronger, not weaker.** This part of the picture — Type 1 genuinely can't fake a dusty power-law across 3.6–8.0 µm — looks like real SKIRTOR-model physics, consistent with the cited literature. But see §3.2: this test only checked whether *Type 1's own* behaviour was an artifact. It didn't check the more important half of the question — Type 2.

### 3.2 The real methodology issue: Mitchell's counter-objection was correct

Mitchell pushed back on §3.1's fix directly: excluding EUV/X-ray from the normalisation on the grounds that "no filter samples it" isn't physically motivated — **total bolometric energy output shouldn't depend on which wavelengths we happen to observe.** That's correct, and it exposes what the real problem is.

**The raw SKIRTOR files for Type 1 and Type 2 are not two independent spectra — they are the same accretion engine** (`t7_p0.5_q0_oa40_R20_Mcl0.97`, fixed reference luminosity and distance in SKIRT's radiative-transfer units), computed at two different viewing angles (i=0° and i=90°). Because torus obscuration is anisotropic by construction, SKIRTOR's own, un-renormalised prediction for how much flux reaches an observer already differs enormously by inclination:

> **Native (as-published) integrated flux — Type 1 (i=0°): 1.031×10⁻³. Type 2 (i=90°): 5.77×10⁻⁵. Ratio ≈ 17.9.**

This ~18× is genuine unified-AGN-model physics (an edge-on observer really does see far less of the same engine's output, due to obscuration) — **not a numerical artifact, and not something that should be normalised away.** But `create_composite_sed`'s `SF = integral_flux(gal)/integral_flux(agn)` is computed *independently* for whichever AGN template is passed in. That means at `alpha=1`, Type 1 and Type 2 are *forced* to inject exactly equal total flux into the composite — as if the edge-on, obscured view of an AGN were intrinsically just as bright as the face-on view of the same engine. **This is the real methodology issue: not which wavelengths count as "total flux" (§3.1), but treating two viewing angles of one physical engine as independently renormalisable rather than sharing one physical luminosity scale.**

**Tested the fix Mitchell's physical instinct actually implies**: compute `SF` once, from Type 1 only, and reuse that identical scaling factor for both types (`alpha` then means "how many Type-1-equivalent luminosities of AGN are injected" — the same physical scale for both types, SKIRTOR's own anisotropy preserved). Reran the full 129-template grid:

| Alpha (shared scale) | IRAC completeness Type1 | IRAC completeness Type2 | UVJ offset Type1 [mag] | UVJ offset Type2 [mag] |
|---|---|---|---|---|
| 0.0 | 0.217 | 0.217 | 0.000 | 0.000 |
| 0.5 | 0.132 | 0.341 | 0.347 | 0.000 |
| 1.0 | 0.054 | **0.395** | 0.541 | **0.000** |

Compare to the paper's current (independently-normalised) numbers at the same nominal `alpha=1`: Type 2 IRAC completeness = **1.000** (vs 0.395 here), Type 2 UVJ offset ≈ 0.002 mag (vs 0.000, i.e. even more strongly null here).

**Type 1's numbers are essentially unchanged** (it's the shared anchor, so this is expected and a good consistency check). **Type 2 looks dramatically different**: under the paper's current normalisation it reaches 100% Lacy-wedge completeness by `alpha≈0.4`; under the energy-conserving, shared-luminosity normalisation it only reaches 39.5% completeness at `alpha=1`, and its UVJ offset is indistinguishable from zero throughout. To match Type 1's `alpha=1` in absolute injected flux, Type 2 needs `alpha≈17.9` on this shared scale — a Type 2 AGN needs to be intrinsically ~18× more luminous than a Type 1 AGN before it perturbs observed host colours by a comparable amount.

**This means the paper's current Type 2 IRAC "success" story (98% completeness by 70% nominal AGN contribution) is substantially an artifact of independently renormalising two viewing-angle spectra of the same engine up to equal apparent brightness — not, as currently framed, a genuine prediction that modest obscured AGN readily dominate mid-IR colours.** Under the physically-motivated convention, Type 2 still eventually wins the wedge faster than Type 1 per unit of *actually injected* flux (none of its light is wasted on invisible EUV/X-ray, per §3.1) — but it needs far more intrinsic AGN power to get there than the current grid explores.

**This is a bigger deal than a caveat sentence — it's a candidate change to the paper's core Part 1 methodology, not just its labelling or its grid range.** No pipeline or paper changes have been made based on this; it needs a decision, not a fix:

1. **Surface this explicitly to the supervisor/Jamie/Vanessa before proceeding** — it changes the central Type 1-vs-Type 2 IRAC comparison the paper is built around, not a side result.
2. **If adopted, it also reframes Section 1 (`alpha` vs `f_AGN`)**: under a shared-luminosity convention, "AGN contribution" would naturally be defined once, on a single physical scale common to both viewing angles, rather than needing a separate per-type `f_AGN` conversion at all.
3. **Open question this notebook does not resolve**: is "anchor the shared scale on Type 1" the right choice, or should the shared reference be something else entirely (e.g. a genuinely inclination-averaged/isotropic-equivalent luminosity, if that can be derived from SKIRTOR's fuller output)? Type 1 was used here because it's the least-obscured, closest-to-intrinsic view, but that's a defensible starting choice, not a definitive one.

## 4. Action item: Vanessa

Loop Vanessa in specifically on:
- The §3.4 CIGALE methodology write-up (still a `% todo` per earlier project docs — check current state before assuming).
- The discrete `fracAGN` grid striation issue (already flagged in current Limitations).
- The `alpha_theory = fracAGN/(1-fracAGN)` calibration finding (§1 above) — she may also know, from her own CIGALE work, why the historical ~29% CIGALE subsample (§2 above) was selected the way it was.

## 5. File reference

| File | Purpose |
|---|---|
| `notebooks/Supervisor_Feedback_Investigation.ipynb` | Full analysis: Sections A (alpha/f_AGN), B (sample cuts), C (IRAC SED figure) |
| `outputs/supervisor_feedback/extended_grid_{irac,uvj}.csv` | Per-template, per-AGN-type, per-f_AGN-gridpoint colours (129 templates × 2 types × 12 grid points) |
| `outputs/supervisor_feedback/corrected_table1_extended_fAGN.csv` | Candidate replacement for paper Table 1/2, keyed on true `f_AGN` |
| `outputs/supervisor_feedback/IRAC_Evolution_ExtendedGrid.pdf`, `UVJ_Fractions_ExtendedGrid.pdf` | Candidate replacement figures for Figs 1–3, extended to `f_AGN=0.99` |
| `outputs/supervisor_feedback/zfourge_sample_cuts.csv` | Raw / `Use==1` / legacy-parent counts per field |
| `outputs/supervisor_feedback/ZFOURGE_sample_selection_bias.pdf` | Redshift/mass distributions, legacy sample vs. dropped `Use==1` galaxies |
| `outputs/supervisor_feedback/IRAC_CompositeSEDs_Type1vType2.pdf` | New Type1-vs-Type2 IRAC composite SED figure |
| `outputs/supervisor_feedback/irac_completeness_sf_window_sensitivity.csv` | Lacy-wedge completeness under the default vs. `>3000 Å` normalisation window (methodology stress test, §3.1) |
| `outputs/supervisor_feedback/shared_sf_energy_conserving_results.csv` | IRAC/UVJ results under the energy-conserving, shared-luminosity normalisation (§3.2) — the more significant finding |
| `outputs/supervisor_feedback/fAGN_vs_cigale_style_fracAGN.csv` | `f_AGN=alpha/(1+alpha)` vs a CIGALE-equivalent (IR-luminosity-only) fracAGN proxy, both AGN types (§1.1) |
