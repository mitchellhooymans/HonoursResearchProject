# GLASS Paper Completion & Action Plan

---

## Executive Summary

This action plan details the step-by-step tasks required to complete [`Context/AGNPaper/paper.tex`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex), resolve all unwritten `% todo` sections, integrate recent August 2026 validation findings (recombination fidelity, torus inclination matching, and Bayesian $\alpha$ calibration), implement the **Figure 9 & Figure 10 plot redesigns**, and outline future research directions for the project.

---

## 🎨 Figure Redesign & Layout Enhancements (Completed in Codebase)

Per your feedback regarding Figure 9 and Figure 10 layout complexity, the following figure updates have been implemented and generated via [`scripts/recreate_redesigned_fig9_fig10.py`](file:///m:/GitHub/HonoursResearchProject/scripts/recreate_redesigned_fig9_fig10.py):

1. **Simplified Figure 9 (`UVJ_CIGALE_fracAGN_distribution_and_offset.pdf`):**
   * **Clean Single Panel (Histogram Removed):** Displays the mean UVJ vector offset [mag] vs CIGALE best-fit $f_{\text{AGN}}$ with 95% bootstrap CIs. The top histogram has been removed for a clean, focused publication plot.

2. **Redesigned Figure 10 (`UVJ_CIGALE_fracAGN_redshift_confound.pdf`):**
   * **Focused Single Panel:** Focused 100% on the quiescent migration fraction vs $f_{\text{AGN}}$, comparing All Redshifts vs the $z < 1.5$ cut. Removes plot clutter and highlights the intermediate-$f_{\text{AGN}}$ non-monotonic hump ($\Delta\text{AIC} = 99$).

3. **New Standalone Figure 10B (`UVJ_CIGALE_fracAGN_redshift_evolution.pdf`):**
   * **Dedicated Redshift Evolution Figure:** Top panel shows median host redshift vs $f_{\text{AGN}}$ ($\rho = 0.32$), while the bottom panel presents the stacked population distribution below vs above $z = 1.5$.

---

## Priority 1: Fill Mandatory Text Gaps in `paper.tex`

These are section placeholders or `% todo` blocks in [`paper.tex`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex) that need writing before the paper can be read end-to-end:

### Task 1: Write §3.4 / §4.4 CIGALE Methodology
* **Location:** [`paper.tex:162`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex#L162) (`\subsection{SED Decomposition via CIGALE}`)
* **Action:** Replace the `% todo` comment with text defining:
  * How CIGALE fits galaxy + AGN SEDs.
  * The discrete $f_{\text{AGN}}$ grid ($\{0, 0.01, 0.1, \dots, 0.9, 0.99\}$).
  * Host-only SED extraction ($F_{\text{host}} = F_{\text{total}} - F_{\text{AGN}}$) for calculating rest-frame intrinsic UVJ colors.
  * The empirical sample size ($N = 6,509$ active AGN hosts out of $10,876$ total ZFOURGE galaxies).

### Task 2: Write §5.3 Implications for Quiescent Demographics
* **Location:** [`paper.tex:382`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex#L382) (`\subsection{Implications for Quiescent Galaxy Demographics}`)
* **Action:** Replace the `% todo` block with text explaining:
  * **Underestimation of Quiescent Fraction:** Uncorrected surveys underestimate passive fractions by up to **+9.3 percentage points** at $z \sim 0.5 - 1.0$.
  * **Quantitative Bias Map (Fig 12):** Highlight the worst-case bias (**+17.9 percentage points** at $z = 0.5 - 1.0$, $f_{\text{AGN}} = 0.4$, $N = 145$).
  * **Practical Survey QA Takeaway:** Intermediate-luminosity AGN ($f_{\text{AGN}} \sim 0.2 - 0.4$) pose the highest risk for silent misclassification, rather than extreme quasars.
  * **Astrophysical Impact:** Consequences for derived quenching timescales, passive stellar mass functions, and morphological quenching models.

### Task 3: Write §5.4 Diagnostic Completeness & Obscuration
* **Location:** [`paper.tex:401`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex#L401) (`\subsection{AGN Obscuration and Diagnostic Completeness}`)
* **Action:** Replace the `% todo` block with text covering:
  * **Complementary Biases:** UVJ is highly sensitive to Type 1 (unobscured) AGN but blind to Type 2 (obscured) AGN. Conversely, the Lacy IRAC wedge selects Type 2 (>98% completeness at $\alpha \ge 0.7$) but misses Type 1 (dropping from 26% to 6%).
  * **Multi-Wavelength Necessity:** Demonstrating why multi-band mid-IR / X-ray selection must be paired with SED decomposition to avoid biased demographic censuses.

### Task 4: Write Section 6 Conclusions
* **Location:** [`paper.tex:421`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex#L421) (`\section{CONCLUSIONS}`)
* **Action:** Draft a 3-paragraph conclusion summarizing:
  1. *Theoretical predictions:* Unobscured AGN inject U-band continuum flux that dilutes $4000~\text{Å}$ breaks, shifting passive host colors bluer into the star-forming locus.
  2. *Empirical ZFOURGE/CIGALE validation:* CIGALE host isolation confirms this shift in real galaxies, demonstrating a strictly one-way recovery of quiescent systems.
  3. *Cosmic time peak & summary:* Intermediate-fraction AGN ($f_{\text{AGN}} \sim 0.1 - 0.5$) drive maximum classification bias, peaking at $z = 0.5 - 1.0$ and dropping to null above $z \sim 1.5$.

---

## Priority 2: Integrate Recent Validation Findings & Future Directions

Incorporate recent project validation findings into **§5.5 (Limitations and Future Work)**:

### Task 5: Integrate Recombination & Matched Inclination Findings
* **Location:** [`paper.tex:411`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex#L411) (`\subsection{Limitations and Future Work}`)
* **Action:** Add 1–2 paragraphs detailing:
  * **Model Self-Consistency:** Re-adding theoretical SKIRTOR AGN models onto CIGALE host SEDs achieves **95.8% aggregate UVJ classification agreement** ([`notebooks/CIGALE_Decomposition_Validation.ipynb`](file:///m:/GitHub/HonoursResearchProject/notebooks/CIGALE_Decomposition_Validation.ipynb)).
  * **Matched Torus Inclinations:** Mention that using CIGALE's actual best-fit inclinations ($i = 30^\circ / 70^\circ$) eliminates fixed Type 1 false negatives ($27.4\% \to 14.9\%$) and yields **85.1% quiescent recovery** ([`notebooks/CIGALE_Decomposition_Validation_GeometryFix.ipynb`](file:///m:/GitHub/HonoursResearchProject/notebooks/CIGALE_Decomposition_Validation_GeometryFix.ipynb)).
  * **Bayesian $\alpha$ Calibration:** Note that $\alpha_{\text{theory}} = f_{\text{AGN}} / (1 - f_{\text{AGN}})$ exhibits log-log slope compression ($\beta_1 \sim 0.55 - 0.78$), providing empirical parameters for converting between CIGALE decomposition parameters and theoretical grid mixing weights ([`notebooks/CIGALE_Decomposition_Validation_BayesianAlpha.ipynb`](file:///m:/GitHub/HonoursResearchProject/notebooks/CIGALE_Decomposition_Validation_BayesianAlpha.ipynb)).

### Task 6: Articulate Future Directions
* **Location:** [`paper.tex:411`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex#L411)
* **Action:** Detail 3 key future avenues:
  1. *JWST NIRCam + MIRI Extension:* Extending rest-frame UVJ and mid-IR diagnostics out to $z \sim 3 - 6$ and performing spatially resolved SED decomposition (pixel-level nuclear PSF subtraction).
  2. *Continuous Bayesian Samplers:* Replacing discrete CIGALE fitting grids with continuous nested samplers (`Prospector` / `bagpipes`) to eliminate UVJ grid striations.
  3. *Optical Drop-Out Selection:* Extending GLASS to optical Lyman-break selection ($u-g$ vs $g-r$) using the pre-computed grid dataset ([`scripts/generate_redshift_grid_data.py`](file:///m:/GitHub/HonoursResearchProject/scripts/generate_redshift_grid_data.py)).

---

## Priority 3: Polish & Minor LaTeX Fixes

### Task 7: Fix Missing Citation
* **Location:** [`paper.tex:359`](file:///m:/GitHub/HonoursResearchProject/Context/AGNPaper/paper.tex#L359)
* **Action:** Fill in the empty `\citep{}` placeholder in §5.1 (e.g., add `\citep{brammer_eazy_2008}` or `\citep{whitaker_newfirm_2011}`).

### Task 8: LaTeX Compilation Check
* **Action:** Recompile `paper.tex` to confirm:
  * No broken `\ref{}` or `\cite{}` cross-references.
  * All figures (including simplified Fig 9, Fig 10, and new Fig 10B) and tables render with proper captions and formatting.
