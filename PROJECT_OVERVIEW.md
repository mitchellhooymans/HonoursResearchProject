# Project Overview: AGN Synthetic SED Modelling & Observational Decomposition

## 1. Mission & Objective
This toolkit investigates the impact of **Active Galactic Nuclei (AGN)** contamination on host galaxy classifications. It achieves this through a dual-track methodology:
1.  **Forward Modeling (Theoretical):** Constructing synthetic composite SEDs by combining pure galaxy templates with AGN models at varying contribution levels ($\alpha$).
2.  **Backward Analysis (Observational):** Decomposing real-world multi-wavelength SEDs from the **ZFOURGE** survey using **CIGALE** to measure how removing the AGN component shifts a galaxy's position in diagnostic diagrams.

The goal is to determine the "contamination thresholds" at which traditional selection methods (UVJ, ugr dropout, Lacy IRAC) become unreliable for high-redshift galaxy surveys.

---

## 2. Core Methodology & Theory

### A. Synthetic Composite Construction
The pipeline mathematically blends host light and AGN light:
$$F_{\text{total}}(\lambda) = F_{\text{galaxy}}(\lambda) + \alpha \cdot \left( \text{SF} \cdot F_{\text{AGN}}(\lambda) \right)$$
- **$\alpha$ (Alpha):** Fractional AGN contribution, ranging from 0.0 (Pure Galaxy) to 1.0 (Integrated AGN flux equals integrated Galaxy flux).
- **SF (Scaling Factor):** Normalizes the AGN model to the host galaxy's bolometric luminosity before scaling by $\alpha$.
- **Wavelength Alignment:** The pipeline performs linear interpolation to align disparate wavelength grids (e.g., SKIRTOR vs GALSEDATLAS) before summing fluxes.

### B. Observational Validation (ZFOURGE + CIGALE)
The project validates theoretical results against the **ZFOURGE (FourStar Galaxy Evolution)** survey:
- **EAZY:** Used for initial SED fitting and rest-frame template extraction from observational photometry.
- **CIGALE Decomposition:** Real SEDs are decomposed into constituent physical components:
    - *Host Components:* Stellar (old/young), Nebular lines, and Dust.
    - *AGN Components:* SKIRTOR polar dust, torus, and disk emission.
- **Comparison:** The pipeline compares the $(\Delta U-V, \Delta V-J)$ vector offsets of theoretical composites against those found in CIGALE-decomposed observed sources.

### C. Redshift Grid Selection Analysis
Traditional dropout selections (e.g., Lyman Break Galaxies) are redshift-dependent. This toolkit simulates this by:
- Redshifting SEDs from **$z=0.0$ to $4.0$** in 0.1 increments.
- Applying the **Giavalisco Wedge** selection criteria to identify **Correct Identification**, **Misidentification**, and **Missed Selection** categories across the $\alpha$ grid.

---

## 3. Data Sources & Integration

- **Galaxy Templates:** 129 templates from **GALSEDATLAS (Brown et al. 2014)**, chosen for their broad wavelength coverage ($10^2$ to $10^7$ Å).
- **AGN Models:** **SKIRTOR (Stalevski et al. 2012)** clumpy-torus models, distinguishing between Type 1 (unobscured) and Type 2 (obscured) AGN.
- **Observational Catalogs:** ZFOURGE FITS files (CDFS, COSMOS, UDS fields) providing photometry, redshifts, and stellar masses.
- **Filter Passbands:** Standardized `.dat` files for Johnson U/V, 2MASS J, SDSS u/g/r, and Spitzer IRAC I1-I4.

---

## 4. Software Architecture (`src/sed_pipeline/`)

The pipeline is fully modularized for reproducibility and AI-driven extension:

- **`config.py`**: Centralizes physics parameters (Lambda-CDM), SKIRTOR settings, and absolute project pathing.
- **`data_io.py`**: High-level readers for SKIRTOR `.dat`, GALSEDATLAS restframe templates, ZFOURGE FITS catalogs, and CIGALE `best_model` outputs.
- **`composite_math.py`**: Physics engine for alignment, flux integration (handles NumPy 1.x/2.x compatibility), and composite summation.
- **`photometry.py`**: Calculates magnitudes and colours (UVJ, ugr, IRAC) by projecting SEDs onto filter transmission curves. Supports artificial redshifting.
- **`analysis.py`**: 
    - **Classification:** Implements logic for UVJ regions, Giavalisco dropouts, and Lacy wedges.
    - **Metrics:** Calculates **Completeness**, **Mean Vector Offsets**, and **Population Migration** fractions.
    - **Bootstrap:** Logic for mass-distribution bootstrapping to match survey parent samples.
    - **Error Propagation:** Implements the corrected flux-to-magnitude error formulas derived in the thesis.
- **`visualization.py`**: Wrappers for density-contoured colour-colour diagrams and multi-segment evolution tracks.

---

## 5. Key Analytical Results

1.  **Completeness Curves:** Graphs showing selection efficiency dropping as $\alpha$ increases.
2.  **Colour Evolution Tracks:** Visual paths of galaxies migrating across selection boundaries due to AGN contamination.
3.  **Migration Tables:** Quantifies how many Star-forming galaxies "masquerade" as Quiescent or Dusty due to AGN-induced reddening or blueing.
4.  **Statistical Validation:** LaTeX-ready tables of vector offsets comparing theoretical predictions to observational data.

---

## 6. Directory Structure
```text
HonoursResearchProject/
├── src/sed_pipeline/        # Core Engine (Modular Package)
├── scripts/                 # Automation (e.g., Redshift Grid Generation)
├── notebooks/               # Presentation (Paper_Results_Master.ipynb)
├── datasets/                # RAW DATA (Immutable)
│   ├── Filters/             # Filter curves
│   ├── Templates/           # Brown & Skirtor models
│   └── zfourge/             # Observational FITS files
├── outputs/                 # ARTIFACTS (PDFs, LaTeX tables, CSVs)
├── Context/                 # Thesis source (.tex) and Reference papers
└── astLib/                  # Astronomy support library (Modified)
```

---

## 7. AI Operations & Grounding
For future sessions using this toolkit:
- **Physics Constant:** Integrated flux MUST use the `_trapezoid` wrapper in `composite_math.py` to maintain cross-environment safety.
- **Unit Safety:** SKIRTOR models are converted from $W/m^2$ to $erg/s/cm^2/Å$ during ingestion.
- **Path Logic:** Always append `config.PROJECT_ROOT` to your `sys.path` when working in `notebooks/` or `scripts/`.
- **Methodology Ref:** Refer to **Section 5.1** of the thesis for theoretical modeling and **Section 5.2** for observational validation details.
