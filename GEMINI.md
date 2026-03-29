# Project Context: AGN Synthetic SED Modelling (Honours Project)
This repository contains the codebase for a research project investigating the effects of Active Galactic Nuclei (AGN) contamination on host galaxy colours, specifically using composite spectral energy distributions (SEDs) and observational data from the ZFOURGE survey.

## 0. Mandatory Context Initialization
**CRITICAL:** Before generating code, refactoring scripts, or suggesting physics logic, the agent MUST:
1. **Scan the `context/` directory:** Read the `.tex` files for the Thesis and Paper to understand the specific mathematical derivations, variable definitions, and physical assumptions (e.g., AGN templates used, extinction laws, and redshift handling).
2. **Verify against 'Ground Truth':** All refactored code in `src/sed_pipeline/` must be cross-referenced with the methodology described in the thesis. If a script's logic contradicts the thesis text, flag the discrepancy immediately.

## 1. Primary Objective
Refactor and consolidate a large collection of exploratory scripts and Jupyter notebooks into a streamlined, redundant-free pipeline. **Crucial Constraint: There must be zero loss of existing functionality.** All data processing, SED alignment, synthetic photometry calculation, and plotting capabilities must be strictly preserved.

## 2. Project Architecture
The agent must adhere to the following structure to organize the project:
* `data/raw/`: Immutable original data (ZFOURGE catalogs, GALSEDATLAS templates, SKIRTOR AGN models).
* `data/processed/`: Generated composite SEDs and filtered catalogs.
* `notebooks/`: Numbered notebooks for final figure generation and interactive exploration.
* `src/sed_pipeline/`: The core refactored Python modules.
* `scripts/`: Consolidated executable scripts (e.g., `run_uvj_analysis.py`, `run_irac_selection.py`).

## 3. Refactoring Directives (Zero Data Loss)
When compiling multiple scripts into the `src/sed_pipeline/` directory, organize the code into these distinct modules:
* `data_io.py`: Centralize all file reading operations. Eliminate duplicate functions that load the ZFOURGE catalog or SKIRTOR models across different scripts.
* `composite_math.py`: Consolidate the core physics logic. This includes SED wavelength alignment via linear interpolation, integral normalization, calculating the scaling factor (SF), and generating the linear combination for the total SED.
* `photometry.py`: Centralize the derivation of synthetic photometry and the definitions for the Lacy wedge (IRAC) and UVJ selection regions (star-forming, quiescent, dusty).
* `analysis.py`: Consolidate the logic for calculating mean vector offsets, completeness fractions, and galaxy population fractions across varying AGN contribution steps (alpha).
* `visualization.py`: Move all Matplotlib/Seaborn code here. Ensure functions are flexible enough to recreate all density plots, vector offset charts, and SED spectra visualizations from the original scripts.

## 4. Coding Standards & Exploration
* **Redundancy Check:** Before writing new code, analyze existing scripts to identify duplicated functions and merge them into a single, robust function in the `src/` directory.
* **Exploration:** While refactoring, flag any potential degeneracies or hardcoded variables (like cosmological parameters H0=70, Omega_M=0.3) and move them to a central configuration file.
* **Documentation:** Ensure all consolidated functions have clear docstrings explaining their parameters and return types so the pipeline is easily readable.
* **PEP 8 Compliance:** All new code must adhere to PEP 8 style guidelines for Python.
- Commenting: Use inline comments to explain complex logic, especially in the physics calculations and data transformations.
- Docstrings: Every function must have a docstring that describes its purpose, parameters, and return values.


## Do not scan legacy
 This contains information that is not relevant to the current project and may cause confusion.
