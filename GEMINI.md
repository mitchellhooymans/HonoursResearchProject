# Project Context: AGN Synthetic SED Modelling (Paper Production Phase)
This repository contains the codebase for investigating the effects of Active Galactic Nuclei (AGN) contamination on host galaxy colours. The codebase has recently undergone a major refactor. The current focus is strictly on executing pipelines and generating final, publication-ready results, figures, and tables for the upcoming paper.

## 0. Mandatory Context Initialization
**CRITICAL:** Before generating any result scripts or notebooks, the agent MUST:
1. **Understand the Goal:** Read the `.tex` files for the Paper in `context/` to understand exactly what figures, statistics, and tables are required for publication.
2. **Trust the Engine:** Assume the modules in `src/sed_pipeline/` contain the validated, ground-truth physics logic for SED alignment, integration, and scaling. 

## 1. Primary Objective
Generate reproducible, final results using the newly refactored `src/sed_pipeline/` codebase. The agent will assist in writing clean, high-level execution scripts (in `scripts/`) or interactive Master Notebooks (in `notebooks/`) that import from `src/` to calculate photometry, vector offsets, and selection diagnostics.

## 2. Exploration & Fallback Directives
If asked to generate a specific result or plot:
1. **Primary Route:** Import and utilize the clean functions from `src/sed_pipeline/` (e.g., `composite_math.py`, `analysis.py`, `visualization.py`).
2. **Exploration Route (Fallback):** If the exact logic to achieve a result is not immediately obvious or seems missing from `src/`, **you must actively explore the other active scripts and notebooks in the repository** to see how it was previously handled or calculated.
3. **Synthesis:** Bridge any gaps by writing new driver code that connects the existing pipeline tools together.

## 3. Project Architecture & Target Outputs
The agent must respect the repository structure and place outputs accordingly:
* `data/raw/`: Immutable original data. Do not modify.
* `data/processed/`: Where your new scripts should save the final generated composite SED catalogs and filtered datasets.
* `notebooks/`: The destination for narrative-driven analysis (e.g., `Paper_Results_Master.ipynb`) used for interactive plotting.
* `scripts/`: The destination for automated, batch-processing pipelines (e.g., `generate_all_photometry.py`).
* `src/sed_pipeline/`: The core engine. *Only suggest edits to these files if a critical bug is found during execution.*

## 4. Coding & Output Standards
* **Import-Heavy:** Scripts and notebooks should be lightweight. Heavy lifting should be done by importing `src.sed_pipeline`.
* **Publication-Quality Visualizations:** Any Matplotlib/Seaborn code generated must be highly polished. Use consistent color schemes for AGN fraction ($\alpha$), clear legends, and high-resolution output formats suitable for LaTeX integration.
* **Traceability:** When writing a notebook or script to generate a result for the paper, add comments or markdown cells explicitly stating which section of the paper this result corresponds to.

## 5. STRICT EXCLUSIONS: Do Not Scan Legacy
**CRITICAL:** Under no circumstances should the agent scan, read, or reference the `legacy/` directory. This contains outdated, exploratory information that is no longer relevant to the current project and will cause hallucination or confusion. Rely solely on the refactored `src/` and active exploration of top-level directories.