"""
config.py

Project-specific configuration for this repo's paper analysis: dataset
paths, cosmology, and model-grid parameters. The reusable GLASS library
itself now lives in the standalone `glass` package (installed as a
dependency) - this module only supplies the paths to the full research
datasets that the bundled sample data in that package doesn't include.
"""

import numpy as np
import os

from glass import config as _glass_config

# ==============================================================================
# Cosmological Parameters (Lambda-CDM)
# ==============================================================================
H0 = 70.0          # Hubble constant (km s^-1 Mpc^-1)
OMEGA_M = 0.3      # Matter density
OMEGA_LAMBDA = 0.7 # Dark energy density

# ==============================================================================
# Model Grid Parameters
# ==============================================================================
# Alpha represents the fractional contribution of the AGN to the composite SED
# Ranging from 0 (pure galaxy) to 1.0 (100% AGN contribution scaled)
ALPHA_VALUES = np.linspace(0, 1, 11)

# ==============================================================================
# File Paths and Directories
# ==============================================================================
# Define base directory relative to this file (repo root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "outputs")

ZFOURGE_CATALOG_DIR = os.path.join(RAW_DATA_DIR, "zfourge")

# GALSEDATLAS (Brown) and SKIRTOR templates are bundled with the `glass`
# package itself (MAST HLSP naming for GALSEDATLAS, SKIRTOR release naming
# for SKIRTOR) - point at those instead of this repo's local datasets/
# copies, which use an older, incompatible GALSEDATLAS file-naming scheme.
GALSEDATLAS_DIR = _glass_config.GALSEDATLAS_DIR
SKIRTOR_DIR = _glass_config.SKIRTOR_DIR

# Consolidated destination for the plots actually referenced by
# Context/AGNPaper/paper.tex - lets the notebooks export just the
# paper-required subset instead of the full working outputs/ directory.
PAPER_FIGURES_DIR = os.path.join(PROCESSED_DATA_DIR, "PaperFigures")

# The gitignored Overleaf checkout (only present locally) and the image
# folder paper.tex's \includegraphics calls actually point at. Notebooks
# export figures into outputs/PaperFigures/ only; they copy into the paper's
# own Images folder ONLY when COPY_FIGURES_INTO_PAPER is True (below), because
# the figures in the paper are curated and must never be replaced by an
# older copy just because a notebook was re-run.
AGNPAPER_DIR = os.path.join(PROJECT_ROOT, "Context", "AGNPaper")
PAPER_OVERLEAF_IMAGES_DIR = os.path.join(AGNPAPER_DIR, "Images", "TheoreticalModelPlots")

# Safety switch: leave False so running a notebook never touches the paper's Images folder.
# Set True only for a deliberate refresh (src/paper_export.py then backs up every replaced file and
# keeps any paper figure that is newer than its source).
COPY_FIGURES_INTO_PAPER = False

# Figures required by paper.tex's \includegraphics calls, keyed by the exact
# filename the paper expects. Value = path relative to PROCESSED_DATA_DIR
# where the source file is currently saved. Update this dict if the paper's
# figures change - it is the single source of truth for "which plots does
# the paper need", derived directly from paper.tex.
PAPER_FIGURE_MANIFEST = {
    "Paper_IRAC_Evolution_Combined.pdf": "Paper_IRAC_Evolution_Combined.pdf",
    "Paper_UVJ_Evolution_Combined.pdf": "Paper_UVJ_Evolution_Combined.pdf",
    "Paper_UVJ_Fractions_Combined.pdf": "Paper_UVJ_Fractions_Combined.pdf",
    "CompositeSEDs_UVJ.pdf": "CompositeSEDs_UVJ.pdf",
    "UVJ_CIGALE_allhosts_fracAGN_colored_combined.pdf": os.path.join("useall", "UVJ_useall_allhosts_fracAGN_colored_combined.pdf"),
    "UVJ_CIGALE_allhosts_fracAGN_colored_redshift_bins.pdf": os.path.join("useall", "UVJ_useall_allhosts_fracAGN_colored_redshift_bins.pdf"),
    "Paper_UVJ_Offset_vs_fAGN.pdf": os.path.join("useall", "Paper_UVJ_Offset_vs_fAGN_useall.pdf"),
    "Paper_UVJ_offset_theory_vs_observed.pdf": os.path.join("useall", "Paper_UVJ_offset_theory_vs_observed_useall.pdf"),
    "UVJ_CIGALE_fracAGN_redshift_confound.pdf": os.path.join("useall", "UVJ_useall_fracAGN_redshift_confound.pdf"),
    "UVJ_CIGALE_LAGN_vs_redshift.pdf": os.path.join("useall", "UVJ_useall_LAGN_vs_redshift.pdf"),
    "UVJ_CIGALE_recombination_validation.pdf": os.path.join("useall", "UVJ_useall_recombination_validation.pdf"),
    "UVJ_CIGALE_movers_vs_population.pdf": os.path.join("useall", "UVJ_useall_movers_vs_population.pdf"),
}

# SKIRTOR AGN model parameters (Type 1 and Type 2 defaults)
# Inclination matched to CIGALE's own best-fit AGN geometry for this ZFOURGE
# sample (docs/cigale_recombination_validation_findings.md Sec 3: scanning all
# 6,509 CIGALE best-fit FITS headers found only two inclinations were ever
# fit -- i=30 (76.1%, n=4956) and i=70 (23.9%, n=1553), both at CIGALE's own
# torus shape p=1,q=1. Only the inclination is matched here; this project's
# own torus shape (p=0.5, q=0) is kept unchanged. At this shape, i=30 is
# still 76.9% direct/unobscured flux (vs 80.2% at the old i=0) and i=70 is
# still 98.9% dust-reprocessed/obscured (vs 99.9% at the old i=90) -- the
# Type 1 (unobscured) / Type 2 (obscured) dichotomy survives, just less
# extreme than the old face-on/edge-on limits.
SKIRTOR_TYPE1_PARAMS = {'optical_depth': 7, 'p': 0.5, 'q': 0, 'opening_angle': 40, 'radius_ratio': 20, 'inclination': 30}
SKIRTOR_TYPE2_PARAMS = {'optical_depth': 7, 'p': 0.5, 'q': 0, 'opening_angle': 40, 'radius_ratio': 20, 'inclination': 70}

# ==============================================================================
# Photometric Definitions
# ==============================================================================
FILTER_DIR = os.path.join(RAW_DATA_DIR, "Filters")

# Filter paths (migrated from the original filter definitions)
FILTER_PATHS = {
    'U': os.path.join(FILTER_DIR, 'Generic_Johnson.U.dat'),
    'V': os.path.join(FILTER_DIR, 'Generic_Johnson.V.dat'),
    'J': os.path.join(FILTER_DIR, '2MASS_2MASS.J.dat'),
    'u': os.path.join(FILTER_DIR, 'Paranal_OmegaCAM.u_SDSS.dat'),
    'g': os.path.join(FILTER_DIR, 'Paranal_OmegaCAM.g_SDSS.dat'),
    'r': os.path.join(FILTER_DIR, 'Paranal_OmegaCAM.r_SDSS.dat'),
    '3.6': os.path.join(FILTER_DIR, 'Spitzer_IRAC.I1.dat'),
    '4.5': os.path.join(FILTER_DIR, 'Spitzer_IRAC.I2.dat'),
    '5.8': os.path.join(FILTER_DIR, 'Spitzer_IRAC.I3.dat'),
    '8.0': os.path.join(FILTER_DIR, 'Spitzer_IRAC.I4.dat')
}
