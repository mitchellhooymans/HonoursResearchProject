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
# folder paper.tex's \includegraphics calls actually point at. When present,
# the paper-figure export step copies straight in here too, so a figure can
# never go stale in the paper just because the manual copy step was skipped.
AGNPAPER_DIR = os.path.join(PROJECT_ROOT, "Context", "AGNPaper")
PAPER_OVERLEAF_IMAGES_DIR = os.path.join(AGNPAPER_DIR, "Images", "TheoreticalModelPlots")

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
    "UVJ_Type1_Density_Evolution.pdf": "UVJ_Type1_Density_Evolution.pdf",
    "ZFOURGE_UVJ_Density_RegionTracks_Combined.pdf": "ZFOURGE_UVJ_Density_RegionTracks_Combined.pdf",
    "UVJ_CIGALE_hidden_quiescent_redshift.pdf": os.path.join("fracAGN_diagnostics", "UVJ_CIGALE_hidden_quiescent_redshift.pdf"),
    "UVJ_CIGALE_fracAGN_distribution_and_offset.pdf": os.path.join("fracAGN_diagnostics", "UVJ_CIGALE_fracAGN_distribution_and_offset.pdf"),
    "UVJ_CIGALE_fracAGN_redshift_confound.pdf": os.path.join("fracAGN_diagnostics", "UVJ_CIGALE_fracAGN_redshift_confound.pdf"),
    "UVJ_CIGALE_allhosts_fracAGN_colored_combined.pdf": os.path.join("fracAGN_diagnostics", "UVJ_CIGALE_allhosts_fracAGN_colored_combined.pdf"),
    "UVJ_CIGALE_allhosts_fracAGN_colored_redshift_bins.pdf": os.path.join("fracAGN_diagnostics", "UVJ_CIGALE_allhosts_fracAGN_colored_redshift_bins.pdf"),
    "Paper_UVJ_Contamination_Bias_Summary.pdf": "Paper_UVJ_Contamination_Bias_Summary.pdf",
}

# SKIRTOR AGN model parameters (Type 1 and Type 2 defaults)
# Based on the project's modelling methodology
SKIRTOR_TYPE1_PARAMS = {'optical_depth': 7, 'p': 0.5, 'q': 0, 'opening_angle': 40, 'radius_ratio': 20, 'inclination': 0}
SKIRTOR_TYPE2_PARAMS = {'optical_depth': 7, 'p': 0.5, 'q': 0, 'opening_angle': 40, 'radius_ratio': 20, 'inclination': 90}

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
