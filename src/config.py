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
GALSEDATLAS_DIR = os.path.join(RAW_DATA_DIR, "Templates")

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
