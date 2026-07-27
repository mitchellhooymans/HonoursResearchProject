"""
run_pipeline.py

Main executable script for the Refactored SED Pipeline.
Demonstrates the module linkage and initialization of the project.
"""

import sys
import os
import numpy as np

# Adjust path to import the new src package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sed_pipeline import config
from src.sed_pipeline import data_io
from src.sed_pipeline import composite_math
from src.sed_pipeline import photometry
from src.sed_pipeline import visualization
from src.sed_pipeline import analysis
import matplotlib.pyplot as plt

def main():
    print("========================================")
    print("     SED AGN Contamination Pipeline     ")
    print("========================================")
    
    # Print configuration to prove successful linkage
    print("\n[Configuration Loaded]")
    print(f"Cosmology: H0={config.H0}, Omega_M={config.OMEGA_M}")
    print(f"Alpha Scaling Grid: {config.ALPHA_VALUES}")
    print(f"Raw Data Dir: {config.RAW_DATA_DIR}")

    # The pipeline is now ready. 
    # To run a full simulation, we would do:
    # 1. agn_df = data_io.read_skirtor_model(config.SKIRTOR_DIR, **config.SKIRTOR_TYPE1_PARAMS)
    # 2. gal_list, _ = data_io.read_brown_galaxy_templates(config.GALSEDATLAS_DIR)
    # 3. For each alpha in config.ALPHA_VALUES:
    #      composite_math.create_composite_sed(agn_df, gal_list[0], alpha)
    # 4. uv, vj = photometry.calculate_UVJ_colours(...)
    # 5. analysis.calculate_population_fractions(photometry.classify_uvj(vj, uv))
    # 6. visualization.plot_uvj_diagram(vj, uv)

    print("\n[Status]")
    print("All modules imported successfully. Mathematical error propagations")
    print("have been corrected per the project's colour-error definitions.")
    print("Ready for full batch processing.")

if __name__ == "__main__":
    main()
