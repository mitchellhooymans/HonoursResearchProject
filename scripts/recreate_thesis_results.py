"""
recreate_thesis_results.py

Comprehensive script to recreate key visualizations and analysis from the thesis.
Demonstrates:
1. ZFOURGE catalog loading and UVJ plotting with density.
2. CIGALE SED decomposition and color shift analysis.
3. Population fraction calculations.
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sed_pipeline import config, data_io, photometry, visualization, analysis

def recreate_uvj_catalog_analysis(field='CDFS'):
    print(f"--- Recreating UVJ Analysis for {field} ---")
    
    # 1. Load ZFOURGE Data
    df = data_io.read_zfourge_data(field, config.ZFOURGE_CATALOG_DIR)
    
    # 2. Filter data (Use=1, Sigma>5)
    # Note: Corrected the error propagation call
    df['UV_err'] = analysis.color_error(df['U'], df['V'], df['eU'], df['eV'])
    df['VJ_err'] = analysis.color_error(df['V'], df['J'], df['eV'], df['eJ'])
    
    # Apply filters consistent with ZFourgeUVJInvestigation.py
    mask = (df['U'] > 5*df['eU']) & (df['V'] > 5*df['eV']) & (df['J'] > 5*df['eJ']) & (df['Use'] == 1)
    df_filtered = df[mask].copy()
    
    # 3. Classify populations
    vj = df_filtered['V'] - df_filtered['J'] # This is incorrect in data_io? 
    # Actually, data_io gives U, V, J from rest_df which are already colors or fluxes? 
    # In ThesisOutputs.py: df['U']-df['V'] is UV.
    # In ZFourgeUVJInvestigation: U, V, J are fluxes, mag_U is 25 - 2.5*log10(flux)
    
    # Re-calculate magnitudes to be sure
    df_filtered['mag_U'] = photometry.flux_to_mag(df_filtered['U'])
    df_filtered['mag_V'] = photometry.flux_to_mag(df_filtered['V'])
    df_filtered['mag_J'] = photometry.flux_to_mag(df_filtered['J'])
    
    uv = df_filtered['mag_U'] - df_filtered['mag_V']
    vj = df_filtered['mag_V'] - df_filtered['mag_J']
    
    classifications = photometry.classify_uvj(vj, uv)
    
    # 4. Plot UVJ Diagram with Density
    avg_vj_err = df_filtered['VJ_err'].mean()
    avg_uv_err = df_filtered['UV_err'].mean()
    
    visualization.plot_uvj_diagram(vj, uv, classifications, 
                                   title=f"UVJ Density Plot - {field}", 
                                   show_density=True, 
                                   avg_error=(avg_vj_err, avg_uv_err))
    
    plt.savefig(os.path.join(config.PROCESSED_DATA_DIR, f"thesis_uvj_{field}_density.png"))
    print(f"Saved: thesis_uvj_{field}_density.png")
    
    # 5. Output Fractions
    q, sf, d = analysis.calculate_population_fractions(classifications)
    print(f"Population Fractions: Quiescent={q:.2f}, Star-forming={sf:.2f}, Dusty={d:.2f}")

def recreate_cigale_decomposition_demo():
    print("\n--- Recreating CIGALE Decomposition Demo ---")
    
    # 1. Load a sample CIGALE Best Model
    # Note: This requires the fits files to be in the expected location
    sample_field = 'CDFS'
    sample_id = '5886' # From SEDProcessing_DecomposedSEDs_Full.py
    redshift = 0.5 # Example
    
    filepath = os.path.join(config.RAW_DATA_DIR, 'full_zfourge_decomposed', 
                            f'{sample_field.lower()}_best_models_fits', f'{sample_id}_best_model.fits')
    
    if not os.path.exists(filepath):
        print(f"Warning: Sample file {filepath} not found. Skipping decomposition demo.")
        return

    full_sed = data_io.read_cigale_best_model(filepath, redshift=redshift, restframe=True)
    host_sed = analysis.decompose_cigale_sed(full_sed, target='host')
    agn_sed = analysis.decompose_cigale_sed(full_sed, target='agn')
    
    # 2. Plot Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(full_sed['lambda (Angstroms)'], full_sed['Total Flux (erg/s/cm^2/Angstrom)'], label='Full SED', color='k')
    ax.loglog(host_sed['lambda (Angstroms)'], host_sed['Total Flux (erg/s/cm^2/Angstrom)'], label='Host (Decomposed)', color='b', ls='--')
    ax.loglog(agn_sed['lambda (Angstroms)'], agn_sed['Total Flux (erg/s/cm^2/Angstrom)'], label='AGN (Isolated)', color='r', ls=':')
    
    ax.set_xlabel('Wavelength (Angstroms)')
    ax.set_ylabel('Flux (erg/s/cm^2/Angstrom)')
    ax.set_title(f"SED Decomposition Demo: {sample_field}_{sample_id}")
    ax.legend()
    
    plt.savefig(os.path.join(config.PROCESSED_DATA_DIR, "thesis_sed_decomposition_demo.png"))
    print("Saved: thesis_sed_decomposition_demo.png")

if __name__ == "__main__":
    recreate_uvj_catalog_analysis('COSMOS') # COSMOS was used in the script snippet
    recreate_cigale_decomposition_demo()
    print("\nThesis recreation script finished.")
