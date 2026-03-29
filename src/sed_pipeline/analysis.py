"""
analysis.py

Consolidates logic for error propagation, calculating mean vector offsets, 
completeness fractions, and galaxy population fractions.
"""

import numpy as np
import pandas as pd

def flux_to_mag_error(flux, error):
    """
    CORRECTED ERROR PROPAGATION (Thesis Fix):
    M = -2.5 * log10(F) + C
    dM = |-2.5 / (F * ln(10))| * dF
    """
    return (2.5 / np.log(10)) * (error / flux)

def color_error(f1, f2, e1, e2):
    """
    CORRECTED COLOR ERROR PROPAGATION (Thesis Fix):
    C = -2.5 * log10(f1/f2)
    dC = (2.5 / ln(10)) * sqrt((e1/f1)^2 + (e2/f2)^2)
    """
    a = 2.5 / np.log(10)
    return a * np.sqrt((e1/f1)**2 + (e2/f2)**2)

def calculate_vector_magnitude(vj1, uv1, vj2, uv2):
    """
    Calculates the Euclidean distance (vector magnitude) between two UVJ positions.
    """
    return np.sqrt((np.asarray(vj1) - np.asarray(vj2))**2 + (np.asarray(uv1) - np.asarray(uv2))**2)

def calculate_mean_vector_offset(vj_alpha, uv_alpha, vj_initial, uv_initial):
    """
    Calculates the mean vector offset across a population.
    M_C(alpha) = 1/N * sum(sqrt((VJa - VJi)^2 + (UVa - UVi)^2))
    """
    vj_alpha = np.asarray(vj_alpha)
    uv_alpha = np.asarray(uv_alpha)
    vj_initial = np.asarray(vj_initial)
    uv_initial = np.asarray(uv_initial)
    
    # Drop NaNs
    mask = ~np.isnan(vj_alpha) & ~np.isnan(uv_alpha) & ~np.isnan(vj_initial) & ~np.isnan(uv_initial)
    
    dvj = vj_alpha[mask] - vj_initial[mask]
    duv = uv_alpha[mask] - uv_initial[mask]
    
    offsets = np.sqrt(dvj**2 + duv**2)
    return np.mean(offsets)

def decompose_cigale_sed(sed_df, target='host'):
    """
    Decomposes a CIGALE SED into its host galaxy or AGN components.
    target: 'host' (galaxy only) or 'agn' (agn components only)
    """
    new_df = sed_df.copy()
    
    agn_components = [
        'agn.SKIRTOR2016_polar_dust', 
        'agn.SKIRTOR2016_torus', 
        'agn.SKIRTOR2016_disk'
    ]
    
    # Available components in CIGALE best_model typically include:
    # stellar.old, stellar.young, nebular.lines_old, nebular.lines_young, etc.
    # To get Host: Total - sum(AGN components)
    # To get AGN: Total - sum(Host components) or just sum(AGN components)
    
    available_agn = [c for c in agn_components if c in new_df.columns]
    
    if target == 'host':
        # Subtract all AGN components from total
        new_df['Total Flux (erg/s/cm^2/Angstrom)'] = new_df['L_lambda_total']
        for comp in available_agn:
            new_df['Total Flux (erg/s/cm^2/Angstrom)'] -= new_df[comp]
    elif target == 'agn':
        # Sum only the AGN components
        new_df['Total Flux (erg/s/cm^2/Angstrom)'] = 0
        for comp in available_agn:
            new_df['Total Flux (erg/s/cm^2/Angstrom)'] += new_df[comp]
            
    return new_df

def run_mass_bootstrap(df, population_col, mass_col, n_trials=5000, sample_size=10):
    """
    Performs bootstrapping to find the mean mass distribution for each population.
    """
    results = {}
    for pop_type in df[population_col].unique():
        pop_df = df[df[population_col] == pop_type]
        means = []
        for _ in range(n_trials):
            if len(pop_df) < sample_size:
                continue
            sample = pop_df[mass_col].sample(sample_size)
            means.append(sample.mean())
        results[pop_type] = np.array(means)
    return results

def calculate_population_fractions(classifications):
    """
    Calculates the fraction of Quiescent (0), Star-forming (1), and Dusty (2) galaxies.
    """
    total = len(classifications)
    if total == 0:
        return 0, 0, 0
    q_frac = np.sum(classifications == 0) / total
    sf_frac = np.sum(classifications == 1) / total
    d_frac = np.sum(classifications == 2) / total
    return q_frac, sf_frac, d_frac
