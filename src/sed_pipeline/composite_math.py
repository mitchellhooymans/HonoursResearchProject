"""
composite_math.py

Consolidates the core physics logic for manipulating SEDs,
including alignment, interpolation, and creating composite combinations.
"""

import numpy as np
import pandas as pd

def integral_flux(wavelengths, flux):
    """
    Computes the integrated flux of an SED over its wavelength range.
    """
    return np.trapz(flux, wavelengths)

def compute_scaling_factor(agn_wavelengths, agn_flux, galaxy_wavelengths, galaxy_flux):
    """
    Computes the Scaling Factor (SF) using integral normalization.
    SF = F_galaxy / F_agn
    """
    integrated_agn_flux = integral_flux(agn_wavelengths, agn_flux)
    integrated_galaxy_flux = integral_flux(galaxy_wavelengths, galaxy_flux)
    
    if integrated_agn_flux == 0:
        return 0
    return integrated_galaxy_flux / integrated_agn_flux

def adjust_wavelength_range(wl1, flux1, wl2, flux2):
    """
    Aligns two SEDs to a common wavelength grid via linear interpolation,
    masking out non-overlapping ranges.
    """
    combined_wavelengths = np.union1d(wl1, wl2)
    
    # Interpolate flux values to the combined grid
    combined_flux1 = np.interp(combined_wavelengths, wl1, flux1, left=np.nan, right=np.nan)
    combined_flux2 = np.interp(combined_wavelengths, wl2, flux2, left=np.nan, right=np.nan)
    
    # Restrict to intersecting range
    min_wavelength = np.max([np.min(wl1), np.min(wl2)])
    max_wavelength = np.min([np.max(wl1), np.max(wl2)])
    
    mask = (combined_wavelengths >= min_wavelength) & (combined_wavelengths <= max_wavelength)
    
    return combined_wavelengths[mask], combined_flux1[mask], combined_flux2[mask]

def create_composite_sed(agn_df, gal_sed, alpha):
    """
    Creates a composite SED combining a galaxy template and an AGN model,
    scaled by alpha.
    """
    wl_agn = agn_df['lambda (Angstroms)'].values
    flux_agn = agn_df['Total Flux (erg/s/cm^2/Angstrom)'].values
    wl_gal = gal_sed['lambda (Angstroms)'].values
    flux_gal = gal_sed['Total Flux (erg/s/cm^2/Angstrom)'].values
    
    # 1. Align SED grids
    combined_wl, aligned_agn_flux, aligned_gal_flux = adjust_wavelength_range(wl_agn, flux_agn, wl_gal, flux_gal)
    
    # 2. Compute Scaling Factor over the aligned overlap
    scaling_factor = compute_scaling_factor(combined_wl, aligned_agn_flux, combined_wl, aligned_gal_flux)
    
    # 3. Apply Scaling and Alpha Contribution
    scaled_agn_flux = aligned_agn_flux * scaling_factor
    combined_flux = aligned_gal_flux + (alpha * scaled_agn_flux)
    
    composite_sed_df = pd.DataFrame({
        'lambda (Angstroms)': combined_wl,
        'Total Flux (erg/s/cm^2/Angstrom)': combined_flux
    })
    
    return composite_sed_df
