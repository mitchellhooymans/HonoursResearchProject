"""
photometry.py

Derivation of synthetic photometry and definitions for the Lacy wedge,
UVJ selection regions, and basic magnitude conversions.
"""

import numpy as np
import matplotlib.path as mpath
from astLib import astSED

def load_passbands(filter_paths):
    """
    Loads multiple passbands from a dictionary of paths.
    """
    return {name: astSED.Passband(path, normalise=False) for name, path in filter_paths.items()}

def flux_to_mag(flux):
    """
    Converts flux to magnitude.
    """
    return 25 - 2.5 * np.log10(flux)

def calculate_UVJ_colours(sed_df, pb_U, pb_V, pb_J):
    """
    Calculates U-V and V-J rest-frame colours using astSED.
    """
    wl = sed_df['lambda (Angstroms)'].values
    fl = sed_df['Total Flux (erg/s/cm^2/Angstrom)'].values
    sed = astSED.SED(wavelength=wl, flux=fl, z=0.0)
    
    uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
    vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
    return uv, vj

def calculate_ugr_colours(sed_df, pb_u, pb_g, pb_r, redshift=0.0):
    """
    Calculates u-g and g-r colours using astSED, with optional redshifting.
    """
    wl = sed_df['lambda (Angstroms)'].values
    fl = sed_df['Total Flux (erg/s/cm^2/Angstrom)'].values
    sed = astSED.SED(wavelength=wl, flux=fl, z=0.0)
    
    if redshift > 0.0:
        sed.redshift(redshift)
        
    ug = astSED.SED.calcColour(sed, pb_u, pb_g, magType='AB')
    gr = astSED.SED.calcColour(sed, pb_g, pb_r, magType='AB')
    return ug, gr

def calculate_irac_colours(sed_df, pb_36, pb_45, pb_58, pb_80, redshift=0.0):
    """
    Calculates IRAC flux ratios for the Lacy wedge: log10(f_5.8/f_3.6) and log10(f_8.0/f_4.5).
    """
    wl = sed_df['lambda (Angstroms)'].values
    fl = sed_df['Total Flux (erg/s/cm^2/Angstrom)'].values
    sed = astSED.SED(wavelength=wl, flux=fl, z=0.0)
    
    if redshift > 0.0:
        sed.redshift(redshift)
        
    f_36 = astSED.SED.calcFlux(sed, pb_36)
    f_45 = astSED.SED.calcFlux(sed, pb_45)
    f_58 = astSED.SED.calcFlux(sed, pb_58)
    f_80 = astSED.SED.calcFlux(sed, pb_80)
    
    # Calculate Lacy wedge colours: x = log10(f58/f36), y = log10(f80/f45)
    f_5836 = np.log10(f_58 / f_36) if f_36 > 0 and f_58 > 0 else np.nan
    f_8045 = np.log10(f_80 / f_45) if f_45 > 0 and f_80 > 0 else np.nan
    
    return f_5836, f_8045

def classify_uvj(vj, uv):
    """
    Classifies a galaxy into Star-forming (1), Quiescent (0), or Dusty (2)
    based on its UVJ rest-frame colors using mpath.
    Returns an array of classifications.
    """
    vj = np.asarray(vj)
    uv = np.asarray(uv)
    points = np.column_stack([vj, uv])
    
    # Define the quiescent region path
    # Vertices: (-0.5, 1.3), (0.85, 1.3), (1.6, 1.95), (1.6, 2.5), (-0.5, 2.5)
    quiescent_verts = [(-0.5, 1.3), (0.85, 1.3), (1.6, 1.95), (1.6, 2.5), (-0.5, 2.5), (-0.5, 1.3)]
    q_path = mpath.Path(quiescent_verts)
    
    is_quiescent = q_path.contains_points(points)
    
    # Selection boundary for Dusty vs Star-forming
    # Traditionally V-J > 1.2 is Dusty
    is_dusty = (~is_quiescent) & (vj > 1.2)
    is_sf = (~is_quiescent) & (vj <= 1.2)
    
    classifications = np.zeros_like(uv, dtype=int)
    classifications[is_sf] = 1
    classifications[is_dusty] = 2
    # Quiescent remains 0
    
    return classifications
