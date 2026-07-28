"""
data_io.py

Centralized module for reading and writing data, templates, and catalogs.
Migrated and cleaned from legacy `carf.py` and loose scripts.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

def read_skirtor_model(folder_path, optical_depth, p, q, opening_angle, radius_ratio, inclination):
    """
    Reads a SKIRTOR model file into a DataFrame and converts units.
    """
    filename = f't{optical_depth}_p{p}_q{q}_oa{opening_angle}_R{radius_ratio}_Mcl0.97_i{inclination}_sed.dat'
    filepath = os.path.join(folder_path, filename)
    data = np.loadtxt(filepath, skiprows=5)
    
    df = pd.DataFrame(data)
    df[0] = df[0] * 10000  # Convert to angstroms
    
    # Convert W/m^2 to erg/s/cm^2/Angstrom
    df.iloc[:, 1:] = df.iloc[:, 1:] * 10**7
    df.iloc[:, 1:] = df.iloc[:, 1:].div(df[0], axis=0)
    
    df.columns = [
        'lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)', 
        'Direct AGN Flux', 'Scattered AGN Flux', 
        'Total Dust Emission Flux', 'Dust Emission Scattered Flux', 'Transparent Flux'
    ]
    return df

def read_brown_galaxy_templates(folder_path):
    """
    Reads all GALSEDATLAS (Brown) restframe templates from the given folder.
    """
    df_list = []
    objname_list = []
    for file in os.listdir(folder_path):
        if file.endswith('_restframe.dat'):
            objname = file.split('_restframe.dat')[0]
            filepath = os.path.join(folder_path, file)
            data = np.loadtxt(filepath)
            df = pd.DataFrame(data)
            
            # Wavelength in microns -> Angstroms
            df[0] = df[0] * 10000
            df.columns = ['lambda (Angstroms)', 'Luminosity (W/Hz)', 'Total Flux (erg/s/cm^2/Angstrom)', 'Source']
            
            df_list.append(df)
            objname_list.append(objname)
            
    return df_list, objname_list

def read_swire_templates(folder_path):
    """
    Reads all SWIRE template files (.sed) from the given folder.
    """
    df_list = []
    objname_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.sed'):
            objname = file.split('_template_norm.sed')[0]
            filepath = os.path.join(folder_path, file)
            data = np.loadtxt(filepath)
            df = pd.DataFrame(data)
            
            # Name columns appropriately
            df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
            
            df_list.append(df)
            objname_list.append(objname)
            
    return df_list, objname_list

def read_zfourge_data(fieldname, folderpath):
    """
    Reads ZFOURGE catalogs and extracts necessary fluxes, identifiers, and properties.
    Merges catalog, restframe, eazy, and sfr files.
    """
    zfourge_fields = {
        'CDFS': ['zf_cdfs.fits', 'zf_cdfs_rest.fits', 'zf_cdfs_eazy.fits', 'zf_cdfs_sfr.fits'],
        'COSMOS': ['zf_cosmos.fits', 'zf_cosmos_rest.fits', 'zf_cosmos_eazy.fits', 'zf_cosmos_sfr.fits'],
        'UDS': ['zf_uds.fits', 'zf_uds_rest.fits', 'zf_uds_eazy.fits', 'zf_uds_sfr.fits'],
    }
    
    catalog_file = os.path.join(folderpath, zfourge_fields[fieldname][0])
    rest_file = os.path.join(folderpath, zfourge_fields[fieldname][1])
    eazy_file = os.path.join(folderpath, zfourge_fields[fieldname][2])
    sfr_file = os.path.join(folderpath, zfourge_fields[fieldname][3])
    
    df = Table.read(catalog_file).to_pandas()
    rest_df = Table.read(rest_file).to_pandas()
    eazy_df = Table.read(eazy_file).to_pandas()
    sfr_df = Table.read(sfr_file).to_pandas()
    
    df.rename(columns={'Seq': 'id'}, inplace=True)
    rest_df.rename(columns={'Seq': 'id', 'FU': 'U', 'e_FU': 'eU', 'FV': 'V', 'e_FV': 'eV', 'FJ': 'J', 'e_FJ': 'eJ'}, inplace=True)
    eazy_df.rename(columns={'Seq': 'id'}, inplace=True)
    sfr_df.rename(columns={'Seq': 'id'}, inplace=True)
    
    df = pd.merge(df, rest_df[['id', 'U', 'eU', 'V', 'eV', 'J', 'eJ']], on='id', suffixes=('_original', '_rest'))
    df = pd.merge(df, eazy_df[['id', 'zpk']], on='id', suffixes=('', '_eazy'))
    df = pd.merge(df, sfr_df[['id', 'lssfr', 'lmass']], on='id', suffixes=('', '_sfr'))
    
    df['field'] = fieldname
    df['full_id'] = fieldname + "_" + df['id'].astype(str)
    
    return df

def read_cigale_best_model(filepath, redshift=None, restframe=True):
    """
    Reads a CIGALE best_model.fits file and converts to F_lambda units.
    Optionally redshifts to restframe.
    """
    df = Table.read(filepath).to_pandas()
    
    # CIGALE wavelength is in nm -> convert to Angstroms
    df['wavelength'] = df['wavelength'] * 10
    
    # Convert mJy (Fnu) to Janskys
    s_nu = df['Fnu'] * 1e-3
    
    if restframe and redshift is not None:
        # 1. Convert S_nu to nu*S_nu (invariant)
        freq_obs = 3e18 / df['wavelength']
        nu_s_nu = s_nu * freq_obs
        
        # 2. Shift wavelength to restframe
        df['wavelength'] = df['wavelength'] / (1 + redshift)
        
        # 3. Convert nu*S_nu back to S_nu at new frequency
        freq_rest = 3e18 / df['wavelength']
        s_nu = nu_s_nu / freq_rest
        
    # Convert S_nu (Janskys) to F_lambda (erg/s/cm^2/Angstrom)
    # F_lambda = S_nu * c / lambda^2
    # c in Angstrom/s is 3e18. S_nu in Jy is 1e-23 erg/s/cm^2/Hz
    # F_lambda = S_nu * 1e-23 * 3e18 / lambda^2 = S_nu * 3e-5 / lambda^2
    df['Flambda'] = s_nu * 3e-5 / (df['wavelength']**2)
    
    # Standardize columns for the pipeline
    df['lambda (Angstroms)'] = df['wavelength']
    df['Total Flux (erg/s/cm^2/Angstrom)'] = df['Flambda']
    
    return df
