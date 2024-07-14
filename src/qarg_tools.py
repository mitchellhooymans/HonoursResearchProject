# This is the codebase for QARG tools, tools to be used by the QUT Astrophysics Research Group
# This file is intended to be used and imported for use in the project. This
# file contains functions, variables and code that will make the project easy 
# to use in the future for this project

# Import packages <-  note the dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# This will need to be adjusted later but we can just move the astSED file into the current directory.
from astLib import astSED
from astropy.io import fits


################################################################################################

# Functions we need in this util file
# Needs to be general enough to be reusable without the exact setup


# Loading data
# Loading models 
# - in particular we are wanting to load SKIRTOR models
# - and we are also wanting to load in template set
# loading passbands based on a particular input
# exporting data to output files



######################################################################################################################



# Given a colourspace, return the required filters
def load_passbands(colorspace, path='datasets\Filters'):
    """
    Load the passbands based on the provided colorspace.

    Args:
        colorspace (str): The colorspace (e.g., 'UVJ', 'ugr', 'IRAC', etc.).
        path (str): The directory where the filter files are located.

    Returns:
        list: A list of astSED.Passband objects representing the loaded filters.
    """

    pb_path = path  # No need for os.path.join if you're already providing the full path

    # Filter paths based on colorspace
    filters = {
        
        # UVJ: Colourspace used for exploiting the bimodality of galaxies, 
        # can be used to seperate quiescent and star-forming galaxies
        'UVJ': {
            'U': 'Generic_Johnson.U.dat',
            'V': 'Generic_Johnson.V.dat',
            'J': '2MASS_2MASS.J.dat'
        },
        
        # ugr: Colourspace used to exploit the lyman break of galaxies (hydrogen absorption line)
        # used a a redshift diagnostic technique
        'ugr': {
            'u': 'Paranal_OmegaCAM.u_SDSS.dat',
            'g': 'Paranal_OmegaCAM.g_SDSS.dat',
            'r': 'Paranal_OmegaCAM.r_SDSS.dat'
        },
        
        # IRAC: Colourspace used to exploit the IR emission of galaxies 
        #
        'IRAC': { 
            'I1' : 'Spitzer_IRAC.I1.dat',
            'I2' : 'Spitzer_IRAC.I2.dat',
            'I3' : 'Spitzer_IRAC.I3.dat',
            'I4' : 'Spitzer_IRAC.I4.dat'
        },
        
        # Add more colorspaces and their filters as needed
        
        # There is the potential here to add the ability to add customer filter sets to the program that can be used to 
        # explore AGN contamination in more detail through a custom filter function. 
        
        
        
    }

    # Check if colorspace is supported
    if colorspace not in filters:
        raise ValueError(f"Unsupported colorspace: {colorspace}")

    # Load the passbands
    passbands = []
    for filter_name, filename in filters[colorspace].items():
        pb_path = os.path.join(path, filename)
        try:
            pb = astSED.Passband(pb_path, normalise=False)
            passbands.append(pb)
        except FileNotFoundError:
            raise FileNotFoundError(f"Filter file not found: {pb_path}")

    return passbands

######################################################################################################################



# TAU = [3, 5, 7, 9, 11] # Optical Depth - tau
# P = [0, 0.5,1, 1.5] # radial gradient of dust density
# Q = [0, 0.5,1, 1.5] # polar dust density gradient 
# OA = [10, 20, 30, 40, 50, 60, 70, 80] # opening angle between equiltorial plane and edge of torus
# RR = [10, 20, 30] # ratio of outer to inner radius
# I = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] # inclination, the viewing angle of instrument w.r.t AGN

# Dictionary to store the names of the models, using the values of the parameters
SKIRTOR_PARAMS = {'tau': [3, 5, 7, 9, 11], 'p': [0, 0.5,1, 1.5], 'q': [0, 0.5,1, 1.5], 'oa': [10, 20, 30, 40, 50, 60, 70, 80], 'rr': [10, 20, 30], 'i': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}


####################################################################################################


# For interfacing with the SKIRTOR AGN models
def read_skirtor_model(optical_depth, p, q, opening_angle, radius_ratio, inclination, folder_path='datasets/Templates/Skirtor'):
    """_summary_

    Args:
        folder_path (string): path to the folder where the models are located
        optical_depth (int): average edge-on optical depth at 9.7 micron; may take values 3, 5, 7, 9
        p (int): power-law exponent that sets radial gradient of dust density; may take values 0, 0.5, 1, 1.5
        q (int): index that sets dust density gradient with polar angle; may take values; 0, 0.5, 1, 1.5
        opening_angle (int): angle measured between the equatorial plan and edge of the torus; may take values 10 to 80, in steps of 10.
        radius_ratio (int): ratio of outer to inner radius, R_out/R_in; may take values 10, 20, 30
        inclination (_type_): viewing angle, i.e. position of the instrument w.r.t. the AGN axis. i=0: face-on, type 1 view; i=90: edge-on, type 2 view. may take on values 0 to 90 in steps of 10
    Returns:
        pd.DataFrame: Returns a dataframe containing the output of the chosen model
    """
    # Define the naming convention for reading in the Skirtor models
    filename = 't'+str(optical_depth)+'_p'+str(p)+'_q'+str(q)+'_oa'+str(opening_angle)+'_R'+str(radius_ratio)+'_Mcl0.97_i'+str(inclination)+'_sed.dat'
    # Join the file to the path and then read in the file
    filepath =os.path.join(folder_path, filename)
    # Read in the file and convert it to a pandas dataframe
    data = np.loadtxt(filepath, skiprows=5)
    
    # Convert it to a pandas dataframe # All fluxes are of the form lambda*F_lambda
    df = pd.DataFrame(data)
    
    # Convert the first column to angstroms
    df[0] = df[0]*10000
    
    
    # for the rest of the columns, we need to convert the fluxes to erg/s/cm^2/Angstrom
    df.iloc[:, 1:]
    
    # Convert W/m2 to erg/s/cm^2/Angstrom
    # first by converting W to erg/s
    df.iloc[:, 1:] = df.iloc[:, 1:]*10**7
        
    # then by converting  ergs/s/m^2 to ergs/s/cm^2
    #df.iloc[:, 1:] = df.iloc[:, 1:]*10**4
        
    # finally by converting ergs/s/cm^2 to ergs/s/cm^2/Angstrom: lambda*f_lambda -> f_lambda
    df.iloc[:, 1:] = df.iloc[:, 1:].div(df[0], axis=0)
    
    # Name each of the columns appropriately # We want to get rid of this and keep the information elsewhere
    #'Wavelength (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)', 'Direct AGN Flux (erg/s/cm^2/Angstrom)', 'Scattered AGN Flux (erg/s/cm^2/Angstrom)', 'Total Dust Emission Flux (erg/s/cm^2/Angstrom)', 'Dust Emission Scattered Flux(erg/s/cm^2/Angstrom)', 'Transparent Flux(erg/s/cm^2/Angstrom)']   
    df.columns = ['Wavelength', 'Total Flux', 'Direct AGN Flux', 'Scattered AGN Flux', 'Total Dust Emission Flux', 'Dust Emission Scattered Flux', 'Transparent Flux']
        
    return df






######################################################################################################################

# Load AGN Models code -this will only use SKIRTOR models for now - can be extended later
def load_agn_model(agn_type, path='datasets/Templates'):
    """
    Loads an AGN model with the appropriate parameters.

    Args:
        agn_type (str): The agn type we are wanting to loa
        path (str): The directory where the AGN model files are located.

    Returns:
        pd.DataFrame: A dataframe containing the output of the chosen model.
    """
    
    

    