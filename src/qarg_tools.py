# This is the codebase for QARG tools, tools to be used by the QUT Astrophysics Research Group
# This file is intended to be used and imported for use in the project. This
# file contains functions, variables and code that will make the project easy 
# to use in the future for this project

# Import packages <-  note the dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
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