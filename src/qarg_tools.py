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

# Function to load and preprocess data

def load_preprocess_data(galaxy_templates, agn_templates, colourspace):
    """Function to load and preprocess data for the analysis pipeline."""
    
    # Load the galaxy templates
    galaxy_templates = load_galaxy_templates(galaxy_template_set)
    
    # Load the AGN templates
    agn_templates = load_agn_model(agn_model_name)
    
    # Load the astronomical colours
    astronomical_colours = load_filters(colourspace)
    
    # Return the data
    return galaxy_templates, agn_templates, astronomical_colours

def load_galaxy_templates(galaxy_templates):
    
    
    # Load the galaxy templates
    galaxy_templates = galaxy_templates
    
    return galaxy_templates

def load_agn_model(agn_templates):

   # Load the AGN templates
    agn_templates = agn_templates
    
    return agn_templates


def load_filters(colourspace):
    filter_1, filter_2, filter_3 = 0, 0, 0
    # Switch statement to load the filters based on the colourspace
    if colourspace == 'UVJ':
        # Load the UVJ filters
        
        astronomical_colours = [a, b, c]
    elif colourspace == 'NUVrK':
        # Load the NUVrK filters
        astronomical_colours = 'NUVrK'
    elif colourspace == 'BzK':
        # Load the BzK filters
        astronomical_colours = 'BzK'
    else:
        # Load the default UVJ filters

    
    return astronomical_colours