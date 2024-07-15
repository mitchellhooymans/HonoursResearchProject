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



###

# Importatant variables

# possible parameters for the SKIRTOR models, any permutations of these parameters can be used to generate a model
skirtor_params = {'tau': [3, 5, 7, 9, 11], 'p': [0, 0.5,1, 1.5], 'q': [0, 0.5,1, 1.5], 'oa': [10, 20, 30, 40, 50, 60, 70, 80], 'rr': [10, 20, 30], 'i': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}






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
        
    
    # use a Try and Except block to catch any file not found errors
    try:
        # Join the file to the path and then read in the file
        filepath =os.path.join(folder_path, filename)
        data = np.loadtxt(filepath, skiprows=5)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    
    
    
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
def load_agn_model(agn_type, path='datasets/Templates/Skirtor'):
    """
    Loads an AGN model with the appropriate parameters.

    Args:
        agn_type (str): The agn type we are wanting to loa
        path (str): The directory where the AGN model files are located.

    Returns:
        pd.DataFrame: A dataframe containing the output of the chosen model.
    """
    
    # TAU = [3, 5, 7, 9, 11] # Optical Depth - tau
    # P = [0, 0.5,1, 1.5] # radial gradient of dust density
    # Q = [0, 0.5,1, 1.5] # polar dust density gradient 
    # OA = [10, 20, 30, 40, 50, 60, 70, 80] # opening angle between equiltorial plane and edge of torus
    # RR = [10, 20, 30] # ratio of outer to inner radius
    # I = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] # inclination, the viewing angle of instrument w.r.t AGN
    agn_type = agn_type.lower()
    agn_types = ['type 1', 'type 2', 'intermediate']
    # Check if agn type is supported
    if agn_type not in agn_types:
        raise ValueError(f"Unsupported AGN type: {agn_type}")    
    else:

        # Given the agn type, we can load the appropriate model
        if agn_type == 'type 1':
            # Load the type 1 model, using the paramaters 
            tau = skirtor_params['tau'][2] # optical depth of 7
            p = skirtor_params['p'][1] # Can make this either 0.5 or 0 (we choose 0.5)
            q = skirtor_params['q'][0] # Make this zero, similar to the Ciesla paper
            oa = skirtor_params['oa'][4] # opening angle of 50 degrees
            rr = skirtor_params['rr'][1] # ratio of 20
            i = skirtor_params['i'][0] # inclination of 0 degrees
            
        elif agn_type == 'type 2':
            tau = skirtor_params['tau'][2] # optical depth of 7
            p = skirtor_params['p'][1] # Can make this either 0.5 or 0 (we choose 0.5)
            q = skirtor_params['q'][0] # Make this zero, similar to the Ciesla paper
            oa = skirtor_params['oa'][4] # opening angle of 50 degrees
            rr = skirtor_params['rr'][1] # ratio of 20
            i = skirtor_params['i'][9] # inclination of 0 degrees
            
        elif agn_type == 'intermediate':
            tau = skirtor_params['tau'][2] # optical depth of 7
            p = skirtor_params['p'][1]  # Can make this either 0.5 or 0 (we choose 0.5)
            q = skirtor_params['q'][0]  # Make this zero, similar to the Ciesla paper
            oa = skirtor_params['oa'][4] # opening angle of 50 degrees
            rr = skirtor_params['rr'][1] # ratio of 20
            i = skirtor_params['i'][4] # inclination of 0 degrees
    
        # print the paramsa
        print(f"Loading AGN model: {agn_type} with parameters: tau={tau}, p={p}, q={q}, oa={oa}, rr={rr}, i={i}")
    
        # Load the model
        model = read_skirtor_model(tau, p, q, oa, rr, i, path)
        
    return model


######################################################################################################################

# Create an astSED object of the AGN model
def create_sed(model):
    """
    Create an astSED object from the model.

    Args:
        model (pd.DataFrame): The model dataframe.

    Returns:
        astSED.SED: The astSED object representing the model.
    """
    # Create the astSED object
    sed = astSED.SED(model[['Wavelength']], model['Total Flux'])
    return sed
    
######################################################################################################################

def read_brown_galaxy_template(folder_path, name):
    """
        Reads in the galaxy template from the Brown 2014 dataset that has been restframed.

    Args:
        folder_path (string): path to the folder where the SED templates are located
        name (string): name of the object
    
    Returns:
        df: Returns a dataframe containing the SED template
        objname: Returns the name of the object
    """
    folder_path = os.path.join(folder_path)
    files_in_folder = os.listdir(folder_path)
    
    for file in files_in_folder:
        # Find filepath and convert to df
        objname = file.split('_restframe.dat')[0]
        if objname == name:
            filepath = os.path.join(folder_path, file)
            data = np.loadtxt(filepath)
            df = pd.DataFrame(data)
            
            # our wavelength is in microns, convert to Angstroms
            df[0] = df[0] * 10000 # microns 10^-6 -> Angstroms 10^-10 
        
            # Name each of the columns appropriately
            df.columns = ['lambda (Angstroms)', 'Luminosity (W/Hz)' , 'Total Flux (erg/s/cm^2/Angstrom)', 'Source']
            
            return df, objname
        
    
    return None, None


######################################################################################################################

def read_brown_galaxy_templates(folder_path='datasets/Templates/Brown/2014/Rest'):
    """_summary_

        Args:
            folder_path (string): path to the folder where the SED templates are located
    
        Returns:
            df_list: Returns a list of dataframes containing the SED templates
            objname_list: Returns a list of the names of the objects
    """
    df_list = []
    objname_list = []
    folder_path = os.path.join(folder_path)
    files_in_folder = os.listdir(folder_path)
    for file in files_in_folder:
        
        # Do a a try and except block to catch any file not found errors
        try:
            # Find filepath and convert to df
            objname = file.split('_restframe.dat')[0]
            filepath = os.path.join(folder_path, file)
            data = np.loadtxt(filepath)
            df = pd.DataFrame(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        
            
        # our wavelength is in microns, convert to Angstroms
        df[0] = df[0] * 10000 # microns 10^-6 -> Angstroms 10^-10 
        
        # Name each of the columns appropriately
        #df.columns = ['lambda (Angstroms)', 'Luminosity (W/Hz)' , 'Total Flux (erg/s/cm^2/Angstrom)', 'Source']
        df.columns = ['Wavelength', 'Luminosity' , 'Total Flux', 'Source']
            
        # Append the dataframe to the list    
        df_list.append(df)
        objname_list.append(objname)
        
        
    return {'dataframe': df_list, 'name': objname_list}


######################################################################################################################

def read_swire_galaxy_templates(folder_path):
    """_summary_

    Args:
        folder_path (string): folder path to the swire templates
        name (string): name of template
    
    Returns:
            df_list: Returns a list of dataframes containing the SED templates
            objname_list: Returns a list of the names of the objects
    """
    df_list = []
    objname_list = []
    folder_path = os.path.join(folder_path)
    files_in_folder = os.listdir(folder_path)
    
    # make sure to only read .sed files
    file_extension = '.sed'
    
    # Filter files based on the specified file extension
    files_in_folder = [file for file in files_in_folder if file.endswith(file_extension)]
    
    for file in files_in_folder:
        
        
        # Do a a try and except block to catch any file not found errors
        try:
            # Find filepath and convert to df
            objname = file.split('_template_norm.sed')[0]
            filepath = os.path.join(folder_path, file)
            data = np.loadtxt(filepath)
            df = pd.DataFrame(data)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Name each of the columns appropriately
        # df.columns = ['Wavelength (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
        df.columns = ['Wavelength', 'Total Flux']
            
        # Append the dataframe to the list    
        df_list.append(df)
        objname_list.append(objname)
    
    return {'dataframe': df_list, 'name': objname_list}


######################################################################################################################

# read SED templates: Should read in whichever set of templates are in the directory and specified
def load_galaxy_template_set(template_set, path):
    """
    Loads the entire galaxy template set.

    Args:
        template_set (str): The template set we are wanting to load

    Returns:
        pd.DataFrame: A collection of templates from the chosen set 
    """
    
    # format string correctly
    template_set = template_set.upper()
    
    # Check if template set is supported
    template_sets = ['SWIRE', 'BROWN']
    
    if template_set not in template_sets:
        raise ValueError(f"Unsupported template set: {template_set}")
    else:
        if template_set == 'SWIRE':
            # Read in swire templates
            templates = read_swire_galaxy_templates(path)
        elif template_set == 'BROWN':
            # Read in brown templates
            templates = read_brown_galaxy_templates(path)
           
    return templates


######################################################################################################################

def calc_flux_integral(sed):
    """
    Calculate the integral of the flux with respect to the wavelength.

    Args:
        sed (SED object): Takes an SED with a flux and wavelength and computes a resulting integral.

    Returns:
        _type_: Returns the result of an integral over the total flux with respect to the wavelength
    """
    return np.trapz(sed.flux, sed.wavelength)


######################################################################################################################

def compute_scaling_factor(agn_sed, galaxy_sed):
    """
    Calculates the scaling factor required to match the flux of the AGN and galaxy SEDs.

    Args:
        agn (SED object): the first SED of the AGN
        galaxy (SED object): the second SED of the galaxy

    Returns:
        float: scaling factor to match the flux of the AGN and galaxy SEDs
    """
    # Integrate the flux to get the total flux
    integrated_agn_flux = calc_flux_integral(agn_sed)
    integrated_galaxy_flux = calc_flux_integral(galaxy_sed)

    scaling_factor = integrated_galaxy_flux/integrated_agn_flux
    
    return scaling_factor

######################################################################################################################
  
def adjust_wavelength_range(sed_1, sed_2):
    """
    Given a particular SED with a particular wavelength range, and another SED with a different but similar wavelength
    range, make a combined wavelength range for both of these values, cutting off both SEDS at the minimum SED value of the two
    and return two SEDs that are comparable.

    Args:
    sed_1 (dataframe): The first sed dataframe
    sed_2 (dataframe): the second sed dataframe
    """
    # Given an SED
    wavelengths_sed1 = sed_1.wavelength
    flux_sed1 = sed_1.flux

    #print(sed_2['lambda (Angstroms)'])
    wavelengths_sed2 = sed_2.wavelength
    flux_sed2 = sed_2.flux

    # Get a shared wavelength range across both SEDS
    combined_wavelengths = np.union1d(wavelengths_sed1, wavelengths_sed2)

    # Interpolate flux values for the combined wavelengths
    combined_flux_sed1 = np.interp(combined_wavelengths, wavelengths_sed1, flux_sed1, left=np.nan, right=np.nan)
    combined_flux_sed2 = np.interp(combined_wavelengths, wavelengths_sed2, flux_sed2, left=np.nan, right=np.nan) 

    # We would like to see which sed has the min wavelength , and max wavelength,
    # Cut the AGN and Galaxy model so they are within range of the original swire model
    min_wavelength = np.max([np.min(wavelengths_sed1), np.min(wavelengths_sed2)])
    max_wavelength = np.min([np.max(wavelengths_sed1), np.max(wavelengths_sed2)])

    # Cut the AGN model
    mask = (combined_wavelengths >= min_wavelength) & (combined_wavelengths <= max_wavelength)
    combined_wavelengths = combined_wavelengths[mask]
    combined_flux_sed1 = combined_flux_sed1[mask]
    combined_flux_sed2 = combined_flux_sed2[mask]

    # Create new sedobjects
    new_sed1 = astSED.SED(combined_wavelengths, combined_flux_sed1)
    new_sed2 = astSED.SED(combined_wavelengths, combined_flux_sed2)
    

    return new_sed1, new_sed2
  
######################################################################################################################  
     
# Function to create a single SED composite from 1 galaxy SED and 1 AGN model
def create_composite_sed(agn_sed, gal_sed, alpha, beta=1):
    """ Creates a composite galaxy/agn sed with a given weight of the AGN SED and Galaxy SED
    mixing can be done if beta = 1 - alpha.
    Args:
        agn_df (astSED.SED object): agn dataframe
        gal_sed (astSED.SED): galaxy dataframe
        alpha (int): value from 0 to 1 the weight of the AGN SED in the composite SED
        beta (int): value from 0 to 1, the weight of the galaxy SED in the composite SED (default to 1 to allow for addition)

    Returns:
        astSED.SED Object: composite SED object
    """
    
    # Ensure that the SED is of the same wavelength range, interpolating as required
    agn_sed, gal_sed = adjust_wavelength_range(agn_sed, gal_sed)
    
    
    # Remove them
    #gal_sed = gal_sed.dropna()
    
    # Normalize the flux of the AGN and Galaxy SEDs
    scaling_factor = compute_scaling_factor(agn_sed, gal_sed)
    
    # Scaled flux densities
    agn_sed.flux = agn_sed.flux * scaling_factor
    
    # Sum the flux values at each wavelength
    combined_flux = alpha * agn_sed.flux + beta * gal_sed.flux
    
    # use the wavelength of the galaxy SED or AGN sed
    combined_wavelengths = gal_sed.wavelength
    
    # create a new astSED object
    composite_sed = astSED.SED(combined_wavelengths, combined_flux)
    
    return composite_sed

######################################################################################################################

# Creating a set of SEDS for a given AGN model and a set of galaxy templates
# function to create a set of composites
def generate_composite_set(agn_sed, gal_sed_list, alpha_list):
    """
    Generate a set of composite SEDs for a given AGN model and a set of galaxy templates.

    Args:
        agn_sed (astSED.SED object): SED object that contains the SED of the AGN model
        gal_sed_list (list): list of SED objects that contains the SED of the galaxy templates
        alpha_list (list): list of alpha values, that each composite will be made with

    Returns:
        _type_: a list of lists of astSED.SED objects that contains the composite SEDs
    """
    # Create a list of composite SEDs
    composite_alpha_list = []
    # Loop through the list of alpha values
    for alpha in alpha_list:
        composite_sed_list = []
        for gal_sed in gal_sed_list:
            agn_model = agn_sed # ensures that each composite is made with the inital AGN model and it isn't adjusted.
            composite_sed = create_composite_sed(agn_model, gal_sed, alpha)
            composite_sed_list.append(composite_sed) # list of composites at a particular alpha value
        composite_alpha_list.append(composite_sed_list)
    return composite_alpha_list # contains a list containing a the list of composites for each alpha value

    