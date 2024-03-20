# Convienient Astrophysics Research Functions (CARF)

# This file is intended to be used and imported for use in the project. This
# file contains functions, variables and code that will make the project easy 
# to use in the future for this project

# Import packages <-  note the dependencies
import matplotlib.pyplot as plt
#import astropy.units as u
import numpy as np
import pandas as pd
import os
#from astLib import astSED
#import astropy.io.fits as fits


SKIRTOR_PARAMS = {
    'optical_depth_list' : [3, 5, 7, 9, 11],
    'p_list' : [0, 0.5,1, 1.5],
    'q_list' : [0, 0.5,1, 1.5],
    'opening_angle_list' : [10, 20, 30, 40, 50, 60, 70, 80],
    'radius_ratio_list' : [10, 20, 30],
    'inclination_list' : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
}



def read_skirtor_model(folder_path, optical_depth, p, q, opening_angle, radius_ratio, inclination):
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
    df = pd.DataFrame(data, columns=['lambda (micron)', 'Total Flux (W/m2)', 'Direct AGN Flux (W/m2)', 'Scattered AGN Flux (W/m2)', 'Total Dust Emission Flux (W/m2)', 'Dust Emission Scattered Flux(W/m2)', 'Transparent Flux(W/m2)'])
    return df


def read_multiple_skirtor_models(folder_path, optical_depth_list, p_list, q_list, opening_angle_list, radius_ratio_list, inclination_list):
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
        pd.DataFrame: Returns a tuple with the a list of dataframes containing each of the models, and a list of the input parameters for each dataframe
    """
    # Create an empty list to store the dataframes
    df_list = []
    parameter_list = []
    # Loop through all the models and store them in the list
    for optical_depth in optical_depth_list:
        for p in p_list:
            for q in q_list:
                for opening_angle in opening_angle_list:
                    for radius_ratio in radius_ratio_list:
                        for inclination in inclination_list:
                            df_list.append(read_skirtor_model(folder_path, optical_depth, p, q, opening_angle, radius_ratio, inclination))
                            # also put the parametrs into another list and return that
                            parameter_list.append([optical_depth, p, q, opening_angle, radius_ratio, inclination])
    return (df_list, parameter_list)


