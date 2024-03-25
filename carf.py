# Convienient Astrophysics Research Functions (CARF)

# This file is intended to be used and imported for use in the project. This
# file contains functions, variables and code that will make the project easy 
# to use in the future for this project

# Import packages <-  note the dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from astLib import astSED

####################################################################################################

# Define global important values

# TAU = [3, 5, 7, 9, 11] # Optical Depth - tau
# P = [0, 0.5,1, 1.5] # radial gradient of dust density
# Q = [0, 0.5,1, 1.5] # polar dust density gradient 
# OA = [10, 20, 30, 40, 50, 60, 70, 80] # opening angle between equiltorial plane and edge of torus
# RR = [10, 20, 30] # ratio of outer to inner radius
# I = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] # inclination, the viewing angle of instrument w.r.t AGN

# Dictionary to store the names of the models, using the values of the parameters
SKIRTOR_PARAMS = {'tau': [3, 5, 7, 9, 11], 'p': [0, 0.5,1, 1.5], 'q': [0, 0.5,1, 1.5], 'oa': [10, 20, 30, 40, 50, 60, 70, 80], 'rr': [10, 20, 30], 'i': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

####################################################################################################


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
    
    # Name each of the columns appropriately 
    df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)', 'Direct AGN Flux (erg/s/cm^2/Angstrom)', 'Scattered AGN Flux (erg/s/cm^2/Angstrom)', 'Total Dust Emission Flux (erg/s/cm^2/Angstrom)', 'Dust Emission Scattered Flux(erg/s/cm^2/Angstrom)', 'Transparent Flux(erg/s/cm^2/Angstrom)']
        
    return df


####################################################################################################

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

####################################################################################################

def read_all_skirtor_models(folder_path):
    """_summary_

    Args:
        folder_path (string): path to the folder where the models are located
    
    Returns:
        pd.DataFrame: Returns a tuple with the a list of dataframes containing all of thee models, and a list of the input parameters for each dataframe
    """
    #(df_list, parameter_list) = read_multiple_skirtor_models(folder_path, SKIRTOR_PARAMS['tau'], SKIRTOR_PARAMS['p'], SKIRTOR_PARAMS['q'], SKIRTOR_PARAMS['oa'], SKIRTOR_PARAMS['rr'], SKIRTOR_PARAMS['i'])
    
    df_list = []
    parameter_list = []
    files_in_folder = os.listdir(folder_path)
    print(files_in_folder)
    for file in files_in_folder:

        # Find filepath
        filepath = os.path.join(folder_path, file)
        data = np.loadtxt(filepath)
        
        # Find the parameters from the filename
        objname = file.split('_sed.dat')[0]
        t = str(objname.split('t')[1].split('_p')[0])
        p = str(objname.split('p')[1].split('_q')[0])
        q = str(objname.split('q')[1].split('_oa')[0])
        oa = str(objname.split('oa')[1].split('_R')[0])
        rr = str(objname.split('R')[1].split('_Mcl')[0])
        i = str(objname.split('i')[1])
        
        # put all of the parameters into a list
        objname = [t, p, q, oa, rr, i]
        
        
        #convert to dataframe 
        df = pd.DataFrame(data)
        
        # Convert wavelength into Angstroms
        df[0] = df[0]*10000
        
        # Convert W/m2 to erg/s/cm^2/Angstrom
        # first by converting W to erg/s
        df[1:] = df[1:]*10**7
        
        # then by converting  ergs/s/m^2 to ergs/s/cm^2
        df[1:] = df[1:]*10**4
        
        # finally by converting ergs/s/cm^2 to ergs/s/cm^2/Angstrom: lambda*f_lambda -> f_lambda
        df[1:] = df[1:]/df[0]
        
        # Name each of the columns appropriately 
        df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)', 'Direct AGN Flux (erg/s/cm^2/Angstrom)', 'Scattered AGN Flux (erg/s/cm^2/Angstrom)', 'Total Dust Emission Flux (erg/s/cm^2/Angstrom)', 'Dust Emission Scattered Flux(erg/s/cm^2/Angstrom)', 'Transparent Flux(erg/s/cm^2/Angstrom)']
            
        df_list.append(df)
        parameter_list.append(objname)
    
    return (df_list, parameter_list)

####################################################################################################

def plot_uvj(uv_colours, vj_colours, path=False, col="red"):
    """_summary_

    Args:
        uv_colours (array):  
        vj_colours (array):
    
    Returns:
        None: plots a UVJ diagram with the given UV and VJ colours
    """
    
    plt.figure(figsize=(10, 10))
    plt.scatter(vj_colours, uv_colours, c=col, s=10, label="Galaxy")
    plt.ylabel('U - V')
    plt.xlabel('V - J')
    plt.title("Restframe UVJ Colours")
    plt.xlim([-0.5,2.2])
    plt.axes.line_width = 4
    plt.ylim([0,2.5])
    
    
    if path:
        # Line plot connecting points
        #plt.plot(vj_colours, uv_colours, color='blue', linewidth=1, linestyle='-', alpha=0.5)
        for i in range(len(uv_colours) - 1):
            dx = vj_colours[i+1] - vj_colours[i]
            dy = uv_colours[i+1] - uv_colours[i]
            plt.arrow(vj_colours[i], uv_colours[i], dx, dy, head_width=0.05, head_length=0.05, fc='black', ec='red')


     # We can use code to make patch selections on the UVJ diagram, selecting Quiescent, Star-forming, and Dusty Galaxies
    # We use the paths as provided below to make the selections.
    path_quiescent = [[-0.5, 1.3],
                        [0.85, 1.3],
                        [1.6, 1.95],
                        [1.6, 2.5],
                        [-0.5, 2.5]]

    path_sf = [[-0.5, 0.0],
                [-0.5, 1.3],
                [0.85, 1.3],
                [1.2, 1.60333],
                [1.2, 0.0]]

    path_sfd = [[1.2, 0.0],
                    [1.2, 1.60333],
                    [1.6, 1.95],
                    [1.6, 2.5],
                    [2.2, 2.5],
                    [2.2, 0.0]]

    plt.gca().add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03),edgecolor='k', linewidth=2, linestyle='solid'))
    plt.gca().add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
    plt.gca().add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

    plt.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5) 


    plt.annotate('Quiescent', (-0.4, 2.4), color='black')
    plt.annotate('Star-forming', (-0.4, 1.2), color='black')
    plt.annotate('Dusty', (1.95, 2.4), color='black')
    plt.show()
    
####################################################################################################
    
def read_brown_galaxy_templates(folder_path):
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
    print(files_in_folder)
    for file in files_in_folder:

        # Find filepath and convert to df
        objname = file.split('_restframe.dat')[0]
        filepath = os.path.join(folder_path, file)
        data = np.loadtxt(filepath)
        df = pd.DataFrame(data)
            
        # our wavelength is in microns, convert to Angstroms
        df[0] = df[0] * 10000 # microns 10^-6 -> Angstroms 10^-10 
        
        # Name each of the columns appropriately
        df.columns = ['lambda (Angstroms)', 'Luminosity (W/Hz)' , 'Total Flux (erg/s/cm^2/Angstrom)', 'Source']
            
        # Append the dataframe to the list    
        df_list.append(df)
        objname_list.append(objname)
        
        
    return (df_list, objname_list)
    
####################################################################################################

def plot_galaxy_sed(wavelengths, fluxes, name, template_set):
        """_summary_
        A tool to quickly plot a galaxy templates, flexible enough to plot templates from different sources
        
        Args:
            df (DataFrame): Dataframe containing the SED information
            name (string): name of the galaxy 
        """
        plt.figure(figsize=(10, 5))
        plt.plot(wavelengths, fluxes, color='blue', linewidth=1, linestyle='-', alpha=0.5)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
        plt.title('Galaxy Template of: '+ name + " (" + template_set + ")")
        plt.grid()
        plt.xscale('log')
        #plt.xlim([10**5, 10**6])
        plt.show()
        
####################################################################################################

def read_swire_templates(folder_path):
    """_summary_

    Args:
        folder_path (string): folder path to the swire templates
    
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
        # Find filepath and convert to df
        objname = file.split('_template_norm.sed')[0]
        filepath = os.path.join(folder_path, file)
        data = np.loadtxt(filepath)
        df = pd.DataFrame(data)
        
        # Name each of the columns appropriately
        df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
            
        # Append the dataframe to the list    
        df_list.append(df)
        objname_list.append(objname)
    
    return df_list, objname_list

####################################################################################################

def normalize_sed(wavelengths, flux, reference_wavelength):
    """
    Normalize the flux of a spectral energy distribution (SED) at a specified reference wavelength.
    If the exact reference wavelength is not found, use the next closest wavelength.

    Parameters:
    - wavelengths (numpy array): Array of wavelengths (in microns).
    - flux (numpy array): Array of flux values corresponding to each wavelength.
    - reference_wavelength (float): Reference wavelength (in microns) to normalize the flux.

    Returns:
    - normalized_flux (numpy array): Normalized flux values of the SED.
    """
    # Find the index of the reference wavelength in the wavelengths array
    ref_index = np.argmin(np.abs(wavelengths - reference_wavelength))
    
    # Get the flux value at the reference wavelength or the next closest wavelength
    ref_flux = flux[ref_index]
    
    # Normalize the flux values by dividing by the reference flux
    normalized_flux = flux / ref_flux
    
    return normalized_flux

####################################################################################################