# Convienient Astrophysics Research Functions (CARF)

# This file is intended to be used and imported for use in the project. This
# file contains functions, variables and code that will make the project easy 
# to use in the future for this project

# Import packages <-  note the dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

####################################################################################################

# Define global important values

TAU = [3, 5, 7, 9, 11] # Optical Depth - tau
P = [0, 0.5,1, 1.5] # radial gradient of dust density
Q = [0, 0.5,1, 1.5] # polar dust density gradient 
OA = [10, 20, 30, 40, 50, 60, 70, 80] # opening angle between equiltorial plane and edge of torus
RR = [10, 20, 30] # ratio of outer to inner radius
I = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] # inclination, the viewing angle of instrument w.r.t AGN


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
    
    df = pd.DataFrame(data, columns=['lambda (micron)', 'Total Flux (W/m2)', 'Direct AGN Flux (W/m2)', 'Scattered AGN Flux (W/m2)', 'Total Dust Emission Flux (W/m2)', 'Dust Emission Scattered Flux(W/m2)', 'Transparent Flux(W/m2)'])
    # Be sure to convert the wavelength column to Angstroms
    data[0] = data[0]*10000
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
    (df_list, parameter_list) = read_multiple_skirtor_models(folder_path, TAU, P, Q, OA, RR, I)
    
    return (df_list, parameter_list)

####################################################################################################

def plot_uvj(uv_colours, vj_colours):
    """_summary_

    Args:
        uv_colours (array):  
        vj_colours (array):
    
    Returns:
        None: plots a UVJ diagram with the given UV and VJ colours
    """
    
    plt.figure(figsize=(10, 10))
    plt.scatter(vj_colours, uv_colours, c="red", s=10, label="Galaxy")
    plt.ylabel('U - V')
    plt.xlabel('V - J')
    plt.title("Restframe UVJ Colours for Brown's Templates")
    plt.xlim([-0.5,2.2])
    plt.axes.line_width = 4
    plt.ylim([0,2.5])


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

        # Find filepath
        objname = file.split('_restframe.dat')[0]
        filepath = os.path.join(folder_path, file)
        data = np.loadtxt(filepath)
        #convert to dataframe 
        df = pd.DataFrame(data)
            
        # our wavelength is in microns, convert to Angstroms
        df[0] = df[0] * 10000 # microns 10^-6 -> Angstroms 10^-10 
        

            
        df_list.append(df)
        objname_list.append(objname)
    return (df_list, objname_list)
    
####################################################################################################

def plot_brown_galaxy_sed(df, name):
        """_summary_
        A tool to quickly plot a galaxy from Brown's templates

        Args:
            df (DataFrame): Dataframe containing the SED information
            name (string): name of the galaxy 
        """
        plt.figure(figsize=(10, 5))
        plt.plot(df[0], df[2])
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
        plt.title('Galaxy Template of: '+ name)
        plt.grid()
        plt.xscale('log')
        #plt.xlim([6500, 6750])
        plt.show()