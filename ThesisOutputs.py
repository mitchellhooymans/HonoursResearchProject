#!/usr/bin/env python
# coding: utf-8

# In[27]:


# Quick script to generate the outputs for the literature review
# specifically I need a couple of figures

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from astLib import astSED
import astropy.io.fits as fits


# In[28]:




# Params -
skirtor_folderpath = os.path.join('datasets', 'Templates', 'Skirtor')

swire_folderpath = os.path.join('datasets', 'Templates', 'SWIRE')


# Filters - IRAC
pb_f3_6_path = os.path.join('datasets', 'filters', 'Spitzer_IRAC.I1.dat')
pb_f4_5_path = os.path.join('datasets', 'filters', 'Spitzer_IRAC.I2.dat')
pb_f5_8_path = os.path.join('datasets', 'filters', 'Spitzer_IRAC.I3.dat')
pb_f8_0_path = os.path.join('datasets', 'filters', 'Spitzer_IRAC.I4.dat')
# In addition also use astSED to create filters
pb_f3_6 = astSED.Passband(pb_f3_6_path, normalise=False)
pb_f4_5 = astSED.Passband(pb_f4_5_path, normalise=False)
pb_f5_8 = astSED.Passband(pb_f5_8_path, normalise=False)
pb_f8_0 = astSED.Passband(pb_f8_0_path, normalise=False)

# Filters -ugr
pb_u_path = os.path.join('datasets', 'filters', 'Paranal_OmegaCAM.u_SDSS.dat')
pb_g_path = os.path.join('datasets', 'filters', 'Paranal_OmegaCAM.g_SDSS.dat')
pb_r_path = os.path.join('datasets', 'filters', 'Paranal_OmegaCAM.r_SDSS.dat')
# In addition also use astSED to create filters
pb_u = astSED.Passband(pb_u_path, normalise=False)
pb_g = astSED.Passband(pb_g_path, normalise=False)
pb_r = astSED.Passband(pb_r_path, normalise=False)





# Filters - UVJ
pb_U_path = os.path.join('datasets', 'Filters', 'Generic_Johnson.U.dat')
pb_V_path = os.path.join('datasets', 'Filters', 'Generic_Johnson.V.dat')
pb_J_path = os.path.join('datasets', 'Filters', '2MASS_2MASS.J.dat')

# In addition also use astSED to create filters
pb_U = astSED.Passband(pb_U_path, normalise=False)
pb_V = astSED.Passband(pb_V_path, normalise=False)
pb_J = astSED.Passband(pb_J_path, normalise=False)



f3_6_arr = np.array(pb_f3_6.asList())
f4_5_arr = np.array(pb_f4_5.asList())
f5_8_arr = np.array(pb_f5_8.asList())
f8_0_arr = np.array(pb_f8_0.asList())


# In[29]:



# Create a function to compute the flux integral
def integral_flux(sed):
    return np.trapz(sed['Total Flux (erg/s/cm^2/Angstrom)'], sed['lambda (Angstroms)'])

agn_composite_models = []
for models in range(1, 3):
    if(models == 1):
        # type 1 AGN
        optical_depth = 3 # fixed
        p = 0 # fixed
        q = 0 # fixed
        opening_angle = 50 # fixed
        radius_ratio = 20 # fixed 
        inclination = 0 # you can adjust this between a value of 0 and 90 (in steps of 10 as per the files available)
    elif(models == 2):
        # type 2 AGN
        optical_depth = 3 # fixed
        p = 0
        q = 0
        opening_angle = 50
        radius_ratio = 20
        inclination = 90
        
    # if neither of the above, exit program
    else:
        break
    


    # read in the Skirtor model of the AGN
    filename = 't'+str(optical_depth)+'_p'+str(p)+'_q'+str(q)+'_oa'+str(opening_angle)+'_R'+str(radius_ratio)+'_Mcl0.97_i'+str(inclination)+'_sed.dat'
    # Join the file to the path and then read in the file
    filepath =os.path.join(skirtor_folderpath, filename)
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


    agn_df = df




    df_list = []
    objname_list = []
    swire_folderpath = os.path.join(swire_folderpath)
    files_in_folder = os.listdir(swire_folderpath)


    # make sure to only read .sed files
    file_extension = '.sed'

    # Filter files based on the specified file extension
    files_in_folder = [file for file in files_in_folder if file.endswith(file_extension)]

    for file in files_in_folder:
        # Find filepath and convert to df
        objname = file.split('_template_norm.sed')[0]
        filepath = os.path.join(swire_folderpath, file)
        data = np.loadtxt(filepath)
        df = pd.DataFrame(data)
        
        # Name each of the columns appropriately
        df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
            
        # Append the dataframe to the list    
        df_list.append(df)
        objname_list.append(objname)


    df = agn_df.copy() # set the df to the AGN model
    print(objname_list)
    # n chooses the galaxy we are interested in
    n = 16
    galaxy_df = df_list[n]

    # Given an SED
    wavelengths_sed1 = galaxy_df['lambda (Angstroms)']
    flux_sed1 = galaxy_df['Total Flux (erg/s/cm^2/Angstrom)']

    # Given a model
    wavelengths_sed2 = df['lambda (Angstroms)']
    flux_sed2 = df['Total Flux (erg/s/cm^2/Angstrom)']

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

    # Create a new dataframe for each SED
    galaxy_df = pd.DataFrame({'lambda (Angstroms)': combined_wavelengths, 'Total Flux (erg/s/cm^2/Angstrom)': combined_flux_sed1}) 
    df = pd.DataFrame({'lambda (Angstroms)': combined_wavelengths, 'Total Flux (erg/s/cm^2/Angstrom)': combined_flux_sed2}) 





    # Calculating the integrated flux for each SED
    integrated_model_flux = integral_flux(df)
    integrated_galaxy_flux = integral_flux(galaxy_df)
    scaling_factor = integrated_galaxy_flux/integrated_model_flux

    # Using this scaling factor, we can now scale the model to the galaxy model
    df['Total Flux (erg/s/cm^2/Angstrom)'] = df['Total Flux (erg/s/cm^2/Angstrom)'] * scaling_factor



    # plt.figure(figsize=(10, 6))
    # plt.loglog(galaxy_df['lambda (Angstroms)'], galaxy_df['Total Flux (erg/s/cm^2/Angstrom)'], label=objname_list[n])
    # plt.loglog(df['lambda (Angstroms)'], df['Total Flux (erg/s/cm^2/Angstrom)'], label='AGN Type 1')
    # plt.xlabel('Wavelength (Angstroms)')
    # plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
    # plt.title('SED of AGN Model')
    # plt.legend()
    # plt.show()

    alpha = np.arange(0, 1.2, 0.2)
    alpha

    # creating a set of composite SEDs
    composite_seds = []
    for a in alpha:
        combined_flux = a * df['Total Flux (erg/s/cm^2/Angstrom)'] + galaxy_df['Total Flux (erg/s/cm^2/Angstrom)']
        
        # use the wavelength of the galaxy SED or AGN sed
        combined_wavelengths = df['lambda (Angstroms)']

        # Create a composite SED DataFrame
        composite_sed_df = pd.DataFrame({'lambda (Angstroms)': combined_wavelengths, 'Total Flux (erg/s/cm^2/Angstrom)': combined_flux})
        
        # add to composite sed list
        composite_seds.append(composite_sed_df)


    # # We can now plot these SEDs and see what an increase of model contribution does to the overal SED
    # plt.figure(figsize=(6, 6))
    # for i, composite_sed in enumerate(composite_seds):
    #     plt.loglog(composite_sed['lambda (Angstroms)'], composite_sed['Total Flux (erg/s/cm^2/Angstrom)'], label=f'Model Contribution: {alpha[i]*100:.0f}%')
    # plt.xlabel('Wavelength (Angstroms)')
    # plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
    # plt.title('Composite SEDs')
    # plt.legend()
    # plt.xlim(1e3, 1e7)
    # plt.ylim(1e-4, 1e2)
    # plt.show()
    
    # Append the composite SEDs to the list
    agn_composite_models.append(composite_seds)


# In[30]:


# Create a plot of the composite SEDs with the filters
plt.figure(figsize=(7, 5))
# for i, composite_sed in enumerate(agn_composite_models[0]):
#     plt.loglog(composite_sed['lambda (Angstroms)'], composite_sed['Total Flux (erg/s/cm^2/Angstrom)'], 'k')

# Plot a single composite SED at AGN contribution 0
plt.loglog(agn_composite_models[0][0]['lambda (Angstroms)'], agn_composite_models[0][0]['Total Flux (erg/s/cm^2/Angstrom)'], 'k', label='Galaxy Template')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (erg/s/cm^2/Angstrom)')

plt.xlim(1e3, 1e7)
plt.ylim(1e-4, 1e2)

sc = 10**-3
# Plot IRAC filters
plt.loglog(pb_f3_6.wavelength, sc*pb_f3_6.transmission, label='IRAC 3.6 Filter')
plt.loglog(pb_f4_5.wavelength, sc*pb_f4_5.transmission, label='IRAC 4.5 Filter')
plt.loglog(pb_f5_8.wavelength, 2*sc*pb_f5_8.transmission, label='IRAC 5.8 Filter')
plt.loglog(pb_f8_0.wavelength, sc*pb_f8_0.transmission, label='IRAC 8.0 Filter')

# plot ugr filters
plt.loglog(pb_u.wavelength, sc*pb_u.transmission, label='u Filter')
plt.loglog(pb_g.wavelength, sc*pb_g.transmission, label='g Filter')  
plt.loglog(pb_r.wavelength, sc*pb_r.transmission, label='r Filter')


# Plot UVJ filters
plt.loglog(pb_U.wavelength, sc*pb_U.transmission, label='U Filter')
plt.loglog(pb_V.wavelength, sc*pb_V.transmission, label='V Filter')
plt.loglog(pb_J.wavelength, sc*pb_J.transmission, label='J Filter')

# Place legend outside on the right

plt.legend()
plt.xlim(1e3, 1e6)
plt.savefig('outputs/Galaxy SED with all filters.png', dpi=300)
plt.tight_layout()
plt.show()


# In[ ]:





# In[31]:



# Now we have both models we can plot them side by side 
# Create subplot of the two models
fig, ax = plt.subplots(2, 1, figsize=(7, 6))
for i, composite_sed in enumerate(agn_composite_models[0]):
   ax[0].loglog(composite_sed['lambda (Angstroms)'], composite_sed['Total Flux (erg/s/cm^2/Angstrom)'], label=f'AGN Percentage: {alpha[i]*100:.0f}%')
ax[0].set_xlabel('Wavelength (Angstroms)')
ax[0].set_ylabel('Flux (erg/s/cm^2/Angstrom)')
ax[0].set_title('Spiral Galaxy with Type 1 AGN')
ax[0].legend()
ax[0].set_xlim(1e3, 1e7)
ax[0].set_ylim(1e-4, 1e2)

for i, composite_sed in enumerate(agn_composite_models[1]):
   ax[1].loglog(composite_sed['lambda (Angstroms)'], composite_sed['Total Flux (erg/s/cm^2/Angstrom)'], label=f'AGN Percentage: {alpha[i]*100:.0f}%')
ax[1].set_xlabel('Wavelength (Angstroms)')
ax[1].set_ylabel('Flux (erg/s/cm^2/Angstrom)')
ax[1].set_title('Spiral Galaxy with Type 2 AGN')
# set legend to the side
ax[1].legend()
ax[1].set_xlim(1e3, 1e7)
ax[1].set_ylim(1e-4, 1e2)

# layout
plt.tight_layout()

# Save the output 
plt.savefig('outputs/sed_agn_contaminiation.png', dpi=300, bbox_inches='tight')

plt.show()



# We would also like to produce a blue and a red galaxy - associated with a elliptical and a spiral galaxy
# plot the swire templates


# In[32]:


# UVJ Plotting Code
# Now that we have all the data in one dataframe we can start to look at the data
# Importantly we would like to plot how different the data is 

# Plotting a master UVJ diagram
def categorise_uvj(df, x, y, alpha=None):
    #x = df['mag_V'] - df['mag_J'] # VJ Colours 
    #y = df['mag_U'] - df['mag_V'] # UV Colours

    # Plot the UVJ diagram
    plt.figure(figsize=(7, 5))

    xmax = 2.2
    ymax = 2.5
    xmin = -0.5
    ymin = 0

    # Set the plotting limits
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Define the points for a quiescent galaxy selection
    x_points = [-0.5, 0.85, 1.6, 1.6]
    y_points = [1.3, 1.3, 1.95, 2.5]

    # Plot the points
    plt.plot(x_points, y_points, linestyle='-')

    # Interpolate the y-value at x=1.2
    x_target = 1.2
    y_target = np.interp(x_target, x_points, y_points)

    # Plot the interpolated point, this separates everything on the right as dusty galaxies,
    # and everything on the left as star-forming galaxies
    plt.plot([x_target, x_target], [0, y_target], linestyle='--')

    quiescent_x = [-0.5, 0.85, 1.6, 1.6, xmin, xmin]
    quiescent_y = [1.3, 1.3, 1.95, 2.5, ymax, 1.3]
    # We want to make a wedge selection for the Quiescent Selection of Galaxies
    points = np.column_stack([x, y])
    verts = np.array([quiescent_x, quiescent_y]).T
    path = mpath.Path(verts)
    
    
    # Define the path for point selection
    #selected_path = mpath.Path([(2, 3), (6, 4), (8, 2), (2, 1), (2, 3)])  # Example path, replace with your own

    # Use path.contains_points to get a boolean array
    points_inside_selection = path.contains_points(np.column_stack([x, y]))
    
    
    
    dusty_condition = (points[:, 0] > x_target) & (~points_inside_selection)
    star_forming_condition = (points[:, 0] < x_target) & (~points_inside_selection)
    
    

    # Filter the DataFrame using the boolean array
    selected_df = df[points_inside_selection] # For quiescent, clean later
    

    # Mark dusty, and star-forming galaxies
    if alpha == None:
        df.loc[dusty_condition, f'GalaxyType'] = 2
        df.loc[star_forming_condition, f'GalaxyType'] = 1
        selected_ids = selected_df['id']
        df.loc[df['id'].isin(selected_ids), f'GalaxyType'] = 0
    else:     
        df.loc[dusty_condition, f'GalaxyType_{int(alpha*100)}'] = 2
        df.loc[star_forming_condition, f'GalaxyType_{int(alpha*100)}'] = 1
        selected_ids = selected_df['id']
        df.loc[df['id'].isin(selected_ids), f'GalaxyType_{int(alpha*100)}'] = 0

    #print(selected_df)
    #print(unselected_df)
    
    # Now we can easily select the quiescent galaxies and set the galaxy type to quiescent - 0 or starforming 1.
    #print(selected_df)
     # This is what makes the selection happen
    
    
    
    # Try do the same for the unselected galaxies, noting that the unselected galaxies with x and y > 1.2 are dusty galaxies, and the rest are star-forming galaxies
    #unselected_ids = unselected_df['id']
    
    
    
    
    # Sort the quiescent and non-quiescent galaxies
    quiescent_points = points[path.contains_points(points)]
    # Find the points from here to categorise dusty, and star-forming galaxies
    non_quiescent_points = points[~path.contains_points(points)]
    dusty_galaxies_points = non_quiescent_points[non_quiescent_points[:, 0] > x_target]
    star_forming_galaxies_points = non_quiescent_points[non_quiescent_points[:, 0] < x_target]
    
    
    #print(non_quiescent_points[0][0])
    
    #print(y)
    # Plot the selected points
    #plt.scatter(x, y, s=3, alpha=0.5, label='Not Quiescent Selection')
    plt.scatter(quiescent_points[:, 0], quiescent_points[:, 1], c='r', s=3, alpha=0.5, label='Quiescent Selection')
    plt.scatter(dusty_galaxies_points[:, 0], dusty_galaxies_points[:, 1], c='g', s=3, alpha=0.5, label='Dusty Galaxies')
    plt.scatter(star_forming_galaxies_points[:, 0], star_forming_galaxies_points[:, 1], c='b', s=3, alpha=0.5, label='Star Forming Galaxies')

    plt.xlabel('Restframe V-J [Mag]')
    plt.ylabel('Restframe U-V [Mag]')
    #plt.title('UVJ Diagram using ZFOURGE Galaxies')
    plt.legend()

    plt.savefig('outputs/uvj_diagram_example_recalculated.png')
    plt.show()
    return df


# In[39]:


# Ensure same colours were used for recalculationn

# Params -
skirtor_folderpath = os.path.join('datasets', 'Templates', 'Skirtor')

swire_folderpath = os.path.join('datasets', 'Templates', 'SWIRE')


# Create a function to compute the flux integral
def integral_flux(sed):
    return np.trapz(sed['Total Flux (erg/s/cm^2/Angstrom)'], sed['lambda (Angstroms)'])

agn_composite_models = []
for models in range(1, 3):
    if(models == 1):
        # type 1 AGN
        optical_depth = 3 # fixed
        p = 0 # fixed
        q = 0 # fixed
        opening_angle = 50 # fixed
        radius_ratio = 20 # fixed 
        inclination = 0 # you can adjust this between a value of 0 and 90 (in steps of 10 as per the files available)
    elif(models == 2):
        # type 2 AGN
        optical_depth = 3 # fixed
        p = 0
        q = 0
        opening_angle = 50
        radius_ratio = 20
        inclination = 90
        
    # if neither of the above, exit program
    else:
        break
    


    # read in the Skirtor model of the AGN
    filename = 't'+str(optical_depth)+'_p'+str(p)+'_q'+str(q)+'_oa'+str(opening_angle)+'_R'+str(radius_ratio)+'_Mcl0.97_i'+str(inclination)+'_sed.dat'
    # Join the file to the path and then read in the file
    filepath =os.path.join(skirtor_folderpath, filename)
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


    agn_df = df




    df_list = []
    objname_list = []
    swire_folderpath = os.path.join(swire_folderpath)
    files_in_folder = os.listdir(swire_folderpath)


    # make sure to only read .sed files
    file_extension = '.sed'

    # Filter files based on the specified file extension
    files_in_folder = [file for file in files_in_folder if file.endswith(file_extension)]

    for file in files_in_folder:
        # Find filepath and convert to df
        objname = file.split('_template_norm.sed')[0]
        filepath = os.path.join(swire_folderpath, file)
        data = np.loadtxt(filepath)
        df = pd.DataFrame(data)
        
        # Name each of the columns appropriately
        df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
            
        # Append the dataframe to the list    
        df_list.append(df)
        objname_list.append(objname)


    df = agn_df.copy() # set the df to the AGN model
    print(objname_list)
    # n chooses the galaxy we are interested in
    n = 16
    galaxy_df = df_list[n]

    # Given an SED
    wavelengths_sed1 = galaxy_df['lambda (Angstroms)']
    flux_sed1 = galaxy_df['Total Flux (erg/s/cm^2/Angstrom)']

    # Given a model
    wavelengths_sed2 = df['lambda (Angstroms)']
    flux_sed2 = df['Total Flux (erg/s/cm^2/Angstrom)']

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

    # Create a new dataframe for each SED
    galaxy_df = pd.DataFrame({'lambda (Angstroms)': combined_wavelengths, 'Total Flux (erg/s/cm^2/Angstrom)': combined_flux_sed1}) 
    df = pd.DataFrame({'lambda (Angstroms)': combined_wavelengths, 'Total Flux (erg/s/cm^2/Angstrom)': combined_flux_sed2}) 





    # Calculating the integrated flux for each SED
    integrated_model_flux = integral_flux(df)
    integrated_galaxy_flux = integral_flux(galaxy_df)
    scaling_factor = integrated_galaxy_flux/integrated_model_flux

    # Using this scaling factor, we can now scale the model to the galaxy model
    df['Total Flux (erg/s/cm^2/Angstrom)'] = df['Total Flux (erg/s/cm^2/Angstrom)'] * scaling_factor



    # plt.figure(figsize=(10, 6))
    # plt.loglog(galaxy_df['lambda (Angstroms)'], galaxy_df['Total Flux (erg/s/cm^2/Angstrom)'], label=objname_list[n])
    # plt.loglog(df['lambda (Angstroms)'], df['Total Flux (erg/s/cm^2/Angstrom)'], label='AGN Type 1')
    # plt.xlabel('Wavelength (Angstroms)')
    # plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
    # plt.title('SED of AGN Model')
    # plt.legend()
    # plt.show()

    alpha = np.arange(0, 1.2, 0.2)
    alpha

    # creating a set of composite SEDs
    composite_seds = []
    for a in alpha:
        combined_flux = a * df['Total Flux (erg/s/cm^2/Angstrom)'] + galaxy_df['Total Flux (erg/s/cm^2/Angstrom)']
        
        # use the wavelength of the galaxy SED or AGN sed
        combined_wavelengths = df['lambda (Angstroms)']

        # Create a composite SED DataFrame
        composite_sed_df = pd.DataFrame({'lambda (Angstroms)': combined_wavelengths, 'Total Flux (erg/s/cm^2/Angstrom)': combined_flux})
        
        # add to composite sed list
        composite_seds.append(composite_sed_df)


    # # We can now plot these SEDs and see what an increase of model contribution does to the overal SED
    # plt.figure(figsize=(6, 6))
    # for i, composite_sed in enumerate(composite_seds):
    #     plt.loglog(composite_sed['lambda (Angstroms)'], composite_sed['Total Flux (erg/s/cm^2/Angstrom)'], label=f'Model Contribution: {alpha[i]*100:.0f}%')
    # plt.xlabel('Wavelength (Angstroms)')
    # plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
    # plt.title('Composite SEDs')
    # plt.legend()
    # plt.xlim(1e3, 1e7)
    # plt.ylim(1e-4, 1e2)
    # plt.show()
    
    # Append the composite SEDs to the list
    agn_composite_models.append(composite_seds)
    
    
# Now we have both models we can plot them side by side 
# Create subplot of the two models
# Create subplot of the two models
fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharey=True, sharex=True)  # Share x-axis for better comparison

# Plotting for Type 1 AGN
for i, composite_sed in enumerate(agn_composite_models[0]):
    ax[0].loglog(composite_sed['lambda (Angstroms)'], 
                 composite_sed['Total Flux (erg/s/cm^2/Angstrom)'], 
                 label=f'AGN: {alpha[i]*100:.0f}%')

ax[0].set_ylabel('Flux (erg/s/cm^2/Angstrom)')
ax[0].legend()
ax[0].set_ylim(1e-4, 1e2)  # Set y-limits only once
ax[0].set_xlim(1e3, 1e7)  # Set x-limits only once
# Plotting for Type 2 AGN
for i, composite_sed in enumerate(agn_composite_models[1]):
    ax[1].loglog(composite_sed['lambda (Angstroms)'], 
                 composite_sed['Total Flux (erg/s/cm^2/Angstrom)'], 
                 label=f'AGN: {alpha[i]*100:.0f}%')

ax[1].set_xlabel('Wavelength (Angstroms)')
ax[1].set_ylabel('Flux (erg/s/cm^2/Angstrom)')

ax[1].set_ylim(1e-4, 1e2)  # Set y-limits only once


# # Get current y-ticks
# yticks = ax[0].get_yticks()  # Assuming both subplots have the same y-ticks

# # Remove the first tick (or adjust as needed)
# new_yticks = np.delete(yticks, 1) 
# print(new_yticks)

# # Set the new y-ticks for both subplots
# ax[0].set_yticks(new_yticks)

# Layout adjustments to make subplots touch
plt.subplots_adjust(hspace=0) 

# Layout
plt.tight_layout(pad=0.5)

# 

# Save the output 
plt.savefig('outputs/sed_agn_contaminiation_.png', dpi=300, bbox_inches='tight')

plt.show()


# In[ ]:





# For the next part of this thesis we need to ensure we are outputting the correct graphs for our analysis. This is incredibly important.
