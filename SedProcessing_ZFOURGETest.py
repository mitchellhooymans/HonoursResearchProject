#!/usr/bin/env python
# coding: utf-8

# # ZFOURGE SED Template Processing 
# Using the generated ZFOURGE SED templates from the cdfs field, we explore how these SEDs will react in the UVJ colour space to see if the UVJ coordinates are still in the sample place.
# 

# In[1]:


import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pandas as pd
import os
from astLib import astSED
import astropy.io.fits as fits
from carf import * # custom module for functions relating to the project
import matplotlib.path as mpath


# So that we can change the helper functions without reloading the kernel
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Before we begin, lets recall what field we are exploring
field = 'COSMOS'


# In[3]:




# We are attempting to read in a new set of SEDs that have been generated using a variation of Michael's ZFOURGE SED processing code.
# the SEDs are in the form of csvs, this can be explored in the zfourge/seds
# directory. The csvs are in the form of:
# Wavelength (microns), Flux Density (10^-19erg_s_cm^2_Angstrom)

zfourge_folderpath = os.path.join('Eazy Template Extractions', 'zfourgeSEDs/'+field+'/')
# We would like to create a function to read this in
def read_zfourge_template(folder_path, name):
    """_summary_

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
        
        # get rid of units in the filename
        objname = file.split('SED_')[1]
        
        objname = objname.split('.csv')[0]
        
        if objname == name:
            print("Found object: ", objname)
            filepath = os.path.join(folder_path, file)
            df = pd.read_csv(filepath)
            
            # drop rows with NaN values
            df = df.dropna()
            
            if np.all(df.iloc[:, 1]) == 0.0:
                print("Not including object: ", objname)
                continue
            # our wavelength is in microns, convert to Angstroms
            
            # for the first column, we want to convert to Angstroms
            df.iloc[:, 0] = df.iloc[:, 0] 
            # Name each of the columns appropriately
            df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
            return df, objname
        
    
    return None, None

df, objname = read_zfourge_template(zfourge_folderpath, 'COSMOS_1') # Test function


# In[4]:


df


# In[5]:



def read_zfourge_galaxy_templates(folder_path):
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
i = 0

for file in files_in_folder:

    # Find filepath and convert to df
    objname = file.split('SED_')[1]
    objname = objname.split('.csv')[0]
    filepath = os.path.join(folder_path, file)
    df = pd.read_csv(filepath)
    
    
    
    
    # check for infinities or null values and replace 
    if np.any(np.isnan(df)):
        df = df.replace([np.inf, -np.inf], np.nan)
    if np.any(np.isinf(df) ):
        df = df.replace([np.inf, -np.inf], np.nan)
    
    
    # Whereever there are NaN values, interpolate
    df = df.interpolate()
    
    
    if np.all(df.iloc[:, 1] == 0.0):
        #print("Not including object: ", objname)
        continue
    
    # Convert microns to angstroms    
    df.iloc[:, 0] = df.iloc[:, 0]
    
    # name each of the columns appropriately
    df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
        
    # Append the dataframe to the list    
    df_list.append(df)
    objname_list.append(objname)
    
    
return df_list, objname_list


# In[6]:


# In addition also use astSED to create filters

# Read in all filters


# UVJ Filters
pb_U_path = os.path.join('datasets', 'Filters', 'Generic_Johnson.U.dat')
pb_V_path = os.path.join('datasets', 'Filters', 'Generic_Johnson.V.dat')
pb_J_path = os.path.join('datasets', 'Filters', '2MASS_2MASS.J.dat')
# Spitzer filters
pb_f3_6_path = os.path.join('datasets', 'filters', 'Spitzer_IRAC.I1.dat')
pb_f4_5_path = os.path.join('datasets', 'filters', 'Spitzer_IRAC.I2.dat')
pb_f5_8_path = os.path.join('datasets', 'filters', 'Spitzer_IRAC.I3.dat')
pb_f8_0_path = os.path.join('datasets', 'filters', 'Spitzer_IRAC.I4.dat')
# ugr filters
pb_u_path = os.path.join('datasets', 'filters', 'Paranal_OmegaCAM.u_SDSS.dat')
pb_g_path = os.path.join('datasets', 'filters', 'Paranal_OmegaCAM.g_SDSS.dat')
pb_r_path = os.path.join('datasets', 'filters', 'Paranal_OmegaCAM.r_SDSS.dat')


# Load all of the filters 
pb_U = astSED.Passband(pb_U_path, normalise=False)
pb_V = astSED.Passband(pb_V_path, normalise=False)
pb_J = astSED.Passband(pb_J_path, normalise=False)
pb_f3_6 = astSED.Passband(pb_f3_6_path, normalise=False)
pb_f4_5 = astSED.Passband(pb_f4_5_path, normalise=False)
pb_f5_8 = astSED.Passband(pb_f5_8_path, normalise=False)
pb_f8_0 = astSED.Passband(pb_f8_0_path, normalise=False)
pb_u = astSED.Passband(pb_u_path, normalise=False)
pb_g = astSED.Passband(pb_g_path, normalise=False)
pb_r = astSED.Passband(pb_r_path, normalise=False)

filter_set = {'U': pb_U, 'V':pb_V, 'J':pb_J, 'IRAC3.6': pb_f3_6,'IRAC4.5': pb_f4_5,'IRAC5.8': pb_f5_8, 'IRAC8.0':pb_f8_0, 'u': pb_u, 'g': pb_g, 'r':pb_r}

# Skirtor models
skirtor_folderpath = os.path.join('datasets', 'Templates', 'Skirtor')


# In[7]:


# In addition also use astSED to create filters
pb_U = astSED.Passband(pb_U_path, normalise=False)
pb_V = astSED.Passband(pb_V_path, normalise=False)
pb_J = astSED.Passband(pb_J_path, normalise=False)


# In[8]:


# For plotting
# convert the passbands to arrays
U_arr = np.array(pb_U.asList())
V_arr = np.array(pb_V.asList())
J_arr = np.array(pb_J.asList())

# Effectvie wavelengths
U_eff = pb_U.effectiveWavelength()
V_eff = pb_V.effectiveWavelength()
J_eff = pb_J.effectiveWavelength()

# print 
print("The effective wavelengths are: U: {}, V: {}, J: {}".format(U_eff, V_eff, J_eff))


# In[9]:


# Read in all the ZFOURGE templates
df_list, objname_list = read_zfourge_galaxy_templates(zfourge_folderpath)


# In[10]:



# print the number of items in df_list
print(len(df_list))


# In[11]:


print(objname_list[2])

# count and print na
print("Number of NaN values in the dataframe: ", df_list[0].isna().sum().sum())


# In[ ]:





# In[12]:



# we know where the NaN values are and we know what values come before and after
# so we can interpolate the values
df_list[0] = df_list[0]
print("Number of NaN values in the dataframe: ", df_list[0].isna().sum().sum())


# In[13]:


df_list[0][2000:2015]


# In[14]:


# Plot all of the SEDS

# for each sed, cut it so the wavelength is between 10^2 and 10^5
# then plot it
#for i in range(len(df_list)):
   # df_list[i] = df_list[i][(df_list[i]['lambda (Angstroms)'] > 10**2) & (df_list[i]['lambda (Angstroms)'] < 10**5)]

# Plot the first 10
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.loglog(df_list[i].iloc[:, 0], df_list[i].iloc[:, 1], label=objname_list[i], marker='o', markersize=1)
#plt.legend()
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
plt.title('ZFOURGE SED Templates')
sc = 10**-3
# additionally plot the relatively locations of the filters
plt.loglog(U_arr[:, 0], sc*U_arr[:,1], label='U filter', c='purple')
plt.loglog(V_arr[:, 0], sc*V_arr[:,1], label='V filter', c='olive')
plt.loglog(J_arr[:, 0], sc*J_arr[:,1], label='J filter', c='red')

plt.ylim(10**-4, 10**3)

plt.legend()

plt.show()



# We can see a very very clear trend of the seds now being alligned in their rest frame. The lyman break is clearly visible and the absorption and emission lines are also very clear. This is a very good sign that the SEDs are being generated correctly.

# In[15]:


# We now have some SED's that can be put through my fitting code to see how the UVJ diagram looks,
# and to see if it behaves as expected

# We will need to convert the SEDs to a format that can be used by the fitting code
# We can attempt to use the composite code, without the need for the SED fitting code
uv_colours = []
vj_colours = []

# for better coverage, create an AGN SED, combine them at 0% and have a look
agn_df, params = create_type1_skirtor_agn(skirtor_folderpath)


# alpha list
alpha_list = np.linspace(0, 1, 11)


# create a composite set similar to the GALSEDATLAS set
# composite_df = create_composite_sed(agn_df, df_list[0], 0) # This is just for one, with no agn juice added
# print(composite_df)
# alpha_list = [0]

# Try now with a composite stack of seds
df_list = generate_composite_set(agn_df, df_list, alpha_list)


# In[16]:


len(df_list)

# Setup is df_list has a length of 11

# each of these 11 elements has a list inside of it
# this list has the number of galaxies we read in
# each of these galaxies has a dataframe with the SED (wavelength, flux) ~ 2700 rows or something


# In[17]:



# # Plot this
# plt.figure(figsize=(10, 10))
# #plt.loglog(composite_df.iloc[:, 0], composite_df.iloc[:, 1], label='Composite', marker='o', markersize=1)
# # Plot agn normalized seds from the df_list
# for i in range(0, 10):
#     plt.loglog(df_list[i].iloc[:, 0], df_list[i].iloc[:, 1], label=objname_list[i], marker='o', markersize=1)


# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
# plt.title('Composite SED')
# plt.legend()
# plt.show()


# Plot one set of composites from alpha 0 to 100 to see if this worked
# Plot this
plt.figure(figsize=(10, 10))
# Plot agn normalized seds from the df_list
v = 8
for i in range(0, 11):
    plt.loglog(df_list[i][v].iloc[:, 0], df_list[i][v].iloc[:, 1], marker='o', markersize=1, label='AGN%:  {}'.format(int(alpha_list[i]*100)))
# data label

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
# Show lims
plt.ylim(10**-1, 10**3)
plt.xlim(10**3, 10**5)

plt.title('Composite SED')
plt.legend()
plt.show()


# In[18]:


# uv_colours = []
# vj_colours = []
# bad_entries = []
# new_objname_list = []
# print(len(df_list))
# for i, sed_data in enumerate(df_list):
#     # Create an SED object using astSED    
#     wl = sed_data['lambda (Angstroms)']
#     fl = sed_data['Total Flux (erg/s/cm^2/Angstrom)']

#     # We are using a restframe SED, so z = 0 - now atleast
#     sed = astSED.SED(wavelength=wl, flux=fl, z=0.0)    
    
print(len(objname_list))  


# In[19]:




#     # # Try to calc uvj, if math domain error occurs, if bad SED, delete it
#     # uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
#     # vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
#     # uv_colours.append(uv)
#     # vj_colours.append(vj)
    
#     # We assume there will be some bad SEDs
#     # Try to create UVJ colours, and if it fails, remove the SED
#     # we expect a math domain error
    
#     try:
#         uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
#         vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
        
#         # Try calculate the u mag and the v mag and the j mag seperately as fluxes
#         # flux
#         # u_mag = astSED.SED.calcMag(sed, pb_U, magType='AB')
#         # v_mag = astSED.SED.calcMag(sed, pb_V, magType='AB')
#         # j_mag = astSED.SED.calcMag(sed, pb_J, magType='AB')
        
#         # uv = u_mag - v_mag
#         # vj = v_mag - j_mag
        
#         uv_colours.append(uv)
#         vj_colours.append(vj)
#         new_objname_list.append(objname_list[i])
#     except:
#         print('Bad SED')
#         print(objname_list[i])
#         bad_entries.append(i)
#         del df_list[i]
#         del objname_list[i]
        
#         continue
    
# We need to use some previous code to calculate the UVJ colours, we need to do this for the composite SEDs
# Because we would like to create a dataframe eventually, with each ID, the associated UV and VJ colours with UV_alpha and VJ_alpha
# at each point, where the alpha represents the contribution. This should then be output as a csv file, which can be read in
# and combined with the zfourge catalogue to get real information, potentially.


# Use previous code to accomplish what we are after
# Create some lists to store the full set of alpha colours
uv_specific_alpha_colours = []
vj_specific_alpha_colours = []
new_objname_alpha_list = []
uv_colours =[]
vj_colours = []
new_objname_list = []
alpha_bad = []
bad_id = []
good_id = []
alpha_good = []

for i in range(len(alpha_list)):
    # This will be the set of composites for the specific alpha value
    sed_alpha_data = df_list[i] # should go through each
    
    for j, sed_data in enumerate(sed_alpha_data): # will go through each of the zfourge galaxies
        # Create an SED object using astSED
        wl = sed_data['lambda (Angstroms)']
        fl = sed_data['Total Flux (erg/s/cm^2/Angstrom)']
        sed = astSED.SED(wavelength=wl, flux=fl, z=0.0)  
        
        # The id of this particular entry is the objname_list[j]
        objname = objname_list[j]
        
        
        # We assume there will be some bad SEDs
        try:
            uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
            vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
            
            # Try calculate the u mag and the v mag and the j mag seperately as fluxes
            # flux
            # u_mag = astSED.SED.calcMag(sed, pb_U, magType='AB')
            # v_mag = astSED.SED.calcMag(sed, pb_V, magType='AB')
            # j_mag = astSED.SED.calcMag(sed, pb_J, magType='AB')
            
            # uv = u_mag - v_mag
            # vj = v_mag - j_mag
            # Append the uv, vj and name to their relevant lists
            uv_colours.append(uv)
            vj_colours.append(vj)
            new_objname_list.append(objname)
        except:
            # Add the bad id to the bad id list
            bad_id.append(j)
            
            #print('Bad SED:', objname)
            #del df_list[i][j] # This should delete the jth entry from the ith list
            #del objname_list[j] # this should delete the associated name
            continue
        
    # Append the uv, and vj colours     
    uv_specific_alpha_colours.append(uv_colours)
    vj_specific_alpha_colours.append(vj_colours)
    alpha_bad.append(bad_id) # append the bad id list - so then we can find the bad idea for each alpha
    new_objname_alpha_list.append(new_objname_list)
    # Reset the colours for the next set of alpha values
    uv_colours = []
    vj_colours = []
    bad_id = []
    new_objname_list = []
    
# We should have a good list of colours after this


# In[25]:


# Check the lengths of the lists
print(len(uv_specific_alpha_colours[0]))
print(len(vj_specific_alpha_colours[0]))

# Add all of these to a dataframe
# essentially each alpha value will have a dataframe with the uv and vj colours
# and the associated name

# We do want a master dataframe
# we want the ids, and the uv and vj colours for each alpha value

# ids will be in the new_objname_alpha_list
# specifically should be the same for each element

# There shouldn't be any bad ideas
ids = new_objname_alpha_list[0]


# add the field prefix to the IDs
ids = [str(x) for x in ids]


# In[26]:



# Create a dataframe with all the id's 
df = pd.DataFrame(data={'ID': ids})


# In[ ]:





# In[96]:


# Take first 7313 elements as the good ids: from objname_list

#df.reset_index(drop=True, inplace=True)


# In[27]:


# Now we can attempt to see
df


# In[28]:


# Now we want to add the uv and vj colours for each alpha value, where the col will have uv_alpha and vj_alpha

# We will need to loop through the uv and vj colours and add them to the dataframe
for i in range(len(alpha_list)):
    df['UV_{}'.format(int(alpha_list[i]*100))] = uv_specific_alpha_colours[i]
    df['VJ_{}'.format(int(alpha_list[i]*100))] = vj_specific_alpha_colours[i]


# In[29]:


df


# In[30]:


# Export the above file as a csv
df.to_csv('composite_uv_vj_colours_'+field+'.csv', index=False)
# Change the names so the alpha values are from 0 to 100
alpha_list


# In[31]:


# Now we can plot these in UVJ space, including their relevant labels, and UV/VJ positions
n = 0
plt.figure(figsize=(10, 10))

plt.scatter(vj_specific_alpha_colours[n],uv_specific_alpha_colours[n], c='r', s=10)

annotations = []

# Annotate points with text and adjust for collisions
#texts = []
#for i, txt in enumerate(new_objname_list):
#    txt = f"{txt}" #({vj_colours[i]:.2f}, {uv_colours[i]:.2f})"
#    texts.append(plt.text(vj_colours[i], uv_colours[i], txt, ha='center'))

# Adjust text to avoid collisions
#adjust_text(texts)

plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours of AGN Composites, ZFOURGE" + field)
plt.xlim([-0.5, 2.2])
plt.ylim([0, 2.5])

# Define paths for selections
path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

# Add patches for selections
plt.gca().add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
plt.gca().add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
plt.gca().add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

# Add vertical line
plt.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

# output the plot, and save it
#plt.savefig('outputs/UVJPlots/RecalculatedUVJ_Positions.png')

plt.show()


# In[ ]:


# # Now we can plot these in UVJ space, including their relevant labels, and UV/VJ positions

# plt.figure(figsize=(10, 10))

# plt.scatter(vj_colours, uv_colours, c='r', s=10)

# # Adjust text to avoid collisions
# adjust_text(texts)

# plt.ylabel('U - V')
# plt.xlabel('V - J')
# plt.title("Restframe UVJ Colours of AGN Composites")
# plt.xlim([-0.5, 2.2])
# plt.ylim([0, 2.5])

# # Define paths for selections
# path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
# path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
# path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

# # Add patches for selections
# plt.gca().add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
# plt.gca().add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
# plt.gca().add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

# # Add vertical line
# plt.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)


# plt.show()


# In[ ]:





# In[34]:



# # Plot the UVJ diagram
# plt.figure(figsize=(10, 10))

# x = df['VJ']
# y = df['UV']
# id = df['ID']

# xmax = 2.5
# ymax = 2.5
# xmin = -0.5
# ymin = 0

# # Set the plotting limits
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)

# # Define the points for a quiescent galaxy selection
# x_points = [-0.5, 0.85, 1.6, 1.6]
# y_points = [1.3, 1.3, 1.95, 2.5]

# # Plot the points
# plt.plot(x_points, y_points, linestyle='-')

# # Interpolate the y-value at x=1.2
# x_target = 1.2
# y_target = np.interp(x_target, x_points, y_points)

# # Plot the interpolated point, this separates everything on the right as dusty galaxies,
# # and everything on the left as star-forming galaxies
# plt.plot([x_target, x_target], [0, y_target], linestyle='--')

# quiescent_x = [-0.5, 0.85, 1.6, 1.6, xmin, xmin]
# quiescent_y = [1.3, 1.3, 1.95, 2.5, ymax, 1.3]
# # We want to make a wedge selection for the Quiescent Selection of Galaxies
# points = np.column_stack([x, y])
# verts = np.array([quiescent_x, quiescent_y]).T
# path = mpath.Path(verts)


# # Define the path for point selection
# #selected_path = mpath.Path([(2, 3), (6, 4), (8, 2), (2, 1), (2, 3)])  # Example path, replace with your own

# # Use path.contains_points to get a boolean array
# points_inside_selection = path.contains_points(np.column_stack([x, y]))



# dusty_condition = (points[:, 0] > x_target) & (~points_inside_selection)
# star_forming_condition = (points[:, 0] < x_target) & (~points_inside_selection)



# # Filter the DataFrame using the boolean array
# selected_df = df[points_inside_selection] # For quiescent, clean later

# # Mark dusty, and star-forming galaxies
# df.loc[dusty_condition, 'GalaxyType'] = 2
# df.loc[star_forming_condition, 'GalaxyType'] = 1




# print(selected_df)
# #print(unselected_df)

# # Now we can easily select the quiescent galaxies and set the galaxy type to quiescent - 0 or starforming 1.
# print(selected_df)
# selected_ids = selected_df['ID']
# df.loc[df['ID'].isin(selected_ids), 'GalaxyType'] = 0 # This is what makes the selection happen



# # Try do the same for the unselected galaxies, noting that the unselected galaxies with x and y > 1.2 are dusty galaxies, and the rest are star-forming galaxies
# #unselected_ids = unselected_df['id']




# # Sort the quiescent and non-quiescent galaxies
# quiescent_points = points[path.contains_points(points)]
# # Find the points from here to categorise dusty, and star-forming galaxies
# non_quiescent_points = points[~path.contains_points(points)]
# dusty_galaxies_points = non_quiescent_points[non_quiescent_points[:, 0] > x_target]
# star_forming_galaxies_points = non_quiescent_points[non_quiescent_points[:, 0] < x_target]


# #print(non_quiescent_points[0][0])

# #print(y)
# # Plot the selected points
# #plt.scatter(x, y, s=3, alpha=0.5, label='Not Quiescent Selection')
# plt.scatter(quiescent_points[:, 0], quiescent_points[:, 1], c='r', s=10, alpha=0.5, label='Quiescent Selection')
# plt.scatter(dusty_galaxies_points[:, 0], dusty_galaxies_points[:, 1], c='g', s=10, alpha=0.5, label='Dusty Galaxies')
# plt.scatter(star_forming_galaxies_points[:, 0], star_forming_galaxies_points[:, 1], c='b', s=10, alpha=0.5, label='Star Forming Galaxies')

# # Plot the names for the selected galaxies, in each reigon
# # text = []
# # for i in range(len(df)):
# #    text.append(plt.text(df.iloc[i]['VJ'], df.iloc[i]['UV'], df.iloc[i]['ID']))
# # adjust_text(text)



# plt.gca().add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
# plt.gca().add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
# plt.gca().add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))


# plt.xlabel('Restframe V-J [Mag]')
# plt.ylabel('Restframe U-V [Mag]')
# plt.title('UVJ Diagram for the CDFS field')
# plt.legend()
# plt.show()


# In[33]:


df


# In[36]:


# Finally export the dataframe to a csv, containing the recalculated UVJ positions and galaxy type
def save_csv_with_numbered_name(df, save_path, file_name):
    # Check if the file already exists
    if os.path.isfile(os.path.join(save_path, f"{file_name}.csv")):
        # Append a number until a unique file name is found
        i = 1
        while os.path.isfile(os.path.join(save_path, f"{file_name}_{i}.csv")):
            i += 1
        file_name = f"{file_name}_{i}"
    
    # Save the DataFrame to CSV with the updated file name
    df.to_csv(os.path.join(save_path, f"{file_name}.csv"), index=False)
    print(f"CSV file saved as {file_name}.csv")


# In[37]:


save_csv_with_numbered_name(df, 'Eazy Template Extractions', field+'_RecalculatedUVJids_full')


# In[ ]:




