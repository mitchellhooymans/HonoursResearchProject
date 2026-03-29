#!/usr/bin/env python
# coding: utf-8

# # Recreating the SED from the Decomposed Components
# This is intended to be a slightly more rigiourous approach to the problem of recreating the SEDs from the decomposed components. We expect to see different types of SEDs, in particular we wil see some AGN type 1, and some AGN type 2. It is essentially that we explore these difference and try to recreate our SEDs from the decomposed components, but plugging the SED into my code and seeing what comes out.

# In[209]:


# Import all relevant libraries
# Begin by importing all relevant libraries and packages
import matplotlib.pyplot as plt
import seaborn as sns
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


# In[210]:


# Filters
pb_U_path = os.path.join('datasets', 'Filters', 'Generic_Johnson.U.dat')
pb_V_path = os.path.join('datasets', 'Filters', 'Generic_Johnson.V.dat')
pb_J_path = os.path.join('datasets', 'Filters', '2MASS_2MASS.J.dat')


pb_U = astSED.Passband(pb_U_path, normalise=False)
pb_V = astSED.Passband(pb_V_path, normalise=False)
pb_J = astSED.Passband(pb_J_path, normalise=False)


# In[211]:


# We will be employing the use of the Skirtor models to add AGN back into the decomposed SEDs
tau = SKIRTOR_PARAMS['tau'][3]
p = SKIRTOR_PARAMS['p'][0]
q = SKIRTOR_PARAMS['q'][0]
oa = SKIRTOR_PARAMS['oa'][4]
rr = SKIRTOR_PARAMS['rr'][2]
i = SKIRTOR_PARAMS['i'][0]

# Skirtor folder
skirtor_folderpath = os.path.join('datasets', 'Templates', 'Skirtor')

# This is our type-1 AGN model
agn_model = read_skirtor_model(skirtor_folderpath, tau, p, q, oa, rr, i)


# In[212]:


# Get the df 
queiscent_transitions = pd.read_csv('outputs\quiescent_transition_ids.csv')



def get_n_seds(df, n, restframe=False, all=False):
    # Select n galaxies
    
    df_list = []
    names = []
    redshifts = []
    if all==False:
        selected_galaxies = df.sample(n)
    else: 
        selected_galaxies = df
        
    # Reset the index
    selected_galaxies = selected_galaxies.reset_index(drop=True)
    
    # name 
    gal_name = selected_galaxies['id'].astype(str)
    
    # field
    gal_field = selected_galaxies['field'].astype(str)
    
    
    names = gal_field + '_' + gal_name
    gal_redshift = selected_galaxies['zpk'].astype(float)

    # Now we will read in the fits files for these galaxies

    for i in range(len(selected_galaxies)):
        path = 'datasets\student_fits_files\\'+ str(gal_field[i]).lower() +'_best_models_fits\\'
        name = str(gal_name[i])+'_best_model.fits'

        galaxy_path = os.path.join(path, name)
        with fits.open(galaxy_path) as data:
            df = pd.DataFrame(np.array(data[1].data).byteswap().newbyteorder())
        
        # Convert to angstroms
        df['wavelength'] = df['wavelength']*10

        if restframe:
            df['Snu'] = df['Fnu']*10**-3 # milliJanksys to Janksys <- J = ergs/(s*(cm^2)*(s^-1))
            # F_nu currently has a frequency dependence, convert to nuFnu by multiplying the the frequency associated
            # with the wavelength, as we are in angstroms, we can use the formula c = f*lambda
            
            
            # This should prevent any issues, but check
            freq = (3*10**18)/df['wavelength'] # in Hz
            # multiply the Snu * nu to get nuSnu
            df['nuSnu'] = df['Snu']*freq
            # Restframe the values of wavelength
            df['wavelength'] = df['wavelength'] / (1 + gal_redshift[i]) # we redshift the values of of wavelength
            # now calculate a new frequency, based on the new wavelength
            freq = (3*10**18)/df['wavelength'] # in Hz
            # divide the nuSnu by the new frequency to get the restframed values
            df['Snu'] = df['nuSnu']/freq
            
            # Convert flux values
            df['Flambda'] = df['Snu']*(3*10**-5)/(df['wavelength']**2) # S_nu to F_lambda <- angstroms 
            
        else:
            # Convert flux values
            df['Snu'] = df['Fnu']*10**-3 # milliJanksys to Janksys <- J = ergs/(s*(cm^2)*(s^-1))
            df['Flambda'] = df['Snu']*(3*10**-5)/(df['wavelength']**2) # S_nu to F_lambda <- angstroms 
            
        
            
            
        redshift_Val = gal_redshift[i]
        redshifts.append(redshift_Val)        
        
        

        
        # For simplicity, just create some extra columns
        df['lambda (Angstroms)'] = df['wavelength']
        df['Total Flux (erg/s/cm^2/Angstrom)'] = df['Flambda']
        
        
        
        df_list.append(df)
        
        
        plt.loglog(df['wavelength'], df['Flambda'])
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux (Fnu)')
    #plt.xlim(1e3, 1e5)
    plt.ylim(1e-30, 1e-2)
    plt.title('SED of galaxies')
    plt.legend()
    plt.show()
    
    print(len(df_list))
    
    return df_list, names, redshifts


# In[213]:


zfourge_decomposed_seds, names, redshifts = get_n_seds(queiscent_transitions, 5, restframe=True, all=True)


# In[214]:


# We only want to investigate the decomposed SED IDs that have been output: Decomposed_UVJ_Ids
# We will use the IDs to select the galaxies we want to investigate
decomposed_UVJ_ids = pd.read_csv('Decomposed_UVJ_Ids.csv')


# In[215]:


decomposed_UVJ_ids['id'] 


# In[216]:


names


# In[217]:


# Using the id's from the decomposed UVJ ids, we can select the galaxies we want to investigate at a particular index


# use the decomposed_UVJ_ids['id'] to selected the appropriate names, `names` and `zfourge_decomposed_seds`
# find 
#decomposed_UVJ_ids['id']
#names
#zfourge_decomposed_seds

# Whereever there is a match between the ids, return the index number in the names df

#decomposed_UVJ_ids['id'].isin(names)
names.isin(decomposed_UVJ_ids['id'])

# count the trues
names.isin(decomposed_UVJ_ids['id']).sum()


# find the index of each of the trues
galaxy_index = names[names.isin(decomposed_UVJ_ids['id'])].index


# In[218]:


# Add names, redshifts, and decomposed seds into a multidimensional array

SED_df = [[], [], []]
print(len(galaxy_index))
for i in galaxy_index:
    print(i)
    SED_df[0].append(names[i]) # ID of galaxy
    SED_df[1].append(redshifts[i]) # Redshift of galaxy
    SED_df[2].append(zfourge_decomposed_seds[i]) # Full SED of galaxy
    print('\n')


# In[219]:


# Now we have our selection of SEDs from our inital sample that contain AGN features
# we will reproccess these these SEDS to ensure 


def decompose_agn_seds(best_fit_seds):
    # Create a list to store the decomposed SEDs
    decomposed_seds = []
    
    # Iterate through the best_fit_seds
    for i in range(len(best_fit_seds)):
        new_df = best_fit_seds[i].copy()
        
        # Drop the zero values - maybe include this potentially
       # new_df = df[df['L_lambda_total'] > 10**-7].copy()

        # AGN components - Remove the galaxy component
        # Create an adjust L_lambda to plot against
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total']

        # Remove the young and old stellar components
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['stellar.old']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['stellar.young']


        # Remove the absopriton and emission lines

        # Absorption
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['nebular.absorption_old']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['nebular.absorption_young']

        # Emission
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['nebular.lines_old']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['nebular.lines_young']


        # Removing the dust component
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['dust']


        # Removing the galactic continuum
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['nebular.continuum_old']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['nebular.continuum_young']

        # Removing the attentuated components
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['attenuation.stellar.old']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['attenuation.stellar.young']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['attenuation.nebular.lines_old']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['attenuation.nebular.lines_young']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['attenuation.nebular.continuum_old']
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['attenuation.nebular.continuum_young']


        # Remove the igm
        new_df['L_lambda_total_decomposed_AGN'] = new_df['L_lambda_total_decomposed_AGN'] - new_df['igm']
        
        integral_total = np.trapz(new_df['L_lambda_total'], new_df['wavelength'])
        integral_decomposed = np.trapz(new_df['L_lambda_total_decomposed_AGN'], new_df['wavelength'])

        # Scaling factor
        scaling_factor = integral_total/integral_decomposed

        # Normalise the decomposed values <- Unsure if we should be doing this, but for complenetess it may be a good idea
        new_df['L_lambda_total_decomposed_AGN_Scaled'] = new_df['L_lambda_total_decomposed_AGN'] * scaling_factor
        
        

        # Galaxy component - Remove the AGN Component
        # Create an adjust L_lambda to plot against
        new_df['L_lambda_total_decomposed'] = new_df['L_lambda_total']

        # To ensure this is done correctly, we also are choosing to consider simply just the AGN components 
        # where the the values are within a range of 10^2 - 10^7
        new_df = new_df[(new_df['wavelength'] > 10**-2) & (new_df['wavelength'] < 10**7)]

        # We would like to remove the AGN from the galaxy if possible
        
        ####
        # Some galaxies will contain a fit for each AGN component
        # and others may not. Try later to explore this.
        ####
        
        
        
        # Removing polar dust
        new_df['L_lambda_total_decomposed'] = new_df['L_lambda_total_decomposed'] - new_df['agn.SKIRTOR2016_polar_dust']

        # Removing torus
        new_df['L_lambda_total_decomposed'] = new_df['L_lambda_total_decomposed'] - new_df['agn.SKIRTOR2016_torus']

        # Removing the accretion disk
        new_df['L_lambda_total_decomposed'] = new_df['L_lambda_total_decomposed'] - new_df['agn.SKIRTOR2016_disk']

        # normalise the values using the integral of the total flux of the original SED
        # We will use the trapezoidal rule to calculate the integral of the total flux
        # We will then divide the decomposed values by this integral to normalise the values
        integral_total = np.trapz(new_df['L_lambda_total'], new_df['wavelength'])
        integral_decomposed = np.trapz(new_df['L_lambda_total_decomposed'], new_df['wavelength'])

        # Scaling factor
        scaling_factor = integral_total/integral_decomposed

        # Normalise the decomposed values <- Unsure if we should be doing this, but for complenetess it may be a good idea
        new_df['L_lambda_total_decomposed_Scaled'] = new_df['L_lambda_total_decomposed'] * scaling_factor
        
        
        # For the purposes of using these SEDs in my code, we will create two new columns which have a specific label
        # wavelength
        new_df['lambda (Angstroms)'] = new_df['wavelength']
        new_df['Total Flux (erg/s/cm^2/Angstrom)'] = new_df['L_lambda_total_decomposed']
        
        
        decomposed_seds.append(new_df)
        
    return decomposed_seds 


# In[220]:


SED_df[2] = decompose_agn_seds(SED_df[2])


# In[221]:


len(SED_df[2])


# Now we want to attempt to recreate some of the results. In essence we want the decomposed SED to be roughly returned to it's previous position
# when we add our AGN model back in
# 
# 
# 
# A bassic approach to this may be to run the before and after UVJ colours: which we already have for each ID.
# Progressively adding the AGN components back in until we see a relatively similar UVJ colour to the original.
# 
# For starters we would like to simply just worry about putting the current SEDs through the original code to generate UVJ colours
# 
# 
# 
# 

# In[222]:


# Now that we have both a set of decomposed SEDs (our decomposed templates in a sense)
# So the set of templates is: SED_df[2]

# set an alpha array
alpha_values = np.linspace(0, 1, 11)


# In[223]:


alpha_values


# In[224]:



SED_decomposed_composites_df = generate_composite_set(agn_model, SED_df[2], alpha_values)


# In[225]:


SED_decomposed_composites_df[0]


# In[226]:


# We can not plot the SEDs
plt.figure(figsize=(10, 10))
# Plot agn normalized seds from the df_list
v = 90
for i in range(0, 11):
    print()
    plt.loglog(SED_decomposed_composites_df[i][v].iloc[:, 0], SED_decomposed_composites_df[i][v].iloc[:, 1], marker='o', markersize=1, label='AGN%:  {}'.format(int(alpha_values[i]*100)))
# data label

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
# Show lims
#plt.ylim(10**-1, 10**3)
#plt.xlim(10**3, 10**5)

plt.title('Composite SED')
plt.legend()
plt.show()


# In[227]:


# We would now like to attempt to recreate some of the colours
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

for i in range(len(alpha_values)):
    # This will be the set of composites for the specific alpha value
    sed_alpha_data = SED_decomposed_composites_df[i] # should go through each
    
    for j, sed_data in enumerate(sed_alpha_data): # will go through each of the zfourge galaxies
        # Create an SED object using astSED
        wl = sed_data['lambda (Angstroms)']
        fl = sed_data['Total Flux (erg/s/cm^2/Angstrom)']
        sed = astSED.SED(wavelength=wl, flux=fl, z=0.0)  
        
        # The id of this particular entry is the objname_list[j]
        objname = names[j]
        
        
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


# In[228]:


print(len(uv_specific_alpha_colours[0]))
print(len(vj_specific_alpha_colours[0]))


# In[240]:


ids = new_objname_alpha_list[0]


# add the field prefix to the IDs
ids = [str(x) for x in ids]


# In[241]:


# Create a dataframe with all the id's 
df = pd.DataFrame(data={'ID': ids})


# In[242]:


df


# In[243]:


for i in range(len(alpha_values)):
    df['UV_{}'.format(int(alpha_values[i]*100))] = uv_specific_alpha_colours[i]
    df['VJ_{}'.format(int(alpha_values[i]*100))] = vj_specific_alpha_colours[i]


# In[244]:


df


# In[248]:


# Read in the decomposed UVJ ids
decomposed_UVJ_ids.rename(columns={'id': 'ID'}, inplace=True)

df


# In[247]:


decomposed_UVJ_ids


# In[249]:



# join these two tables together, adding the columns VJ_withAGN	UV_withAGN	VJ_withoutAGN	UV_withoutAGN to the df table
new_df = pd.merge(df, decomposed_UVJ_ids, on='ID')


# In[250]:





# In[237]:


# Creating the same output but in the other two fields
alpha_list = np.linspace(0, 1, 11)


fig = plt.figure(figsize=(6, 6))

# Plot each value of alpha from the decomposed original UVJ colour
for alpha in alpha_list:
    plt.scatter(df['VJ_{}'.format(int(alpha*100))], df['UV_{}'.format(int(alpha*100))], c='b', s=3, alpha=0.5)

# Plot the UV_withAGN and VJ_withAGN
plt.scatter(df['VJ_withAGN'], df['UV_withAGN'], c='r', s=3, alpha=0.5)

# And UV_withoutAGN and VJ_withoutAGN
plt.scatter(df['VJ_withoutAGN'], df['UV_withoutAGN'], c='g', s=3, alpha=0.5)


    

plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours of AGN Composites")
plt.xlim([-0.5, 2.2])
plt.ylim([0, 2.5])

# Plot the names
# for i in range(len(selected_ids_df)):
#     #plt.text(selected_ids_df['vj'][i], selected_ids_df['uv'][i], selected_ids_df['id'][i], fontsize=12)
#     plt.text(recalculated_df['VJ'][i], recalculated_df['UV'][i], recalculated_df['ID'][i], fontsize=12)

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

plt.show()


# In[ ]:


# Now, based on where the inital AGN uvj value is, and where the recaluclated one is at alpha = 0. draw an arrow 
# to show the direction of the change

