#!/usr/bin/env python
# coding: utf-8

# # Theoretical Modelling of Composite Galaxies 
# This code is intended to output a dataframe containing the colours of the theoretical galaxy+AGN composites.
# 
# 

# In[13]:


# Read in an AGN template
# Read in all required libraries
# Import in all of the required libraries
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pandas as pd
import os
from astLib import astSED
import astropy.io.fits as fits
import matplotlib.path as mpath
from carf import * # custom module for functions relating to the project

# So that we can change the helper functions without reloading the kernel
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[14]:


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


# In[15]:


# Skirtor models
skirtor_folderpath = os.path.join('datasets', 'Templates', 'Skirtor')

# Swire templates
swire_folderpath = os.path.join('datasets', 'Templates', 'SWIRE')

# Brown templates
brown_folderpath = os.path.join('datasets', 'Templates', 'Brown', '2014','Rest')


# In[16]:


# Read in the AGN templates
type1_agn, type1_params = create_type1_skirtor_agn(skirtor_folderpath)
type2_agn, type2_params = create_type2_skirtor_agn(skirtor_folderpath)

# Read in the template set of choice
swire_templates, template_names = read_swire_templates(swire_folderpath)

# Read in the brown templates
brown_templates, brown_template_names = read_brown_galaxy_templates(brown_folderpath)


# In[17]:


# Choose running parameters

# Template set and AGN type
template_set_name = 'SWIRE'
agn_model_name = 'Type1AGN'

# Setup how many alpha values we want to explore
alpha_values = np.linspace(0, 1, 11)


# In[18]:


# Choose an AGN model
if agn_model_name == 'Type1AGN':
    agn_model = type1_agn
elif agn_model_name == 'Type2AGN':
    agn_model = type2_agn
else:
    print('AGN model not recognised')

if template_set_name == 'SWIRE':
    template_set = swire_templates
    template_names = template_names
elif template_set_name == 'Brown':
    template_set = brown_templates
    template_names = brown_template_names
else:
    print('Template set not recognised')


# In[ ]:





# In[19]:


# Create all of the composites
composites = generate_composite_set(agn_model, template_set, alpha_values)


# In[20]:


# # # Save all the sampled colours to a data frame

# Create an empty data frame for each template that has the filters with the alpha values o.e u_0, u_10, u_20 etc for each filter
column_names = ['id', 'z'] # add an inital column for the redshift and for the ID
# filters 
for filter in filter_set.keys():
    # Check filter we are looking at
    print(filter)
    
    for alpha_val in alpha_values:
    # Add filter into a data frame
        column_names.append(filter + '_' + str(int(round(alpha_val, 2)*100)))

# Turn into a dataframe  
composite_fluxes = pd.DataFrame(columns=column_names)


# In[21]:


composite_fluxes


# In[22]:


# To make this work for the redshifts we are going to need to create a new dataframe for each redshift
# This will be a list of dataframes

# for each redshift create a new data frame
redshifts = np.arange(0, 4, 0.1)

# Create a list of dataframes for each redshift
composite_fluxes_list = []


for redshift in redshifts:    
    # Create an empty data frame for each template that has the filters with the alpha values o.e u_0, u_10, u_20 etc for each filter
    flux_df = composite_fluxes.copy()
    
    
    for i in range(len(alpha_values)):
        # This will be the set of composites for the specific alpha value
        sed_alpha_data = composites[i]
        
        for j, sed_data in enumerate(sed_alpha_data):
            # Create an SED object using astSED
        
            #print(sed_data['lambda (Angstroms)'])
            wl = sed_data['lambda (Angstroms)']
            fl = sed_data['Total Flux (erg/s/cm^2/Angstrom)']
            
            # Create an SED object
            sed = astSED.SED(wavelength=wl, flux=fl, z=0.0)
            
            # Redshift the SED
            sed.redshift(redshift)
            
            # Add template name to the data frame in the id column
            flux_df.loc[j, 'id'] = template_names[j]
            
            # Only turn this on to output all seds
            # Only necessary to output the CSV for the rest frame
            
            if redshift == 0:
                sed_data.to_csv(f'outputs/composite_seds/{template_set_name}/'+template_names[j]+ f'{agn_model_name}'+ 'composite_' +str(int(round(alpha_values[i], 2)*100))+'.csv')
            
            # Calculate each filter value for the specific alpha value
            for filter in filter_set.keys():
                # Calculate the magnitude for the filter
                
                if filter == 'IRAC3.6' or filter == 'IRAC4.5' or filter == 'IRAC5.8' or filter == 'IRAC8.0':
                    # Calculate the magnitude for the IRAC filters
                    obs_flux = astSED.SED.calcFlux(sed, filter_set[filter])  
                    # Add the magnitude to the data frame
                    flux_df.loc[j, filter + '_' + str(int(round(alpha_values[i], 2)*100))] = obs_flux
                    
                else:
                    # Calculate the magnitude for the other filters
                    mag = astSED.SED.calcMag(sed, filter_set[filter], magType='AB')

                    # Add the magnitude to the data frame
                    flux_df.loc[j, filter + '_' + str(int(round(alpha_values[i], 2)*100))] = mag
                
    # Add the redshift to the data frame
    flux_df['z'] = redshift
    composite_fluxes_list.append(flux_df)

            
            
            


# In[23]:


# We now want to join each of the data frames together
composite_fluxes = pd.concat(composite_fluxes_list)


# In[24]:



# Output as required
composite_choice = f'{template_set_name}_theoretical_composite_fluxes_{agn_model_name}'
composite_fluxes.to_csv(f'outputs\composite_seds\{composite_choice}.csv')   


# In[ ]:




