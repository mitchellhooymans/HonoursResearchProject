#!/usr/bin/env python
# coding: utf-8

# # SED Processing SWIRE Composite Tests
# This is an extension of the previous notebook (SED Processing_Test.ipynb) which tests the functions that have been developed for processing and creating composite SEDS
# The ides is that these functions will be useable enough, and will be flexible enough to allow for resuseability with different SEDs and with different AGN models

# In[473]:


# Import in all of the required libraries
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pandas as pd
import os
from astLib import astSED
import astropy.io.fits as fits
from carf import * # custom module for functions relating to the project

# So that we can change the helper functions without reloading the kernel
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[474]:


# setup all the directories we are getting our data from 

# Skirtor models
skirtor_folderpath = os.path.join('datasets', 'Templates', 'Skirtor')

# Swire templates
swire_folderpath = os.path.join('datasets', 'Templates', 'SWIRE')

# Brown templates
brown_folderpath = os.path.join('datasets', 'Templates', 'Brown', '2014','Rest')

# Filters
pb_U_path = os.path.join('datasets', 'Filters', 'Generic_Johnson.U.dat')
pb_V_path = os.path.join('datasets', 'Filters', 'Generic_Johnson.V.dat')
pb_J_path = os.path.join('datasets', 'Filters', '2MASS_2MASS.J.dat')


# In[475]:


# In addition also use astSED to create filters
pb_U = astSED.Passband(pb_U_path, normalise=False)
pb_V = astSED.Passband(pb_V_path, normalise=False)
pb_J = astSED.Passband(pb_J_path, normalise=False)


# # SWIRE + Skirtor Composite SEDs
# This part of the document aims to test the combining proccess of the Skirtor AGN models, with the SWIRE templates. These templates have a shortened wavelength range so the functionality in the helper package should make sure these are both compatible and allow for composite seds to be made.
# 

# In[476]:


# Begin by attempting to create a composite of just 1 model
# For this we will define a type 1 agn model using the appropriate Skirtor parameters
# Begin by importing the first AGN model we would like to use
# 1. Type 1 AGN
tau = SKIRTOR_PARAMS['tau'][3]
p = SKIRTOR_PARAMS['p'][0]
q = SKIRTOR_PARAMS['q'][0]
oa = SKIRTOR_PARAMS['oa'][4]
rr = SKIRTOR_PARAMS['rr'][2]
i = SKIRTOR_PARAMS['i'][0]

agn_model = read_skirtor_model(skirtor_folderpath, tau, p, q, oa, rr, i)
wl_agn = agn_model['lambda (Angstroms)']
fl_agn = agn_model['Total Flux (erg/s/cm^2/Angstrom)']


# In[477]:


agn_model


# In[478]:



# plot the agn model
plot_galaxy_sed(wl_agn, fl_agn, "Type 1 AGN Model", "SKIRTOR")


# In[479]:


# Similarly we can plot a single swire template, we will use M82
m82_template, objname = read_swire_template(swire_folderpath, 'M82')
m82_template


wl_m82 = m82_template['lambda (Angstroms)']
fl_m82 = m82_template['Total Flux (erg/s/cm^2/Angstrom)']

# Plot the SED of M82
plot_galaxy_sed(wl_m82, fl_m82, 'M82', 'SWIRE')


# In[480]:


plt.figure(figsize=(10, 5))
plt.loglog(wl_m82, fl_m82, color='red', linewidth=1, linestyle='-', alpha=0.5)
plt.loglog(wl_agn, fl_agn, color='blue', linewidth=1, linestyle='-', alpha=0.5)
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
plt.title('Comparison of SED fluxes')
plt.grid()
plt.show()


# From the graph above it is obvious that we can se that both of these SEDs don't exactly align in a way that will make sense or work for us. To rectify this we need to make sure that both of these SEDs exsist over the same wavelength range (are interpolated over a set of combined wavelengths, and are also within the same range of values. We see that the AGN has a lot more values in the shorter wavelength range, so for this to work this information must be cut off. This may be a limitation in the SWIRE templates but further investigation is needed. Our step now is to correct the wavelength range for these two SEDS

# In[481]:


# We use our function two change the wavelenth range of the two SEDs, rectifying any missing wavelength values
m82_template, agn_model = adjust_wavelength_range(m82_template, agn_model)

# We can now attempt to replot these both on the same graph
wl_m82 = m82_template['lambda (Angstroms)']
fl_m82 = m82_template['Total Flux (erg/s/cm^2/Angstrom)']
wl_agn = agn_model['lambda (Angstroms)']
fl_agn = agn_model['Total Flux (erg/s/cm^2/Angstrom)']


# Plotting the adjusted SEDS
plt.figure(figsize=(10, 5))
plt.loglog(wl_m82, fl_m82, color='red', linewidth=1, linestyle='-', alpha=0.5)
plt.loglog(wl_agn, fl_agn, color='blue', linewidth=1, linestyle='-', alpha=0.5)
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
plt.title('Comparison of SED fluxes')
plt.grid()
plt.show()


# We can see that this has correctly put the SEDs into the same wavelength range and in addition to this have made both of the SEDs compareable in the wavelength range. Now we need to make sure that we get the fluxes to be comparable. We do this by integrating the fluxes across the wavelength range, and then perform a normalization based on the integral of the flux.

# In[482]:


# We now perform integral normalization on both of the SEDS
agn_model = normalize_flux_integral(agn_model)
m82_template = normalize_flux_integral(m82_template)


# In[483]:


# We can attempt to plot these again 

# We can now attempt to replot these both on the same graph
wl_m82 = m82_template['lambda (Angstroms)']
fl_m82 = m82_template['integral normalized flux']
wl_agn = agn_model['lambda (Angstroms)']
fl_agn = agn_model['integral normalized flux']


# Plotting the adjusted SEDS
plt.figure(figsize=(10, 5))
plt.loglog(wl_m82, fl_m82, color='red', linewidth=1, linestyle='-', alpha=0.5)
plt.loglog(wl_agn, fl_agn, color='blue', linewidth=1, linestyle='-', alpha=0.5)
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Integral Normalized Flux')
plt.title('Comparison of SED fluxes')
plt.grid()
plt.show()


# In[484]:


# Now that we have a galaxy and a AGN which have integral normalized flux units, we can use our code to create composite agn+galaxies
# We begin with the example above

# Create an entirely new composite galaxy() with params alpha = 0.5, beta is default to 1, but set to (1-alpha for this)
# thus this is an galaxy composite created through sed addition: 50% AGN, 50% M82 (Starburst)
alpha = 0.5
agn_starburst_composite = create_gal_agn_composite_sed(agn_model, m82_template, alpha, (1-alpha))


agn_starburst_composite


# In[485]:


# We can now plot this composite
wl_comp = agn_starburst_composite['lambda (Angstroms)']
fl_comp = agn_starburst_composite['Total Flux (erg/s/cm^2/Angstrom)']


# Plotting the adjusted SEDS
plt.figure(figsize=(10, 5))
plt.loglog(wl_comp, fl_comp, color='red', linewidth=1, linestyle='-', alpha=0.5)
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
plt.title('Comparison of SED fluxes')
plt.grid()
plt.show()


# In[486]:


# Plotting this with it's component AGN bits
# We can now attempt to replot these both on the same graph
wl_m82 = m82_template['lambda (Angstroms)']
fl_m82 = m82_template['Total Flux (erg/s/cm^2/Angstrom)']
wl_agn = agn_model['lambda (Angstroms)']
fl_agn = agn_model['Total Flux (erg/s/cm^2/Angstrom)']


# Plotting the adjusted SEDS
plt.figure(figsize=(10, 5))
plt.loglog(wl_m82, fl_m82, color='red', linewidth=1, linestyle='-', alpha=0.5, label='M82')
plt.loglog(wl_agn, fl_agn, color='blue', linewidth=1, linestyle='-', alpha=0.5, label='Type 1 AGN')
plt.loglog(wl_comp, fl_comp, color='purple', linewidth=1, linestyle='-', alpha=0.5, label='AGN+M82 Composite')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Integral Normalized Flux')
plt.title('Comparison of SED fluxes')
plt.grid()
plt.legend()
plt.show()


# In[487]:


# Now we see exactly how these work we also note that this is not what we want
# We again try but instead we use a scaling factor derived from the galaxy to determine how to scale the AGN


# We attempt this using the parameters alpha = 0.5, beta = 1
# where the scaling is done internally in the function
alpha = 0.5
agn_starburst_composite_add = create_gal_agn_composite_sed(agn_model, m82_template, alpha)
agn_starburst_composite_add


# In[488]:


# We can now attempt to plot this
wl_comp = agn_starburst_composite_add['lambda (Angstroms)']
fl_comp = agn_starburst_composite_add['Total Flux (erg/s/cm^2/Angstrom)']

# Plotting the adjusted SEDS
plt.figure(figsize=(10, 5))
plt.loglog(wl_comp, fl_comp, color='red', linewidth=1, linestyle='-', alpha=0.5)
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
plt.title('Comparison of SED fluxes')
plt.grid()
plt.show()


# In[489]:


# We attempt this method with multiple different values of alpha
alpha = np.linspace(0, 1, 6)


# In[490]:


alpha


# In[491]:


# Creating many composites
composites = []
for a in alpha:
    print(f'Creating composite with alpha = {a}')
    composite_galaxy = create_gal_agn_composite_sed(agn_model, m82_template, a, 1)
    composites.append(composite_galaxy)

# Plotting the adjusted SEDS
plt.figure(figsize=(10, 6))
for i, comp in enumerate(composites):
    wl_comp = comp['lambda (Angstroms)']
    fl_comp = comp['Total Flux (erg/s/cm^2/Angstrom)']
    plt.loglog(wl_comp, fl_comp, linewidth=1, linestyle='-', alpha=0.5, label=f'{round(alpha[i]*100)}% AGN')
plt.legend()
plt.title("Hybrid SED with Incremental AGN Contribution Normalized")
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
plt.show()


# In[492]:


# Now we have a list of composite SEDs we can attempt to convert these into restframe UVJ colours and ultimately plot these on a UVJ diagram
colour_list = []
uv_list = []
vj_list = []



for n in range(6):
    
    # use the wavelength and flux of the sed
    wl = composites[n].iloc[:, 0].values
    fl = composites[n].iloc[:, 1].values
    

    # create an SED object containing the SED of the galaxy
    # in addition to this use the relevant wavelength and flux
    sed = astSED.SED(wavelength=wl, flux=fl) # z = 0.0 as these are restframe SEDs

    # Using the astSED library calculate the UVJ colours using the U, V, and J passbands. 
    # We will use the AB magnitude system
    uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
    vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
    
    uv_list.append(uv)
    vj_list.append(vj)


# In[493]:


alpha


# In[494]:


uv_colours = uv_list
vj_colours = vj_list
col = 'r'

plt.figure(figsize=(10, 10))
plt.scatter(vj_colours, uv_colours, c=alpha, s=10, label="M82 + Type 1 AGN Composites")
plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours")
plt.xlim([-0.5, 2.2])
plt.ylim([0, 2.5])

# Set colourbar label
plt.colorbar().set_label('AGN Contribution')


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

plt.legend()
#Add text labels from name_list
#print(name_list[0])

plt.show()


# In[495]:


# Again we can attempt to create a composite using a different galaxy templates.

# Now that we have a proof of concept that this works for one galaxy, we can generalise this and work with multiple galaxies
# Read in a list of swire tempaltes, and append them into a list
# Read in the subset of the swire templates


# ## Type 1 AGN + Swire Composites
# This particular code will be used to create a set of Type 1 + SWIRE Composites

# In[496]:


# To extend this we are now dealing with more than one galaxy so we need to make sure our code is flexible enough to be able to create
# useful composites that are correctly scaled. To do this we need to make sure every composite we generate is scaled accurately against a useful

# Follow a similar proccess as before

# Define the particular AGN Model we are exploring
# # 1. Type 1 AGN
tau = SKIRTOR_PARAMS['tau'][2]
p = SKIRTOR_PARAMS['p'][0]
q = SKIRTOR_PARAMS['q'][0]
oa = SKIRTOR_PARAMS['oa'][4]
rr = SKIRTOR_PARAMS['rr'][2]
i = SKIRTOR_PARAMS['i'][0]


# This is again, a new model
agn_model = read_skirtor_model(skirtor_folderpath, tau, p, q, oa, rr, i)

# Print the parameters used in the AGN model
print(f'Tau: {tau}, p: {p}, q: {q}, oa: {oa}, rr: {rr}, i: {i}')

# 2. Swire Templates
swire_templates = []
objname_list = []

# In this template set we have 3 ellipticals, 4 spirals, 2 star bursts, and a Seyfert 2 galaxy (which inherently has AGN contributions)
template_names = ['Ell2', 'Ell5', 'S0', 'Sb', 'Sc', 'M82', 'N6090', 'Sdm', 'Sey2']
for name in template_names:
    
    template, obj_name = read_swire_template(swire_folderpath, name)
    swire_templates.append(template)
    objname_list.append(obj_name)

# Make sure AGN are correctly scaled against the galaxy range   
# Have an original AGN model to adjust against 
type_1_agn = agn_model.copy()

type1agn_models = []


# When adjusting, we need to make sure we interpolate the data correctly at each point, 
# so for a specific AGN model this needs to be interpolated correctly against the relevant galaxy template
# essentially each AGN model will be specifically cut for it's template.
for i, template in enumerate(swire_templates):
    swire_templates[i], agn_model = adjust_wavelength_range(template, agn_model)   
    type1agn_models.append(agn_model)
    agn_model = type_1_agn.copy()


# Using the same alpha as before, we can create a composite of the AGN and the galaxy


# In[497]:


agn_model


# In[498]:


swire_templates[0]


# In[499]:


# We can plot the fluxes all on one graph to see if they are within the same wavelength range

# Use this as a comparison, but in actuality each AGN model will be adjusted to the specific galaxy template
wl_agn = type1agn_models[0]['lambda (Angstroms)']
fl_agn = type1agn_models[0]['Total Flux (erg/s/cm^2/Angstrom)']


plt.figure(figsize=(10, 5))
for i, template in enumerate(swire_templates):
    wl = template['lambda (Angstroms)']
    fl = template['Total Flux (erg/s/cm^2/Angstrom)']
    plt.loglog(wl, fl, label=objname_list[i])
plt.loglog(wl_agn, fl_agn, label='Type 1 AGN', color='black', linestyle='--')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
plt.title('Comparison of SED fluxes')
plt.grid()
plt.legend()
plt.show()


# This makes sense for this range of SEDs, we can now use our code to create a composite of these SEDs, creating the code to work
# with only 1 galaxy, then generalising it to 2



# In[500]:


# Before looping through the galaxies we can test the code with just one galaxy
# We attempt this method with multiple different values of alpha, as defined before
alpha = np.linspace(0, 3, 8)

n = 8

# Creating many composites 
composites = []
for a in alpha:
    print(f'Creating composite with alpha = {a}')
    composite_galaxy = create_gal_agn_composite_sed(type1agn_models[n], swire_templates[n], a, 1)
    composites.append(composite_galaxy)


# In[501]:



# Plotting the adjusted SEDS
plt.figure(figsize=(10, 6))
for i, comp in enumerate(composites):
    wl_comp = comp['lambda (Angstroms)']
    fl_comp = comp['Total Flux (erg/s/cm^2/Angstrom)']
    plt.loglog(wl_comp, fl_comp, linewidth=1, linestyle='-', label=f'{round(alpha[i]*100)}% AGN')
plt.legend()
plt.title("Hybrid SED with Incremental AGN Contribution Normalized: " + objname_list[n])
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
plt.show()


# In[502]:


# Now that we have a working composite set, we aim to do this for all templates, creating a list of composites, 
# with each of those composites also having values from 0 to 1 for alpha, defined earlier
spacing = 6
alpha = np.linspace(0, 1, spacing)
type1composites = []

# So for every swire template
for i, template in enumerate(swire_templates):
    print(f'Creating composite for {objname_list[i]}')
    agn_model = type1agn_models[i]
    type1composite_list = []
    for a in alpha:
        composite_galaxy = create_gal_agn_composite_sed(agn_model, template, a, 1)
        type1composite_list.append(composite_galaxy)
    type1composites.append(type1composite_list)
    
    


# In[503]:


# We should have composites which work now. To check that this works, we can plot a 3x3 grid of the composites
# Plotting the composite SEDs 
plt.figure(figsize=(15, 15))
for i in range(9):
    plt.subplot(3, 3, i+1)
    for j, comp in enumerate(type1composites[i]):
        wl_comp = comp['lambda (Angstroms)']
        fl_comp = comp['Total Flux (erg/s/cm^2/Angstrom)']
        plt.loglog(wl_comp, fl_comp, linewidth=1, linestyle='-', label=f'{round(alpha[j]*100)}% AGN')
    plt.legend()
    plt.title(objname_list[i])
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
    plt.grid()
    
plt.tight_layout()

# Save to outputs folder
#plt.savefig('outputs/Type1AGNSWIREComposites.png', bbox_inches='tight', pad_inches=0.1, facecolor='white')

plt.show()


# In[ ]:





# ## Type 2 AGN + Swire Composites

# In[504]:


# Define the particular AGN Model we are exploring
# # 1. Type 2 AGN
tau = SKIRTOR_PARAMS['tau'][2]
p = SKIRTOR_PARAMS['p'][0]
q = SKIRTOR_PARAMS['q'][0]
oa = SKIRTOR_PARAMS['oa'][4]
rr = SKIRTOR_PARAMS['rr'][2]
i = SKIRTOR_PARAMS['i'][9]


# This is again, a new model
agn_model = read_skirtor_model(skirtor_folderpath, tau, p, q, oa, rr, i)

# Print the parameters used in the AGN model
print(f'Tau: {tau}, p: {p}, q: {q}, oa: {oa}, rr: {rr}, i: {i}')

# 2. Swire Templates
swire_templates = []
objname_list = []

# In this template set we have 3 ellipticals, 4 spirals, 2 star bursts, and a Seyfert 2 galaxy (which inherently has AGN contributions)
template_names = ['Ell2', 'Ell5', 'S0', 'Sb', 'Sc', 'M82', 'N6090', 'Sdm', 'Sey2']
for name in template_names:
    
    template, obj_name = read_swire_template(swire_folderpath, name)
    swire_templates.append(template)
    objname_list.append(obj_name)

# Make sure AGN are correctly scaled against the galaxy range   
# Have an original AGN model to adjust against 
type_2_agn = agn_model.copy()

type2agn_models = []

# When adjusting, we need to make sure we interpolate the data correctly at each point, 
# so for a specific AGN model this needs to be interpolated correctly against the relevant galaxy template
# essentially each AGN model will be specifically cut for it's template.
for i, template in enumerate(swire_templates):
    swire_templates[i], agn_model = adjust_wavelength_range(template, agn_model)   
    type2agn_models.append(agn_model)
    agn_model = type_2_agn.copy()


# In[505]:


# We can plot the fluxes all on one graph to see if they are within the same wavelength range

# Use this as a comparison, but in actuality each AGN model will be adjusted to the specific galaxy template
wl_agn = type2agn_models[0]['lambda (Angstroms)']
fl_agn = type2agn_models[0]['Total Flux (erg/s/cm^2/Angstrom)']


plt.figure(figsize=(10, 5))
for i, template in enumerate(swire_templates):
    wl = template['lambda (Angstroms)']
    fl = template['Total Flux (erg/s/cm^2/Angstrom)']
    plt.loglog(wl, fl, label=objname_list[i])
plt.loglog(wl_agn, fl_agn, label='Type 1 AGN', color='black', linestyle='--')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
#plt.title('Comparison of SED fluxes')
plt.grid()
plt.legend()
#plt.savefig('Flux Comparisons.png', bbox_inches='tight', pad_inches=0.1, facecolor='white')
plt.show()


# In[506]:


# Creating my full set of composites
# Now that we have a working composite set, we aim to do this for all templates, creating a list of composites, 
# with each of those composites also having values from 0 to 1 for alpha, defined earlier
spacing = 6
alpha = np.linspace(0, 1, spacing)
type2composites = []

# So for every swire template
for i, template in enumerate(swire_templates):
    print(f'Creating composite for {objname_list[i]}')
    agn_model = type2agn_models[i]
    type2composite_list = []
    for a in alpha:
        composite_galaxy = create_gal_agn_composite_sed(agn_model, template, a, 1)
        type2composite_list.append(composite_galaxy)
    type2composites.append(type2composite_list)
    


# In[507]:


# Now we have our new set of composites we can plot these to see how they look
# Plotting the composite SEDs
plt.figure(figsize=(15, 15))
for i in range(9):
    plt.subplot(3, 3, i+1)
    for j, comp in enumerate(type2composites[i]):
        wl_comp = comp['lambda (Angstroms)']
        fl_comp = comp['Total Flux (erg/s/cm^2/Angstrom)']
        plt.loglog(wl_comp, fl_comp, linewidth=1, linestyle='-', label=f'{round(alpha[j]*100)}% AGN')
    plt.legend()
    plt.title(objname_list[i])
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
    plt.grid()
    
plt.tight_layout()

# Save to outputs folder
#plt.savefig('outputs/Type2AGNComposites.png', bbox_inches='tight', pad_inches=0.1, facecolor='white')

plt.show()


# # Intermediate Type AGN + Swire Composites
# We are now doing the same thing for the intermediate type AGN models

# In[508]:


# Define the particular AGN Model we are exploring
# # 1. Intermediate AGN
tau = SKIRTOR_PARAMS['tau'][2]
p = SKIRTOR_PARAMS['p'][1]
q = SKIRTOR_PARAMS['q'][0]
oa = SKIRTOR_PARAMS['oa'][4]
rr = SKIRTOR_PARAMS['rr'][1]
i = SKIRTOR_PARAMS['i'][4]


# This is again, a new model
agn_model = read_skirtor_model(skirtor_folderpath, tau, p, q, oa, rr, i)

# Print the parameters used in the AGN model
print(f'Tau: {tau}, p: {p}, q: {q}, oa: {oa}, rr: {rr}, i: {i}')

# 2. Swire Templates
swire_templates = []
objname_list = []

# In this template set we have 3 ellipticals, 4 spirals, 2 star bursts, and a Seyfert 2 galaxy (which inherently has AGN contributions)
template_names = ['Ell2', 'Ell5', 'S0', 'Sb', 'Sc', 'M82', 'N6090', 'Sdm', 'Sey2']
for name in template_names:
    
    template, obj_name = read_swire_template(swire_folderpath, name)
    swire_templates.append(template)
    objname_list.append(obj_name)

# Make sure AGN are correctly scaled against the galaxy range   
# Have an original AGN model to adjust against 
intermediatet_agn = agn_model.copy()

intermediateagn_models = []

# When adjusting, we need to make sure we interpolate the data correctly at each point, 
# so for a specific AGN model this needs to be interpolated correctly against the relevant galaxy template
# essentially each AGN model will be specifically cut for it's template.
for i, template in enumerate(swire_templates):
    swire_templates[i], agn_model = adjust_wavelength_range(template, agn_model)   
    intermediateagn_models.append(agn_model)
    agn_model = intermediatet_agn.copy()


# In[509]:


# Use this as a comparison, but in actuality each AGN model will be adjusted to the specific galaxy template
wl_agn = intermediateagn_models[0]['lambda (Angstroms)']
fl_agn = intermediateagn_models[0]['Total Flux (erg/s/cm^2/Angstrom)']


plt.figure(figsize=(10, 5))
for i, template in enumerate(swire_templates):
    wl = template['lambda (Angstroms)']
    fl = template['Total Flux (erg/s/cm^2/Angstrom)']
    plt.loglog(wl, fl, label=objname_list[i])
plt.loglog(wl_agn, fl_agn, label='Intermediate Type AGN', color='black', linestyle='--')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
plt.title('Comparison of SED fluxes')
plt.grid()
plt.legend()
plt.show()


# In[510]:


# Creating my full set of composites
# Now that we have a working composite set, we aim to do this for all templates, creating a list of composites, 
# with each of those composites also having values from 0 to 1 for alpha, defined earlier
spacing = 6
alpha = np.linspace(0, 1, spacing)
intermediatecomposites = []

# So for every swire template
for i, template in enumerate(swire_templates):
    print(f'Creating composite for {objname_list[i]}')
    agn_model = intermediateagn_models[i]
    intermediatecomposite_list = []
    for a in alpha:
        composite_galaxy = create_gal_agn_composite_sed(agn_model, template, a, 1)
        intermediatecomposite_list.append(composite_galaxy)
    intermediatecomposites.append(intermediatecomposite_list)


# In[511]:


# Now we have our new set of composites we can plot these to see how they look
# Plotting the composite SEDs
plt.figure(figsize=(15, 15))
for i in range(9):
    plt.subplot(3, 3, i+1)
    for j, comp in enumerate(intermediatecomposites[i]):
        wl_comp = comp['lambda (Angstroms)']
        fl_comp = comp['Total Flux (erg/s/cm^2/Angstrom)']
        plt.loglog(wl_comp, fl_comp, linewidth=1, linestyle='-', label=f'{round(alpha[j]*100)}% AGN')
    plt.legend()
    plt.title(objname_list[i])
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
    plt.grid()
    
plt.tight_layout()

# Save to outputs folder
#plt.savefig('outputs/IntermediateAGNComposites.png', bbox_inches='tight', pad_inches=0.1, facecolor='white')

plt.show()


# ## Exploring UVJ Colour Space with Composites SEDs

# In[512]:


# Now we have our three sets of composites, what we need to do is to plot these on a UVJ diagram to see how they compare
# To do this well we need to convert all of our SEDs to photometry using our filter sets.

# First let's see how the UVJ diagram changes under the influence of the Type1 Contamination
# We can do this by plotting the UVJ diagram for the type 1 composites

# Calculate the UVJ colours for each composite
uv_list = []
vj_list = []

        
# For each Type1 Composite, of verying values calculate the UVJ colours

# First index is the galaxy base, second index is the alpha value
type1composites[7][5] # <- i.e the 8th galaxy, with 6th alpha value (Sdm +  100% AGN)

# For each composite, calculate the UVJ colours for each alpha value

# for i in range(9):
#     uv_list = []
#     vj_list = []
#     for j in range(6):
#         wl = type1composites[i][j].iloc[:, 0].values
#         fl = type1composites[i][j].iloc[:, 1].values
#         sed = astSED.SED(wavelength=wl, flux=fl) # z = 0.0 as these are restframe SEDs
#         uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
#         vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
#         uv_list.append(uv)
#         vj_list.append(vj)
#     uv_colours = uv_list
#     vj_colours = vj_list
#     col = 'r'
#     plt.figure(figsize=(10, 10))
#     #plt.scatter(vj_colours, uv_colours, c=alpha, s=10, label="")
#     # Plotting the colours, with the alpha values as the colour
#     plt.scatter(vj_colours, uv_colours, c=alpha, s=10, label="")
    
#     # Add text to the first galaxy with no contribution of AGN, showing which SWIRE template is being used
#     plt.text(vj_colours[0], uv_colours[0], objname_list[i], fontsize=12)
    
#     plt.ylabel('U - V')
#     plt.xlabel('V - J')
#     plt.title("Restframe UVJ Colours")
#     plt.xlim([-0.5, 2.2])
#     plt.ylim([0, 2.5])
#     plt.colorbar().set_label('AGN Contribution')
# plt.show()

# Setup a basic general composite list 

agn_type = 'Type1'


if agn_type == 'Type1':
    composites = type1composites
elif agn_type == 'Type2':
    composites = type2composites
elif agn_type == 'Intermediate':
    composites = intermediatecomposites

# Similar to before but plotting all of the composites on the same graph
uv_list = []
vj_list = []

# For each composite, calculate the UVJ colours for each alpha value
plt.figure(figsize=(6, 6))
for i in range(9):
    uv_list = []
    vj_list = []
    for j in range(6):
        wl = composites[i][j].iloc[:, 0].values
        fl = composites[i][j].iloc[:, 1].values
        sed = astSED.SED(wavelength=wl, flux=fl) # z = 0.0 as these are restframe SEDs
        uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
        vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
        uv_list.append(uv)
        vj_list.append(vj)
    uv_colours = uv_list
    vj_colours = vj_list
    col = 'r'
    #plt.scatter(vj_colours, uv_colours, c=alpha, s=10, label="")
    # Plotting the colours, with the alpha values as the colour
    plt.scatter(vj_colours, uv_colours, c=alpha, s=15, label="")
    
    # Add a faint line to show the path of the composites
    plt.plot(vj_colours, uv_colours, alpha=0.5, color='black')
    
    # Add text to the first galaxy with no contribution of AGN, showing which SWIRE template is being used
    #plt.text(vj_colours[0], uv_colours[0], objname_list[i], fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
plt.ylabel('U - V')
plt.xlabel('V - J')
#plt.title("Restframe UVJ Colours of "+ agn_type + " AGN Composites")
plt.xlim([-0.5, 2.2])
plt.ylim([0, 2.5])

# Create the colorbar and store it in a variable
cbar = plt.colorbar()

# Now make all your modifications to this single colorbar
cbar.set_label('AGN Contribution (%)')
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_ticklabels(['0', '20', '40', '60', '80', '100']) 

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

# Save to outputs folder
#plt.savefig('outputs/SWIRE'+agn_type+'AGNUVJ.png', bbox_inches='tight', pad_inches=0.1, facecolor='white')

plt.show()


# In[513]:


# To ensure that we are not missing something we will plot the filters on the wavelength range of the SEDs, while not to scale on the flux 
# they will be plotted and sclaed to the bottom of the plot to show where the filters are in relation to the SEDs
# and thus where the photometry is being calculated from

# Plotting the composite SEDs

# We can also plot the passbands for the UVJ
U_arr = np.array(pb_U.asList())
V_arr = np.array(pb_V.asList())
J_arr = np.array(pb_J.asList())

plt.figure(figsize=(20, 20))
sc = 1e-4
# Plot the SEDs
for i in range(9):
    plt.subplot(3, 3, i+1)
    for j, comp in enumerate(composites[i]):
        wl_comp = comp['lambda (Angstroms)']
        fl_comp = comp['Total Flux (erg/s/cm^2/Angstrom)']
        plt.loglog(wl_comp, fl_comp, linewidth=1, linestyle='-', label=f'{round(alpha[j]*100)}% AGN')
        plt.loglog(U_arr[:, 0], sc*U_arr[:,1])
        plt.loglog(V_arr[:, 0], sc*V_arr[:,1])
        plt.loglog(J_arr[:, 0], sc*J_arr[:,1])
    plt.legend()
    plt.title(objname_list[i])
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
    plt.grid()
    plt.xlim([1e3, 1e5])
    plt.ylim([1e-6, 1e1])



# plt.loglog(U_arr[:, 0], U_arr[:,1], label='U')
# plt.loglog(V_arr[:, 0], V_arr[:,1], label='V')
# plt.loglog(J_arr[:, 0], J_arr[:,1], label='J')
# plt.title("Passbands for the UVJ Filters")
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Transmission')

# Set some limits


plt.tight_layout()

#plt.savefig('outputs/'+agn_type+'AGNSEDsWithUVJFilters.png', bbox_inches='tight', pad_inches=0.1, facecolor='white')

plt.show()




# In[514]:


# Generalise the above code to iterate through the inclination angles
# Define the particular AGN Model we are exploring
# # 1. Intermediate AGN

comp_inc_list = []
tau_val = 2
for n in range(len(SKIRTOR_PARAMS['i'])):
    print("Creating Composites for i = ", SKIRTOR_PARAMS['i'][n])
    tau = SKIRTOR_PARAMS['tau'][tau_val]
    p = SKIRTOR_PARAMS['p'][1] # Can make this either 0.5 or 0
    q = SKIRTOR_PARAMS['q'][0] # Make this zero, similar to the Ciesla paper
    oa = SKIRTOR_PARAMS['oa'][4]
    rr = SKIRTOR_PARAMS['rr'][1]
    i = SKIRTOR_PARAMS['i'][n]


    # This is again, a new model
    agn_model = read_skirtor_model(skirtor_folderpath, tau, p, q, oa, rr, i)

    # Print the parameters used in the AGN model
    print(f'Tau: {tau}, p: {p}, q: {q}, oa: {oa}, rr: {rr}, i: {i}')

    # 2. Swire Templates
    swire_templates = []
    objname_list = []

    # In this template set we have 3 ellipticals, 4 spirals, 2 star bursts, and a Seyfert 2 galaxy (which inherently has AGN contributions)
    template_names = ['Ell2', 'Ell5', 'S0', 'Sb', 'Sc', 'M82', 'N6090', 'Sdm', 'Sey2']
    for name in template_names:
        
        template, obj_name = read_swire_template(swire_folderpath, name)
        swire_templates.append(template)
        objname_list.append(obj_name)

    # Make sure AGN are correctly scaled against the galaxy range   
    # Have an original AGN model to adjust against 
    intermediatet_agn = agn_model.copy()

    intermediateagn_models = []

    # When adjusting, we need to make sure we interpolate the data correctly at each point, 
    # so for a specific AGN model this needs to be interpolated correctly against the relevant galaxy template
    # essentially each AGN model will be specifically cut for it's template.
    for i, template in enumerate(swire_templates):
        swire_templates[i], agn_model = adjust_wavelength_range(template, agn_model)   
        intermediateagn_models.append(agn_model)
        agn_model = intermediatet_agn.copy()
    
    
    spacing = 13
    alpha = np.linspace(0, 1, spacing)
    intermediatecomposites = []

    # So for every swire template
    for i, template in enumerate(swire_templates):
        print(f'Creating composite for {objname_list[i]}')
        agn_model = intermediateagn_models[i]
        intermediatecomposite_list = []
        for a in alpha:
            composite_galaxy = create_gal_agn_composite_sed(agn_model, template, a, 1)
            intermediatecomposite_list.append(composite_galaxy)
        intermediatecomposites.append(intermediatecomposite_list)
    
        
    comp_inc_list.append(intermediatecomposites)
    


# In[515]:


# Now we have our full set of composites for all of the inclinations, we can plot these to see how they look
# Skipping the SED step, we convert to UVJ colours and plot a UVJ diagram for each of the inclinations
# Setup a basic general composite list

# each of our comp_inc_list contains a list of AGN models for each galaxy, and each of these AGN models has been combined with the galaxy template 
# using a range of alpha values


# Similar to before but plotting all of the composites on the same graph
# As we are plotting this for each inclination loop over the inclinations

# For each inclination
type1_uv = []
type1_vj = []

for n in range(len(SKIRTOR_PARAMS['i'])):
    # For each composite, calculate the UVJ colours for each alpha value
    plt.figure(figsize=(6, 6))

    for i in range(len(objname_list)):
        uv_list = []
        vj_list = []
        for j in range(len(alpha)):
            wl = comp_inc_list[n][i][j].iloc[:, 0].values
            fl = comp_inc_list[n][i][j].iloc[:, 1].values
            sed = astSED.SED(wavelength=wl, flux=fl) # z = 0.0 as these are restframe SEDs
            uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
            vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
            uv_list.append(uv)
            vj_list.append(vj)
        uv_colours = uv_list
        vj_colours = vj_list
        col = 'r'
        
        #plt.scatter(vj_colours, uv_colours, c=alpha, s=10, label="")
        # Plotting the colours, with the alpha values as the colour
        plt.scatter(vj_colours, uv_colours, c=alpha, s=10, label="")
        
        # Add a faint line to show the path of the composites
        plt.plot(vj_colours, uv_colours, alpha=0.5, color='black')
        
        # Add text to the first galaxy with no contribution of AGN, showing which SWIRE template is being used
        #plt.text(vj_colours[0], uv_colours[0], objname_list[i], fontsize=12)
        
    plt.ylabel('U - V')
    plt.xlabel('V - J')
    #plt.title("Restframe UVJ Colours of "+ str(SKIRTOR_PARAMS['i'][n]) + "% Inclincation AGN Composites")
    plt.xlim([-0.5, 2.2])
    plt.ylim([0, 2.5])
    plt.colorbar().set_label('AGN Contribution')
        
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
    
    # Add a textbox with the model params
    plt.text(-0.4, 0.1, f'Tau: {tau}, p: {p}, q: {q}, oa: {oa}, rr: {rr}, i: {SKIRTOR_PARAMS["i"][n]}', fontsize=12)

    # Save to outputs folder
   # plt.savefig('outputs/UVJPlots/SWIRE_Composite'+str(SKIRTOR_PARAMS['i'][n]) +'i_'+str(SKIRTOR_PARAMS['tau'][tau_val])+'tau_AGNUVJ.png', bbox_inches='tight', dpi=300, facecolor='white')

    plt.show()
    


# In[516]:


# # Generalise this idea further so I can print out the UVJ diagrams for all of the AGN types
# # We would like to have a particular AGN, with parameters, and go through the entire parameter space, creating UVJ diagrams for each of the AGN types
# # that a composite has been created for


### Do not run this, this will create all of the composites but will take a long time to run
### and may not be necessary for the final report
### This will however create composites for everything in the parameter space


# comp_inc_list = []
# comp_tau_list = []
# comp_open_angle_list = []
# comp_p_list = []   
# comp_q_list = []
# comp_rr_list = []

# for n in range(len(SKIRTOR_PARAMS['i'])):
#     print("Creating Composites for i = ", SKIRTOR_PARAMS['i'][n])
    
#     for m in range(len(SKIRTOR_PARAMS['tau'])):
        
#         print("Creating Composites for tau = ", SKIRTOR_PARAMS['tau'][m])
        
#         for o in range(len(SKIRTOR_PARAMS['oa'])):
            
#             print("Creating Composites for oa = ", SKIRTOR_PARAMS['oa'][o])
            
#             for p_val in range(len(SKIRTOR_PARAMS['p'])):
                
#                 print("Creating Composites for p = ", SKIRTOR_PARAMS['p'][p_val])
                
#                 for q_val in range(len(SKIRTOR_PARAMS['q'])):
                    
#                     print("Creating Composites for q = ", SKIRTOR_PARAMS['q'][q_val])
                    
#                     for rr in range(len(SKIRTOR_PARAMS['rr'])):
                        
#                         print("Creating Composites for rr = ", SKIRTOR_PARAMS['rr'][rr])
        
#                         tau = SKIRTOR_PARAMS['tau'][m]
#                         p = SKIRTOR_PARAMS['p'][p_val]
#                         q = SKIRTOR_PARAMS['q'][q_val]
#                         oa = SKIRTOR_PARAMS['oa'][o]
#                         rr = SKIRTOR_PARAMS['rr'][rr]
#                         i = SKIRTOR_PARAMS['i'][n]


#                         # This is again, a new model
#                         agn_model = read_skirtor_model(skirtor_folderpath, tau, p, q, oa, rr, i)

#                         # Print the parameters used in the AGN model
#                         print(f'Tau: {tau}, p: {p}, q: {q}, oa: {oa}, rr: {rr}, i: {i}')

#                         # 2. Swire Templates
#                         swire_templates = []
#                         objname_list = []

#                         # In this template set we have 3 ellipticals, 4 spirals, 2 star bursts, and a Seyfert 2 galaxy (which inherently has AGN contributions)
#                         template_names = ['Ell2', 'Ell5', 'S0', 'Sb', 'Sc', 'M82', 'N6090', 'Sdm', 'Sey2']
#                         for name in template_names:
                            
#                             template, obj_name = read_swire_template(swire_folderpath, name)
#                             swire_templates.append(template)
#                             objname_list.append(obj_name)

#                         # Make sure AGN are correctly scaled against the galaxy range   
#                         # Have an original AGN model to adjust against 
#                         intermediatet_agn = agn_model.copy()

#                         intermediateagn_models = []

#                         # When adjusting, we need to make sure we interpolate the data correctly at each point, 
#                         # so for a specific AGN model this needs to be interpolated correctly against the relevant galaxy template
#                         # essentially each AGN model will be specifically cut for it's template.
#                         for i, template in enumerate(swire_templates):
#                             swire_templates[i], agn_model = adjust_wavelength_range(template, agn_model)   
#                             intermediateagn_models.append(agn_model)
#                             agn_model = intermediatet_agn.copy()
                        
                        
#                         spacing = 6
#                         alpha = np.linspace(0, 1, spacing)
#                         intermediatecomposites = []

#                         # So for every swire template
#                         for i, template in enumerate(swire_templates):
#                             print(f'Creating composite for {objname_list[i]}')
#                             agn_model = intermediateagn_models[i]
#                             intermediatecomposite_list = []
#                             for a in alpha:
#                                 composite_galaxy = create_gal_agn_composite_sed(agn_model, template, a, 1)
#                                 intermediatecomposite_list.append(composite_galaxy)
#                             intermediatecomposites.append(intermediatecomposite_list)
                        
                            
#                         comp_inc_list.append(intermediatecomposites)
#                     comp_tau_list.append(tau)
#                 comp_open_angle_list.append(oa)
#             comp_p_list.append(p_val)
#         comp_q_list.append(q_val)
#     comp_rr_list.append(rr)
            
                    
            




# In[517]:


# We would also like to create a mean position of each of the composites, i.e for all galaxies at a particular alpha value.
# Get their position, and generate a mean position. We can then create a new UVJ plot, and plot the mean positions with a line
# To see how each of these galaxies move through the UVJ colour space.

# To do this we need to explore the UVJ colours of each of the composites, and then calculate the mean position of each of the composites
# We can then plot this mean position on the UVJ diagram to see how the composites move through the UVJ space

# We can do this by plotting the UVJ diagram for the type 1 composites

# We also go back to only considering the type 1, intermediate, and type 2 AGN


alpha = np.linspace(0, 1, 13)
for n in range(len(SKIRTOR_PARAMS['i'])):
    # For each composite, calculate the UVJ colours for each alpha value
    plt.figure(figsize=(10, 10))

    # For each inclination we would like to look at the alpha value, 
    # For each alpha value, check each galaxy, and calculate the UVJ colours
    # get a mean value of the UVJ colours for each alpha value, and plot it
    # then move to the next alpha value
    
    uv_mean = []
    vj_mean = []
    
    type1agn_uv = []
    type1agn_vj = []
    for j in range(len(alpha)):
        # Calculate the UVJ colours for all of the composites and then average the position
        uv_list = []
        vj_list = []
        for i in range(len(objname_list)):
            wl = comp_inc_list[n][i][j].iloc[:, 0].values
            fl = comp_inc_list[n][i][j].iloc[:, 1].values
            sed = astSED.SED(wavelength=wl, flux=fl)
            uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
            vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
            uv_list.append(uv)
            vj_list.append(vj)
        uv_colours = uv_list
        vj_colours = vj_list
        uv_mean.append(np.mean(uv_colours))
        vj_mean.append(np.mean(vj_colours))
        
        # Save these for plotting later
        if(j == 0):
            type1agn_uv.append(uv_colours)
            type1agn_vj.append(vj_colours)
        
    plt.scatter(vj_mean, uv_mean, c=alpha, s=10, label="")
        
    # add the connecting line
    plt.plot(vj_mean, uv_mean, alpha=0.5, color='black')
    
    # Add text to the first galaxy with no contribution of AGN, showing which SWIRE template is being used
    plt.text(vj_mean[0], uv_mean[0], 'Mean UVJ P', fontsize=12)
       
    plt.ylabel('U - V')
    plt.xlabel('V - J')
    plt.title("Restframe UVJ Colours of "+ str(SKIRTOR_PARAMS['i'][n]) + "% Inclincation AGN Composites")
    plt.xlim([-0.5, 2.2])
    plt.ylim([0, 2.5])
    plt.colorbar().set_label('AGN Contribution')
        
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
    
    # Add a textbox with the model params
    plt.text(-0.4, 0.1, f'Tau: {tau}, p: {p}, q: {q}, oa: {oa}, rr: {rr}, i: {SKIRTOR_PARAMS["i"][n]}', fontsize=12)

    # Save to outputs folder
    #plt.savefig('outputs/UVJPlots/meanposition_composite_full_'+str(SKIRTOR_PARAMS['i'][n]) +'_i_AGNUVJ.png', bbox_inches='tight', pad_inches=0.1, facecolor='white')

    plt.show()


# In[518]:


# To double check that the outputs are okay, plot the swire templates on UVJ space without any changes
templates, template_names = read_swire_templates(swire_folderpath)

# Now we have a list of composite SEDs we can attempt to convert these into restframe UVJ colours and ultimately plot these on a UVJ diagram
colour_list = []
uv_list = []
vj_list = []

print(range(len(templates)))

for n in range(len(templates)):
    
    # use the wavelength and flux of the sed
    wl = templates[n].iloc[:, 0].values
    fl = templates[n].iloc[:, 1].values
    

    # create an SED object containing the SED of the galaxy
    # in addition to this use the relevant wavelength and flux
    sed = astSED.SED(wavelength=wl, flux=fl) # z = 0.0 as these are restframe SEDs

    # Using the astSED library calculate the UVJ colours using the U, V, and J passbands. 
    # We will use the AB magnitude system
    uv = astSED.SED.calcColour(sed, pb_U, pb_V, magType='AB')
    vj = astSED.SED.calcColour(sed, pb_V, pb_J, magType='AB')
    
    uv_list.append(uv)
    vj_list.append(vj)


# In[519]:


plt.figure(figsize=(10, 10))


        
        # Plot a connecting line
        #if j != 0:
            #plt.plot([vj[i][j-1], vj[i][j]], [uv[i][j-1], uv[i][j]], color='black', alpha=alpha_values[j])
plt.scatter(vj_list, uv_list, c='r', s=10)
uv_cols = []
vj_cols = []
    # Plotting a connecting line only between the first and last point of a particular composite
# add names
for i in range(len(template_names)):
    plt.text(vj_list[i], uv_list[i], template_names[i], fontsize=12)
plt.colorbar().set_label('AGN Contribution')
plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours of AGN Composites")
plt.xlim([-0.5, 2.2])
plt.ylim([0, 2.5])
#plt.colorbar().set_label('AGN Contribution')
    
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




