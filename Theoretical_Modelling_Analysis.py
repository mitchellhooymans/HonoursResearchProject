#!/usr/bin/env python
# coding: utf-8

# # Theoretical Modelling Analysis
# This script will be used to generate plots for my thesis, in particular I'll generate each of the plots i need for the theoretical modelling analysis. This should be versatile enough to work for the ZFOURGE observational (semi-empirical) models, and the pure theoretical models.

# In[365]:


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


# In[366]:


# Choose if you want to save the plots
save_figures = True
fig_size_params = (10, 6)

# Alternative params
fig_size_params = (7, 5)


# In[367]:


# We would like to investigate the effects of the different filters.
# Read in the dataframe

template_set_name = 'Brown'
agn_model_name = 'Type2AGN'

composite_choice = f'{template_set_name}_theoretical_composite_fluxes_{agn_model_name}'

# read in the dataframe and ensure that the unamed: 0 column is the index
composite_choice = f'{template_set_name}_theoretical_composite_fluxes_{agn_model_name}'
composite_fluxes = pd.read_csv(f'outputs\composite_seds\{composite_choice}.csv', index_col=0)   
# Reset the index
composite_fluxes.reset_index(drop=True, inplace=True)
# Drop the first col
#composite_fluxes = composite_fluxes.drop(columns=['Unnamed: 0.1'])


# In[368]:


# Check outputs
composite_fluxes

# Check index column 
len(composite_fluxes.index.unique())


# In[369]:


# Check which IRAC filters are available (whatever begins with IRAC)
irac_filters = [col for col in composite_fluxes.columns if col.startswith('IRAC')]
print(irac_filters)


# In[370]:


# Check for each entry of a particular filter, how many variations of it there are i.e U_0, U_10
# Allow us to have a robust way of investigating the effects of the different alpha values automatically, without recoding the filter names/alpha values

# Get the filters
filters = composite_fluxes.columns[2:]

# For the first filter, see how many variations there are
filter_choice = filters[0].split('_')[0]


# Get the variations
filter_variations = [filter for filter in filters if filter_choice in filter]

# add the alpha values to a new array
alpha_values = [int(filter.split('_')[1]) for filter in filter_variations]

# Print the alpha values
print(alpha_values)


# ## UVJ
# ### Plot
# The code below will generate a subplot for the UVJ diagrams of the theoretical models. 

# In[371]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.path as mpath

# --- Customize these parameters ---
fig_size_params = (14, 9)  # Adjust figure size if needed
font_size = 11  # Adjust text size if needed

# --- Assuming you have your data in a DataFrame called 'composite_fluxes' ---
# Example alpha values (replace with your actual values)
alpha_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 

spacing = len(alpha_values)
colours_theme = plt.cm.viridis(np.linspace(0, 1, spacing))

# Define paths for selections (do this once, outside the loop)
path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]
path_quiescent_obj = mpath.Path(path_quiescent)
path_sf_obj = mpath.Path(path_sf)
path_sfd_obj = mpath.Path(path_sfd)

# Galaxy Fractions - Quiescent, Star-forming, Dusty for each alpha value
galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []}
fraction_errors = {'Star-forming': [], 'Quiescent': [], 'Dusty': []}
num_galaxies = len(composite_fluxes[composite_fluxes['z'] == 0])

fig, axs = plt.subplots(3, 4, figsize=fig_size_params, sharex=True, sharey=True)
for n in range(len(alpha_values)):
    # Get the filter magnitudes
    U_mag = composite_fluxes[f'U_{alpha_values[n]}']
    V_mag = composite_fluxes[f'V_{alpha_values[n]}']
    J_mag = composite_fluxes[f'J_{alpha_values[n]}']
    
    # Create the colours
    uv = U_mag - V_mag
    vj = V_mag - J_mag
    
    # Restframe colours
    uv = uv[composite_fluxes['z'] == 0]
    vj = vj[composite_fluxes['z'] == 0]
    
    axs[n//4, n%4].scatter(vj, uv, c='blue', s=5, alpha=0.5)
    
    axs[n//4, n%4].set_xlim(-0.5, 2.2)
    axs[n//4, n%4].set_ylim(0, 2.5)
    
    # Plot AGN Contribution in the title 
    axs[n//4, n%4].set_title(rf'$\alpha$ = {alpha_values[n]}%', fontsize=font_size)
    
    # Add patches for selections
    axs[n//4, n%4].add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
    axs[n//4, n%4].add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
    axs[n//4, n%4].add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))
    
    # Add vertical line
    axs[n//4, n%4].axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)
    
    # Add axis labels
    if n//4 == 2:
        axs[n//4, n%4].set_xlabel("V - J", fontsize=font_size)
    if n%4 == 0:
        axs[n//4, n%4].set_ylabel("U - V", fontsize=font_size)
        
    # Create a DataFrame with just the uv and vj columns
    uvj_data = pd.DataFrame({'vj': vj, 'uv': uv})

    # Perform the selection
    quiescent_seds = uvj_data[path_quiescent_obj.contains_points(uvj_data.values)]
    sf_seds = uvj_data[path_sf_obj.contains_points(uvj_data.values)]
    sfd_seds = uvj_data[path_sfd_obj.contains_points(uvj_data.values)]

    # Calculate the fractions
    quiescent_fraction = len(quiescent_seds) / num_galaxies
    sf_fraction = len(sf_seds) / num_galaxies
    sfd_fraction = len(sfd_seds) / num_galaxies
    
    # Store the fractions
    galaxy_fractions['Quiescent'].append(quiescent_fraction)
    galaxy_fractions['Star-forming'].append(sf_fraction)
    galaxy_fractions['Dusty'].append(sfd_fraction)

    # --- Calculate and store errors using Adjusted Wald ---
    for category, fraction in zip(['Quiescent', 'Star-forming', 'Dusty'], 
                                  [quiescent_fraction, sf_fraction, sfd_fraction]):
        k = fraction * num_galaxies
        k_adj = k + 0.5
        n_adj = num_galaxies + 1
        p_adj = k_adj / n_adj
        se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
        margin_of_error = 1.96 * se 
        fraction_errors[category].append(margin_of_error)

    # Plot the fractions with errors in the corner of each section
    
    axs[n//4, n%4].text(0.05, 0.8, f'{quiescent_fraction:.2f} ± {fraction_errors["Quiescent"][-1]:.2f}', transform=axs[n//4, n%4].transAxes, color='k', fontsize=font_size)
    axs[n//4, n%4].text(0.05, 0.35, f'{sf_fraction:.2f} ± {fraction_errors["Star-forming"][-1]:.2f}', transform=axs[n//4, n%4].transAxes, color='k', fontsize=font_size)
    axs[n//4, n%4].text(0.65, 0.05, f'{sfd_fraction:.2f} ± {fraction_errors["Dusty"][-1]:.2f}', transform=axs[n//4, n%4].transAxes, color='k', fontsize=font_size)  

# --- Plot Colour Evolution ---
for m in range(len(alpha_values)):

    # Get the filter magnitudes
    U_mag = composite_fluxes[f'U_{alpha_values[m]}']
    V_mag = composite_fluxes[f'V_{alpha_values[m]}']
    J_mag = composite_fluxes[f'J_{alpha_values[m]}']
    
    # Create the colours
    uv = U_mag - V_mag
    vj = V_mag - J_mag
    
    # Recall that this is only a restframe diagram, so only look at the restframe colours
    uv = uv[composite_fluxes['z'] == 0]
    vj = vj[composite_fluxes['z'] == 0]
    
    # Plot scatter
    axs[2, 3].scatter(vj, uv, c=colours_theme[m], s=10)
    
    
# Add patches for selections
axs[2, 3].add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
axs[2, 3].add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
axs[2, 3].add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

# Add vertical line
axs[2, 3].axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

# Add axis labels
axs[2, 3].set_xlabel("V - J", fontsize=font_size)

# AGN Colour Evolution title
axs[2, 3].set_title("Colour Evolution", fontsize=font_size)

# Plot colorbar for AGN contribution
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
cbar = plt.colorbar(sm, ax=axs[2, 3])

# Name the colorbar
cbar.set_label('AGN Contribution', fontsize=font_size)

# Make the subplots closer together
plt.subplots_adjust(wspace=0.05, hspace=0.3)

# Save the output
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UVJ_evolution_{agn_model_name}_{template_set_name}.png', dpi=600, bbox_inches='tight')

# Show the output
plt.show()


# In[372]:



spacing = len(alpha_values)
colours_theme = plt.cm.viridis(np.linspace(0, 1, spacing))

# Galaxy Fractions - Quiescent, Star-forming, Dusty for each alpha value
galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
num_galaxies = len(composite_fluxes[composite_fluxes['z'] == 0]) # number of galaxies in the composite_flux in the rest frame

fig, axs = plt.subplots(3, 4, figsize=fig_size_params, sharex=True, sharey=True)
for n in range(len(alpha_values)):
    # Get the filter magnitudes
    U_mag = composite_fluxes[f'U_{alpha_values[n]}']
    V_mag = composite_fluxes[f'V_{alpha_values[n]}']
    J_mag = composite_fluxes[f'J_{alpha_values[n]}']
    
    # Create the colours
    uv = U_mag - V_mag
    vj = V_mag - J_mag
    
    # Recall that this is only a restframe diagram, so only look at the restframe colours
    uv = uv[composite_fluxes['z'] == 0]
    vj = vj[composite_fluxes['z'] == 0]
    
    axs[n//4, n%4].scatter(vj, uv, c='blue', s=5, alpha=0.5)
    
    axs[n//4, n%4].set_xlim(-0.5, 2.2)
    axs[n//4, n%4].set_ylim(0, 2.5)
    
    # Plot AGN Contribution in the title 
    # alpha needs to be the alpha symbol
    #alpha_symbol = r'$\alpha$'
    axs[n//4, n%4].set_title(rf'$\alpha$ = {alpha_values[n]}%')
    
    # Define paths for selections
    path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
    path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
    path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]
    
    # Add patches for selections
    axs[n//4, n%4].add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
    axs[n//4, n%4].add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
    axs[n//4, n%4].add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))
    
    # Add vertical line
    axs[n//4, n%4].axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)
    
    # Add axis labels
    if n//4 == 2:
        axs[n//4, n%4].set_xlabel("V - J")
    if n%4 == 0:
        axs[n//4, n%4].set_ylabel("U - V")
        
    
    # We will obtain the Star-forming, Quiescent, and Dusty Fractions for this particular alpha value     
    # Obtain the number of galaxies with sf region dictated by the sf path
    # Create Path objects from your path coordinates (do this once, outside the loop)
    path_quiescent_obj = mpath.Path(path_quiescent)
    path_sf_obj = mpath.Path(path_sf)
    path_sfd_obj = mpath.Path(path_sfd)

    # Create a DataFrame with just the uv and vj columns for easier selection
    uvj_data = pd.DataFrame({'vj': vj, 'uv': uv})

    # Perform the selection
    quiescent_seds = uvj_data[path_quiescent_obj.contains_points(uvj_data.values)]
    sf_seds = uvj_data[path_sf_obj.contains_points(uvj_data.values)]
    sfd_seds = uvj_data[path_sfd_obj.contains_points(uvj_data.values)]

    # Calculate the fractions
    galaxy_fractions['Quiescent'].append(len(quiescent_seds) / num_galaxies)
    galaxy_fractions['Star-forming'].append(len(sf_seds) / num_galaxies)
    galaxy_fractions['Dusty'].append(len(sfd_seds) / num_galaxies)
    
    # Calculate the fractions
    quiescent_fraction = len(quiescent_seds)/num_galaxies
    sf_fraction = len(sf_seds)/num_galaxies
    sfd_fraction = len(sfd_seds)/num_galaxies
    
    
        # Plot the fractions in the corner of each section on the UVJ
    axs[n//4, n%4].text(0.05, 0.8, f'{quiescent_fraction:.2f}', transform=axs[n//4, n%4].transAxes, color='k')
    axs[n//4, n%4].text(0.05, 0.35, f'{sf_fraction:.2f}', transform=axs[n//4, n%4].transAxes, color='k')
    axs[n//4, n%4].text(0.70, 0.05, f'{sfd_fraction:.2f}', transform=axs[n//4, n%4].transAxes, color='k')
        

    
for m in range(len(alpha_values)):
    # Get the filter magnitudes
    U_mag = composite_fluxes[f'U_{alpha_values[m]}']
    V_mag = composite_fluxes[f'V_{alpha_values[m]}']
    J_mag = composite_fluxes[f'J_{alpha_values[m]}']
    
    # Create the colours
    uv = U_mag - V_mag
    vj = V_mag - J_mag
    
    # Recall that this is only a restframe diagram, so only look at the restframe colours
    uv = uv[composite_fluxes['z'] == 0]
    vj = vj[composite_fluxes['z'] == 0]
    
    # Plot scatter
    axs[2, 3].scatter(vj, uv, c=colours_theme[m], s=10)
    
    
# Add patches for selections
axs[2, 3].add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
axs[2, 3].add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
axs[2, 3].add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

# Add vertical line
axs[2, 3].axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

# Add axis labels
axs[2, 3].set_xlabel("V - J")

# AGN Colour Evolution title
axs[2, 3].set_title("Colour Evolution")



# Plot colorbar for AGN contribution
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
cbar = plt.colorbar(sm, ax=axs[2, 3])

# Name the colorbar
cbar.set_label('AGN Contribution')

    
# Make the subplots closer together
plt.subplots_adjust(wspace=0.05, hspace=0.3)
    

# # Save the output
# if save_figures:
#     plt.savefig(f'outputs/ThesisPlots/UVJ_evolution_{agn_model_name}_{template_set_name}.png', dpi=300)

# # Show the output
plt.show()


# Use the adjusted wald test to determine if the fractions are significantly different
# We will compare the fractions of the first alpha value to the last alpha value
# We will use the adjusted wald test to determine if the fractions are significantly different

# # use adjust walf
# from statsmodels.stats.proportion import proportions_ztest

# # Get the fractions
# quiescent_fractions = galaxy_fractions['Quiescent']
# starforming_fractions = galaxy_fractions['Star-forming']
# dusty_fractions = galaxy_fractions['Dusty']

# # Get the first and last alpha values
# first_alpha = alpha_values[0]
# last_alpha = alpha_values[-1]

# # Get the fractions for the first and last alpha values
# quiescent_fractions_first = quiescent_fractions[0]
# starforming_fractions_first = starforming_fractions[0]

# quiescent_fractions_last = quiescent_fractions[-1]
# starforming_fractions_last = starforming_fractions[-1]

# # Perform the test
# # Quiescent Fractions
# count = np.array([quiescent_fractions_first*num_galaxies, quiescent_fractions_last*num_galaxies])
# nobs = np.array([num_galaxies, num_galaxies])
# stat, pval = proportions_ztest(count, nobs)

# print(f'Quiescent Fractions: p-value = {pval}')

# # Star-forming Fractions
# count = np.array([starforming_fractions_first*num_galaxies, starforming_fractions_last*num_galaxies])
# nobs = np.array([num_galaxies, num_galaxies])
# stat, pval = proportions_ztest(count, nobs)

# print(f'Star-forming Fractions: p-value = {pval}')

# # Dusty Fractions
# count = np.array([dusty_fractions[0]*num_galaxies, dusty_fractions[-1]*num_galaxies])
# nobs = np.array([num_galaxies, num_galaxies])
# stat, pval = proportions_ztest(count, nobs)

# print(f'Dusty Fractions: p-value = {pval}')


# # Calculate a confidence interval for the fractions
# from statsmodels.stats.proportion import proportion_confint

# # Quiescent Fractions
# quiescent_ci = proportion_confint(quiescent_fractions_first*num_galaxies, num_galaxies, alpha=0.05, method='normal')
# starforming_ci = proportion_confint(starforming_fractions_first*num_galaxies, num_galaxies, alpha=0.05, method='normal')
# dusty_ci = proportion_confint(dusty_fractions[0]*num_galaxies, num_galaxies, alpha=0.05, method='normal')

# # Print the confidence intervals
# print(f'Quiescent Fractions: {quiescent_ci}')
# print(f'Star-forming Fractions: {starforming_ci}')
# print(f'Dusty Fractions: {dusty_ci}')


# In[ ]:





# In[373]:


# Check the fractions 
galaxy_fractions

fraction_figsize = (3, 4)
# Plot the fractions
fig, ax = plt.subplots(figsize=fraction_figsize)
ax.plot(alpha_values, galaxy_fractions['Quiescent'], label='Quiescent', color='red')
ax.plot(alpha_values, galaxy_fractions['Star-forming'], label='Star-forming', color='blue')
ax.plot(alpha_values, galaxy_fractions['Dusty'], label='Dusty', color='Green')

# Add labels
ax.set_xlabel(f'AGN Contribution (%)')
ax.set_ylabel('Fraction')
#ax.set_title('Fraction of Galaxies in UVJ Diagram')
ax.legend()

# Save the output
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UVJ_galfractions_{agn_model_name}_{template_set_name}.png')
    
# Show the output
plt.show()

# Also convert the fractions to a dataframe
galaxy_fractions_df = pd.DataFrame(galaxy_fractions, index=alpha_values)



if save_figures:
    galaxy_fractions_df.to_csv(f'outputs/ThesisPlots/UVJ_fractions_{agn_model_name}_{template_set_name}.csv')

galaxy_fractions_df


# ### Metrics
# These will be used to analyze the results from the UVJ diagram above

# ## ugr
# ### Plot

# In[374]:


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import matplotlib.path as mpath

# # --- Customize these parameters ---
# fig_size_params = (16, 12)  # Adjust figure size if needed
# spacing = len(alpha_values)
# colours_theme = plt.cm.viridis(np.linspace(0, 1, spacing))





# # --- UGR Selection Statistics ---

# selection_statistics = {'Missed Selection': [], 'Misidentification': [], 'Correct Identification': [], 'Correct Non-UGR Identification': []}
# selection_errors = {'Missed Selection': [], 'Misidentification': [], 'Correct Identification': [], 'Correct Non-UGR Identification': []}
# completeness_stats = {'Completeness': [] }
# completeness_errors = []
# num_galaxies = len(composite_fluxes)

# marker_size_ugr=5

# # ugr Diagram 
# fig, axs = plt.subplots(3, 4, figsize=fig_size_params, sharex=True, sharey=True)
# for n in range(len(alpha_values)):
#     # Get the u, g, and r magnitudes for the specific alpha value
#     u_col_name = f'u_{int(round(alpha_values[n], 2))}'
#     g_col_name = f'g_{int(round(alpha_values[n], 2))}'
#     r_col_name = f'r_{int(round(alpha_values[n], 2))}'
    
#     # Create the colours for the UGR diagram
#     ug_specific_alpha_colours = composite_fluxes[u_col_name] - composite_fluxes[g_col_name]
#     gr_specific_alpha_colours = composite_fluxes[g_col_name] - composite_fluxes[r_col_name]

#     # Scatter plots with color-coded redshift ranges
#     axs[n // 4, n % 4].scatter(gr_specific_alpha_colours[composite_fluxes['z'] > 3.5], 
#                               ug_specific_alpha_colours[composite_fluxes['z'] > 3.5], 
#                               c="c", s=marker_size_ugr, label="z > 3.5", alpha=0.5)

#     axs[n // 4, n % 4].scatter(gr_specific_alpha_colours[(composite_fluxes['z'] > 2.6) & (composite_fluxes['z'] <= 3.5)], 
#                               ug_specific_alpha_colours[(composite_fluxes['z'] > 2.6) & (composite_fluxes['z'] <= 3.5)], 
#                               c="y", s=marker_size_ugr, label="2.6 < z < 3.5", alpha=0.5)

#     axs[n // 4, n % 4].scatter(gr_specific_alpha_colours[composite_fluxes['z'] < 2.6], 
#                               ug_specific_alpha_colours[composite_fluxes['z'] < 2.6], 
#                               c="m", s=marker_size_ugr, label="z < 2.6", alpha=0.5)
    
    
#     # Set the AGN Contribution title 
#     axs[n//4, n%4].set_title(rf'<span class="math-inline">\\alpha</span> = {alpha_values[n]}%')
    
#     # Set the axis labels
#     if n//4 == 2:
#         axs[n//4, n%4].set_xlabel("G-R")
#     if n%4 == 0:
#         axs[n//4, n%4].set_ylabel("U-G")

#     # UGR selection criteria
#     U_rule = [[1.2,9], [1.2,2.2], [0.6,1.6], [-3,1.6], [-3,9]]
#     axs[n//4, n%4].add_patch(plt.Polygon(U_rule, closed=True, fill=True, facecolor=(1,0,0,0.05), edgecolor=(0,0,0,1), linewidth=2, linestyle='solid'))

#     # Create Path object
#     path = mpath.Path(U_rule)
    
#     # Create a DataFrame with just the u-g and g-r columns
#     ugr_data = pd.DataFrame({'gr': gr_specific_alpha_colours, 'ug': ug_specific_alpha_colours})
    
#     # Append the associated redshifts to the ugr data
#     redshifts = composite_fluxes['z']
    
#     # Perform the selection
#     selected_seds = ugr_data[path.contains_points(ugr_data.values)]
#     non_selected_seds = ugr_data[~path.contains_points(ugr_data.values)]
    
#     # Using the id of the selected seds, find the associated redshift values
#     redshifts_selected = redshifts[selected_seds.index]
#     redshifts_non_selected = redshifts[non_selected_seds.index]
    
#     # Append the redshifts to the selected seds
#     selected_seds['redshift'] = redshifts_selected
#     non_selected_seds['redshift'] = redshifts_non_selected
    
#     # Calculate the fractions
#     correct_ugr_selection = selected_seds[(selected_seds['redshift'] >= 2.6) & (selected_seds['redshift'] <= 3.5)]
#     correct_nonugr_selection = non_selected_seds[(non_selected_seds['redshift'] < 2.6) | (non_selected_seds['redshift'] > 3.5)]
    
#     misidentification = selected_seds[(selected_seds['redshift'] < 2.6) | (selected_seds['redshift'] > 3.5)]
#     missed_selection = non_selected_seds[(non_selected_seds['redshift'] >= 2.6) & (non_selected_seds['redshift'] <= 3.5)]

#     # Append the numbers
#     selection_statistics['Correct Identification'].append(len(correct_ugr_selection))
#     selection_statistics['Correct Non-UGR Identification'].append(len(correct_nonugr_selection))
#     selection_statistics['Misidentification'].append(len(misidentification))
#     selection_statistics['Missed Selection'].append(len(missed_selection))
    
#     # Calculate completeness
#     completeness = len(correct_ugr_selection) / (len(correct_ugr_selection) + len(missed_selection))
#     completeness_stats['Completeness'].append(completeness)

#     # --- Calculate and store errors using Adjusted Wald ---
#     for category, count in selection_statistics.items():
#         k = count[n] 
#         k_adj = k + 0.5
#         n_adj = num_galaxies + 1
#         p_adj = k_adj / n_adj
#         se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
#         margin_of_error = 1.96 * se
#         selection_errors[category].append(margin_of_error)

#     # Calculate Adjusted Wald error for completeness
#     k = completeness_stats['Completeness'][n] * num_galaxies
#     k_adj = k + 0.5
#     n_adj = num_galaxies + 1
#     p_adj = k_adj / n_adj
#     se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
#     margin_of_error = 1.96 * se
#     completeness_errors.append(margin_of_error)

#     # Plot the completeness with error within the selection region (with larger font size)
#     axs[n//4, n%4].text(0.05, 0.8, f'{completeness:.2f} ± {completeness_errors[-1]:.2f}', transform=axs[n//4, n%4].transAxes, color='k')

# # change axis limits
# axs[0, 0].set_xlim(-3, 1.6)


# # Make the subplots closer together
# plt.subplots_adjust(wspace=0.05, hspace=0.12)

# # # Save the output
# # if save_figures:
# #     plt.savefig(f'outputs/ThesisPlots/UGR_evolution_{agn_model_name}_{template_set_name}.png', dpi=300, bbox_inches='tight')
    
# # Show the output
# plt.show()


# In[375]:


# We can do this a few ways
# We can either do a similar plot to above
# or we can do similar to sam's thesis work

# Essentially we want to quantify the values for missed selection, misidentifications, and correct identifications
# and see how those change with the alpha values

selection_statistics = {'Missed Selection': [], 'Misidentification': [], 'Correct Identification': [], 'Correct Non-UGR Identification': []}
selection_errors = {'Missed Selection': [], 'Misidentification': [], 'Correct Identification': [], 'Correct Non-UGR Identification': []}
completeness_stats = {'Completeness': [] }
completeness_errors = []
# Realistically we are only interested in the correct identification and missed selection, but we will include all.
num_galaxies = len(composite_fluxes) # Because we consider all redshit ranges

# Completeness and contamination table
completeness_stats = {'Completeness': [] }

# The true values exsist for each of these regions - as we have developed the code to see if a galaxy is in a particular region
# and we know the redshift (as we artifically redshifted the galaxy

print(composite_fluxes['z'].unique())

marker_size_ugr=5

# ugr Diagram 
fig, axs = plt.subplots(3, 4, figsize=fig_size_params, sharex=True, sharey=True)
for n in range(len(alpha_values)):
    # Get the u, g, and r magnitudes for the specific alpha value
    u_col_name = f'u_{int(round(alpha_values[n], 2))}'
    g_col_name = f'g_{int(round(alpha_values[n], 2))}'
    r_col_name = f'r_{int(round(alpha_values[n], 2))}'
    
    # Create the colours for the UGR diagram
    ug_specific_alpha_colours = composite_fluxes[u_col_name] - composite_fluxes[g_col_name]
    gr_specific_alpha_colours = composite_fluxes[g_col_name] - composite_fluxes[r_col_name]

    # Scatter plots with color-coded redshift ranges
    axs[n // 4, n % 4].scatter(gr_specific_alpha_colours[composite_fluxes['z'] > 3.5], 
                               ug_specific_alpha_colours[composite_fluxes['z'] > 3.5], 
                               c="c", s=marker_size_ugr, label="z > 3.5", alpha=0.5)

    axs[n // 4, n % 4].scatter(gr_specific_alpha_colours[(composite_fluxes['z'] > 2.6) & (composite_fluxes['z'] <= 3.5)], 
                               ug_specific_alpha_colours[(composite_fluxes['z'] > 2.6) & (composite_fluxes['z'] <= 3.5)], 
                               c="y", s=marker_size_ugr, label="2.6 < z < 3.5", alpha=0.5)

    axs[n // 4, n % 4].scatter(gr_specific_alpha_colours[composite_fluxes['z'] < 2.6], 
                               ug_specific_alpha_colours[composite_fluxes['z'] < 2.6], 
                               c="m", s=marker_size_ugr, label="z < 2.6", alpha=0.5)
    
    
    # Set the AGN Contribution title 
    axs[n//4, n%4].set_title(rf'$\alpha$ = {alpha_values[n]}%')
    
    plt.xlabel("")
    plt.ylabel("")
    #plt.title("UGR Diagram for SED Templates at Different Redshifts(0 < z < 4) for alpha = {}".format(alphas[n]))

      
    # Set the axis labels
    if n//4 == 2:
        axs[n//4, n%4].set_xlabel("G-R")
    if n%4 == 0:
        axs[n//4, n%4].set_ylabel("U-G")
    
    
    # add the alpha value information onto the plot somewhere
    #axs[n//4, n%4].text(0.5, 0.9, f"alpha = {round(alpha[n], 2)}", horizontalalignment='center', verticalalignment='center', transform=axs[n//4, n%4].transAxes)

    # Additionally we can plot the UGR selection criteria on the UGR diagram
    U_rule = [[1.2,9], [1.2,2.2], [0.6,1.6], [-3,1.6], [-3,9]]
    axs[n//4, n%4].add_patch(plt.Polygon(U_rule, closed=True, fill=True, facecolor=(1,0,0,0.05), edgecolor=(0,0,0,1), linewidth=2, linestyle='solid')) # This looks like the correct U dropout technique
    
    
    # The path here is the UGR selection criteria, we can find the selected points in this particular population using the path
    # Create Path objects from your path coordinates (do this once, outside the loop)
    path = mpath.Path(U_rule)
    
    # Create a DataFrame with just the u-g and g-r columns for easier selection
    ugr_data = pd.DataFrame({'gr': gr_specific_alpha_colours, 'ug': ug_specific_alpha_colours})
    
    # append the associated redshifts to the ugr data
    redshifts = composite_fluxes['z']
    
    
    
    # Perform the selection
    selected_seds = ugr_data[path.contains_points(ugr_data.values)]
    non_selected_seds = ugr_data[~path.contains_points(ugr_data.values)]
    
    # Using the id of the selected seds, we can find the associated redshift values
    redshifts_selected = redshifts[selected_seds.index]
    redshifts_non_selected = redshifts[non_selected_seds.index]
    
    # We can then append the redshifts to the selected seds
    selected_seds['redshift'] = redshifts_selected
    non_selected_seds['redshift'] = redshifts_non_selected
    
    # using the above data we can calculate the fractions
    correct_ugr_selection = selected_seds[(selected_seds['redshift'] >= 2.6) & (selected_seds['redshift'] <= 3.5)]
    correct_nonugr_selection = non_selected_seds[(non_selected_seds['redshift'] < 2.6) | (non_selected_seds['redshift'] > 3.5)]
    
    # Combine the above to be a correct identification
    #correct_identification = correct_ugr_selection.append(correct_nonugr_selection)
    
    # Misidentification - a selected sed that was not in the redshift range we were after - incorrect redshift range
    misidentification = selected_seds[(selected_seds['redshift'] < 2.6) | (selected_seds['redshift'] > 3.5)]
    
    # Missed Selection - a non-selected sed SED that was in the redshift range we were after, something we missed.
    missed_selection = non_selected_seds[(non_selected_seds['redshift'] >= 2.6) & (non_selected_seds['redshift'] <= 3.5)]
    
    
    # Print the numbers without the fractions
    print(f"Correct Identification: {len(correct_ugr_selection)}")
    print(f"Correct Non-UGR Identification: {len(correct_nonugr_selection)}")
    print(f"Misidentification: {len(misidentification)}")
    print(f"Missed Selection: {len(missed_selection)}")
    
    
    
    
    # # Calculate the fractions
    # selection_statistics['Correct Identification'].append(len(correct_ugr_selection) / num_galaxies)
    # selection_statistics['Correct Non-UGR Identification'].append(len(correct_nonugr_selection) / num_galaxies)
    # selection_statistics['Misidentification'].append(len(misidentification) / num_galaxies)
    # selection_statistics['Missed Selection'].append(len(missed_selection) / num_galaxies)
    
    # Instead of fractions, we can just append the numbers
    selection_statistics['Correct Identification'].append(len(correct_ugr_selection))
    selection_statistics['Correct Non-UGR Identification'].append(len(correct_nonugr_selection))
    selection_statistics['Misidentification'].append(len(misidentification))
    selection_statistics['Missed Selection'].append(len(missed_selection))
    
    # We can also quantify the completeness and contamination of the selection
    # Completeness = Correct Identification / (Correct Identification + Missed Selection)
    # Contamination = Misidentification / (Correct Identification + Misidentification)
    
    completeness = len(correct_ugr_selection) / (len(correct_ugr_selection) + len(missed_selection))
    #contamination = len(misidentification) / (len(correct_ugr_selection) + len(misidentification))
    
    # Append the values
    completeness_stats['Completeness'].append(completeness)
    #completeness_stats['Contamination'].append(contamination)
    
    
    
    # --- Calculate and store errors using Adjusted Wald ---
    for category, count in selection_statistics.items():
        k = count[n] 
        k_adj = k + 0.5
        n_adj = num_galaxies + 1
        p_adj = k_adj / n_adj
        se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
        margin_of_error = 1.96 * se
        selection_errors[category].append(margin_of_error)

    # Calculate Adjusted Wald error for completeness
    k = completeness_stats['Completeness'][n] * num_galaxies
    k_adj = k + 0.5
    n_adj = num_galaxies + 1
    p_adj = k_adj / n_adj
    se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
    margin_of_error = 1.96 * se
    completeness_errors.append(margin_of_error)

    # Plot the completeness with error within the selection region (with larger font size)
    axs[n//4, n%4].text(0.05, 0.8, f'{completeness:.2f} ± {completeness_errors[-1]:.2f}', transform=axs[n//4, n%4].transAxes, color='k')
    
    # # Plot the completeness within the selection region
    # axs[n//4, n%4].text(0.05, 0.8, f'{completeness:.2f}', transform=axs[n//4, n%4].transAxes, color='k')
    
    
    axs[n//4, n%4].set_xlim(-1, 4)
    axs[n//4, n%4].set_ylim(-1, 8.5)
    
for m in range(len(alpha_values)):
    
    
    
    # Recall that this is only a restframe diagram, so only look at the restframe colours
    u_col_name = f'u_{int(round(alpha_values[m], 2))}'
    g_col_name = f'g_{int(round(alpha_values[m], 2))}'
    r_col_name = f'r_{int(round(alpha_values[m], 2))}'
    
    # Create the colours for the UGR diagram
    ug_specific_alpha_colours = composite_fluxes[u_col_name] - composite_fluxes[g_col_name]
    gr_specific_alpha_colours = composite_fluxes[g_col_name] - composite_fluxes[r_col_name]
    
    # Scatter plots with color-coded redshift ranges
    axs[2, 3].scatter(gr_specific_alpha_colours, 
                               ug_specific_alpha_colours, 
                               c=colours_theme[m], s=10)


    
    
    # In addition we can plot a mean position for on each of the alpha plots. This mean value will be the mean of the ugr

# Additionally we can plot the UGR selection criteria on the UGR diagram
U_rule = [[1.2,9], [1.2,2.2], [0.6,1.6], [-3,1.6], [-3,9]]
axs[2, 3].add_patch(plt.Polygon(U_rule, closed=True, fill=True, facecolor=(1,0,0,0.05), edgecolor=(0,0,0,1), linewidth=2, linestyle='solid')) # This looks like the correct U dropout technique

axs[2, 3].set_xlim(-1, 4)
axs[2, 3].set_ylim(-1, 8.5)

axs[2, 3].set_xlabel("G-R")



# Add a colorbar for the AGN contribution
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
cbar = plt.colorbar(sm, ax=axs[2, 3])

# Add the colorbar label
cbar.set_label('AGN Contribution')

# add the final subplot labels
axs[2, 3].set_title("Colour Evolution")


# # Make all the plots touch

    
# Make the subplots closer together
plt.subplots_adjust(wspace=0.05, hspace=0.3)
    

if save_figures:
    plt.savefig(f'outputs/ThesisPlots/ugr_evolution_{agn_model_name}_{template_set_name}.png', dpi=300, bbox_inches='tight')
    
# Plot
plt.show()
    


# In[376]:


selection_statistics_df = pd.DataFrame(selection_statistics, index=alpha_values)

# Plot the fractions
fig, ax = plt.subplots(figsize=(3, 4))
ax.plot(alpha_values, selection_statistics['Correct Identification'], label='Correct Identification', color='blue')
ax.plot(alpha_values, selection_statistics['Misidentification'], label='Misidentification', color='red')
ax.plot(alpha_values, selection_statistics['Missed Selection'], label='Missed Selection', color='green')
ax.plot(alpha_values, selection_statistics['Correct Non-UGR Identification'], label='Correct Non-UGR Identification', color='purple')

# Add labels
ax.set_xlabel(f'AGN Contribution (%)')
ax.set_ylabel('Count')
#ax.set_title('Fraction of Galaxies in UGR Diagram')
# ax.legend()

# Save the output
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/ugr_selection_fractions_{agn_model_name}_{template_set_name}.png')
    
# Show the output
plt.show()

selection_statistics_df

# We would like to output the selection statistics to a csv file
if save_figures:
    selection_statistics_df.to_csv(f'outputs/ThesisPlots/ugr_selection_fractions_{agn_model_name}_{template_set_name}.csv')
    


# In[377]:


# Also plot the completeness and contamination statistics for each alpha
completeness_stats_df = pd.DataFrame(completeness_stats, index=alpha_values)

# add new column for completeness (convert to percentages)
completeness_stats_df['Completeness (%)'] = completeness_stats_df['Completeness'] * 100

# Plot the fractions
fig, ax = plt.subplots(figsize=(3, 4))

# Plot the completeness percentages
ax.plot(alpha_values, completeness_stats_df['Completeness (%)'], label='Completeness', color='blue')

# Add labels
ax.set_xlabel(f'AGN Contribution (%)')
ax.set_ylabel('Completeness (%)')
#ax.set_title('Fraction of Galaxies in UGR Diagram')
ax.legend()


# Set the ylim to 100
ax.set_ylim(0, 100)

# Save the output
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/ugr_completeness_contamination_{agn_model_name}_{template_set_name}.png')
    
# Show the output
plt.show()

completeness_stats_df

# change the index to a column named alpha
# Create new column for alpha
completeness_stats_df['AGN Contribution (%)'] = completeness_stats_df.index
completeness_stats_df.reset_index(inplace=False).copy()

# Save the completeness percentages and AGN contribution to a csv file
if save_figures:
    completeness_stats_df[['AGN Contribution (%)', 'Completeness (%)']].to_csv(f'outputs/ThesisPlots/ugr_completeness_Contribution_{agn_model_name}_{template_set_name}.csv')


# In[378]:


# # # For each of the alpha values, we can plot the counts

# fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

# # Plot the correct identification
# axs[0, 0].bar(alpha_values, selection_statistics['Correct Identification'], color='blue')
# axs[0, 0].set_title('Correct Identification')

# # Plot the misidentification
# axs[0, 1].bar(alpha_values, selection_statistics['Misidentification'], color='red')
# axs[0, 1].set_title('Misidentification')

# # Plot the missed selection
# axs[1, 0].bar(alpha_values, selection_statistics['Missed Selection'], color='green')
# axs[1, 0].set_title('Missed Selection')

# # Plot the correct non-ugr identification
# axs[1, 1].bar(alpha_values, selection_statistics['Correct Non-UGR Identification'], color='purple')
# axs[1, 1].set_title('Correct Non-UGR Identification')

# # Add labels
# for ax in axs.flat:
#     ax.set(xlabel='AGN Contribution (%)', ylabel='Counts')
    
# # Make the subplots closer together
# plt.subplots_adjust(wspace=0.05, hspace=0.3)

# # Save the output
# if save_figures:
#     plt.savefig(f'outputs/ThesisPlots/ugr_counts_{agn_model_name}_{template_set_name}.png')

    
# # Show the output
# plt.show()

# Plot


# ### Metrics

# ## IRAC 
# ### Plot
# 

# In[379]:


# we will need to create a list of dictionaries
completeness_values = []

lower_redshift_limit = 0
upper_redshift_limit = 0


for n in range(len(alpha_values)):
    # at this alpha value, get the coordoinates
    # Get the IRAC magnitudes
    f_36 = composite_fluxes[f'IRAC3.6_{alpha_values[n]}']
    f_45 = composite_fluxes[f'IRAC4.5_{alpha_values[n]}']
    f_58 = composite_fluxes[f'IRAC5.8_{alpha_values[n]}']
    f_80 = composite_fluxes[f'IRAC8.0_{alpha_values[n]}']

    # Create the colours for the IRAC diagram
    f_5836_specific_alpha_colours = np.log10(f_58/f_36)
    f_8045_specific_alpha_colours = np.log10(f_80/f_45)

    # Redefine x and y for the first alpha value to plot the wedge
    x = f_5836_specific_alpha_colours
    y = f_8045_specific_alpha_colours

    # Can change this to ensure we are only looking at the rest frame stuff
    x = x[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
    y = y[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]

    print(len(x))
    # # Reimplement the selection criteria
    lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)
    
    num_sources_selected = x[lacy_selection_condition].notna().sum()
    
    print(num_sources_selected)
    # get the number of sources that are selected by the lacy wedge
    # num_sources_selected = len()
    # print(num_sources_selected)
    # for the first value of alpha there will be no relevance as there are no AGN, 
    # but we can instead check how many of all sources were selected
    if n == 0:
        num_sources_all = len(x)
        completeness = num_sources_selected/num_sources_all
        completeness_values.append({'alpha': alpha_values[n], 'completeness': completeness})
    else:
        completeness = num_sources_selected/num_sources_all
        completeness_values.append({'alpha': alpha_values[n], 'completeness': completeness})
        
# Create a pd.DataFrame
completeness_df = pd.DataFrame(completeness_values)


# save
if save_figures:
    completeness_df.to_csv(f'outputs/ThesisPlots/IRAC_completeness_{agn_model_name}_{template_set_name}.csv')

completeness_df


# Calculate the adjust wald as before
# We can do this a few ways

# Calculate the adjusted Wald values
completeness_errors = []
for n in range(len(alpha_values)):
    completeness = completeness_values[n]['completeness']
    k = completeness * num_sources_all  # Number of "successes"
    k_adj = k + 0.5
    n_adj = num_sources_all + 1
    p_adj = k_adj / n_adj
    se = np.sqrt(p_adj * (1 - p_adj) / n_adj)
    margin_of_error = 1.96 * se  # For 95% confidence interval
    completeness_errors.append(margin_of_error)

# Add the errors to the DataFrame
completeness_df['completeness_error'] = completeness_errors


# In[380]:


# Choose the diagram min and max
xmax = 0.8
ymax = 1.2
xmin = -0.6
ymin = -0.8

# Can change this to ensure we are only looking at the rest frame stuff
upper_redshift_limit = 0
lower_redshift_limit = 0
# setup 
fig, axs = plt.subplots(3, 4, figsize=fig_size_params, sharex=True, sharey=True)
for n in range(len(alpha_values)):
    
    # Get the IRAC magnitudes
    f_36 = composite_fluxes[f'IRAC3.6_{alpha_values[n]}']
    f_45 = composite_fluxes[f'IRAC4.5_{alpha_values[n]}']
    f_58 = composite_fluxes[f'IRAC5.8_{alpha_values[n]}']
    f_80 = composite_fluxes[f'IRAC8.0_{alpha_values[n]}']
    
    # Create the colours for the IRAC diagram
    f_5836_specific_alpha_colours = np.log10(f_58/f_36)
    f_8045_specific_alpha_colours = np.log10(f_80/f_45)

    # Redefine x and y for the first alpha value to plot the wedge
    x = f_5836_specific_alpha_colours
    y = f_8045_specific_alpha_colours
    
    
    
    x = x[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
    y = y[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
    
    
    
    # Reimplement the selection criteria 
    lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)
    
    # Colour code for each alpha
    axs[n//4, n%4].scatter(x[lacy_selection_condition], y[lacy_selection_condition], c='blue', s=5, alpha=0.5)
    axs[n//4, n%4].scatter(x[~lacy_selection_condition], y[~lacy_selection_condition], c='grey', s=5, alpha=0.5)
    
    
    # Get the IRAC magnitudes
    f_36 = composite_fluxes[f'IRAC3.6_{alpha_values[0]}']
    f_45 = composite_fluxes[f'IRAC4.5_{alpha_values[0]}']
    f_58 = composite_fluxes[f'IRAC5.8_{alpha_values[0]}']
    f_80 = composite_fluxes[f'IRAC8.0_{alpha_values[0]}']
    
    # Create the colours for the IRAC diagram
    f_5836_specific_alpha_colours = np.log10(f_58/f_36)
    f_8045_specific_alpha_colours = np.log10(f_80/f_45)
    
    
    # Redefine x and y for the first alpha value to plot the wedge
    x = f_5836_specific_alpha_colours
    y = f_8045_specific_alpha_colours
    
    # # Ensure we are only looking at the rest frame stuff
    # x = x[composite_fluxes['z'] == 0]
    # y = y[composite_fluxes['z'] == 0]
      # Can change this to ensure we are only looking at the rest frame stuff
    x = x[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
    y = y[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
    
    # # Reimplement the selection criteria
    lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)
    wedge_vertices = [
        (xmax, -0.2),
        (-0.1, -0.2),
        (-0.1, 0.8*np.nanmin(x[lacy_selection_condition]) + 0.5 if np.any(lacy_selection_condition) else -0.2),  # Handle empty selection
        (((ymax-0.5)/0.8), ymax)
    ] 
    wedge_polygon = plt.Polygon(wedge_vertices, closed=False, edgecolor='r', facecolor=(1,0,0,0.05), linestyle='-', linewidth=2, label='Lacy Wedge')
    # Plot the polygon on the subplot
    axs[n//4, n%4].add_patch(wedge_polygon)
    
    # Add the completeness with error (larger font size)
    axs[n//4, n%4].text(0.5+0.10, 0.6, f'{completeness_df["completeness"][n]:.2f} ± {completeness_errors[n]:.2f}', 
                        transform=axs[n//4, n%4].transAxes, color='k', fontsize=12)

    
    
    
    # Set the AGN Contribution title
    axs[n//4, n%4].set_title(rf'$\alpha$ = {alpha_values[n]}%')
    
    # Set the axis labels
    if n//4 == 2:
        axs[n//4, n%4].set_xlabel("[5.8] - [3.6]")
    if n%4 == 0:
        axs[n//4, n%4].set_ylabel("[8.0] - [4.5]")
    
    # Set the axis limits
    axs[n//4, n%4].set_xlim(xmin, xmax)
    axs[n//4, n%4].set_ylim(ymin, ymax)
    
# We want an AGN evolution plot for
for m in range(len(alpha_values)):
    
    # Get the IRAC magnitudes
    f_36 = composite_fluxes[f'IRAC3.6_{alpha_values[m]}']
    f_45 = composite_fluxes[f'IRAC4.5_{alpha_values[m]}']
    f_58 = composite_fluxes[f'IRAC5.8_{alpha_values[m]}']
    f_80 = composite_fluxes[f'IRAC8.0_{alpha_values[m]}']
    
    # Create the colours for the IRAC diagram
    f_5836_specific_alpha_colours = np.log10(f_58/f_36)
    f_8045_specific_alpha_colours = np.log10(f_80/f_45)
    
    # Redefine x and y for the first alpha value to plot the wedge
    x = f_5836_specific_alpha_colours
    y = f_8045_specific_alpha_colours
    
    # Can change this to ensure we are only looking at the rest frame stuff
    x = x[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
    y = y[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
    
    # Reimplement the selection criteria 
    lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)
    
    # Colour code for each alpha
    axs[2, 3].scatter(x, y, c=colours_theme[m], s=10)
    
# Get the IRAC magnitudes
f_36 = composite_fluxes[f'IRAC3.6_{alpha_values[0]}']
f_45 = composite_fluxes[f'IRAC4.5_{alpha_values[0]}']
f_58 = composite_fluxes[f'IRAC5.8_{alpha_values[0]}']
f_80 = composite_fluxes[f'IRAC8.0_{alpha_values[0]}']

# Create the colours for the IRAC diagram
f_5836_specific_alpha_colours = np.log10(f_58/f_36)
f_8045_specific_alpha_colours = np.log10(f_80/f_45)

# Redefine x and y for the first alpha value to plot the wedge
x = f_5836_specific_alpha_colours
y = f_8045_specific_alpha_colours

# Can change this to ensure we are only looking at the rest frame stuff
x = x[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
y = y[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]

# # Reimplement the selection criteria
lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)

wedge_vertices = [
    (xmax, -0.2),
    (-0.1, -0.2),
    (-0.1, 0.8*np.nanmin(x[lacy_selection_condition]) + 0.5 if np.any(lacy_selection_condition) else -0.2),  # Handle empty selection
    (((ymax-0.5)/0.8), ymax)
]

axs[2, 3].add_patch(plt.Polygon(wedge_vertices, closed=False, edgecolor='r', facecolor=(1,0,0,0.05), linestyle='-', linewidth=2, label='Lacy Wedge'))

# Set the axis labels
axs[2, 3].set_xlabel("[5.8] - [3.6]")


# Set the axis limits
axs[2, 3].set_xlim(xmin, xmax)
axs[2, 3].set_ylim(ymin, ymax)

# Add a colorbar for the AGN contribution
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
cbar = plt.colorbar(sm, ax=axs[2, 3])

# Add the colorbar label
cbar.set_label('AGN Contribution')

# add the final subplot labels
axs[2, 3].set_title("Colour Evolution")

    
    
        
# Make the subplots closer together
plt.subplots_adjust(wspace=0.05, hspace=0.3)
    
# Save the output
if save_figures:
    
    plt.savefig(f'outputs/ThesisPlots/IRAC_evolution_{agn_model_name}_{template_set_name}.png', dpi=300, bbox_inches='tight')
    
      
# Show the output
plt.show()


# ### Metrics

# In[381]:


# we will need to create a list of dictionaries
completeness_values = []

for n in range(len(alpha_values)):
    # at this alpha value, get the coordoinates
    # Get the IRAC magnitudes
    f_36 = composite_fluxes[f'IRAC3.6_{alpha_values[n]}']
    f_45 = composite_fluxes[f'IRAC4.5_{alpha_values[n]}']
    f_58 = composite_fluxes[f'IRAC5.8_{alpha_values[n]}']
    f_80 = composite_fluxes[f'IRAC8.0_{alpha_values[n]}']

    # Create the colours for the IRAC diagram
    f_5836_specific_alpha_colours = np.log10(f_58/f_36)
    f_8045_specific_alpha_colours = np.log10(f_80/f_45)

    # Redefine x and y for the first alpha value to plot the wedge
    x = f_5836_specific_alpha_colours
    y = f_8045_specific_alpha_colours

    # Can change this to ensure we are only looking at the rest frame stuff
    x = x[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
    y = y[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]

    print(len(x))
    # # Reimplement the selection criteria
    lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)
    
    num_sources_selected = x[lacy_selection_condition].notna().sum()
    
    print(num_sources_selected)
    # get the number of sources that are selected by the lacy wedge
    # num_sources_selected = len()
    # print(num_sources_selected)
    # for the first value of alpha there will be no relevance as there are no AGN, 
    # but we can instead check how many of all sources were selected
    if n == 0:
        num_sources_all = len(x)
        completeness = num_sources_selected/num_sources_all
        completeness_values.append({'alpha': alpha_values[n], 'completeness': completeness})
    else:
        completeness = num_sources_selected/num_sources_all
        completeness_values.append({'alpha': alpha_values[n], 'completeness': completeness})
        
# Create a pd.DataFrame
completeness_df = pd.DataFrame(completeness_values)


# save
if save_figures:
    completeness_df.to_csv(f'outputs/ThesisPlots/IRAC_completeness_{agn_model_name}_{template_set_name}.csv')

completeness_df


# In[382]:


# In addition to the restframe we can also see how aritifically redshifting the galaxy affects the modelling of the SED - the lacy selection seems to break down at redshifts above 0.5
# 
# each entry in the table will be the completeness of the selection at that redshift
completeness_values = [[], [], [], [], [], [], [], []]


# Choose the diagram min and max
xmax = 0.8
ymax = 1.2
xmin = -0.6
ymin = -0.8

# Can change this to ensure we are only looking at the rest frame stuff
upper_redshift_limit = 0.5
lower_redshift_limit = 0

print(colours_theme)
# Redshift bins

bin_1 = [0, 0.5]
bin_2 = [0.5, 1]
bin_3 = [1, 1.5]
bin_4 = [1.5, 2]
bin_5 = [2, 2.5]
bin_6 = [2.5, 3]
bin_7 = [3, 3.5]
bin_8 = [3.5, 4]

# all bins
redshift_bins = [bin_1, bin_2, bin_3, bin_4, bin_5, bin_6, bin_7, bin_8]

# setup
fig, axs = plt.subplots(2, 4, figsize=fig_size_params, sharex=True, sharey=True)

for n in range(len(redshift_bins)):
    lower_redshift_limit = redshift_bins[n][0]
    upper_redshift_limit = redshift_bins[n][1]
    # We want an AGN evolution plot for
    for m in range(len(alpha_values)):
        
        # Show the upper and lower redshift limits
        axs[n//4, n%4].set_title(f"{lower_redshift_limit} < z < {upper_redshift_limit}")
        
        
        # Get the IRAC magnitudes
        f_36 = composite_fluxes[f'IRAC3.6_{alpha_values[m]}']
        f_45 = composite_fluxes[f'IRAC4.5_{alpha_values[m]}']
        f_58 = composite_fluxes[f'IRAC5.8_{alpha_values[m]}']
        f_80 = composite_fluxes[f'IRAC8.0_{alpha_values[m]}']
        
        # Create the colours for the IRAC diagram
        f_5836_specific_alpha_colours = np.log10(f_58/f_36)
        f_8045_specific_alpha_colours = np.log10(f_80/f_45)
        
        # Redefine x and y for the first alpha value to plot the wedge
        x = f_5836_specific_alpha_colours
        y = f_8045_specific_alpha_colours
        
        # Can change this to ensure we are only looking at the rest frame stuff
        x = x[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
        y = y[(composite_fluxes['z'] >= lower_redshift_limit) & (composite_fluxes['z'] <= upper_redshift_limit)]
        
        # Reimplement the selection criteria 
        lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)
        
        # Colour code for each alpha
        axs[n//4, n%4].scatter(x, y, c=[colours_theme[m]], s=10)
            
        
        num_sources_selected = x[lacy_selection_condition].notna().sum()

        
        
        if m == 0:
            num_sources_all = len(x)
            completeness = num_sources_selected/num_sources_all
            
        else:
            completeness = num_sources_selected/num_sources_all
            
            # Add the completeness value to the list, for the redshift bin
            completeness_values[n].append({'alpha': alpha_values[m], 'completeness': completeness})    
        

        
        
        
         # # Get the IRAC magnitudes
        f_36 = composite_fluxes[f'IRAC3.6_{alpha_values[0]}']
        f_45 = composite_fluxes[f'IRAC4.5_{alpha_values[0]}']
        f_58 = composite_fluxes[f'IRAC5.8_{alpha_values[0]}']
        f_80 = composite_fluxes[f'IRAC8.0_{alpha_values[0]}']

        # Create the colours for the IRAC diagram
        f_5836_specific_alpha_colours = np.log10(f_58/f_36)
        f_8045_specific_alpha_colours = np.log10(f_80/f_45)
        
        x = f_5836_specific_alpha_colours
        y = f_8045_specific_alpha_colours
        
                
        # Ensure a consistent selection region defined by the first redshift bin
        x = x[(composite_fluxes['z'] >= redshift_bins[0][0]) & (composite_fluxes['z'] <= redshift_bins[0][1])]
        y = y[(composite_fluxes['z'] >= redshift_bins[0][0]) & (composite_fluxes['z'] <= redshift_bins[0][1])]
        
        # Reimplement the selection criteria
        lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)        
        
        wedge_vertices = [
            (xmax, -0.2),
            (-0.1, -0.2),
            (-0.1, 0.8*np.nanmin(x[lacy_selection_condition]) + 0.5 if np.any(lacy_selection_condition) else -0.2),  # Handle empty selection
            (((ymax-0.5)/0.8), ymax)
        ]
    
            
    # Plot the completeness in the redshift slection
   # axs[n//4, n%4].text(0.05, 0.8, f'{completeness:.2f}', transform=axs[n//4, n%4].transAxes, color='k')  

    axs[n//4, n%4].add_patch(plt.Polygon(wedge_vertices, closed=False, edgecolor='r', facecolor=(1,0,0,0.05), linestyle='-', linewidth=2, label='Lacy Wedge'))

        # Set the axis labels
    axs[n//4, n%4].set_xlabel("[5.8] - [3.6]")


    # Set the axis limits
    axs[n//4, n%4].set_xlim(xmin, xmax)
    axs[n//4, n%4].set_ylim(ymin, ymax)

        # # Add a colorbar for the AGN contribution
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
        # cbar = plt.colorbar(sm, ax=axs[n//4, n%4])

        # # Add the colorbar label
        # cbar.set_label('AGN Contribution')

# Show a single colorbar for the entire plot
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('AGN Contribution')

       
        
    
        
# Make the subplots closer together
plt.subplots_adjust(wspace=0.05, hspace=0.3)
    
# Save the output
if save_figures:
    
    plt.savefig(f'outputs/ThesisPlots/IRAC_AGNRedshift_evolution_{agn_model_name}_{template_set_name}.png')
     
# Show the output
plt.show()


# In[383]:


# Check the redshift completeness outputs for each redshift bin

# Print the results
for n in range(len(redshift_bins)):
    print(f"Redshift Bin: {redshift_bins[n]}")
    completeness_df = pd.DataFrame(completeness_values[n])
    print(completeness_df)
    print("\n\n")
    
    # Save the completeness values for each redshift bin
    if save_figures:
        completeness_df.to_csv(f'outputs/redshift_bins/IRAC_completeness_{agn_model_name}_{template_set_name}_z_{redshift_bins[n][0]}_{redshift_bins[n][1]}.csv')


# In[ ]:





# # Brown Templates Average Moving Position
# We would also like to see some other metrics. For example we would like to see how far each of the points oved in space in each diagram and for each model type (Type 1, Type 2)
# 
# What the values in this table will represent are the average change in position from the initial starting position. I.e the offset from the initial position.
# 
# Showing agn contribution and movement
# 

# In[ ]:





# In[384]:


# Read in both the type 1 and type 2 AGN models
template_set_name = 'Brown'
agn_model_names = ['Type1AGN', 'Type2AGN']


# read in the dataframe and ensure that the unamed: 0 column is the index
composite_choice_1 = f'{template_set_name}_theoretical_composite_fluxes_{agn_model_names[0]}'
composite_choice_2 = f'{template_set_name}_theoretical_composite_fluxes_{agn_model_names[1]}'
composite_fluxes_type1 = pd.read_csv(f'outputs\composite_seds\{composite_choice_1}.csv', index_col=0)   
composite_fluxes_type2 = pd.read_csv(f'outputs\composite_seds\{composite_choice_2}.csv', index_col=0)   


# In[385]:


composite_fluxes_type1


# In[386]:


composite_fluxes_type2


# In[387]:



# Create the a table where it is 

#            | Alpha | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 
# Vector Mag | Type 1|
#   Diff     | Type 2|

# Create table
vector_magnitude_table = pd.DataFrame(columns=alpha_values[1:], index=['Type 1', 'Type 2'])



# types
types = ['Type 1', 'Type 2']


# Create a figure to plot the UVJ diagram
fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)

for type in types: 
    if type == 'Type 1':
        composite_fluxes = composite_fluxes_type1
    else:
        composite_fluxes = composite_fluxes_type2
        
    for alpha in alpha_values:
        # Get the filter magnitudes
        U_mag = composite_fluxes[f'U_{alpha}']
        V_mag = composite_fluxes[f'V_{alpha}']
        J_mag = composite_fluxes[f'J_{alpha}']

        # Create the colours
        uv = U_mag - V_mag
        vj = V_mag - J_mag

        # Recall that this is only a restframe diagram, so only look at the restframe colours
        uv = uv[composite_fluxes['z'] == 0]
        vj = vj[composite_fluxes['z'] == 0]
        
        
        # Plot the UVJ diagram
        axs[types.index(type)].scatter(vj, uv, c=[colours_theme[alpha_values.index(alpha)]], s=10)
        
    
        
        
        if alpha == 0:
            # uv and vj initial values
            uv_initial = uv
            vj_initial = vj
            
            # Get the average location
            uv_mean_inital = np.mean(uv_initial)
            vj_mean_inital = np.mean(vj_initial)
            
            # print the mean location
            #print(f"Mean Location for alpha = {alpha_values[m]}: ({vj_mean_inital}, {vj_mean_inital})")
            
        else:
            
            # Get the vector magnitude
            # Get the average location
            uv_mean = np.mean(uv)
            vj_mean = np.mean(vj)
            
            # print the mean location
            #print(f"Mean Location for alpha = {alpha_values[m]}: ({vj_mean}, {uv_mean})")
            
            # calculate the absolute vector magnitude difference
            vector_magnitude_diff = np.sqrt((vj_mean - vj_mean_inital)**2 + (uv_mean - uv_mean_inital)**2)
                
            # Add the vector magnitude to the table
            vector_magnitude_table.loc[type, alpha] = vector_magnitude_diff
    
    
    # Plot a connecting line between each set of alpha values to the next
    for m in range(len(alpha_values) - 1):
        # Get the filter magnitudes
        U_mag = composite_fluxes[f'U_{alpha_values[m]}']
        V_mag = composite_fluxes[f'V_{alpha_values[m]}']
        J_mag = composite_fluxes[f'J_{alpha_values[m]}']

        # Create the colours
        uv = U_mag - V_mag
        vj = V_mag - J_mag

        # Recall that this is only a restframe diagram, so only look at the restframe colours
        uv = uv[composite_fluxes['z'] == 0]
        vj = vj[composite_fluxes['z'] == 0]
        
        # Get the filter magnitudes
        U_mag_next = composite_fluxes[f'U_{alpha_values[m+1]}']
        V_mag_next = composite_fluxes[f'V_{alpha_values[m+1]}']
        J_mag_next = composite_fluxes[f'J_{alpha_values[m+1]}']

        # Create the next colorus
        uv_next = U_mag_next - V_mag_next
        vj_next = V_mag_next - J_mag_next
        
        # Recall that this is only a restframe diagram, so only look at the restframe colours for the next colorus
        uv_next = uv_next[composite_fluxes['z'] == 0]
        vj_next = vj_next[composite_fluxes['z'] == 0]
        
        # Plot the connecting line between the sources that are selected and their relative next points for each source
        for i in range(len(uv)):
            axs[types.index(type)].plot([vj[i], vj_next[i]], [uv[i], uv_next[i]], c='black', alpha=0.1)
            
            
            
    # Add patches for selections
    axs[types.index(type)].add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
    axs[types.index(type)].add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
    axs[types.index(type)].add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))
    
    # Add vertical line
    axs[types.index(type)].axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)
    
    # Add axis labels
    axs[types.index(type)].set_xlabel("V - J")
    axs[0].set_ylabel("U - V")
    
    # AGN Colour Evolution title
    axs[types.index(type)].set_title(f"{type} Colour Evolution")

    # ensure the axis limits are the same
    axs[types.index(type)].set_xlim(-0.5, 2.2)
    axs[types.index(type)].set_ylim(0, 2.5)
    
    # angle the x axis labels
    axs[types.index(type)].tick_params(axis='x', rotation=45)
    
    # Change the font size of the labels and the ticks
    axs[types.index(type)].tick_params(axis='both', labelsize=12)
    axs[types.index(type)].set_xlabel('V - J', fontsize=12)
    axs[0].set_ylabel('U - V', fontsize=12)
    
    # Ensure each section of the plot is labelled with sstar formation, quiescent and dusty star forming
    
    # Star-forming
    axs[types.index(type)].text(-0.2, 2.3, 'Quiescent', color='k', fontsize=12)
    # Quiescent
    axs[types.index(type)].text(-0.2, 0.1, 'Star-forming', color='k', fontsize=12)
    # Dusty Star Forming
    axs[types.index(type)].text(1.5, 0.1, 'Dusty ', color='k', fontsize=12)
    
# Add colourbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
cbar = plt.colorbar(sm, ax=axs[1])

# Add the colorbar label for the second plot
cbar.set_label('AGN Contribution (%)')
        
# Make the subplots closer together
plt.subplots_adjust(wspace=0, hspace=0.3)


        



if save_figures:
    vector_magnitude_table.to_csv(f'outputs/ThesisPlots/UVJ_VectorMagnitudeDifference_{agn_model_names[0]}_{agn_model_names[1]}.csv')
    
    # sace the plot also
    plt.savefig(f'outputs/ThesisPlots/UVJ_bothcolours_{agn_model_names[0]}_{agn_model_names[1]}.png', dpi=300, bbox_inches='tight')
   
# show
plt.show()
 
# Check the output
vector_magnitude_table


# Now we know this works, we do the same for the other colour spaces.

# In[388]:


# ugr space
vector_magnitude_table = pd.DataFrame(columns=alpha_values[1:], index=['Type 1', 'Type 2'])
fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)
for type in types:
    if type == 'Type 1':
        composite_fluxes = composite_fluxes_type1
    else:
        composite_fluxes = composite_fluxes_type2
        
    for alpha in alpha_values:
        # Get the u, g, and r magnitudes for the specific alpha value
        u_col_name = f'u_{int(round(alpha, 2))}'
        g_col_name = f'g_{int(round(alpha, 2))}'
        r_col_name = f'r_{int(round(alpha, 2))}'

        # Create the colours for the UGR diagram
        ug_specific_alpha_colours = composite_fluxes[u_col_name] - composite_fluxes[g_col_name]
        gr_specific_alpha_colours = composite_fluxes[g_col_name] - composite_fluxes[r_col_name]
        
        
        # Similarly we create two seperate ugr plots
        # Plot the UGR diagram for the specific alpha value
        axs[types.index(type)].scatter(gr_specific_alpha_colours, ug_specific_alpha_colours, c=[colours_theme[alpha_values.index(alpha)]], s=10)
        
        
                
            
            
        if alpha == 0:
            # uv and vj initial values
            ug_initial = ug_specific_alpha_colours
            gr_initial = gr_specific_alpha_colours
            
            # Get the average location
            ug_mean_inital = np.mean(ug_initial)
            gr_mean_inital = np.mean(gr_initial)
            
            # print the mean location
            #print(f"Mean Location for alpha = {alpha_values[m]}: ({vj_mean_inital}, {vj_mean_inital})")
            
        else:
            # Get the vector magnitude
            # Get the average location
            ug_mean = np.mean(ug_specific_alpha_colours)
            gr_mean = np.mean(gr_specific_alpha_colours)
            
            # print the mean location
            #print(f"Mean Location for alpha = {alpha_values[m]}: ({vj_mean}, {uv_mean})")
            
            # calculate the absolute vector magnitude difference
            vector_magnitude_diff = np.sqrt((ug_mean - ug_mean_inital)**2 + (gr_mean - gr_mean_inital)**2)
                
            # Add the vector magnitude to the table
            vector_magnitude_table.loc[type, alpha] = vector_magnitude_diff
    
    # Additionally we can plot the UGR selection criteria on the UGR diagram
    U_rule = [[1.2,9], [1.2,2.2], [0.6,1.6], [-3,1.6], [-3,9]]
    
    # Add onto the plot
    axs[types.index(type)].plot([i[0] for i in U_rule], [i[1] for i in U_rule], c='black', linestyle='-', linewidth=2)
    
    
    axs[types.index(type)].set_xlim(-1, 3.9)
    axs[types.index(type)].set_ylim(-1, 8.5)

    # add text inside the selection region specifying redshift range
    axs[types.index(type)].text(-0.8, 7.5, f"Target Redshift:\n(2.6 < z < 3.5)", fontsize=9)
          
if save_figures:
    vector_magnitude_table.to_csv(f'outputs/ThesisPlots/UGR_VectorMagnitudeDifference_{agn_model_names[0]}_{agn_model_names[1]}.csv')

# change subplot spacing
plt.subplots_adjust(wspace=0, hspace=0.3)

# label the x and y axis
axs[0].set_xlabel("g - r", fontsize=12)

axs[1].set_xlabel("g - r", fontsize=12)
axs[0].set_ylabel("u - g", fontsize=12)

# ensure the axis also have the same fontsize
axs[0].tick_params(axis='both', labelsize=12)
axs[1].tick_params(axis='both', labelsize=12)


# Plot the colourbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
cbar = plt.colorbar(sm, ax=axs[1])
cbar.set_label('AGN Contribution (%)')

# Add the title
axs[0].set_title("Type 1 UGR Colour Evolution",fontsize=12)
axs[1].set_title("Type 2 UGR Colour Evolution", fontsize=12)
# Save
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UGR_bothcolours_{agn_model_names[0]}_{agn_model_names[1]}.png', dpi=300, bbox_inches='tight')

# Show
plt.show()



# Check the output
vector_magnitude_table


# Lastly we look at the IRAC colour space - here we expect less of an effect as the IRAC colour space from Type 1, but more from Type 2.
# 

# In[389]:


# IRAC space
vector_magnitude_table = pd.DataFrame(columns=alpha_values[1:], index=['Type 1', 'Type 2'])

for type in types:
    if type == 'Type 1':
        composite_fluxes = composite_fluxes_type1
    else:
        composite_fluxes = composite_fluxes_type2
        
    for alpha in alpha_values:
        # Get the IRAC magnitudes
        f_36 = composite_fluxes[f'IRAC3.6_{alpha}']
        f_45 = composite_fluxes[f'IRAC4.5_{alpha}']
        f_58 = composite_fluxes[f'IRAC5.8_{alpha}']
        f_80 = composite_fluxes[f'IRAC8.0_{alpha}']

        # Create the colours for the IRAC diagram
        f_5836_specific_alpha_colours = np.log10(f_58/f_36)
        f_8045_specific_alpha_colours = np.log10(f_80/f_45)

        if alpha == 0:
            # uv and vj initial values
            f_5836_initial = f_5836_specific_alpha_colours
            f_8045_initial = f_8045_specific_alpha_colours
            
            # Get the average location
            f_5836_mean_inital = np.mean(f_5836_initial)
            f_8045_mean_inital = np.mean(f_8045_initial)
            
            # print the mean location
            #print(f"Mean Location for alpha = {alpha_values[m]}: ({vj_mean_inital}, {vj_mean_inital})")
            
        else:
            # Get the vector magnitude
            # Get the average location
            f_5836_mean = np.mean(f_5836_specific_alpha_colours)
            f_8045_mean = np.mean(f_8045_specific_alpha_colours)
            
            # print the mean location
            #print(f"Mean Location for alpha = {alpha_values[m]}: ({vj_mean}, {uv_mean})")
            
            # calculate the absolute vector magnitude difference
            vector_magnitude_diff = np.sqrt((f_5836_mean - f_5836_mean_inital)**2 + (f_8045_mean - f_8045_mean_inital)**2)
                
            # Add the vector magnitude to the table
            vector_magnitude_table.loc[type, alpha] = vector_magnitude_diff
            
if save_figures:
    vector_magnitude_table.to_csv(f'outputs/ThesisPlots/IRAC_VectorMagnitudeDifference_{agn_model_names[0]}_{agn_model_names[1]}.csv')
    
# Check the output
vector_magnitude_table


# In[390]:


# composite_fluxes = composite_fluxes_type1

# # Gen figure
# fig, axs = plt.subplots(1, 1, figsize=fig_size_params)

# # Set figure limits
# axs.set_xlim(-0.5, 2.5)
# axs.set_ylim(0, 2.5)


# for m in range(len(alpha_values)):
    
    
    
    
#     # Get the filter magnitudes
#     U_mag = composite_fluxes[f'U_{alpha_values[m]}']
#     V_mag = composite_fluxes[f'V_{alpha_values[m]}']
#     J_mag = composite_fluxes[f'J_{alpha_values[m]}']
    
#     # Create the colours
#     uv = U_mag - V_mag
#     vj = V_mag - J_mag
    
#     # Recall that this is only a restframe diagram, so only look at the restframe colours
#     uv = uv[composite_fluxes['z'] == 0]
#     vj = vj[composite_fluxes['z'] == 0]
    
    
#     # Find the inital position in the UVJ
#     if m == 0:
        
#         # uv and vj initial values
#         uv_initial = uv
#         vj_initial = vj
        
#         # Get the average location
#         uv_mean_inital = np.mean(uv_initial)
#         vj_mean_inital = np.mean(vj_initial)
        
#         # print the mean location
#         print(f"Mean Location for alpha = {alpha_values[m]}: ({vj_mean}, {uv_mean})")
        
#         # Plot the mean location
#         axs.scatter(vj_mean_inital, uv_mean_inital, c='red', s=20)
        
#     else: 
        
#         # Get the average location
#         uv_mean = np.mean(uv)
#         vj_mean = np.mean(vj)
        
#         # print the mean location
#         print(f"Mean Location for alpha = {alpha_values[m]}: ({vj_mean}, {uv_mean})")
        
#         # Plot the mean location
#         axs.scatter(vj_mean, uv_mean, c='red', s=20)
        
#         # calculate the absolute vector magnitude difference
#         vector_magnitude_diff = np.sqrt((vj_mean - vj_mean_inital)**2 + (uv_mean - uv_mean_inital)**2)
        
#         # Print the vector magnitude difference
#         print(f"Vector Magnitude Difference for alpha = {alpha_values[m]}: {vector_magnitude_diff}")

        
#     # Plot scatter
#     axs.scatter(vj, uv, c='grey', s=10)
    
    
# # Add patches for selections
# axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
# axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
# axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

# # Add vertical line
# axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

# # Add axis labels
# axs.set_xlabel("V - J")

# # AGN Colour Evolution title
# axs.set_title("Colour Evolution")



# # Plot colorbar for AGN contribution
# sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values)))
# cbar = plt.colorbar(sm, ax=axs)

# # Name the colorbar
# cbar.set_label('AGN Contribution')

    
# # Make the subplots closer together
# plt.subplots_adjust(wspace=0.05, hspace=0.3)
    

# # Save the output
# if save_figures:
#     plt.savefig(f'outputs/ThesisPlots/UVJ_evolution_{agn_model_name}_{template_set_name}.png')

# # Show the output
# plt.show()


# In[ ]:




