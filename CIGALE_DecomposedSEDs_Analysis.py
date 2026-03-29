# %% [markdown]
# # Data Analysis and Visulisation ZFOURGE SED Decomposition Data from CIGALE 
# This script is to be used for the data analysis of my project. This will contain some of the data analysis and visulisations for the decomposed sed data that will be used in my thesis. This will be one part of a bigger section on data analysis and will be covered later in other notebooks.
# 
# It is intended for the data analysis in this notebook to be exclusively used for my thesis work.

# %%
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pandas as pd
import os
from astLib import astSED
import astropy.io.fits as fits
from carf import * # custom module for functions relating to the project
import matplotlib.path as mpath
import seaborn as sns
from scipy.stats import gaussian_kde
import scipy as sp

# So that we can change the helper functions without reloading the kernel
%load_ext autoreload
%autoreload 2

# %%
figsize_params = (7, 5)

save_figures = True

# %%
# We need to ensure we are reading in our final dataframe we will be doing our analysis on. This dataframe should contain
zfourge_full = pd.read_csv('datasets/full_zfourge_decomposed/zfourge_full_final.csv')

# %%
# Check the zpk_x
zfourge_full['zpk_x'].value_counts()

# %%

# Identify the columns we are interested in
for col in zfourge_full.columns:
    print(col)
# UV_Full
# VJ_Full
# UV_Decomposed
# VJ_Decomposed
# ID
# RAJ2000
# DEJ2000
# zspec
# zpk_x ? or zpk_y
# lssfr
# lmass
# mag_U
# mag_V
# mag_J
# e_mag_U
# e_mag_V
# e_mag_J
# field
# UV, VJ? id what these are probs just fully clalulated UV and VJ colours from above

# Create a subset of the dataset we are interested
zfourge_subset = zfourge_full[['ID', 'UV_Full', 'VJ_Full', 'UV_Decomposed', 'VJ_Decomposed', 'UG_Full', 'GR_Full', 'UG_Decomposed', 'GR_Decomposed', 'lssfr', 'lmass', 'mag_U', 'mag_V', 'mag_J', 'field', 'zpk_x', 'eUV_Full', 'eVJ_Full', 'eUV_Decomposed', 'eVJ_Decomposed']]


# Pairplot of a subset of the dataset 
# Create a pairplot of the dataset
sns.pairplot(zfourge_subset[['UV_Full', 'VJ_Full', 'UV_Decomposed', 'VJ_Decomposed', 'UG_Full', 'GR_Full', 'UG_Decomposed', 'GR_Decomposed', 'lssfr', 'lmass', 'zpk_x']], diag_kind='kde')



# %%
#galaxy_mode = ['Decomposed', 'Full']

# or reverse order

galaxy_mode = ['Full', 'Decomposed']


# %%
# Now that we have this we can explore
# Plot a histogram of the redshifts
plt.figure(figsize=(8, 3))
plt.hist(zfourge_subset['zpk_x'], bins=50)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshifts')
plt.show()

# Max redshift
max_redshift = zfourge_subset['zpk_x'].max()

print('Max redshift:', max_redshift)

# %% [markdown]
# Exploring the UVJ Diagram using the Decomposed Colours obtained using CIGALE using data from the ZFOURGE Survey.

# %%
# Maybe just plot an all redshift UVJ diagram 
fig, axs = plt.subplots(1, 1, figsize=(5, 6), sharex=True, sharey=True)

# Reset the galaxy fractions
galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
num_galaxies = len(zfourge_subset) # number of galaxies in the composite_flux in the rest frame

vj = zfourge_subset['VJ_Full']
uv = zfourge_subset['UV_Full']

# Plot the UVJ diagram
axs.scatter(vj, uv, c='blue', s=10, alpha=0.5, marker='o')


# Plot the error bars, error is taken from eUV_Full, eVJ_Full, eUV_Decomposed, eVJ_Decomposed
# full 
avg_error_UV = np.mean(zfourge_full[f'eUV_Full'])
avg_error_VJ = np.mean(zfourge_full[f'eVJ_Full'])

# Plot the error bar
axs.errorbar(0.5, 1.5, xerr=avg_error_VJ, yerr=avg_error_UV, fmt='o', color='black', markersize=2, capsize=4, capthick=2)

# Define paths for selections
path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

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

quiescent_fraction = galaxy_fractions['Quiescent'][0]
sf_fraction = galaxy_fractions['Star-forming'][0]
sfd_fraction = galaxy_fractions['Dusty'][0]


    # Add patches for selections
axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

# Add vertical line
axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

# Add axis labels
axs.set_xlabel("V - J")
axs.set_ylabel("U - V")

# Plot the fractions in the corner of each section on the UVJ
axs.text(0.05, 0.8, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
axs.text(0.05, 0.35, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
axs.text(0.70, 0.1, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
# AGN Colour Evolution title - just call it redshift
# axs.set_title(f"Colour Evolution for ZFOURGE Composites")

# Ensure limits
axs.set_xlim(-0.5, 2.2)
axs.set_ylim(0, 2.5)
    # Make the subplots closer together
# plt.subplots_adjust(wspace=0.05, hspace=0.3)


plt.tight_layout()




# Save 
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UVJ_agn_evolution_CIGALE_ZFOURGE.png')

plt.show()

# %%
# Maybe just plot an all redshift UVJ diagram 
fig, axs = plt.subplots(1, 1, figsize=(5, 6), sharex=True, sharey=True)

# Reset the galaxy fractions
galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
num_galaxies = len(zfourge_subset) # number of galaxies in the composite_flux in the rest frame

vj = zfourge_subset['VJ_Decomposed']
uv = zfourge_subset['UV_Decomposed']

# Plot the UVJ diagram
axs.scatter(vj, uv, c='blue', s=10, alpha=0.5, marker='o')

# Define paths for selections
path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

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

quiescent_fraction = galaxy_fractions['Quiescent'][0]
sf_fraction = galaxy_fractions['Star-forming'][0]
sfd_fraction = galaxy_fractions['Dusty'][0]


    # Add patches for selections
axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

# Add vertical line
axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

# Add axis labels
axs.set_xlabel("V - J")
axs.set_ylabel("U - V")

# Plot the fractions in the corner of each section on the UVJ
axs.text(0.05, 0.8, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
axs.text(0.05, 0.35, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
axs.text(0.70, 0.1, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
# AGN Colour Evolution title - just call it redshift
# axs.set_title(f"Colour Evolution for ZFOURGE Composites")

# Ensure limits
axs.set_xlim(-0.5, 2.2)
axs.set_ylim(0, 2.5)
    # Make the subplots closer together
# plt.subplots_adjust(wspace=0.05, hspace=0.3)


plt.tight_layout()




# Save 
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UVJ_agn_evolution_CIGALE_ZFOURGE.png')

plt.show()

# %%
# Plot a UVJ diagram showing both the full and decomposed UVJ colours
fig, ax = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True)



for i in range(2):
    # Maybe just plot an all redshift UVJ diagram 
    axs = ax[i]
    # Reset the galaxy fractions
    galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
    num_galaxies = len(zfourge_subset) # number of galaxies in the composite_flux in the rest frame

    vj = zfourge_subset[f'VJ_{galaxy_mode[i]}']
    uv = zfourge_subset[f'UV_{galaxy_mode[i]}']

    # Plot the UVJ diagram
    axs.scatter(vj, uv, c='blue', s=10, alpha=0.5, marker='o')

    avg_error_UV = np.mean(zfourge_full[f'eUV_{galaxy_mode[i]}'])
    avg_error_VJ = np.mean(zfourge_full[f'eVJ_{galaxy_mode[i]}'])

    # Plot the error bar
    axs.errorbar(0, 2, xerr=avg_error_VJ, yerr=avg_error_UV, fmt='o', color='black', markersize=4, capsize=5, capthick=1)


    # Define paths for selections
    path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
    path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
    path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

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

    quiescent_fraction = galaxy_fractions['Quiescent'][0]
    sf_fraction = galaxy_fractions['Star-forming'][0]
    sfd_fraction = galaxy_fractions['Dusty'][0]


        # Add patches for selections
    axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
    axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
    axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

    # Add vertical line
    axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

    # Add axis labels
    axs.set_xlabel("V - J")
    
    # add title 
    axs.set_title(f"{galaxy_mode[i]} Galaxy " if i == 0 else f"AGN Removed")

    # Plot the fractions in the corner of each section on the UVJ
    axs.text(0.05, 0.9, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
    axs.text(0.05, 0.45, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
    axs.text(0.80, 0.05, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
    # AGN Colour Evolution title - just call it redshift
    # axs.set_title(f"Colour Evolution for ZFOURGE Composites")

    # Ensure limits
    axs.set_xlim(-0.5, 2.2)
    axs.set_ylim(0, 2.5)
        # Make the subplots closer together
    # plt.subplots_adjust(wspace=0.05, hspace=0.3)
    
    # adjust the x axis labels (45 degrees
    axs.set_xticklabels(axs.get_xticks(), rotation=45)
    

ax[0].set_ylabel("U - V")

# layout together
# Make the subplots closer together
plt.subplots_adjust(wspace=0, hspace=0.3)





# Save 
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UVJ_agn_evolution_CIGALE_ZFOURGE_comparison.png', dpi=300)

plt.show()


# This is a very cool plot showing some of the interesting behaviour of the UVJ diagram

# %%
# Rather than doing it like this, only plot 1 UVJ diagram with the full and decomposed UVJ colours being colourcoded as different colours, and an arrow between the points where there is a movement greater than 0.05
# Plot a UVJ diagram showing both the full and decomposed UVJ colours
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, sharey=True)



for i in range(2):
    # Maybe just plot an all redshift UVJ diagram 
    axs = ax
    # Reset the galaxy fractions
    galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
    num_galaxies = len(zfourge_subset) # number of galaxies in the composite_flux in the rest frame

    vj = zfourge_subset[f'VJ_{galaxy_mode[i]}']
    uv = zfourge_subset[f'UV_{galaxy_mode[i]}']

    # Plot the UVJ diagram
    axs.scatter(vj, uv, c=('blue' if i==0 else 'red'), s=10, alpha=0.2, marker='o', label=f'{galaxy_mode[i]}')

    avg_error_UV = np.mean(zfourge_full[f'eUV_{galaxy_mode[i]}'])
    avg_error_VJ = np.mean(zfourge_full[f'eVJ_{galaxy_mode[i]}'])

    # Plot the error bar
    axs.errorbar(0, 1.7, xerr=avg_error_VJ, yerr=avg_error_UV, fmt='o', color='black', markersize=4, capsize=5, capthick=1)
    
    # check if the vector offset is greater than 0.05 if it is plot an arrow from the associated point
    # if i == 0:
    #     for j in range(len(vj)):
    #         if np.abs(vj[j] - zfourge_subset['VJ_Decomposed'][j]) > 0.2:
    #             axs.arrow(zfourge_subset['VJ_Full'][j], zfourge_subset['UV_Full'][j], zfourge_subset['VJ_Decomposed'][j] - zfourge_subset['VJ_Full'][j], zfourge_subset['UV_Decomposed'][j] - zfourge_subset['UV_Full'][j], head_width=0.05, head_length=0.05, fc='k', ec='k')
    
    # Plot the mean vector offset for this colourspace for galaxies that moved more than a vector offset of 0.02
    # Calculate the vector offset
    vector_offset = np.sqrt((zfourge_subset[f'VJ_{galaxy_mode[i]}'] - zfourge_subset[f'VJ_{galaxy_mode[1-i]}'])**2 + (zfourge_subset[f'UV_{galaxy_mode[i]}'] - zfourge_subset[f'UV_{galaxy_mode[1-i]}'])**2)
    # Calculate the mean vector offset
    mean_vector_offset = np.mean(vector_offset)
    # Calculate the standard deviation of the vector offset
    std_vector_offset = np.std(vector_offset)
    # Calculate the number of galaxies that moved more than 0.02
    
    # Calculate the current cutoff based on the error in the UVJ diagram 
    cutoff = np.sqrt(avg_error_UV**2 + avg_error_VJ**2)
    
    num_moved = np.sum(vector_offset > cutoff)
    
    # Plot an arrow from the mean of the full UVJ to the mean of the decomposed UVJ
    if i == 0:
        # Subset the dataframe to only look at the galaxies moved
        moved_df = zfourge_subset[vector_offset > cutoff]
        # Calculate the mean of the full UVJ
        mean_full_uvj = [np.mean(moved_df['VJ_Full']), np.mean(moved_df['UV_Full'])]
        # Calculate the mean of the decomposed UVJ
        mean_decomposed_uvj = [np.mean(moved_df['VJ_Decomposed']), np.mean(moved_df['UV_Decomposed'])]
        # Plot the arrow, ensuring it is nice and thick
        axs.arrow(mean_full_uvj[0], mean_full_uvj[1], mean_decomposed_uvj[0] , mean_decomposed_uvj[1], head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=3)
        
        # Make another arrow with a different colour not black
    # also do the same thing but reversed from decomposed to full
    else:
        # print
        print('')
        # # Subset the dataframe to only look at the galaxies moved
        # moved_df = zfourge_subset[vector_offset > cutoff]
        # # Calculate the mean of the full UVJ
        # mean_full_uvj = [np.mean(moved_df['VJ_Full']), np.mean(moved_df['UV_Full'])]
        # # Calculate the mean of the decomposed UVJ
        # mean_decomposed_uvj = [np.mean(moved_df['VJ_Decomposed']), np.mean(moved_df['UV_Decomposed'])]
        # # Plot the arrow, ensuring it is nice and thick
        # axs.arrow(mean_decomposed_uvj[0], mean_decomposed_uvj[1], mean_full_uvj[0] - mean_decomposed_uvj[0], mean_full_uvj[1] - mean_decomposed_uvj[1], head_width=0.05, head_length=0.05, fc='k', ec='k', linewidth=5)
    
        
    # Define paths for selections
    path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
    path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
    path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

    # We will obtain the Star-forming, Quiescent, and Dusty Fractions for this particular alpha value     
    # Obtain the number of galaxies with sf region dictated by the sf path
    # Create Path objects from your path coordinates (do this once, outside the loop)
    path_quiescent_obj = mpath.Path(path_quiescent)
    path_sf_obj = mpath.Path(path_sf)
    path_sfd_obj = mpath.Path(path_sfd)

    # Create a DataFrame with just the uv and vj columns for easier selection
    uvj_data = pd.DataFrame({'vj': vj, 'uv': uv})

  
    # add legend
    axs.legend(fontsize=12)

    # Add patches for selections
    axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
    axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
    axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

    # Add vertical line
    axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

    # Add axis labels
    axs.set_xlabel("V - J", fontsize=12)

    # ensure axis font size is same
    axs.tick_params(axis='both', which='major', labelsize=12)
    
    
    # Ensure limits
    axs.set_xlim(-0.5, 2.2)
    axs.set_ylim(0, 2.5)
        # Make the subplots closer together
    # plt.subplots_adjust(wspace=0.05, hspace=0.3)
    
    # adjust the x axis labels (45 degrees
    axs.set_xticklabels(axs.get_xticks(), rotation=45)
    

ax.set_ylabel("U - V", fontsize=12)


# Save 
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UVJ_agn_evolution_CIGALE_ZFOURGE_single_comparative.png', dpi=300, bbox_inches='tight')

plt.show()


# %%
# Explore the results of the sed decomposition in different mass/redshift bins
# For every mas redshift bin we look at we will also plot the above UVJ diagram, and see if there is a characteristic behaviour

# Create a heap of redshift bins


#
# begin by seperating master dataframe into redshift ranges of 0.2 to 0.8, 0.8 to 1.4, 1.4 to 2.0, 2.0 to 2.6, 2.6 to 3.2
# df_0_2 = df[(df['zpk'] >= 0.2) & (df['zpk'] < 0.8)]
# df_0_8 = df[(df['zpk'] >= 0.8) & (df['zpk'] < 1.4)]
# df_1_4 = df[(df['zpk'] >= 1.4) & (df['zpk'] < 2.0)]
# df_2_0 = df[(df['zpk'] >= 2.0) & (df['zpk'] < 2.6)]
# df_2_6 = df[(df['zpk'] >= 2.6) & (df['zpk'] < 3.2)]

# Redshift bins
z_bins = [(0.2, 0.8), (0.8, 1.4), (1.4, 2.0), (2.0, 2.6), (2.6, 3.2), (3.2, 3.8), (3.8, 4.4)]
#lmass_bins = [(8.5, 9.5), (9.5, 10.5), (10.5, 11.5), (11.5, 12.5), (12.5, 13.5)]
# maybe a reduce mass bin (9.25 to 9.75), 9.75 to 10.25, 10.25 to 10.75, 10.75 to 11.25, 11.25 to 11.75, 11.75 to 12.25, 12.25 to 12.75, 12.75 to 13.25
lmass_bins = [(8.75, 9.25),(9.25, 9.75),(9.75, 10.25), (10.25, 10.75), (10.75, 11.25), (11.25, 11.75)] # Target redshit bins

# Drop the lass mass bin
lmass_bins = [(8.75, 9.25),(9.25, 9.75),(9.75, 10.25), (10.25, 10.75), (10.75, 11.25), (11.25, 11.75), (11.75, 12.25), (12.25, 12.75), (12.75, 13.25)] # Target


# Alternatively create new mass bins to make less plot
lmass_bins = [(8.75, 9.75), (9.75, 10.75), (10.75, 11.75)] # Target

# Plot the distribution of masses
plt.figure(figsize=(8, 3))
plt.hist(zfourge_subset['lmass'], bins=50)
plt.xlabel('Log Stellar Mass')
plt.ylabel('Frequency')
plt.title('Histogram of Stellar Masses')
plt.show()






# %%
# How about for all redshifts plot the masses in bins

# Calculate the number of rows needed for subplots
num_rows = len(lmass_bins) 

# Create the figure and axes
fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows), sharex=True, sharey=True) 



for row_idx, lmass_bin in enumerate(lmass_bins):
    zfourge_subset_massbin = zfourge_subset[(zfourge_subset['lmass'] >= lmass_bin[0]) & (zfourge_subset['lmass'] < lmass_bin[1])].copy()
    print("lmass Bin:", lmass_bin)

    for col_idx in range(2):
        axs = axes[row_idx, col_idx]  # Get the axis for this subplot
        # Maybe just plot an all redshift UVJ diagram 

        # Reset the galaxy fractions
        galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
        num_galaxies = len(zfourge_subset_massbin) # number of galaxies in the composite_flux in the rest frame

        vj = zfourge_subset_massbin[f'VJ_{galaxy_mode[col_idx]}']
        uv = zfourge_subset_massbin[f'UV_{galaxy_mode[col_idx]}']

        # Plot the UVJ diagram
        axs.scatter(vj, uv, c='blue', s=10, alpha=0.5, marker='o')

        # Define paths for selections
        path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
        path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
        path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

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

        quiescent_fraction = (galaxy_fractions['Quiescent'][0])
        sf_fraction = galaxy_fractions['Star-forming'][0]
        sfd_fraction = galaxy_fractions['Dusty'][0]


            # Add patches for selections
        axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
        axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
        axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

        # Add vertical line
        axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

        # Add axis labels
        axs.set_xlabel("V - J")
        
        # # Plot the title on each subplot 
        # if galaxy_mode[col_idx] == 'Decomposed':
        #     axs.set_title(f"AGN Removed Galaxies")
        # else:
        #     axs.set_title(f"Unaltered Galaxies")
        # #For each plot, plot themass range
        axs.set_title(f"lmass: {lmass_bin[0]} to {lmass_bin[1]}")
        
        # 

        # Plot the fractions in the corner of each section on the UVJ
        axs.text(0.05, 0.9, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
        axs.text(0.05, 0.45, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
        axs.text(0.80, 0.05, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
        # AGN Colour Evolution title - just call it redshift
        # axs.set_title(f"Colour Evolution for ZFOURGE Composites")

        # Ensure limits
        axs.set_xlim(-0.5, 2.2)
        axs.set_ylim(0, 2.5)
            # Make the subplots closer together
        # plt.subplots_adjust(wspace=0.05, hspace=0.3)

        ax[0].set_ylabel("U - V")
plt.tight_layout()

# Save
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UVJ_agn_evolution_CIGALE_ZFOURGE_massbins.png')
    

plt.show()





# %%
# # For all masses plot the redshift in bins
# # Do the same thing but instead do redshift bins
# # How about for all redshifts plot the masses in bins

# # Calculate the number of rows needed for subplots
# num_rows = len(z_bins) 

# # Create the figure and axes
# fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows), sharex=True, sharey=True) 


# for row_idx, z_bin in enumerate(z_bins):
#     zfourge_subset_zbin = zfourge_subset[(zfourge_subset['zpk_x'] >= z_bin[0]) & (zfourge_subset['zpk_x'] < z_bin[1])]
#     print("z Bin:", z_bin)

#     for col_idx in range(2):
#         axs = axes[row_idx, col_idx]  # Get the axis for this subplot
#         # Maybe just plot an all redshift UVJ diagram 

#         # Reset the galaxy fractions
#         galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
#         num_galaxies = len(zfourge_subset_massbin) # number of galaxies in the composite_flux in the rest frame

#         vj = zfourge_subset_massbin[f'VJ_{galaxy_mode[col_idx]}']
#         uv = zfourge_subset_massbin[f'UV_{galaxy_mode[col_idx]}']

#         # Plot the UVJ diagram
#         axs.scatter(vj, uv, c='blue', s=10, alpha=0.5, marker='o')

#         # Define paths for selections
#         path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
#         path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
#         path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

#         # We will obtain the Star-forming, Quiescent, and Dusty Fractions for this particular alpha value     
#         # Obtain the number of galaxies with sf region dictated by the sf path
#         # Create Path objects from your path coordinates (do this once, outside the loop)
#         path_quiescent_obj = mpath.Path(path_quiescent)
#         path_sf_obj = mpath.Path(path_sf)
#         path_sfd_obj = mpath.Path(path_sfd)

#         # Create a DataFrame with just the uv and vj columns for easier selection
#         uvj_data = pd.DataFrame({'vj': vj, 'uv': uv})

#         # Perform the selection
#         quiescent_seds = uvj_data[path_quiescent_obj.contains_points(uvj_data.values)]
#         sf_seds = uvj_data[path_sf_obj.contains_points(uvj_data.values)]
#         sfd_seds = uvj_data[path_sfd_obj.contains_points(uvj_data.values)]

#         # Calculate the fractions
#         galaxy_fractions['Quiescent'].append(len(quiescent_seds) / num_galaxies)
#         galaxy_fractions['Star-forming'].append(len(sf_seds) / num_galaxies)
#         galaxy_fractions['Dusty'].append(len(sfd_seds) / num_galaxies)

#         quiescent_fraction = (galaxy_fractions['Quiescent'][0])
#         sf_fraction = galaxy_fractions['Star-forming'][0]
#         sfd_fraction = galaxy_fractions['Dusty'][0]


#             # Add patches for selections
#         axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
#         axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
#         axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

#         # Add vertical line
#         axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

#         # Add axis labels
#         axs.set_xlabel("V - J")
        
#         # # Plot the title on each subplot 
#         # if galaxy_mode[col_idx] == 'Decomposed':
#         #     axs.set_title(f"AGN Removed Galaxies")
#         # else:
#         #     axs.set_title(f"Unaltered Galaxies")
#         # #For each plot, plot themass range
#         axs.set_title(f"z: {z_bin[0]} to {z_bin[1]}")
        
#         # 

#         # Plot the fractions in the corner of each section on the UVJ
#         axs.text(0.05, 0.9, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
#         axs.text(0.05, 0.45, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
#         axs.text(0.80, 0.05, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
#         # AGN Colour Evolution title - just call it redshift
#         # axs.set_title(f"Colour Evolution for ZFOURGE Composites")

#         # Ensure limits
#         axs.set_xlim(-0.5, 2.2)
#         axs.set_ylim(0, 2.5)
#             # Make the subplots closer together
#         # plt.subplots_adjust(wspace=0.05, hspace=0.3)

#         ax[0].set_ylabel("U - V")
# plt.tight_layout()

# # Save
# if save_figures:
#     plt.savefig(f'outputs/ThesisPlots/UVJ_agn_evolution_CIGALE_ZFOURGE_zbins.png')
    

# plt.show()


# %%

for z_bin in z_bins:
    # Plot a UVJ diagram showing both the full and decomposed UVJ colours
    fig, ax = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True)

    zfourge_subset_zbin = zfourge_subset[(zfourge_subset['zpk_x'] >= z_bin[0]) & (zfourge_subset['zpk_x'] < z_bin[1])].copy()
    print("z Bin:", z_bin)
    for i in range(2):
        # Maybe just plot an all redshift UVJ diagram 
        axs = ax[i]
        # Reset the galaxy fractions
        galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
        num_galaxies = len(zfourge_subset_zbin) # number of galaxies in the composite_flux in the rest frame

        vj = zfourge_subset_zbin[f'VJ_{galaxy_mode[i]}']
        uv = zfourge_subset_zbin[f'UV_{galaxy_mode[i]}']

        # Plot the UVJ diagram
        axs.scatter(vj, uv, c='blue', s=10, alpha=0.5, marker='o')
        
        
        # Calculate the mean and std deviation for the Uv and VJ colours
        # ploting the mean and std deviation of the UVJ colours somewhere on the plot
        # Calculate the mean and std deviation for the Uv and VJ colours
        # mean_uv = np.mean(uv)
        # mean_vj = np.mean(vj)
        
        # std_uv = np.std(uv)
        # std_vj = np.std(vj)
        
        # # Plot the mean and std deviation of the UVJ colours
        # # plot a point and error bar for the mean and std deviation of the UVJ colours
        # axs.errorbar(mean_vj, mean_uv, xerr=std_vj, yerr=std_uv, fmt='o', color='black', markersize=5)
        
        # axs.text(0.05, 0.05, f'Mean U-V: {mean_uv:.2f}, Mean V-J: {mean_vj:.2f}', transform=axs.transAxes, color='k')
        # axs.text(0.05, 0.15, f'Std U-V: {std_uv:.2f}, Std V-J: {std_vj:.2f}', transform=axs.transAxes, color='k')
        
        # Define paths for selections
        path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
        path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
        path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

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
        
        
        quiescent_fraction = (galaxy_fractions['Quiescent'][0])
        sf_fraction = galaxy_fractions['Star-forming'][0]
        sfd_fraction = galaxy_fractions['Dusty'][0]
            
            
                # Add patches for selections
        axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
        axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
        axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))
        
        # Add vertical line
        axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)
        
        # Add axis labels
        axs.set_xlabel("V - J")
        
        # Plot the title on each subplot
        if galaxy_mode[i] == 'Decomposed':
            axs.set_title(f"AGN Removed Galaxies")
        else:
            axs.set_title(f"Unaltered Galaxies")
            
        # Plot the fractions in the corner of each section on the UVJ
        axs.text(0.05, 0.9, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
        axs.text(0.05, 0.45, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
        axs.text(0.80, 0.05, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
        
        # Ensure limits
        axs.set_xlim(-0.5, 2.2)
        axs.set_ylim(0, 2.5)
            # Make the subplots closer together
        # plt.subplots_adjust(wspace=0.05, hspace=0.3)
        
        ax[0].set_ylabel("U - V")
    plt.tight_layout()
    
    plt.show()
    
    

# %%
# Instead of plotting a bunch of UVJ diagrams, we can instead plot the entire normal and decomposed, 
# and for each of them we can plot the avergae (uvj redshift/mass bins and see the evolution)

# Colours for each redshift bin
colors = ['cyan', 'green', 'red', 'purple', 'orange', 'black', 'pink']

# Plot a UVJ diagram showing both the full and decomposed UVJ colours
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, sharey=True)


markers = {'Full': 'o', 'Decomposed': '^'}

plot_redshift_avg = True
plot_mass_avg = False
plot_redshift_mass_avg = False

for i in range(2):
    # Maybe just plot an all redshift UVJ diagram 
    axs = ax
    # Reset the galaxy fractions
    galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
    num_galaxies = len(zfourge_subset) # number of galaxies in the composite_flux in the rest frame

    vj = zfourge_subset[f'VJ_{galaxy_mode[i]}']
    uv = zfourge_subset[f'UV_{galaxy_mode[i]}']

    # Only ploy the full UVJ diagram
    if galaxy_mode[i] == 'Full':
        axs.scatter(vj, uv, c='blue', s=10, alpha=0.5, marker='o')

    # Redshift Bins
    if plot_redshift_avg:
        # Plot the average UVJ diagram for each redshift bin
        for z_bin in z_bins:
            zfourge_subset_zbin = zfourge_subset[(zfourge_subset['zpk_x'] >= z_bin[0]) & (zfourge_subset['zpk_x'] < z_bin[1])]
            print("z Bin:", z_bin)
            
            # We plot the average point for each redshift bin
            avg_vj = np.mean(zfourge_subset_zbin[f'VJ_{galaxy_mode[i]}'])
            avg_uv = np.mean(zfourge_subset_zbin[f'UV_{galaxy_mode[i]}'])
            
            # Use the color corresponding to the redshift bin
            color = colors[z_bins.index(z_bin)]  # Get color based on bin index

            if i == 0:
                
                
                # For the average point, plot the average error within that redshift bin from eUV_ and eVJ_
                avg_euv = np.mean(zfourge_subset_zbin[f'eUV_{galaxy_mode[i]}'])
                avg_evj = np.mean(zfourge_subset_zbin[f'eVJ_{galaxy_mode[i]}'])
            
                axs.errorbar(avg_vj, avg_uv, xerr=avg_evj, yerr=avg_euv, 
             fmt=markers[galaxy_mode[i]], color=color, markersize=10,  
             capsize=0, elinewidth=1,  
             alpha=0.90, label=f'{z_bin[0]} to {z_bin[1]}', 
             markeredgecolor='black')  # Changed 'edgecolor' to 'markeredgecolor'
                
                # axs.errorbar(avg_vj, avg_uv, xerr=avg_evj, yerr=avg_euv, fmt=markers[galaxy_mode[i]], 
                #      color=color, markersize=5)

                # axs.scatter(avg_vj, avg_uv, s=125, marker=markers[galaxy_mode[i]], 
                #     alpha=0.90, color=color, label=f'{z_bin[0]} to {z_bin[1]}', edgecolor='black')
                
            else:
                
                
                # For the average point, plot the average error within that redshift bin from eUV_ and eVJ_
                avg_euv = np.mean(zfourge_subset_zbin[f'eUV_{galaxy_mode[i]}'])
                avg_evj = np.mean(zfourge_subset_zbin[f'eVJ_{galaxy_mode[i]}'])
                
                axs.errorbar(avg_vj, avg_uv, xerr=avg_evj, yerr=avg_euv, 
             fmt=markers[galaxy_mode[i]], color=color, markersize=10,  
             capsize=0, elinewidth=1,  
             alpha=0.90, 
             markeredgecolor='black')  # Changed 'edgecolor' to 'markeredgecolor'
                
                # axs.errorbar(avg_vj, avg_uv, xerr=avg_evj, yerr=avg_euv, fmt=markers[galaxy_mode[i]], 
                #      color=color, markersize=5)

                # axs.scatter(avg_vj, avg_uv, s=125, marker=markers[galaxy_mode[i]], 
                #     alpha=0.90, color=color, label=f'{z_bin[0]} to {z_bin[1]}', edgecolor='black')

                
                
    # Mass Bins
    if plot_mass_avg:
        # Plot the average UVJ diagram for each mass bin
        for lmass_bin in lmass_bins:
            zfourge_subset_massbin = zfourge_subset[(zfourge_subset['lmass'] >= lmass_bin[0]) & (zfourge_subset['lmass'] < lmass_bin[1])]
            print("lmass Bin:", lmass_bin)
            
            # Plot the average point for each mass bin
            avg_vj = np.mean(zfourge_subset_massbin[f'VJ_{galaxy_mode[i]}'])
            avg_uv = np.mean(zfourge_subset_massbin[f'UV_{galaxy_mode[i]}'])
            
             # Use the color corresponding to the redshift bin
            color = colors[lmass_bins.index(lmass_bin)]  # Get color based on bin index

            
            if i == 0:
                axs.scatter(avg_vj, avg_uv, s=125, marker=markers[galaxy_mode[i]], 
                            alpha=0.90, color=color, label=f'{lmass_bin[0]} to {lmass_bin[1]}', edgecolor='black')
                
            else:
                axs.scatter(avg_vj, avg_uv, s=125, marker=markers[galaxy_mode[i]], 
                        alpha=0.90, color=color, edgecolor='black')

    
    # # Redshift/Mass Bins <- Could be even more releastic to plot the mass path for multiple redshifts bins on seperate plots
    
    # if plot_redshift_avg & plot_mass_avg:
    #     # Plot the average UVJ diagram for each redshift and mass bin
    #     for z_bin in z_bins:
    #         zfourge_subset_zbin = zfourge_subset[(zfourge_subset['zpk_x'] >= z_bin[0]) & (zfourge_subset['zpk_x'] < z_bin[1])]
    #         print("z Bin:", z_bin)
    #         for lmass_bin in lmass_bins:
    #             zfourge_subset_massbin = zfourge_subset_zbin[(zfourge_subset_zbin['lmass'] >= lmass_bin[0]) & (zfourge_subset_zbin['lmass'] < lmass_bin[1])]
    #             print("lmass Bin:", lmass_bin)

    # Plot label for the redshift bins
    if plot_redshift_avg | plot_mass_avg:
        axs.legend(title='Redshift Bins', loc='upper left')
        

    # Define paths for selections
    path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
    path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
    path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

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

    quiescent_fraction = galaxy_fractions['Quiescent'][0]
    sf_fraction = galaxy_fractions['Star-forming'][0]
    sfd_fraction = galaxy_fractions['Dusty'][0]


        # Add patches for selections
    axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
    axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
    axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

    # Add vertical line
    axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

    # Add axis labels
    axs.set_xlabel("V - J")
    axs.set_ylabel("U - V")

    # Plot the fractions in the corner of each section on the UVJ
    # axs.text(0.05, 0.9, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
    # axs.text(0.05, 0.45, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
    # axs.text(0.80, 0.05, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
    # AGN Colour Evolution title - just call it redshift
    # axs.set_title(f"Colour Evolution for ZFOURGE Composites")

    # Ensure limits
    axs.set_xlim(-0.5, 2.2)
    axs.set_ylim(0, 2.5)
        # Make the subplots closer together
    # plt.subplots_adjust(wspace=0.05, hspace=0.3)

#ax[0].set_ylabel("U - V")
plt.tight_layout()



bin_type = ''
if plot_mass_avg:
    bin_type = 'mass'
elif plot_redshift_avg:
    bin_type = 'redshift'

# Save 
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UVJ_agn_evolution_CIGALE_ZFOURGE_comparison_{bin_type}.png')

plt.show()

# %%
# We will do a similar thing, but instead we will create many redshift plots with the mass bins on them
# Instead of plotting a bunch of UVJ diagrams, we can instead plot the entire normal and decomposed, 
# and for each of them we can plot the avergae (uvj redshift/mass bins and see the evolution)
for z_bin in z_bins:
    
    # zfourge_subset to be the redshift bin
    zfourge_subset_zbin = zfourge_subset[(zfourge_subset['zpk_x'] >= z_bin[0]) & (zfourge_subset['zpk_x'] < z_bin[1])]
    
    # Plot a UVJ diagram showing both the full and decomposed UVJ colours
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharex=True, sharey=True)



    plot_redshift_avg = False
    plot_mass_avg = True
   # plot_redshift_mass_avg = False

    for i in range(2):
        # Maybe just plot an all redshift UVJ diagram 
        axs = ax
        # Reset the galaxy fractions
        galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
        num_galaxies = len(zfourge_subset) # number of galaxies in the composite_flux in the rest frame
        
        
        
        vj = zfourge_subset_zbin[f'VJ_{galaxy_mode[1]}']
        uv = zfourge_subset_zbin[f'UV_{galaxy_mode[1]}']

        
        if i == 0:    
            # Plot the UVJ diagram
            axs.scatter(vj, uv, c='blue', s=10, alpha=0.5, marker='o')

        
                            
        # Mass Bins
        if plot_mass_avg:
            # Plot the average UVJ diagram for each mass bin
            for lmass_bin in lmass_bins:
                zfourge_subset_massbin = zfourge_subset_zbin[(zfourge_subset_zbin['lmass'] >= lmass_bin[0]) & (zfourge_subset_zbin['lmass'] < lmass_bin[1])]

                
                # Plot the average point for each mass bin
                avg_vj = np.mean(zfourge_subset_massbin[f'VJ_{galaxy_mode[i]}'])
                avg_uv = np.mean(zfourge_subset_massbin[f'UV_{galaxy_mode[i]}'])
                
                axs.scatter(avg_vj, avg_uv, s=75, marker='o', alpha=0.90, label=f'{galaxy_mode[i]}: {lmass_bin[0]}-{lmass_bin[1]}')
     
        
        # # Redshift/Mass Bins <- Could be even more releastic to plot the mass path for multiple redshifts bins on seperate plots
        
        # if plot_redshift_avg & plot_mass_avg:
        #     # Plot the average UVJ diagram for each redshift and mass bin
        #     for z_bin in z_bins:
        #         zfourge_subset_zbin = zfourge_subset[(zfourge_subset['zpk_x'] >= z_bin[0]) & (zfourge_subset['zpk_x'] < z_bin[1])]
        #         print("z Bin:", z_bin)
        #         for lmass_bin in lmass_bins:
        #             zfourge_subset_massbin = zfourge_subset_zbin[(zfourge_subset_zbin['lmass'] >= lmass_bin[0]) & (zfourge_subset_zbin['lmass'] < lmass_bin[1])]
        #             print("lmass Bin:", lmass_bin)

        # Plot label for the redshift bins
        if plot_redshift_avg | plot_mass_avg:
            axs.legend()
            

        # Define paths for selections
        path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
        path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
        path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

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

        quiescent_fraction = galaxy_fractions['Quiescent'][0]
        sf_fraction = galaxy_fractions['Star-forming'][0]
        sfd_fraction = galaxy_fractions['Dusty'][0]


            # Add patches for selections
        axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
        axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
        axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))

        # Add vertical line
        axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)

        # Add axis labels
        axs.set_xlabel("V - J")
        

        # Plot the fractions in the corner of each section on the UVJ
        axs.text(0.05, 0.9, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
        axs.text(0.05, 0.45, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
        axs.text(0.80, 0.05, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
        # AGN Colour Evolution title - just call it redshift
        # axs.set_title(f"Colour Evolution for ZFOURGE Composites")

        # Ensure limits
        axs.set_xlim(-0.5, 2.2)
        axs.set_ylim(0, 2.5)
            # Make the subplots closer together
        # plt.subplots_adjust(wspace=0.05, hspace=0.3)

    #ax[0].set_ylabel("U - V")
    plt.tight_layout()

    bin_type = ''
    if plot_mass_avg:
        bin_type = 'mass'
    elif plot_redshift_avg:
        bin_type = 'redshift'

    # Save 
    if save_figures:
        plt.savefig(f'outputs/ThesisPlots/UVJ_agn_evolution_CIGALE_ZFOURGE_comparison_{bin_type}.png')

    plt.show()

# %%
# Try a new plot style
def plotdecomposeduvj(df):
    # Plot a UVJ diagram showing both the full and decomposed UVJ colours
    fig, axs = plt.subplots(1, 1, figsize=(7, 5), sharex=True, sharey=True)


    # Maybe just plot an all redshift UVJ diagram 

    # Reset the galaxy fractions
    galaxy_fractions = {'Star-forming': [], 'Quiescent': [], 'Dusty': []} # each key will have a list of fractions for each alpha value
    num_galaxies = len(zfourge_subset_zbin) # number of galaxies in the composite_flux in the rest frame

    vj_deocomposed = zfourge_subset_zbin[f'VJ_{galaxy_mode[0]}']
    uv_decomposed = zfourge_subset_zbin[f'UV_{galaxy_mode[0]}']

    vj_full = zfourge_subset_zbin[f'VJ_{galaxy_mode[1]}']
    uv_full = zfourge_subset_zbin[f'UV_{galaxy_mode[1]}']
    


    # Plot the UVJ diagram
    axs.scatter(vj_deocomposed, uv_decomposed, c='blue', s=10, alpha=0.5, marker='o')
    axs.scatter(vj_full, uv_full, c='red', s=10, alpha=0.5, marker='x')
    
    # # Plot an arrow from the full to the decomposed
    # for i in range(len(vj_deocomposed)):
    #     axs.arrow(vj_full[i], uv_full[i], vj_deocomposed[i] - vj_full[i], uv_decomposed[i] - uv_full[i], color='black', head_width=0.05)
        
     
    # # Define paths for selections
    # path_quiescent = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5]]
    # path_sf = [[-0.5, 0.0], [-0.5, 1.3], [0.85, 1.3], [1.2, 1.60333], [1.2, 0.0]]
    # path_sfd = [[1.2, 0.0], [1.2, 1.60333], [1.6, 1.95], [1.6, 2.5], [2.2, 2.5], [2.2, 0.0]]

    # # We will obtain the Star-forming, Quiescent, and Dusty Fractions for this particular alpha value     
    # # Obtain the number of galaxies with sf region dictated by the sf path
    # # Create Path objects from your path coordinates (do this once, outside the loop)
    # path_quiescent_obj = mpath.Path(path_quiescent)
    # path_sf_obj = mpath.Path(path_sf)
    # path_sfd_obj = mpath.Path(path_sfd)

    # # Create a DataFrame with just the uv and vj columns for easier selection
    # uvj_data = pd.DataFrame({'vj': vj, 'uv': uv})
    
    # # Perform the selection
    # quiescent_seds = uvj_data[path_quiescent_obj.contains_points(uvj_data.values)]
    # sf_seds = uvj_data[path_sf_obj.contains_points(uvj_data.values)]
    # sfd_seds = uvj_data[path_sfd_obj.contains_points(uvj_data.values)]
    
    # # Calculate the fractions
    # galaxy_fractions['Quiescent'].append(len(quiescent_seds) / num_galaxies)
    # galaxy_fractions['Star-forming'].append(len(sf_seds) / num_galaxies)
    # galaxy_fractions['Dusty'].append(len(sfd_seds) / num_galaxies)
    
    
    # quiescent_fraction = (galaxy_fractions['Quiescent'][0])
    # sf_fraction = galaxy_fractions['Star-forming'][0]
    # sfd_fraction = galaxy_fractions['Dusty'][0]
        
        
    #         # Add patches for selections
    # axs.add_patch(plt.Polygon(path_quiescent, closed=True, fill=True, facecolor=(1, 0, 0, 0.03), edgecolor='k', linewidth=2, linestyle='solid'))
    # axs.add_patch(plt.Polygon(path_sf, closed=True, fill=True, facecolor=(0, 0, 1, 0.03)))
    # axs.add_patch(plt.Polygon(path_sfd, closed=True, fill=True, facecolor=(1, 1, 0, 0.03)))
    
    # # Add vertical line
    # axs.axvline(1.2, color='black', linestyle='--', ymin=0, ymax=1.60333/2.5)
    
    # # Add axis labels
    # axs.set_xlabel("V - J")
    
    # # Plot the title on each subplot
    # if galaxy_mode[i] == 'Decomposed':
    #     axs.set_title(f"AGN Removed Galaxies")
    # else:
    #     axs.set_title(f"Unaltered Galaxies")
        
    # # Plot the fractions in the corner of each section on the UVJ
    # axs.text(0.05, 0.9, f'{quiescent_fraction:.2f}', transform=axs.transAxes, color='k')
    # axs.text(0.05, 0.45, f'{sf_fraction:.2f}', transform=axs.transAxes, color='k')
    # axs.text(0.80, 0.05, f'{sfd_fraction:.2f}', transform=axs.transAxes, color='k')
    
    # Ensure limits
    axs.set_xlim(-0.5, 2.2)
    axs.set_ylim(0, 2.5)
        # Make the subplots closer together
    # plt.subplots_adjust(wspace=0.05, hspace=0.3)
    
    axs.set_ylabel("U - V")
plt.tight_layout()
plt.show()


# %%
plotdecomposeduvj(zfourge_subset)

# %% [markdown]
# Now we aim to look at the ugr diagram to see how, if anything redshift can be affecting the colours of the galaxies.
# 

# %%
fig, axs = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True)



selection_statistics = {'Missed Selection': [], 'Misidentification': [], 'Correct Identification': [], 'Correct Non-UGR Identification': []}
# Realistically we are only interested in the correct identification and missed selection, but we will include all.
num_galaxies = len(zfourge_subset) # Because we consider all redshit ranges

# Completeness and contamination table
completeness_stats = {'Completeness': [] }


# # Get the u, g, and r magnitudes for the specific alpha value

for i in range(2):
    
    ax = axs[i]
    
    # Create the colours for the UGR diagram
    ug_specific_alpha_colours = zfourge_subset[f'UG_{galaxy_mode[i]}']
    gr_specific_alpha_colours = zfourge_subset[f'GR_{galaxy_mode[i]}']


    # Scatter plots with color-coded redshift ranges
    ax.scatter(gr_specific_alpha_colours[zfourge_subset['zpk_x'] > 3.5], 
                                ug_specific_alpha_colours[zfourge_subset['zpk_x'] > 3.5], 
                                c="c", s=10)

    ax.scatter(gr_specific_alpha_colours[(zfourge_subset['zpk_x'] > 2.6) & (zfourge_subset['zpk_x'] <= 3.5)], 
                                ug_specific_alpha_colours[(zfourge_subset['zpk_x'] > 2.6) & (zfourge_subset['zpk_x'] <= 3.5)], 
                                c="y", s=10)

    ax.scatter(gr_specific_alpha_colours[zfourge_subset['zpk_x'] < 2.6], 
                                ug_specific_alpha_colours[zfourge_subset['zpk_x'] < 2.6], 
                                c="m", s=10)
    
    # Additionally we can plot the UGR selection criteria on the UGR diagram
    U_rule = [[1.2,9], [1.2,2.2], [0.6,1.6], [-3,1.6], [-3,9]]

    # Add patch to correct plot
    ax.add_patch(plt.Polygon(U_rule, closed=True, fill=True, facecolor=(1,0,0,0.05), edgecolor=(0,0,0,1), linewidth=2, linestyle='solid')) # This looks like the correct U dropout

    
    # Set limits
    plt.xlim(-1, 4)
    plt.ylim(-1, 8)
    
    # Set labels
    ax.set_xlabel('g - r')
    # Only set the y label for the left plot
    if i == 0:
        ax.set_ylabel('u - g')
    
    
    
    # Do the completeness 
    path = mpath.Path(U_rule)
    
    # Create a DataFrame with just the u-g and g-r columns for easier selection
    ugr_data = pd.DataFrame({'gr': gr_specific_alpha_colours, 'ug': ug_specific_alpha_colours})
    
    # append the associated redshifts to the ugr data
    redshifts = zfourge_subset['zpk_x']
    
    
    
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
    contamination = len(misidentification) / (len(correct_ugr_selection) + len(misidentification))
    
    # Append the values
    completeness_stats['Completeness'].append(completeness)
    #completeness_stats['Contamination'].append(contamination)
    
    # add to plot
    ax.text(0.1, 0.85, f'{completeness:.2f}', transform=ax.transAxes, color='k')

    
plt.tight_layout()    
    
# save
if save_figures:
    plt.savefig(f'outputs/ThesisPlots/UGR_agn_evolution_CIGALE_ZFOURGE_comparison.png')
        
        

plt.show()


# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 


