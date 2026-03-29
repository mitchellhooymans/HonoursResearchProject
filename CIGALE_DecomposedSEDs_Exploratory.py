# %% [markdown]
# # Decomposed ZFOURGE Galaxies UVJ Colour Analysis
# This script is intended to be used to perform a colour analysis on the UVJ colours that have been calculated in the script SEDProcessing_DecomposedSEDs_Full. This will extend the analysis and will atempt to create some metric to be able to explore the UVJ colour space. Additinally we also will attempt to read in and combine the UVJ colours as per the id's with actual data from the ZFOURGE survey to explore the UVJ colour space in more detail. This will be intended to be used as a base for the rest of my analysis in other colour spaces.

# %%
# Read in required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from carf import *


# %%
# Now we can read in the colours of the galaxies
decomposed_colours = pd.read_csv('datasets/full_zfourge_decomposed/decomposed_colours.csv')

# Check length of the output
print(decomposed_colours.shape)


# %%
# Check where na <- this may have been an inccorrect sampling.
print(decomposed_colours.isna().sum())


# %%
decomposed_colours.isna().sum()

# Drop the rows with missing values
decomposed_colours = decomposed_colours.dropna()

# recount the number of missing values
decomposed_colours.isna().sum()


# Check the length of the output
print(decomposed_colours.shape)


# %%
# Quickly create a pairplot to se the relationships between the colours, from fully decomposed to the original galaxy colours
sns.pairplot(decomposed_colours)

# %%
# Now we would like to make two plots, one for the fully decomposed colours and one for the original galaxy colours. These should be UVJ colour plots and should highlight the difference between the two sets of colours.

# extract the fully decomposed colours
vj_galaxy_colours = decomposed_colours['VJ_Decomposed']
uv_galaxy_colours = decomposed_colours['UV_Decomposed']

# extract the original galaxy colours
vj_full_colours = decomposed_colours['VJ_Full']
uv_full_colours = decomposed_colours['UV_Full']



plt.figure(figsize=(6, 6))

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

# as all the points have associated names, plot the names
#for i, txt in enumerate(df['ID']):
#    plt.annotate(txt, (x[i], y[i]))
    
# plot the points
plt.scatter(vj_full_colours, uv_full_colours, s=3, alpha=0.5, label='Galaxies (with AGN)')

plt.scatter(vj_galaxy_colours, uv_galaxy_colours, s=3, alpha=0.5, label='Galaxies (No AGN)')


# Instead of plotting all arrows, define an average arrow 
# we want to get the average location of each set of points, and plot the transition


# Get the average location of each set of points
avg_vj_full = np.mean(vj_full_colours)
avg_uv_full = np.mean(uv_full_colours)

avg_vj_galaxy = np.mean(vj_galaxy_colours)
avg_uv_galaxy = np.mean(uv_galaxy_colours)


print(f'Average VJ Full: {avg_vj_full}'
      f'Average UV Full: {avg_uv_full}'
      f'Average VJ Galaxy: {avg_vj_galaxy}'
      f'Average UV Galaxy: {avg_uv_galaxy}')

# Plot the average arrow
plt.arrow(avg_vj_full, avg_uv_full, avg_vj_galaxy - avg_vj_full, avg_uv_galaxy - avg_uv_full, head_width=0.025, head_length=0.05, fc='k', ec='k')



# Interpolate the y-value at x=1.2
x_target = 1.2
y_target = np.interp(x_target, x_points, y_points)

# Plot the interpolated point, this separates everything on the right as dusty galaxies,
# and everything on the left as star-forming galaxies
plt.plot([x_target, x_target], [0, y_target], linestyle='--')

plt.xlabel('Restframe V-J [Mag]')
plt.ylabel('Restframe U-V [Mag]')
plt.title('UVJ Diagram for the all fields')
plt.legend()
#plt.savefig('outputs/UVJ_Diagram_with_AGN_decom_Allfields_Avg.png')
plt.show()

# %%
# We would like to extract subsets of this data
# extract the fully decomposed colours


# Only select rows which have a UV colour of 1 or great AND a VJ colour of 0.75 or greater

uv_thresh = 1
vj_thresh = 1

condition = (decomposed_colours['UV_Decomposed'] >= uv_thresh) & (decomposed_colours['VJ_Decomposed'] >= vj_thresh)

# additional constraint on full colours
condition = condition & (decomposed_colours['UV_Full'] >= uv_thresh) & (decomposed_colours['VJ_Full'] >= vj_thresh)


decomposed_colours_filtered = decomposed_colours[condition]


vj_galaxy_colours = decomposed_colours_filtered['VJ_Decomposed']
uv_galaxy_colours = decomposed_colours_filtered['UV_Decomposed']

# extract the original galaxy colours
vj_full_colours = decomposed_colours_filtered['VJ_Full']
uv_full_colours = decomposed_colours_filtered['UV_Full']


# We would like to filter the colours to only include the galax


plt.figure(figsize=(6, 6))

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

# as all the points have associated names, plot the names
#for i, txt in enumerate(df['ID']):
#    plt.annotate(txt, (x[i], y[i]))
    
# plot the points
plt.scatter(vj_full_colours, uv_full_colours, s=3, alpha=0.5, label='Galaxies (with AGN)')

plt.scatter(vj_galaxy_colours, uv_galaxy_colours, s=3, alpha=0.5, label='Galaxies (No AGN)')


# Instead of plotting all arrows, define an average arrow 
# we want to get the average location of each set of points, and plot the transition


# Get the average location of each set of points
avg_vj_full = np.mean(vj_full_colours)
avg_uv_full = np.mean(uv_full_colours)

avg_vj_galaxy = np.mean(vj_galaxy_colours)
avg_uv_galaxy = np.mean(uv_galaxy_colours)


print(f'Average VJ Full: {avg_vj_full}'
      f'Average UV Full: {avg_uv_full}'
      f'Average VJ Galaxy: {avg_vj_galaxy}'
      f'Average UV Galaxy: {avg_uv_galaxy}')

# Plot the average arrow
plt.arrow(avg_vj_full, avg_uv_full, avg_vj_galaxy - avg_vj_full, avg_uv_galaxy - avg_uv_full, head_width=0.025, head_length=0.05, fc='k', ec='k')



# Interpolate the y-value at x=1.2
x_target = 1.2
y_target = np.interp(x_target, x_points, y_points)

# Plot the interpolated point, this separates everything on the right as dusty galaxies,
# and everything on the left as star-forming galaxies
plt.plot([x_target, x_target], [0, y_target], linestyle='--')

plt.xlabel('Restframe V-J [Mag]')
plt.ylabel('Restframe U-V [Mag]')
plt.title('UVJ Diagram for the all fields')
plt.legend()
#plt.savefig('outputs/UVJ_Diagram_with_AGN_decom_Allfields_Avg.png')
plt.show()

# %%
# We want to do a similar technique but instead  we want to create 10 different thresholds ranging from 0.5 to 1 in steps of 0.05
thresholds = np.arange(0.5, 1.05, 0.05)
print(thresholds)

# %%
for threshold in thresholds:
    uv_thresh = threshold
    vj_thresh = threshold

    condition = (decomposed_colours['UV_Decomposed'] >= uv_thresh) & (decomposed_colours['VJ_Decomposed'] >= vj_thresh)

    # additional constraint on full colours
    condition = condition & (decomposed_colours['UV_Full'] >= uv_thresh) & (decomposed_colours['VJ_Full'] >= vj_thresh)


    decomposed_colours_filtered = decomposed_colours[condition]


    vj_galaxy_colours = decomposed_colours_filtered['VJ_Decomposed']
    uv_galaxy_colours = decomposed_colours_filtered['UV_Decomposed']

    # extract the original galaxy colours
    vj_full_colours = decomposed_colours_filtered['VJ_Full']
    uv_full_colours = decomposed_colours_filtered['UV_Full']


    # We would like to filter the colours to only include the galax


    plt.figure(figsize=(6, 6))

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

    # as all the points have associated names, plot the names
    #for i, txt in enumerate(df['ID']):
    #    plt.annotate(txt, (x[i], y[i]))
        
    # plot the points
    plt.scatter(vj_full_colours, uv_full_colours, s=3, alpha=0.5, label='Galaxies (with AGN)')

    plt.scatter(vj_galaxy_colours, uv_galaxy_colours, s=3, alpha=0.5, label='Galaxies (No AGN)')


    # Instead of plotting all arrows, define an average arrow 
    # we want to get the average location of each set of points, and plot the transition


    # Get the average location of each set of points
    avg_vj_full = np.mean(vj_full_colours)
    avg_uv_full = np.mean(uv_full_colours)

    avg_vj_galaxy = np.mean(vj_galaxy_colours)
    avg_uv_galaxy = np.mean(uv_galaxy_colours)


    print(f'Average VJ Full: {avg_vj_full}'
        f'Average UV Full: {avg_uv_full}'
        f'Average VJ Galaxy: {avg_vj_galaxy}'
        f'Average UV Galaxy: {avg_uv_galaxy}')

    # Plot the average arrow
    plt.arrow(avg_vj_full, avg_uv_full, avg_vj_galaxy - avg_vj_full, avg_uv_galaxy - avg_uv_full, head_width=0.025, head_length=0.05, fc='k', ec='k')

    # Additionally plot some text next to the average arrow displaying the magnitude of the length of the arrow
    plt.text(avg_vj_full + 0.1, avg_uv_full + 0.1, f'{round(np.sqrt((avg_vj_galaxy - avg_vj_full)**2 + (avg_uv_galaxy - avg_uv_full)**2), 2)}', fontsize=8, color='red')

    # also calculate and plot the degrees
    angle = np.arctan2(avg_uv_galaxy - avg_uv_full, avg_vj_galaxy - avg_vj_full) * 180 / np.pi
    
    # Plot below the arrow 
    plt.text(avg_vj_full + 0.1, avg_uv_full - 0.1, f'{round(angle, 2)}', fontsize=8, color='red')
    
    # Interpolate the y-value at x=1.2
    x_target = 1.2
    y_target = np.interp(x_target, x_points, y_points)

    # Plot the interpolated point, this separates everything on the right as dusty galaxies,
    # and everything on the left as star-forming galaxies
    plt.plot([x_target, x_target], [0, y_target], linestyle='--')

    # Include text with the threshold value
    plt.text(-0.25, 0.5, f' UV/VJ Threshold: {round(threshold, 2)}', fontsize=7, color='red')

    plt.xlabel('Restframe V-J [Mag]')
    plt.ylabel('Restframe U-V [Mag]')
    plt.title('UVJ Diagram for the all fields')
    plt.legend()
    # Save the plot with the thresh value
    plt.savefig(f'outputs/UVJ_Diagram_with_AGN_decom_Allfields_Avg_{round(threshold, 2)}.png')
plt.show()

# %% [markdown]
# From running this analysis we see that by taking slices of this decomposition based on the threshold of values we are looking at, we can effectively see how the inital UVJ colours change. Notably we can see that mostly red quiescent galaxies, and dusty galaxies show the largest amount of change to their colurrs and moving up into the right most area shows that this transition will take place. 

# %%
# We would now like to combine this dataframe, with the cdfs, uds, and cosmos dataframes, so that we can generate insights into the information
# of the actual night sky sources.

#cdfs
zfourge_path = 'datasets/zfourge/'

# Read in ZFourge Data in each field

#CDFS, COSMOS, UDS
cdfs_df = read_zfourge_data('CDFS', zfourge_path)
cosmos_df = read_zfourge_data('COSMOS', zfourge_path)
uds_df = read_zfourge_data('UDS', zfourge_path)


# %%
# Now I can choose to combine these dataframes into a master frame base
master_df = pd.concat([cdfs_df, cosmos_df, uds_df])


# %%
master_df

# rename id to ID
master_df = master_df.rename(columns={'id': 'ID'})

# %%
decomposed_colours

# %%
# Now we would like to merge the decomposed colours with the master dataframe

master_df = master_df.merge(decomposed_colours, on='ID', how='inner')

# %%
master_df

# %%
# Check where zpk_x and zpk_y are different
## master_df['zpk_x'] = master_df['zpk_x'].fillna(master_df['zpk_y'])


# %%
# Show a distribution of the redshifts


plt.hist(master_df['zpk_y'], bins=100, color='blue', alpha=0.5)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Redshift Distribution of the CIGALE ZFOURGE Data')

plt.show()

# %%
# Now that we gave a dataframe containing enough of the information that we are concerend with we should be able to generate some insights
# This is a useful dataframe. We should choose to export this so way may look through the data and analyze it

master_df.to_csv('datasets/full_zfourge_decomposed/zfourge_full.csv', index=False)

# %%



