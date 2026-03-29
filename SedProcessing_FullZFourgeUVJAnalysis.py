#!/usr/bin/env python
# coding: utf-8

# # Full Processing of UVJ Data for the ZFourge Survey
# This script is used to perform the processing of data generated using the template extraction script. This script is used to investigate how the UVJ positions of ZFOURGE change with an increasing amount of contamination from the AGN. In addition to this, this script will be used to choose a selection of IDs across each survey which will the be used to generate some more plots for the final paper.

# In[1]:


# Import all relevant libraries
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


# We would like to read in the fits files that we are exploring in this project
# This is related to the data that we are using
# we will be looking at all fields so it will be easier to read in all required fits files, and all recalculated IDs and combine these 
# into three master dataframes.
# From here we will be able to check the best values for each and eventually select some reliabile samples

zfourge_path = 'datasets/zfourge/'

# Read in ZFourge Data in each field

#CDFS, COSMOS, UDS
cdfs_df = read_zfourge_data('CDFS', zfourge_path)
cosmos_df = read_zfourge_data('COSMOS', zfourge_path)
uds_df = read_zfourge_data('UDS', zfourge_path)


# In[3]:


# Read in all the recalculated UVJ colours for the id's
# as a dataframe
cdfs_recalcUVJids = pd.read_csv('datasets/zfourge/CDFS_RecalculatedUVJids_full.csv')
cosomos_recalcUVJids = pd.read_csv('datasets/zfourge/COSMOS_RecalculatedUVJids_full.csv')
uds_recalcUVJids = pd.read_csv('datasets/zfourge/UDS_RecalculatedUVJids_full.csv')

# cigale & astSED calculated uvj colours
cigale_uvj_ids = pd.read_csv('outputs/cigale_colours_ids.csv')


# Rename id's to lowercase
cdfs_recalcUVJids = cdfs_recalcUVJids.rename(columns={'ID':'id'})
cosomos_recalcUVJids = cosomos_recalcUVJids.rename(columns={'ID':'id'})
uds_recalcUVJids = uds_recalcUVJids.rename(columns={'ID':'id'})


# In[4]:


# Now that we have both the inital dataframe + the recalculated UVJ id's we can merge these together
# after this we should merge all of our dataframes together to create a master dataframe
# This will allow us to easily access all of the data that we need
#cdfs_df = pd.merge(cdfs_df, cdfs_recalcUVJids, on='id')
#cosmos_df = pd.merge(cosmos_df, cosomos_recalcUVJids, on='id')
#uds_df = pd.merge(uds_df, uds_recalcUVJids, on='id')
cigale_uvj_ids = cigale_uvj_ids.rename(columns={'ID':'id'})


# In[5]:


cigale_uvj_ids


# In[6]:


cdfs_df


# In[7]:


cdfs_recalcUVJids


# In[8]:


# Merge the recalculated UVJ id's with the original dataframes
cdfs_df = pd.merge(cdfs_df, cdfs_recalcUVJids, on='id')
cosmos_df = pd.merge(cosmos_df, cosomos_recalcUVJids, on='id')
uds_df = pd.merge(uds_df, uds_recalcUVJids, on='id')


# In[9]:


# See if this worked
# We are now able to put all the dataframes into one dataframe
# This will allow us to easily access all the data that we need

# Making one dataframe
frames = [cdfs_df, cosmos_df, uds_df]
zfourge_df = pd.concat(frames)


# In[10]:


# Before we continue we must look to see the distribution of values that have moved based on calculations
# We will look at the recalculated UVJ values and see how many have changed with the new method
# Removing significant changes for the time being 

zfourge_df['vector_magnitude_original'] = np.sqrt(zfourge_df['UV']**2 + zfourge_df['VJ']**2)
zfourge_df['vector_magnitude_recalculated'] = np.sqrt(zfourge_df['UV_0']**2 + zfourge_df['VJ_0']**2)


# now we can calculate the difference between the two
zfourge_df['vector_magnitude_difference'] = abs(zfourge_df['vector_magnitude_original'] - zfourge_df['vector_magnitude_recalculated'])


# Do the same for cigale uvj colours
cigale_uvj_ids['vector_magnitude_original'] = np.sqrt(cigale_uvj_ids['UV']**2 + cigale_uvj_ids['VJ']**2)

# create a zfourge_df subset containing uvj id's from cigale
zfourge_df_cigale = zfourge_df[zfourge_df['id'].isin(cigale_uvj_ids['id'])]
# reset index
zfourge_df_cigale = zfourge_df_cigale.reset_index(drop=True)


# In[11]:


zfourge_df


# In[12]:


zfourge_df_cigale['vector_magnitude_recalculated_cigale'] = abs(cigale_uvj_ids['vector_magnitude_original'] - zfourge_df_cigale['vector_magnitude_original'])
zfourge_df_cigale['vector_magnitude_difference_cigale'] = abs(cigale_uvj_ids['vector_magnitude_original']  - zfourge_df_cigale['vector_magnitude_recalculated'])


# In[13]:


# Use seaborn to plot the histograms
import seaborn as sns


# In[14]:


zfourge_df[zfourge_df['field'] == 'CDFS']['vector_magnitude_difference']


# In[15]:


# Plot the histograms
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
sns.histplot(zfourge_df[zfourge_df['field'] == 'CDFS']['vector_magnitude_difference'], bins=100, color='blue', ax=ax)
ax.set_xlabel('Vector Magnitude Difference')
ax.set_ylabel('Count')
ax.set_xlim([0, 1])
plt.savefig('outputs/vector_magnitude_difference_cdfs_example.png')
plt.show()


# In[16]:


# Plot the histograms
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
sns.histplot(zfourge_df_cigale['vector_magnitude_recalculated_cigale'], bins=100, color='blue', ax=ax)
ax.set_xlabel('Vector Magnitude Difference')
ax.set_ylabel('Count')
ax.set_title("recalculated UVJ colours using cigale SED fit")
#ax.set_xlim([0, 1])
plt.show()


# In[17]:


# Plot the histograms
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
sns.histplot(zfourge_df_cigale['vector_magnitude_difference_cigale'], bins=100, color='blue', ax=ax)
ax.set_xlabel('Vector Magnitude Difference')
ax.set_ylabel('Count')
ax.set_title("recalculated UVJ colours using cigale SED fit")
#ax.set_xlim([0, 1])
plt.show()


# In[18]:


# Plot the histograms
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
sns.histplot(zfourge_df[zfourge_df['field'] == 'COSMOS']['vector_magnitude_difference'], bins=100, color='blue', ax=ax)
ax.set_xlabel('Vector Magnitude Difference')
ax.set_ylabel('Count')
ax.set_xlim([0, 1])
plt.show()


# In[19]:


# Plot the histograms
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
sns.histplot(zfourge_df[zfourge_df['field'] == 'UDS']['vector_magnitude_difference'], bins=100, color='blue', ax=ax)
ax.set_xlabel('Vector Magnitude Difference')
ax.set_ylabel('Count')
ax.set_xlim([0, 1])
plt.show()


# In[20]:


# We would like to drop all values that have a vector magnitude difference greater than 0.2

# Dropping these values
#zforuge_df = zfourge_df[zfourge_df['vector_magnitude_difference'] < 0.2]



# In[21]:


# Plot the UVJ diagram

def plot_uvj_nocategorise(df, x, y):
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
    plt.title('UVJ Diagram for the CDFS field')
    plt.legend()
    plt.show()


# In[22]:


# Plot the UVJ diagram

def plot_uvj_nocategorise(df, x, y):
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
    plt.title('UVJ Diagram for the CDFS field')
    plt.legend()
    plt.show()


# In[ ]:





# In[30]:


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

    plt.savefig('outputs/uvj_diagram_example_original.png')
    plt.show()
    return df


# In[31]:


zfourge_df = categorise_uvj(zfourge_df, zfourge_df['VJ'], zfourge_df['UV'])


# In[25]:


# For each value of alpha, plot the UVJ diagram
alpha_list = np.linspace(0, 1, 11)


# In[26]:


alpha_list


# In[ ]:


alpha = 0
zfourge_df = categorise_uvj(zfourge_df, zfourge_df['VJ_{}'.format(int(alpha*100))], zfourge_df['UV_{}'.format(int(alpha*100))], alpha)


# In[500]:


for alpha in alpha_list:
    zfourge_df = categorise_uvj(zfourge_df, zfourge_df['VJ_{}'.format(int(alpha*100))], zfourge_df['UV_{}'.format(int(alpha*100))], alpha)


# In[54]:



def categorise_uvj_both(df1, df2, x_col1='x1', y_col1='y1', x_col2='x2', y_col2='y2', alpha=None):
    """
    Plots a UVJ diagram with two DataFrames side-by-side, categorizing galaxies and highlighting selections.

    Args:
        df1, df2: DataFrames containing the galaxy data.
        x_col1, y_col1: Column names in df1 for the x and y values of the UVJ diagram.
        x_col2, y_col2: Column names in df2 for the x and y values of the UVJ diagram.
        alpha: (Optional) Parameter for AGN contamination, used for column naming if provided.
    """

    # Plotting limits and quiescent selection area (same for both plots)
    xmax = 2.2
    ymax = 2.5
    xmin = -0.5
    ymin = 0

    x_points = [-0.5, 0.85, 1.6, 1.6]
    y_points = [1.3, 1.3, 1.95, 2.5]
    x_target = 1.2
    y_target = np.interp(x_target, x_points, y_points)

    quiescent_x = [-0.5, 0.85, 1.6, 1.6, xmin, xmin]
    quiescent_y = [1.3, 1.3, 1.95, 2.5, ymax, 1.3]
    verts = np.array([quiescent_x, quiescent_y]).T
    path = mpath.Path(verts)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(6, 6), sharey=True, sharex=True)

    # Helper function to classify and plot for a single DataFrame
    def classify_and_plot(ax, df, x_col, y_col, alpha):
        x, y = df[x_col], df[y_col]
        points = np.column_stack([x, y])
        points_inside_selection = path.contains_points(points)

        dusty_condition = (points[:, 0] > x_target) & (~points_inside_selection)
        star_forming_condition = (points[:, 0] < x_target) & (~points_inside_selection)

        # Classify galaxies 
        if alpha is None:
            df.loc[dusty_condition, 'GalaxyType'] = 2
            df.loc[star_forming_condition, 'GalaxyType'] = 1
            selected_ids = df[points_inside_selection]['id'] 
            df.loc[df['id'].isin(selected_ids), 'GalaxyType'] = 0
        else:
            df.loc[dusty_condition, f'GalaxyType_{int(alpha*100)}'] = 2
            df.loc[star_forming_condition, f'GalaxyType_{int(alpha*100)}'] = 1
            selected_ids = df[points_inside_selection]['id']
            df.loc[df['id'].isin(selected_ids), f'GalaxyType_{int(alpha*100)}'] = 0

        # Plot 
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.plot(x_points, y_points, linestyle='-')
        ax.plot([x_target, x_target], [0, y_target], linestyle='--')

        quiescent_points = points[path.contains_points(points)]
        non_quiescent_points = points[~path.contains_points(points)]
        dusty_galaxies_points = non_quiescent_points[non_quiescent_points[:, 0] > x_target]
        star_forming_galaxies_points = non_quiescent_points[non_quiescent_points[:, 0] < x_target]

        ax.scatter(quiescent_points[:, 0], quiescent_points[:, 1], c='r', s=3, alpha=0.5, label='Quiescent')
        ax.scatter(dusty_galaxies_points[:, 0], dusty_galaxies_points[:, 1], c='g', s=3, alpha=0.5, label='Dusty')
        ax.scatter(star_forming_galaxies_points[:, 0], star_forming_galaxies_points[:, 1], c='b', s=3, alpha=0.5, label='Star-forming')
        ax.set_xlabel('Restframe V-J [Mag]')
    #    ax.legend()

    # Plot for DataFrame 1
    classify_and_plot(axs[0], df1, x_col1, y_col1, alpha)
    axs[0].set_ylabel('Restframe U-V [Mag]')
    
    # Plot for DataFrame 2
    classify_and_plot(axs[1], df2, x_col2, y_col2, alpha)
    
    # Add one legend
    axs[0].legend()
    # Layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1) 

    # Save or show the plot
    plt.savefig('outputs/uvj_subplots.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Return both modified DataFrames
    return df1, df2


# In[55]:


# Attempt to plot the UVJ diagram for the original and recalculated UVJ colours
alpha = 0
categorise_uvj_both(zfourge_df, zfourge_df, 'VJ', 'UV', 'VJ_{}'.format(int(alpha*100)), 'UV_{}'.format(int(alpha*100)))


# In[501]:


# Now we will look at the best 20 sources of each region, in each field.
# that did not move much based on their vector magnitude difference

# For each field, and each region, select the 20 sources with the lowest magnitude difference
n = 300
# CDFS
# Quiescent
cdfs_quiescent = zfourge_df[(zfourge_df['field'] == 'CDFS') & (zfourge_df['GalaxyType'] == 0)].sort_values(by='vector_magnitude_difference').head(n)
# Dusty
cdfs_dusty = zfourge_df[(zfourge_df['field'] == 'CDFS') & (zfourge_df['GalaxyType'] == 2)].sort_values(by='vector_magnitude_difference').head(n)
# Star Forming
cdfs_star_forming = zfourge_df[(zfourge_df['field'] == 'CDFS') & (zfourge_df['GalaxyType'] == 1)].sort_values(by='vector_magnitude_difference').head(n)

# COSMOS
# Quiescent
cosmos_quiescent = zfourge_df[(zfourge_df['field'] == 'COSMOS') & (zfourge_df['GalaxyType'] == 0)].sort_values(by='vector_magnitude_difference').head(n)
# Dusty
cosmos_dusty = zfourge_df[(zfourge_df['field'] == 'COSMOS') & (zfourge_df['GalaxyType'] == 2)].sort_values(by='vector_magnitude_difference').head(n)
# Star Forming
cosmos_star_forming = zfourge_df[(zfourge_df['field'] == 'COSMOS') & (zfourge_df['GalaxyType'] == 1)].sort_values(by='vector_magnitude_difference').head(n)

# UDS
# Quiescent
uds_quiescent = zfourge_df[(zfourge_df['field'] == 'UDS') & (zfourge_df['GalaxyType'] == 0)].sort_values(by='vector_magnitude_difference').head(n)
# Dusty
uds_dusty = zfourge_df[(zfourge_df['field'] == 'UDS') & (zfourge_df['GalaxyType'] == 2)].sort_values(by='vector_magnitude_difference').head(n)
# Star Forming
uds_star_forming = zfourge_df[(zfourge_df['field'] == 'UDS') & (zfourge_df['GalaxyType'] == 1)].sort_values(by='vector_magnitude_difference').head(n)


# Combining these all back into a more refined dataframe,

frames = [cdfs_quiescent, cdfs_dusty, cdfs_star_forming, cosmos_quiescent, cosmos_dusty, cosmos_star_forming, uds_quiescent, uds_dusty, uds_star_forming]
best_sources_df = pd.concat(frames)

best_sources_df = zfourge_df


# In[502]:


# Checking out the vector magnitude histrogram for the best sources
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
sns.histplot(best_sources_df['vector_magnitude_difference'], bins=100, color='blue', ax=ax)
ax.set_xlabel('Vector Magnitude Difference')
ax.set_ylabel('Count')
ax.set_title('Vector Magnitude Difference for the Best Sources n={}'.format(n))
ax.set_xlim([0, 0.5])
plt.show()


# In[503]:


# Now we have a selection of sources that can be considered good sources for the UVJ diagram

# Plot these sources on a UVJ diagram, all of them 

plt.scatter(best_sources_df['VJ_0'], best_sources_df['UV_0'], c='r', s=3, alpha=0.5, label='Best Sources')

plt.scatter(best_sources_df['VJ'], best_sources_df['UV'], c='b', s=3, alpha=0.5, label='All Sources')
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
plt.legend()
plt.show()


# In[504]:


# So there is a slight shift but it isn't drastically important for this as we are looking at the statistics behind what is actualy happening. We not however that in terms of what is generally tending to be miscalculated is the middle section of the UVJ diagram
# we see that this is one of the thinner parts of the diagram but is important as there is signitifcant shift of approx 1.0 dex due to something it could be due to the way things are being calculated in regards to the middle region or a fitting error that occurs in astSED vs Eazy


# In[505]:


# In doing this we continue and keep the sources here
# Plotting again with a lssfr overlay for each to see if it iin the same general type of spots
# do subplotting to see these graphs side by side

# Create 2 plots, plotting the UVJ diagram for the best sources with inital colours and not inital colours
plt.scatter(best_sources_df['VJ_0'], best_sources_df['UV_0'], c=best_sources_df['lssfr'], cmap='viridis', s=3, alpha=0.5, label='Best Sources')

# limit the colourbar between -8 and -11
plt.colorbar(label='log(SSFR)')

# limit cbar values
plt.clim(-11, -8)

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
plt.legend()
plt.show()


# In[506]:




plt.scatter(best_sources_df['VJ'], best_sources_df['UV'], c=best_sources_df['lssfr'], cmap='viridis', s=3, alpha=0.5, label='All Sources')

# limit the colourbar between -8 and -11
plt.colorbar(label='log(SSFR)')

# limit cbar values
plt.clim(-11, -8)

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
plt.legend()
plt.show()


# In[507]:


# We now want to look at this distribution, and make a informed selection on each region, recalling that we have actually used the previous code to classify the galaxy type based on UVJ
# we extend this by actually looking at UV_0 and VJ_0

best_sources_df = categorise_uvj(best_sources_df, best_sources_df['VJ_0'], best_sources_df['UV_0'], 0)


# We note the expected output, with the UVJ diagram looking much as intended aside from a rather large amount of scatter missing at extreme values of VJ/UV

# In[508]:


# Using this we may select a quiescent selection from each of the three fields. For completeness we may also select a dusty, and star formaing selection from each of the three fields. Evenetually we will out some id's to see how this evolves the colour space.
# for each of the 3 populations, plot the redshift distributions

# Quiescent: plotting all fields on one histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
#sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 0)]['zpk'], bins=100, color='blue', ax=ax, label='All Fields')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 0) & (best_sources_df['field'] == 'CDFS')]['zpk'], bins=100, color='red', ax=ax, label='CDFS')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 0) & (best_sources_df['field'] == 'COSMOS')]['zpk'], bins=100, color='green', ax=ax, label='COSMOS')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 0) & (best_sources_df['field'] == 'UDS')]['zpk'], bins=100, color='blue', ax=ax, label='UDS')
ax.set_xlabel('Redshift')
ax.set_ylabel('Count')
ax.set_title('Redshift Distribution for Quiescent Galaxies')
plt.legend()
plt.show()


# In[509]:


# Dusty: plotting all fields on one histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
#sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 2)]['zpk'], bins=100, color='blue', ax=ax, label='All Fields')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 2) & (best_sources_df['field'] == 'CDFS')]['zpk'], bins=100, color='red', ax=ax, label='CDFS')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 2) & (best_sources_df['field'] == 'COSMOS')]['zpk'], bins=100, color='green', ax=ax, label='COSMOS')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 2) & (best_sources_df['field'] == 'UDS')]['zpk'], bins=100, color='blue', ax=ax, label='UDS')
ax.set_xlabel('Redshift')
ax.set_ylabel('Count')
ax.set_title('Redshift Distribution for Dusty Galaxies')
plt.legend()
plt.show()


# In[510]:


# Finally for starforming
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
#sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 1)]['zpk'], bins=100, color='blue', ax=ax, label='All Fields')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 1) & (best_sources_df['field'] == 'CDFS')]['zpk'], bins=100, color='red', ax=ax, label='CDFS')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 1) & (best_sources_df['field'] == 'COSMOS')]['zpk'], bins=100, color='green', ax=ax, label='COSMOS')
sns.histplot(best_sources_df[(best_sources_df['GalaxyType_0'] == 1) & (best_sources_df['field'] == 'UDS')]['zpk'], bins=100, color='blue', ax=ax, label='UDS')
ax.set_xlabel('Redshift')
ax.set_ylabel('Count')
ax.set_title('Redshift Distribution for Star-forming Galaxies')
plt.legend()
plt.show()


# In[511]:


# Now we have an idea of redshift distribution we can look at finidng some good galaxies to investigate
# We importnatly would like to explore and investigate the quiescent galaxies in zfourge.

# To explore this further we split our data into the three fields, and then look at the quiescent galaxies in each field
# We then look at the redshift distribution of these galaxies

# Simply for quiescent galaxies, we want to see how the UVJ colours evolve with redshift

# Plotting the UVJ diagram for the quiescent galaxies in each field
# in redshift bins of 0.2, colour coding by field

# redshift bins
z_bins = np.linspace(0.2, 2, 10)

# for our calculated values, plot the redshift change
# Create 2 plots, plotting the UVJ diagram for the best sources with inital colours and not inital colours

# for each bin, plot the UVJ diagram
for i in range(len(z_bins)-1):
    # Create a dataframe for the redshift bin
    
    # Target each field with a specific colour, specifically looking at the quiescent galaxies
    zfourge_df_bin = zfourge_df[(zfourge_df['zpk'] > z_bins[i]) & (zfourge_df['zpk'] < z_bins[i+1]) & (zfourge_df['GalaxyType_0'] == 0)]
    # Plot the UVJ diagram
    plt.scatter(zfourge_df_bin[zfourge_df_bin['field'] == 'CDFS']['VJ_0'], zfourge_df_bin[zfourge_df_bin['field'] == 'CDFS']['UV_0'], c='r', s=3, alpha=0.5, label='CDFS')
    plt.scatter(zfourge_df_bin[zfourge_df_bin['field'] == 'COSMOS']['VJ_0'], zfourge_df_bin[zfourge_df_bin['field'] == 'COSMOS']['UV_0'], c='g', s=3, alpha=0.5, label='COSMOS')
    plt.scatter(zfourge_df_bin[zfourge_df_bin['field'] == 'UDS']['VJ_0'], zfourge_df_bin[zfourge_df_bin['field'] == 'UDS']['UV_0'], c='b', s=3, alpha=0.5, label='UDS')
    


    # limit cbar values
    plt.clim(-11, -8)

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
    plt.legend()
    plt.show()



# In[512]:


# We see a very particular redshift distribution, with high densities found at earlier redshifts, potentially due to observational bias.
# We now explore more of the UVJ, looking at the paths of the galaxies in the UVJ diagram
# with increasing AGN contribution

# For each alpha value, plot the UVJ diagram, plot the id, with a connecting line (limiting to best 100 sources)
# Quiestcent galaxies
alpha_list = np.linspace(0, 1, 11)


# Redfine cdfs quiescent on the UV_0 and VJ_0 colours
cdfs_quiescent = zfourge_df[(zfourge_df['field'] == 'CDFS') & (zfourge_df['GalaxyType_0'] == 0)]

fig = plt.figure(figsize=(6, 6))

# Plot each value of alpha
for alpha in alpha_list:
    plt.scatter(cdfs_quiescent['VJ_{}'.format(int(alpha*100))], cdfs_quiescent['UV_{}'.format(int(alpha*100))], c='b', s=3, alpha=0.5)
    

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





# In[513]:


# Creating the same output but in the other two fields
alpha_list = np.linspace(0, 1, 11)


# Redfine cdfs quiescent on the UV_0 and VJ_0 colours
cosmos_quiescent = zfourge_df[(zfourge_df['field'] == 'COSMOS') & (zfourge_df['GalaxyType_0'] == 0)]

fig = plt.figure(figsize=(6, 6))

# Plot each value of alpha
for alpha in alpha_list:
    plt.scatter(cosmos_quiescent['VJ_{}'.format(int(alpha*100))], cosmos_quiescent['UV_{}'.format(int(alpha*100))], c='b', s=3, alpha=0.5)
    

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


# In[514]:


# Finally for UDS
# Redfine cdfs quiescent on the UV_0 and VJ_0 colours
uds_quiescent = zfourge_df[(zfourge_df['field'] == 'UDS') & (zfourge_df['GalaxyType_0'] == 0)]

fig = plt.figure(figsize=(6, 6))

# Plot each value of alpha
for alpha in alpha_list:
    plt.scatter(uds_quiescent['VJ_{}'.format(int(alpha*100))], uds_quiescent['UV_{}'.format(int(alpha*100))], c='b', s=3, alpha=0.5)
    


plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours of AGN Composites")
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

plt.show()


# Again we have very similar shapes between each of these. Notably we don't see much of a difference between the fields. For this analysis I would suspect that we would like to consider
# galaxies that may have started in the UVJ diagram as purely quiescent, and have moved into the star forming or dusty region when the AGN contribution is 50% or below. We should expect to
# find a sample of these galaxies, which we can then plot.

# In[515]:


# We look at the quiescents that enter the SF region at agn contribution of 0.5 or below

# For each field, select the quiescent galaxies that enter the star-forming region at an AGN contribution of 0.5 or below, knnowing that this must be compared
# with the inital designation of 0, so check if quiescents GalaxyType_alpha becomes one at alpha = 0.5 or below
# We are also interested in if there are ones that become dusty or starforming
cutoff = 50 # This determines exactly at what cutoff we are investigating

# CDFS
cdfs_quiescent_transition = cdfs_quiescent[(cdfs_quiescent['GalaxyType_0'] == 0) & ((cdfs_quiescent['GalaxyType_{}'.format(cutoff)] == 1) | (cdfs_quiescent['GalaxyType_{}'.format(cutoff)] == 2))]

# COSMOS
cosmos_quiescent_transition = cosmos_quiescent[(cosmos_quiescent['GalaxyType_0'] == 0) & ((cosmos_quiescent['GalaxyType_{}'.format(cutoff)] == 1) | (cosmos_quiescent['GalaxyType_{}'.format(cutoff)] == 2))]

# UDS
uds_quiescent_transition = uds_quiescent[(uds_quiescent['GalaxyType_0'] == 0) & ((uds_quiescent['GalaxyType_{}'.format(cutoff)] == 1) | (uds_quiescent['GalaxyType_{}'.format(cutoff)] == 2))]


# In[516]:


uds_quiescent_transition # Seeing this, we can see that with an AGN contribution of 0.5 or below we see approx 100 galaxies from each region that enter the SFR

# We want to find the galaxies that move the most in the UVJ diagram, so we can investigate these further
# We can do this by looking at the vector magnitude difference between the inital and final UVJ colours
# We can then select the top 20 sources from each field

# For each field, and each region, select the 20 sources with the highest vector magnitude difference
n = 150
# CDFS
# Quiescent to star-forming
cdfs_quiescent_transition = cdfs_quiescent_transition.sort_values(by='vector_magnitude_difference', ascending=False).head(n)

cosmos_quiescent_transition = cosmos_quiescent_transition.sort_values(by='vector_magnitude_difference', ascending=False).head(n)

uds_quiescent_transition = uds_quiescent_transition.sort_values(by='vector_magnitude_difference', ascending=False).head(n)



# In[517]:


fig = plt.figure(figsize=(6, 6))

# Plot each value of alpha
for alpha in alpha_list:
    plt.scatter(cdfs_quiescent_transition['VJ_{}'.format(int(alpha*100))], cdfs_quiescent_transition['UV_{}'.format(int(alpha*100))], c='b', s=3, alpha=0.5)
    


plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours of AGN Composites")
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

plt.show()


# In[518]:


fig = plt.figure(figsize=(6, 6))

# Plot each value of alpha
for alpha in alpha_list:
    plt.scatter(cosmos_quiescent_transition['VJ_{}'.format(int(alpha*100))], cosmos_quiescent_transition['UV_{}'.format(int(alpha*100))], c='b', s=3, alpha=0.5)
    


plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours of AGN Composites")
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

plt.show()


# In[519]:


# We again replot these on the UVJ

fig = plt.figure(figsize=(6, 6))

# Plot each value of alpha
for alpha in alpha_list:
    plt.scatter(uds_quiescent_transition['VJ_{}'.format(int(alpha*100))], uds_quiescent_transition['UV_{}'.format(int(alpha*100))], c='b', s=3, alpha=0.5)
    


plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours of AGN Composites")
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

plt.show()


# In[520]:


# For each field let us plot an average UVJ diagram for the quiescent galaxies that move to the star-forming region/dusty
fig = plt.figure(figsize=(6, 6))

# Plot each value of alpha
cdfs_mean_values = []
cosmos_mean_values = []
uds_mean_values = []
for alpha in alpha_list:
    # CDFS
    # Calculate the mean values for each value of alpha, and then plot
    cdfs_mean_values.append([cdfs_quiescent_transition['VJ_{}'.format(int(alpha*100))].mean(), cdfs_quiescent_transition['UV_{}'.format(int(alpha*100))].mean()])
    cosmos_mean_values.append([cosmos_quiescent_transition['VJ_{}'.format(int(alpha*100))].mean(), cosmos_quiescent_transition['UV_{}'.format(int(alpha*100))].mean()])
    uds_mean_values.append([uds_quiescent_transition['VJ_{}'.format(int(alpha*100))].mean(), uds_quiescent_transition['UV_{}'.format(int(alpha*100))].mean()])
    
    
    #plt.scatter(uds_quiescent_transition['VJ_{}'.format(int(alpha*100))], uds_quiescent_transition['UV_{}'.format(int(alpha*100))], c='b', s=3, alpha=0.5)
# Plot those mean values
# Plot the values on the uvj
plt.scatter([i[0] for i in cdfs_mean_values], [i[1] for i in cdfs_mean_values], c='r', label='CDFS', s=7)
plt.plot([i[0] for i in cdfs_mean_values], [i[1] for i in cdfs_mean_values], c='k')
plt.scatter([i[0] for i in cosmos_mean_values], [i[1] for i in cosmos_mean_values], c='g', label='COSMOS', s=7)
plt.plot([i[0] for i in cosmos_mean_values], [i[1] for i in cosmos_mean_values], c='k')
plt.scatter([i[0] for i in uds_mean_values], [i[1] for i in uds_mean_values], c='b', label='UDS', s=7)
plt.plot([i[0] for i in uds_mean_values], [i[1] for i in uds_mean_values], c='k')

plt.ylabel('U - V')
plt.xlabel('V - J')
plt.title("Restframe UVJ Colours of AGN Composites")
plt.xlim([-0.5, 2.2])
plt.ylim([0, 2.5])
plt.legend()

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


# In[547]:


# Check the line length, from the inital to the final point
# For each field, calculate the line length for each galaxy, and then plot the distribution of these line lengths
# CDFS
linelength_cdfs = np.sqrt((cdfs_quiescent_transition['VJ_0'] - cdfs_quiescent_transition['VJ_{}'.format(cutoff)])**2 + (cdfs_quiescent_transition['UV_0'] - cdfs_quiescent_transition['UV_{}'.format(cutoff)])**2)
linelength_cosmos = np.sqrt((cosmos_quiescent_transition['VJ_0'] - cosmos_quiescent_transition['VJ_{}'.format(cutoff)])**2 + (cosmos_quiescent_transition['UV_0'] - cosmos_quiescent_transition['UV_{}'.format(cutoff)])**2)
linelength_uds = np.sqrt((uds_quiescent_transition['VJ_0'] - uds_quiescent_transition['VJ_{}'.format(cutoff)])**2 + (uds_quiescent_transition['UV_0'] - uds_quiescent_transition['UV_{}'.format(cutoff)])**2)
# remove the outlier above 1 line length
linelength_uds = linelength_uds[linelength_uds < 1]


# In[548]:


# plot the histrograms for each
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
sns.histplot(linelength_cdfs, bins=25, color='red', ax=ax, label='CDFS')
sns.histplot(linelength_cosmos, bins=25, color='green', ax=ax, label='COSMOS')
sns.histplot(linelength_uds, bins=25, color='blue', ax=ax, label='UDS')
ax.set_xlabel('Line Length')
ax.set_ylabel('Count')
ax.set_title('Line Length Distribution for Quiescent Galaxies')
plt.legend()
plt.show()



# calculate the mean for each
print('CDFS: {}'.format(linelength_cdfs.mean()))
print('COSMOS: {}'.format(linelength_cosmos.mean()))
print('UDS: {}'.format(linelength_uds.mean()))


# In[521]:


# Now that we know this, we can potentially find a good selection of galaxies to investigate further
cdfs_quiescent_transition



# In[522]:


cosmos_quiescent_transition


# In[523]:


uds_quiescent_transition


# In[524]:


# We would now like to the ids for each of these galaxies, their redshifts (photometric), and their field they were selected from
# we would like to do this for each field

# CDFS
cdfs_quiescent_transition[['id', 'zpk', 'field']]
# COSMOS
cosmos_quiescent_transition[['id', 'zpk', 'field']]
# UDS
uds_quiescent_transition[['id', 'zpk', 'field']]



# strip the field id from the id, and then concatenate
cdfs_quiescent_transition['id'] = cdfs_quiescent_transition['id'].str[5:]
cosmos_quiescent_transition['id'] = cosmos_quiescent_transition['id'].str[7:]
uds_quiescent_transition['id'] = uds_quiescent_transition['id'].str[4:]


# In[528]:


# Concatenate these, and export
quiescent_transition = pd.concat([cdfs_quiescent_transition, cosmos_quiescent_transition, uds_quiescent_transition])


# In[531]:


quiescent_transition_ids = quiescent_transition[['id', 'zpk', 'field']]

# plot the distributions of redshifts
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot a histogram for each field: UDS, COSMOS, CDFS
sns.histplot(quiescent_transition_ids[quiescent_transition_ids['field'] == 'CDFS']['zpk'], bins=100, color='red', ax=ax, label='CDFS')
sns.histplot(quiescent_transition_ids[quiescent_transition_ids['field'] == 'COSMOS']['zpk'], bins=100, color='green', ax=ax, label='COSMOS')
sns.histplot(quiescent_transition_ids[quiescent_transition_ids['field'] == 'UDS']['zpk'], bins=100, color='blue', ax=ax, label='UDS')

ax.set_xlabel('Redshift')
ax.set_ylabel('Count')
ax.set_title('Redshift Distribution for Quiescent Galaxies')
plt.legend()
plt.show()


# This shows a very similar distribution to the population distribution from before. Thus what we see if that the values
# that move through this space just so happen to be well distributed across the population.

# In[532]:


# We may now export the DF 
quiescent_transition_ids.to_csv('outputs/quiescent_transition_ids.csv', index=False)

