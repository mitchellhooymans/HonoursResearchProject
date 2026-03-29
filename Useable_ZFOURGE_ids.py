#!/usr/bin/env python
# coding: utf-8

# # Useable ZFOURGE Id's 
# This will be used to get the useable ZFOURGE Id's that will be used in the output code
#  

# In[12]:


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


# In[13]:


# We would like to read in the fits files that we are exploring in this project
# This is related to the data that we are using
# we will be looking at all fields so it will be easier to read in all required fits files, and all recalculated IDs and combine these 
# into three master dataframes.
# From here we will be able to check the best values for each and eventually select some reliabile samples

zfourge_path = 'datasets/zfourge/'

# Read in ZFourge Data in each field


# In[14]:


def read_zfourge_data(fieldname, folderpath): # Define a function to read in zfourge data, this will be added to the helper package later
    # Dictionary to read from
    zfourge_fields = {
    'CDFS': ['zf_cdfs.fits', 'zf_cdfs_rest.fits', 'zf_cdfs_eazy.fits', 'zf_cdfs_sfr.fits'],
    'COSMOS': ['zf_cosmos.fits', 'zf_cosmos_rest.fits', 'zf_cosmos_eazy.fits', 'zf_cosmos_sfr.fits'],
    'UDS': ['zf_uds.fits', 'zf_uds_rest.fits', 'zf_uds_eazy.fits', 'zf_uds_sfr.fits'],
}
    
    folder = folderpath
    
    # Construct file paths using os.path.join() to make it platform-independent
    catalog_file = os.path.join(folder, zfourge_fields[fieldname][0])
    rest_file = os.path.join(folder, zfourge_fields[fieldname][1])
    eazy_file = os.path.join(folder, zfourge_fields[fieldname][2])
    sfr_file = os.path.join(folder, zfourge_fields[fieldname][3])
    
    # Open the fits files <- needs to be different for CAT vs Fits files
    catalog_fits = fits.open(catalog_file)
    rest_fits = fits.open(rest_file)
    sfr_fits = fits.open(sfr_file)
    
    # Read the files into DataFrames <- needs to be different for CAT vs Fits files
    df = pd.DataFrame(np.array(catalog_fits[1].data).byteswap().newbyteorder()) 
    rest_df = pd.DataFrame(np.array(rest_fits[1].data).byteswap().newbyteorder())
    eazy_df = pd.DataFrame(np.array(fits.open(eazy_file)[1].data).byteswap().newbyteorder())
    sfr_df = pd.DataFrame(np.array(sfr_fits[1].data).byteswap().newbyteorder())
    
    
    # Rename the Seq column to id for consistency
    df.rename(columns={'Seq':'id'}, inplace=True)
    rest_df.rename(columns={'Seq':'id', 'FU':'U', 'e_FU':'eU', 'FV':'V', 'e_FV':'eV', 'FJ':'J','e_FJ':'eJ'}, inplace=True)
    eazy_df.rename(columns={'Seq':'id'}, inplace=True)
    sfr_df.rename(columns={'Seq':'id'}, inplace=True)
    
    
    # We now merge the two dataframes into one dataframe, adding a suffix _rest if columns clash
    #df = pd.merge(df, rest_df, on='id', suffixes=('', '_rest'))
    
    # we now merge rest and df into one
    #df = pd.concat([df, rest_df], axis=1)
    df = pd.merge(df, rest_df[['id', 'U', 'eU', 'V', 'eV', 'J','eJ']], on='id', suffixes=('_original', '_rest'))
    df = pd.merge(df, eazy_df[['id', 'zpk']], on='id', suffixes=('', '_eazy'))
    df = pd.merge(df, sfr_df[['id', 'lssfr', 'lmass']], on='id', suffixes=('', '_sfr'))
    
    
    # Create a new column to mark the field that this data is from
    df['field'] = fieldname + "_"
    
    # In this scenario we don't need to append which field it's from
    fieldname = ""
    
    # rename the number in the id column to be prefixed by the fieldname, this is to avoid confusion when merging dataframes
    df['id'] = fieldname + df['id'].astype(str)
    
    # return the created dataframe
    return df


# In[15]:



#CDFS, COSMOS, UDS
cdfs_df = read_zfourge_data('CDFS', zfourge_path)
cosmos_df = read_zfourge_data('COSMOS', zfourge_path)
uds_df = read_zfourge_data('UDS', zfourge_path)


# In[16]:


df_names = ['cdfs', 'cosmos', 'uds']
i = 0
# create a dictionary
id_dict = {'id': [], 'z': [], 'field': []}
for df in [cdfs_df, cosmos_df, uds_df]:
    # Filter the data as necessay
    # flux filtering, set a sigma value for the flux error ratio
    sigma = 3

# optionally we can filter the data into fields
#field = 'CDFS'
#df = df[df['field'] == field].copy()

    # filter for uvj colours, making sure that there isn't a flux below 0
    df = df[(df['U'] > 0) & (df['V'] > 0) & (df['J'] > 0) & (df['eU'] > 0) & (df['eV'] > 0) & (df['eJ'] > 0) & (df['Use']==1)].copy()

    # we also need to filter by the redshift, Z-fourge only has reliable accuracy for redshifts between 0.2 and 3.2 ~ potentially up to a redshift of 4
    # As my project will be investigating redshifts of galaxies where z-0.5~2 we should instead use this range
    min_redshift = 0
    max_redshift = 5

    df = df[(df['zpk'] > min_redshift) & (df['zpk'] < max_redshift)].copy()



    # Propogate errors from each of the fluxes to the UVJ diagram to a ratio of sigma
    df = df[(df['U'] >= sigma * df['eU']) & (df['V'] >= sigma * df['eV']) & (df['J'] >= sigma * df['eJ'])].copy()
    
    
    # append the id, z and field to the dictionary
    id_dict['id'].extend(df['id'])
    id_dict['z'].extend(df['zpk'])
    id_dict['field'].extend([df_names[i]]*len(df))
    
    
    
        
    print(f'{df_names[i]} has {len(df)} useable sources')
    i+=1


# Checkout the 


# In[17]:


id_dict

# Check length of each array in the dict
print(len(id_dict['id']), len(id_dict['z']), len(id_dict['field']))


# In[18]:


# Have a look at the uds


# In[19]:


id_df = pd.DataFrame(id_dict)


# In[20]:


# 


# In[23]:


# Output the ids to a csv file
id_df.to_csv('outputs/useable_zfourge_ids.csv', index=False)
# Output the ids to a csv file
id_df.to_csv('Eazy Template Extractions/useable_zfourge_ids.csv', index=False)


# In[22]:


# Check the max redshift for the uds


