#!/usr/bin/env python
# coding: utf-8

# In[4]:


# We would like to investigate the effects of the different filters.
# Read in the dataframe

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pandas as pd
import os
from astLib import astSED
import astropy.io.fits as fits
from carf import * # custom module for functions relating to the project
import matplotlib.path as mpath


# In[5]:





master_df = []

#field = 'COSMOS'
agn_model_name = 'Type2AGN'

fields = ['COSMOS', 'UDS', 'CDFS']

for field in fields:
    composite_choice = f'{field}_obsevational_composites_fluxes{agn_model_name}'
    composite_fluxes = pd.read_csv(f'outputs\composite_seds\{composite_choice}.csv', index_col=0)   
    
    # Add field name in a new column
    composite_fluxes['field'] = field
    
    # Add to the master dataframe
    master_df.append(composite_fluxes)
    
master_df = pd.concat(master_df)
master_df.reset_index(drop=True, inplace=True)


# In[6]:


# Save the master dataframe
master_df.to_csv(f'outputs/composite_seds/ZFOURGE_obsevational_composites_fluxes{agn_model_name}.csv')


# In[ ]:




