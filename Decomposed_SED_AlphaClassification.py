# %%
# Import all required packages
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

# refresh

# So that we can change the helper functions without reloading the kernel
%load_ext autoreload
%autoreload 2

# %%
# This is a quick script intended to be used to classify the sources into Alpha bins based on moved fraction


# Based on something like this. In our quick analysis we are only considering the Alpha bins from 0 to 50

# Alpha	Quiescent	Star Forming (VJ >0.5)	Star Forming (VJ <0.5)	Dusty
# 0	0.0	0.000000	0.000000	0.000000	0.000000
# 1	10.0	0.100288	0.053567	0.026047	0.137492
# 2	20.0	0.186021	0.096860	0.048387	0.247190
# 3	30.0	0.260914	0.134219	0.068025	0.337960
# 4	40.0	0.327321	0.167330	0.085585	0.414995
# 5	50.0	0.386867	0.197124	0.101484	0.481609
# 6	60.0	0.440738	0.224211	0.116019	0.540057
# 7	70.0	0.489829	0.249023	0.129406	0.591938
# 8	80.0	0.534838	0.271889	0.141811	0.638432
# 9	90.0	0.576318	0.293066	0.153363	0.680434
# 10	100.0	0.614721	0.312760	0.164167	0.718636

# read in data
decomposed_ids = pd.read_csv('Decomposed_UVJ_Ids.csv')

# %%
decomposed_ids


# %%

# get rid of the first column
decomposed_ids = decomposed_ids.drop('Unnamed: 0', axis=1)

# %%
def classify_alpha(distance_moved):
    if distance_moved <0.1:
        return 0
    elif distance_moved <0.186:
        return 1
    elif distance_moved <0.261:
        return 2
    elif distance_moved <0.327:
        return 3
    elif distance_moved <0.387:
        return 4
    elif distance_moved <0.441:
        return 5
    elif distance_moved <0.49:
        return 6
    elif distance_moved <0.535:
        return 7
    elif distance_moved <0.576:
        return 8
    elif distance_moved <0.615:
        return 9
    else:
        return 10
    

# %%
# classify our individual galaxies into bins
decomposed_ids['Alpha'] = decomposed_ids['vector_magnitude'].apply(lambda x: classify_alpha(x))

# %%
decomposed_ids

# %%
# We would like to check this distribution
decomposed_ids['Alpha'].value_counts()

# %%
# Lets investigate the distribution of the Alpha bins: exploring alpha values of 2, 3, and 4
decomposed_ids[decomposed_ids['Alpha'].isin([2,3,4, 5, 6, 7, 8, 9, 10])]

# %%
# plot the distribution
plt.hist(decomposed_ids['Alpha'], bins=11, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Alpha')
plt.ylabel('Number of Galaxies')
plt.title('Distribution of Alpha bins')

plt.show()

# %%
# 



