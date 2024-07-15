# # This is the master SED Proccessing file that will do the majority of the SED Proccessing,
# # It will condense the logic from most of the Preliminary Project Work notebooks into one notebook.


# # Import the necessary packages
import sys
import os

import numpy as np
import pandas as pd
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import qarg_tools as qt


# # Parameters


# # Load Data
# #filters = qt.load_passbands('UVJ')
# #print(filters)


# Load UVJ filters
filters = qt.load_passbands('UVJ')
print(filters[0])


# load Type 1 model
agn_model = qt.load_agn_model('Type 1')

print(agn_model)


# Create agn_model SED
agn_sed = qt.create_sed(agn_model)

print(agn_sed)

# # Plot the AGN Model SED wl against flux
# plt.figure(figsize=(10,6))
# plt.loglog(agn_sed.wavelength, agn_sed.flux)  
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Flux (erg/s/cm^2/A)')
# plt.title('AGN Model SED')
# plt.show()



# Attempt to read in the SWIRE templates using the functions in qarg_tools.py
swire_templates = qt.load_galaxy_template_set('SWIRE', 'datasets/templates/SWIRE')

print(swire_templates['dataframe'][2])

# Create an SED from this
swire_sed = qt.create_sed(swire_templates['dataframe'][15])

# # Plot the SED
# plt.figure(figsize=(10,6))
# plt.loglog(swire_sed.wavelength, swire_sed.flux)
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Flux (erg/s/cm^2/A)')
# plt.title('SWIRE SED')
# plt.show()




# Attempt to read in the SWIRE templates using the functions in qarg_tools.py
brown_templates = qt.load_galaxy_template_set('BROWN', 'datasets/templates/BROWN/2014/rest')

print(brown_templates['dataframe'][2])

# Create an SED from this
brown_sed = qt.create_sed(brown_templates['dataframe'][15])

# # Plot the SED
# plt.figure(figsize=(10,6))
# plt.loglog(brown_sed.wavelength, brown_sed.flux)
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Flux (erg/s/cm^2/A)')
# plt.title('SWIRE SED')
# plt.show()



# combine the AGN and brown seds then plot
combined_sed = qt.create_composite_sed(agn_sed, brown_sed, 0.5)


# Plot the SED
plt.figure(figsize=(10,6))
plt.loglog(combined_sed.wavelength, combined_sed.flux, label='Combined SED')

# plot the constituent SEDs
plt.loglog(agn_sed.wavelength, agn_sed.flux, label='AGN SED')
plt.loglog(brown_sed.wavelength, brown_sed.flux, label='Brown SED')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (erg/s/cm^2/A)')
plt.title('Combined SED')
plt.show()


