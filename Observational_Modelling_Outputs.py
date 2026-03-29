# %% [markdown]
# # ZFOURGE SED Template Processing 
# Using the generated ZFOURGE SED templates from the cdfs field, we explore how these SEDs will react in the UVJ colour space to see if the UVJ coordinates are still in the sample place.
# 

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
import random

# Set a seed for the random number generator 
random.seed(42) 

# So that we can change the helper functions without reloading the kernel
%load_ext autoreload
%autoreload 2

# %%
save_outputs = False

# %%
# Skirtor models
skirtor_folderpath = os.path.join('datasets', 'Templates', 'Skirtor')

# Read in the AGN templates
type1_agn, type1_params = create_type1_skirtor_agn(skirtor_folderpath)
type2_agn, type2_params = create_type2_skirtor_agn(skirtor_folderpath)

# %%


# Choose running parameters

# Template set and AGN type
# Before we begin, lets recall what field we are exploring

field = 'COSMOS'
agn_model_name = 'Type2AGN'

# Setup how many alpha values we want to explore
alpha_values = np.linspace(0, 1, 11)

# %%
# Choose an AGN model
if agn_model_name == 'Type1AGN':
    agn_model = type1_agn
elif agn_model_name == 'Type2AGN':
    agn_model = type2_agn
else:
    print('AGN model not recognised')

# if field  == 'SWIRE':
#     template_set = swire_templates
#     template_names = template_names
# elif template_set_name == 'Brown':
#     template_set = brown_templates
#     template_names = brown_template_names
# else:
#     print('Template set not recognised')

# %%


# We are attempting to read in a new set of SEDs that have been generated using a variation of Michael's ZFOURGE SED processing code.
# the SEDs are in the form of csvs, this can be explored in the zfourge/seds
# directory. The csvs are in the form of:
# Wavelength (microns), Flux Density (10^-19erg_s_cm^2_Angstrom)

zfourge_folderpath = os.path.join('Eazy Template Extractions', 'zfourgeSEDs/'+field+'/')
# We would like to create a function to read this in
def read_zfourge_template(folder_path, name):
    """_summary_

    Args:
        folder_path (string): path to the folder where the SED templates are located
        name (string): name of the object
    
    Returns:
        df: Returns a dataframe containing the SED template
        objname: Returns the name of the object
    """
    folder_path = os.path.join(folder_path)
    files_in_folder = os.listdir(folder_path)

    for file in files_in_folder:
        # Find filepath and convert to df
        
        # get rid of units in the filename
        objname = file.split('SED_')[1]
        
        objname = objname.split('.csv')[0]
        
        if objname == name:
            print("Found object: ", objname)
            filepath = os.path.join(folder_path, file)
            df = pd.read_csv(filepath)
            
            # drop rows with NaN values
            df = df.dropna()
            
            if np.all(df.iloc[:, 1]) == 0.0:
                print("Not including object: ", objname)
                continue
            # our wavelength is in microns, convert to Angstroms
            
            # for the first column, we want to convert to Angstroms
            df.iloc[:, 0] = df.iloc[:, 0] 
            # Name each of the columns appropriately
            df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
            return df, objname
        
    
    return None, None

#df, objname = read_zfourge_template(zfourge_folderpath, 'COSMOS_1') # Test function

# %%

def read_zfourge_galaxy_templates(folder_path):
    """_summary_

        Args:
            folder_path (string): path to the folder where the SED templates are located
    
        Returns:
            df_list: Returns a list of dataframes containing the SED templates
            objname_list: Returns a list of the names of the objects
    """
    df_list = []
    objname_list = []
    folder_path = os.path.join(folder_path)
    files_in_folder = os.listdir(folder_path)
    i = 0
    df_dict = {}
    for file in files_in_folder:

        # Find filepath and convert to df
        objname = file.split('SED_')[1]
        objname = objname.split('.csv')[0]
        filepath = os.path.join(folder_path, file)
        df = pd.read_csv(filepath)
        
        
        
        
        # check for infinities or null values and replace 
        if np.any(np.isnan(df)):
            df = df.replace([np.inf, -np.inf], np.nan)
        if np.any(np.isinf(df) ):
            df = df.replace([np.inf, -np.inf], np.nan)
        
        
        # Whereever there are NaN values, interpolate
        df = df.interpolate()
        
        
        if np.all(df.iloc[:, 1] == 0.0):
            #print("Not including object: ", objname)
            continue
        
        # Convert microns to angstroms    
        df.iloc[:, 0] = df.iloc[:, 0]
        
        # name each of the columns appropriately
        df.columns = ['lambda (Angstroms)', 'Total Flux (erg/s/cm^2/Angstrom)']
            
        # Append the dataframe to the list    
        #df_list.append(df)
        #objname_list.append(objname)
        
        # Turn into dictionary
        df_dict[objname] = df
        
    return df_dict


# %%
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


# %%
# For plotting
# convert the passbands to arrays
U_arr = np.array(pb_U.asList())
V_arr = np.array(pb_V.asList())
J_arr = np.array(pb_J.asList())

# Effectvie wavelengths
U_eff = pb_U.effectiveWavelength()
V_eff = pb_V.effectiveWavelength()
J_eff = pb_J.effectiveWavelength()

# print 
print("The effective wavelengths are: U: {}, V: {}, J: {}".format(U_eff, V_eff, J_eff))

# %%
# Read in all the ZFOURGE templates
df_dict = read_zfourge_galaxy_templates(zfourge_folderpath)


# %%

# print the number of items in df_list
print(len(df_dict))

print(df_dict)

# %%
# Check for null values within the dictionary
for key in df_dict.keys():
    if np.any(np.isnan(df_dict[key])):
        print("Found null values in: ", key)
        

# %%
# Plot all of the SEDS

# for each sed, cut it so the wavelength is between 10^2 and 10^5
# then plot it
#for i in range(len(df_list)):
   # df_list[i] = df_list[i][(df_list[i]['lambda (Angstroms)'] > 10**2) & (df_list[i]['lambda (Angstroms)'] < 10**5)]

# Plot the first 10
plt.figure(figsize=(7, 4))
# Plot the first 10 entries of the dictonary
for i, key in enumerate(list(df_dict.keys())[10:15]):
    plt.loglog(df_dict[key].iloc[:, 0], df_dict[key].iloc[:, 1], label=key)
    
#plt.legend()
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Total Flux (erg/s/cm^2/Angstrom)')
#plt.title('ZFOURGE SED Templates')
sc = 10**-3
# additionally plot the relatively locations of the filters
# plt.loglog(U_arr[:, 0], sc*U_arr[:,1], label='U filter', c='purple')
# plt.loglog(V_arr[:, 0], sc*V_arr[:,1], label='V filter', c='olive')
# plt.loglog(J_arr[:, 0], sc*J_arr[:,1], label='J filter', c='red')

plt.ylim(10**-4, 10**3)

plt.legend()
plt.tight_layout()

# Save
plt.savefig('outputs/ZFOURGE_SED_Templates_EAZYFIT.png', dpi=300)
    

plt.show()




# %% [markdown]
# We can see a very very clear trend of the seds now being alligned in their rest frame. The lyman break is clearly visible and the absorption and emission lines are also very clear. This is a very good sign that the SEDs are being generated correctly.

# %%
# Use a similar method to the one use in the theoretical modelling output notebook.

# First read in the ZFOURGE seds from a fits file and only use id's with good photometry
# read in the useable ids
useable_ids_df = pd.read_csv('outputs/useable_zfourge_ids.csv')


# filter to only zfourge ids for the field we are after
useable_ids_df = useable_ids_df[useable_ids_df['field'] == field.lower()]


# %%
# create dictionary
df_dict

# %%


# plot redshifts
plt.hist(useable_ids_df['z'], bins=20)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Redshifts of ZFOURGE objects')
plt.show()

useable_ids_df

# %%
from sklearn.utils import resample

# Extract the redshifts from the main dataframe
main_redshifts = useable_ids_df['z']

# Define the number of samples you want in the subset
n_samples = 2000 # Adjust this number as needed

# Perform stratified sampling to create a subset with a similar redshift distribution
subset_redshifts = resample(main_redshifts, n_samples=n_samples, stratify=main_redshifts, random_state=42)

# Convert the subset to a dataframe
subset_df = useable_ids_df[useable_ids_df['z'].isin(subset_redshifts)]

# Print the subset dataframe
print(subset_df)

# %%
# Check the distribution of redshifts in the subset
subset_redshifts.hist()
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Redshift distribution of the subset')
plt.show()

# Check the max
max(subset_redshifts)



# %%

useable_ids_df = subset_df

# %%
useable_ids_df

# %%


print(useable_ids_df['z'].describe())

# %%
useable_ids_df

# Append the field to the front of the id
# Make the id column have the field appended to the front
useable_ids_df['id'] = useable_ids_df['field'] + '_' + useable_ids_df['id'].astype(str)

# Uppercase everything
useable_ids_df['id'] = useable_ids_df['id'].str.upper()

# %%
useable_ids_df

# Plot the overal distrobution of redshifts
plt.hist(useable_ids_df['z'], bins=20)

# %%
# # Create the redshift bins
# redshift_bins = np.linspace(0, 5, 11)

# print(redshift_bins)


# # Sample 50 entries from the useable ids in each redshift bin
# # Create a dictionary to store the samples given the overall shape of the redshift distribution,
# # sample  1000 entries ensuring the sample distribution reflects the overall distribution

# # Create a dictionary to store the samples
# sample_dict = {}

# # Ensure our sampling reflects the population distribution
# for i in range(len(redshift_bins) - 1):
#     z_min = redshift_bins[i]
#     z_max = redshift_bins[i + 1]
    
#     # sample 50 entries from the useable ids in each redshift bin
#     sample = useable_ids_df[(useable_ids_df['z'] >= z_min) & (useable_ids_df['z'] < z_max)].sample(50)
    
#     # add to the dictionary
#     sample_dict['z_{}_{}'.format(z_min, z_max)] = sample
    
# Now make the shape o 

# %%
# # subset the original useable ids dataframe to only include the ids that are in the sample_dict
# useable_ids_df = useable_ids_df[useable_ids_df['id'].isin(np.concatenate(list(sample_dict.values())))]
# useable_ids_df

# print(useable_ids_df['z'])

# %%
print(useable_ids_df['id'])

# check the redshift range
print(useable_ids_df['z'].describe())

# Plot a histogram of the redshifts
plt.hist(useable_ids_df['z'], bins=20)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshifts')
plt.show()


# %%
# Check if the id is in the dictionary
df_dict.keys()

# %%
for key in df_dict.keys():
    if key in useable_ids_df['id'].values:
        print("Found: ", key)
    

# %%
    
# redshift
redshifts = []

# Create a new dictionary with only the useable ids
useable_dict = {}
for key in df_dict.keys():
    if key in useable_ids_df['id'].values:
        useable_dict[key] = df_dict[key]
        
        redshifts.append(useable_ids_df[useable_ids_df['id'] == key]['z'].values[0])
        
        # I suspect that eazy had trouble fitting higher redshfit templates and potentially ignored them.
        # This code is causing the redshifts to be taken away
# Check redshift for first entry
print(redshifts[0])

# %%
# describe the redshifts
print(pd.Series(redshifts).describe())

# %%
# Check the redshift range
print("The redshift range is: ", min(redshifts), max(redshifts))
# Plot the redshifts
plt.hist(redshifts, bins=20)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshifts')
plt.show()


# %%
# Check the number of items in the dictionary
print(len(useable_dict))
print(len(redshifts))

# Look at the redshifts
plt.hist(redshifts, bins=20)
plt.show()
# print the largest redshift
print(max(redshifts))

# %%
# Create a new dictionary with only the useable ids in it added 
useable_df_dict = useable_dict

# %%
# Total useable ids
print(len(useable_df_dict))

# %%
# Now that we have all of the useable ids we can now create the models. As we we are intending to use the same sort of method as the theoretical modelling, this will take up a lot of computational resources. 
# To that end, we will randomly sample 150 ids from the useable ids and then create the models for those ids.
# Create an empty data frame for each template that has the filters with the alpha values o.e u_0, u_10, u_20 etc for each filter
column_names = ['id', 'z'] # add an inital column for the redshift and for the ID
# filters 
for filter in filter_set.keys():
    # Check filter we are looking at

    for alpha_val in alpha_values:
    # Add filter into a data frame
        column_names.append(filter + '_' + str(int(round(alpha_val, 2)*100)))



# Turn into a dataframe  
composite_fluxes = pd.DataFrame(columns=column_names)

# %%
composite_fluxes # There will be no redshift information in this dataframe (this will need to be added at a later date)

# %%

# Split the dictionary up again - into sed/name
useable_seds = list(useable_df_dict.values())
useable_names = list(useable_df_dict.keys())



# %%
print(len(useable_seds))
print(len(useable_names))
print(len(redshifts))

# associated redshift for the useable_names
# Read in the redshifts -  redshifts are in the 

# %%
# redshifts

# %%
# useable_names

# %%
# useable_ids_df[useable_ids_df['id'] == 'COSMOS_1042']

# %%
# Create the composites
# Create all of the composites
composites = generate_composite_set(agn_model, useable_seds, alpha_values)

# %%
len(composites[0])

# %%
# Find index of specific ids
ids = useable_names
redshifts = redshifts

# %%
# Turn into a df
ids_df = pd.DataFrame({'id': ids, 'z': redshifts})

# %%
ids_df # This is the useable set of SEDs.

# %%
# Reset index for useable_ids_df


# %%
# # Redshift bins


# # Sample 500 random ids from each redshift bins
# # Create a new dataframe to store the sampled ids
# sampled_ids = pd.DataFrame(columns=['id', 'z'])

# # Sample 500 ids from each redshift bin
# for i in range(len(redshift_bins) - 1):
#     # Get the ids for the redshift bin
#     ids = useable_ids_df[(useable_ids_df['z'] >= redshift_bins[i]) & (useable_ids_df['z'] < redshift_bins[i+1])]
    
#     # Sample 500 ids
#     sampled_ids = pd.concat([sampled_ids, ids.sample(50)])


# %%
# Check distribution
plt.hist(ids_df['z'], bins=20)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshifts')
plt.show()

# %%



# %%

# For each in the dictionary
flux_df = composite_fluxes.copy()
composite_fluxes_list = []
added_ids = 0
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
            

            # We could also add a a section here and we create a new column for a redshift of not 0.
            
            # Check what we are adding
            
            # print('Adding id: ', ids_df['id'][j])
            # print('Adding redshift: ', ids_df['z'][j])
            # print('count: ', added_ids)
            
            added_ids += 1
            # id information
            flux_df.loc[j, 'id'] = ids_df['id'][j]
            
            # Add redshift information
            flux_df.loc[j, 'z'] = ids_df['z'][j]
            
            
            # Only turn this on to output all seds
            # Only necessary to output the CSV for the rest frame
            
            # Maybe add this when we subset what we want 
            #sed_data.to_csv(f'outputs/composite_seds/{template_set_name}/'+template_names[j]+ f'{agn_model_name}'+ 'composite_' +str(int(round(alpha_values[i], 2)*100))+'.csv') <- don't do this but if i need to explore get them to be printed out later 
            # or get a new script to perform the calculation on specific ones again.
            
            
            # Check the flux_Df
           # print(flux_df)
           
            
            
            
            
            # Calculate each filter value for the specific alpha value
            for filter in filter_set.keys():
                # Calculate the magnitude for the filter
                
                #print('Calculating filter: ', filter)
                
                
                # We will need to create the restframe colours using a restframe SED
                if filter == 'U' or filter == 'V' or filter == 'J':
                    
                    # Ensure the sed is restframe
                    sed.redshift(0)
                    
                    # Calculate the magnitude for the filter
                    mag = astSED.SED.calcMag(sed, filter_set[filter], magType='AB')
                    
                    # Add the magnitude to the data frame
                    flux_df.loc[j, filter + '_' + str(int(round(alpha_values[i], 2)*100))] = mag
                    
                else:
                    # Check redshift, if restframed then shift to appropriate redshift 
                    sed.redshift(redshifts[j]) 
                    
                    # IRAC fluxes are observed frame, calculate the flux.
                    if filter == 'IRAC3.6' or filter == 'IRAC4.5' or filter == 'IRAC5.8' or filter == 'IRAC8.0':
                        
                        # Redshift the SED to the specified redshift
                        #sed.z = redshift
                        
                        # Calculate the magnitude for the IRAC filters
                        obs_flux = astSED.SED.calcFlux(sed, filter_set[filter])  
                        # Add the magnitude to the data frame
                        flux_df.loc[j, filter + '_' + str(int(round(alpha_values[i], 2)*100))] = obs_flux
                    
                    else:    
                        # Calculate the magnitude for the other filters
                        mag = astSED.SED.calcMag(sed, filter_set[filter], magType='AB')
                        # Add the magnitude to the data frame
                        flux_df.loc[j, filter + '_' + str(int(round(alpha_values[i], 2)*100))] = mag
            #print("Added filter data", flux_df)
            #
    # Add the redshift to the data frame
    #flux_df['z'] = redshift
#        composite_fluxes_list.append(flux_df)


# %%
flux_df

# Check the redshifts histogram
plt.hist(flux_df['z'], bins=20)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshifts')
plt.show()

# %%

# This ensures we have restframe colours for the U, V and J filters, and observed frame fluxes for the IRAC/ugr filters. 
# This will provide a good testing grounds.
composite_fluxes = flux_df

# %%

# Drop duplicates
composite_fluxes = composite_fluxes.drop_duplicates(subset='id')


# %%
composite_fluxes


# %%


# Output as required
composite_choice = f'{field}_obsevational_composites_fluxes{agn_model_name}'
composite_fluxes.to_csv(f'outputs\composite_seds\{composite_choice}.csv')   



# %%
# Have a look for the largest redshift
composite_fluxes['z'].max()


# %%
# We would then like to export this to a csv file with the appropriate naming convention


# %%


# %%



