# %% [markdown]
# # Error Propagation for SED Modelling
# This script is intended to be used to perform error propogation with monte carlo simulations. The intent of this script will be to use a sample galaxy sed from cigale and create variations of through pertubing it with random noise. 

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
full_cdfs_ids = pd.read_csv('datasets/zfourge/full_CDFS_ids.csv')
full_cosmos_ids = pd.read_csv('datasets/zfourge/full_COSMOS_ids.csv')
full_uds_ids = pd.read_csv('datasets/zfourge/full_UDS_ids.csv')


# %%

# Using dataframes is wildly inefficient. 
# A most robust approach would be to use only the 2 columns we are interested in, specified in the function call
# and then read these in as numpy arrays


def get_n_seds(df, n, field, restframe=False, all=False):
    # Select n galaxies
    
    df_list = []
    names = []
    redshifts = []
    if all==False:
        selected_galaxies = df.sample(n)
    else: 
        selected_galaxies = df
        
    # Reset the index
    selected_galaxies = selected_galaxies.reset_index(drop=True)
    
    # name 
    gal_name = selected_galaxies['id'].astype(str)
    
    # field
    gal_field = field#selected_galaxies['field'].astype(str)
    
    
    names = gal_field + '_' + gal_name
    gal_redshift = selected_galaxies['zpk'].astype(float)

    # Now we will read in the fits files for these galaxies

    for i in range(len(selected_galaxies)):
        path = 'datasets\\full_zfourge_decomposed\\'+ str(gal_field).lower() +'_best_models_fits\\'
        name = str(gal_name[i])+'_best_model.fits'

        galaxy_path = os.path.join(path, name)
        with fits.open(galaxy_path) as data:
            df = pd.DataFrame(np.array(data[1].data).byteswap().newbyteorder())
        
        # Convert to angstroms
        df['wavelength'] = df['wavelength']*10

        if restframe:
            df['Snu'] = df['Fnu']*10**-3 # milliJanksys to Janksys <- J = ergs/(s*(cm^2)*(s^-1))
            # F_nu currently has a frequency dependence, convert to nuFnu by multiplying the the frequency associated
            # with the wavelength, as we are in angstroms, we can use the formula c = f*lambda
            
            
            # This should prevent any issues, but check
            freq = (3*10**18)/df['wavelength'] # in Hz
            # multiply the Snu * nu to get nuSnu
            df['nuSnu'] = df['Snu']*freq
            # Restframe the values of wavelength
            df['wavelength'] = df['wavelength'] / (1 + gal_redshift[i]) # we redshift the values of of wavelength
            # now calculate a new frequency, based on the new wavelength
            freq = (3*10**18)/df['wavelength'] # in Hz
            # divide the nuSnu by the new frequency to get the restframed values
            df['Snu'] = df['nuSnu']/freq
            
            # Convert flux values
            df['Flambda'] = df['Snu']*(3*10**-5)/(df['wavelength']**2) # S_nu to F_lambda <- angstroms 
            
        else:
            # Convert flux values
            df['Snu'] = df['Fnu']*10**-3 # milliJanksys to Janksys <- J = ergs/(s*(cm^2)*(s^-1))
            df['Flambda'] = df['Snu']*(3*10**-5)/(df['wavelength']**2) # S_nu to F_lambda <- angstroms 
            
        
            
            
        redshift_Val = gal_redshift[i]
        redshifts.append(redshift_Val)        
        
        

        
        # For simplicity, just create some extra columns
        df['lambda (Angstroms)'] = df['wavelength']
        df['Total Flux (erg/s/cm^2/Angstrom)'] = df['Flambda']
        
        
        
        df_list.append(df)
        
        
        plt.loglog(df['wavelength'], df['Flambda'])
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux (Fnu)')
    #plt.xlim(1e3, 1e5)
    plt.ylim(1e-30, 1e-2)
    plt.title('SED of galaxies')
    plt.legend()
    plt.show()
    
    print(len(df_list))
    
    return df_list, names, redshifts


# %%
#np.random.seed(42)

def get_n_seds(df, n, field, restframe=False, all=False):

    # Select n galaxies (only the necessary columns)
    if all:
        selected_indices = np.arange(len(df))
    else:
        selected_indices = np.random.choice(len(df), n, replace=False)

    names = []
    redshifts = []
    df_list = []
    
    print(selected_indices)
    
    for i in selected_indices:
        name = f"{field}_{df['id'][i]}"

        redshift = df['zpk'][i]
        names.append(name)
        redshifts.append(redshift)

        # Load FITS file (using only the necessary columns)
        path = f'datasets\\full_zfourge_decomposed\\{field.lower()}_best_models_fits\\'
        file = f"{df['id'][i]}_best_model.fits"

        with fits.open(os.path.join(path, file)) as data:
            wavelength = data[1].data['wavelength'] * 10  # Convert to Angstroms
            fnu = data[1].data['Fnu']

        # Convert and rest-frame data (in-place operations for efficiency)
        snu = fnu * 1e-3  # milliJansky to Jansky
        flambda = snu * 3e-5 / wavelength**2

        if restframe:
            wavelength /= (1 + redshift)
            freq = 3e18 / wavelength
            snu = snu * freq / freq  # Avoid division by zero
            flambda = snu * 3e-5 / wavelength**2

        # Create DataFrame (only with necessary columns)
        df_galaxy = pd.DataFrame({
            'lambda (Angstroms)': wavelength,
            'Total Flux (erg/s/cm^2/Angstrom)': flambda
        })

        df_list.append(df_galaxy)
        
        # plotting, keep if needed
        #plt.loglog(df_galaxy['lambda (Angstroms)'], df_galaxy['Total Flux (erg/s/cm^2/Angstrom)'])

    # Plotting (only once after the loop)
    # plt.xlabel('Wavelength (Angstroms)')
    # plt.ylabel('Flux (Fnu)')
    # plt.ylim(1e-30, 1e-2)
    # plt.title('SED of galaxies')
    # plt.legend()
    # plt.show()

    return df_list, names, redshifts


# %%
def perturb_flux(flux, relative_error):
    """
    Perturbs a flux value using Gaussian noise based on a relative error.

    Args:
        flux: The original flux value.
        relative_error: The relative error in the flux (error / flux).

    Returns:
        perturbed_flux: The flux value with added Gaussian noise.
    """

    # Calculate the absolute error from the relative error
    absolute_error = flux * relative_error

    # Generate Gaussian noise with mean 0 and standard deviation equal to the absolute error
    noise = np.random.normal(0, absolute_error)

    # Add the noise to the original flux
    perturbed_flux = flux + noise

    return perturbed_flux


# %%
# Get one SED for testing from the CDFS field
test_sed_list, names, redshifts = get_n_seds(full_cdfs_ids, 1, 'CDFS', restframe=True, all=False)

# %%
# Now we can safely select the test sed

k = 0

test_sed = test_sed_list[k]
test_name = names[k]
test_redshift = redshifts[k]


# %%
# Print info
print(f"Name: {test_name}")
print(f"Redshift: {test_redshift}")
test_sed.head()


# %%
# We can plot the SED with the redshift information
plt.loglog(test_sed['lambda (Angstroms)'], test_sed['Total Flux (erg/s/cm^2/Angstrom)'])
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (Fnu)')
plt.ylim(1e-30, 1e-2)
plt.title("SED of Galaxy: " + test_name + " at z = " + str(test_redshift))
plt.legend()
plt.show()


# %%
# Now we can attempt to read in the CDFS fits file and extract the associated data
cdfs_df =read_zfourge_data('CDFS','datasets/zfourge')

# %%
# Get the associated row containing the information for the test galaxy
test_galaxy = cdfs_df[cdfs_df['id'] == test_name]

# %%
test_galaxy

# %%
cdfs_df

# %%
# Print each of the column names
for col in cdfs_df.columns:
    print(col)

# %%


# perturbed photometrix data points for each photometric filter
perturbed_photometry =  {'U': [], 'V': [], 'J':[]}
non_pertubed_photometry =  {'U': [], 'V': [], 'J':[]}




# We can plot the SED with the redshift information
plt.loglog(test_sed['lambda (Angstroms)'], test_sed['Total Flux (erg/s/cm^2/Angstrom)'], c='k')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (Fnu)')
plt.ylim(1e-30, 1e-2)
plt.title("SED of Galaxy: " + test_name + " at z = " + str(test_redshift))

# create a subset of filters for UVJ only
uvj_filters = {'U': pb_U, 'V':pb_V, 'J':pb_J}

# For each of the filters get the effectiveWavelength
for filter in uvj_filters.keys():
    print(filter)
    
    # Get the effective wavelength
    eff_wl = filter_set[filter].effectiveWavelength()
    
    # print 
    print(eff_wl)
    sed_flux = test_sed.iloc[(test_sed['lambda (Angstroms)']-eff_wl).abs().argsort()[:1]]['Total Flux (erg/s/cm^2/Angstrom)'].values[0] 
    
    # Go to zfourge to get relative error from the df
    zf_flux = test_galaxy[f'{filter}'].values[0]
    zf_error = test_galaxy[f'e{filter}'].values[0]
    
    # calculate the relative error
    relative_error = zf_error/zf_flux
    
    # Print the relative error
    print(relative_error)
    
    
    # perturb the flux
    perturbed_sed_flux = perturb_flux(sed_flux, relative_error)
    
    
    # plot the sed flux
    plt.scatter(eff_wl, sed_flux)
    
    # plot the perturbed flux
    plt.scatter(eff_wl, perturbed_sed_flux)
    
    # add the perturbed flux to the dictionary
    perturbed_photometry[filter].append(perturbed_sed_flux)
    # add the non perturbed flux to the dictionary
    non_pertubed_photometry[filter].append(sed_flux)
    
plt.legend()
plt.show()





# Calculate absolute errors for perturbed fluxes
perturbed_flux_errors = [
    test_galaxy[f'e{band}'].values[0] for band in uvj_filters.keys()  # Assuming e_mag_error is in magnitudes
]

# Dictionary with effective wavelengths (replace with your actual values)
filter_effective_wavelengths = {
    'U': filter_set['U'].effectiveWavelength(), 
    'V': filter_set['V'].effectiveWavelength(),
    'J': filter_set['J'].effectiveWavelength()
}

# model wavelengths
model_wavelengths = test_sed['lambda (Angstroms)'].values
model_sed_fluxes = test_sed['Total Flux (erg/s/cm^2/Angstrom)'].values


# %%
# perturbed flux errors
perturbed_flux_errors

# %%
# for each of the filters, check the pertubation vs non-pertubation
for filter in uvj_filters.keys():
    print('The filter: ',filter)
    print('Perturbed photometry',perturbed_photometry[filter])
    print('Perturbed photometry',non_pertubed_photometry[filter])
    # print the percentage error
    print('Percentage error:', (abs(perturbed_photometry[filter][0] - non_pertubed_photometry[filter][0]))/non_pertubed_photometry[filter][0]*100)
    print('\n\n')

# %%
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

def scale_sed_to_photometry(model_wavelengths, model_fluxes, 
                             observed_bands, observed_fluxes, observed_errors,
                             filter_effective_wavelengths):
    """
    Scales a model SED to match perturbed photometric fluxes.

    Args:
        model_wavelengths: Array of model SED wavelengths.
        model_fluxes: Array of model SED fluxes.
        observed_bands: List of observed filter names (e.g., ['U', 'V', 'J']).
        observed_fluxes: Array of observed (perturbed) fluxes.
        observed_errors: Array of errors associated with the observed fluxes.
        filter_effective_wavelengths: Dictionary containing effective wavelengths 
                                      for each band (e.g., filter_effective_wavelengths['U']).

    Returns:
        scaled_model_fluxes: Array of scaled model fluxes.
    """

    def chi2(scale_factor):
        """
        Calculates the chi-squared statistic for a given scale factor.
        """
        scaled_model_fluxes = model_fluxes * scale_factor

        chi2 = 0
        for band, flux, err in zip(observed_bands, observed_fluxes, observed_errors):
            eff_wl = filter_effective_wavelengths[band]
            model_interp = interp1d(model_wavelengths, scaled_model_fluxes, bounds_error=False, fill_value=0.0)
            model_flux_at_eff_wl = model_interp(eff_wl)
            chi2 += ((flux - model_flux_at_eff_wl) / err)**2
            
        return chi2

    # Find the optimal scale factor
    result = minimize_scalar(chi2, bounds=(0, 1), method='bounded')
    # print(result)
    print(result.x)
    
    best_scale_factor = result.x
    
    
    # Scale the model SED
    scaled_model_fluxes = model_fluxes * best_scale_factor

    return scaled_model_fluxes


# perturbed fluxes
perturbed_fluxes = [perturbed_photometry[band] for band in uvj_filters.keys()] * 10000



# %%
[perturbed_photometry[band] for band in uvj_filters.keys()][2]

# %%


# Example usage (assuming you have the data loaded)
scaled_sed_fluxes = scale_sed_to_photometry(
    model_wavelengths, model_sed_fluxes, 
    ['U', 'V', 'J'],  # Observed bands
    perturbed_fluxes, perturbed_flux_errors,
    filter_effective_wavelengths  # Dictionary with effective wavelengths
)

# %%
# plot the scaled SED according to the perturbed photometry
plt.loglog(model_wavelengths, model_sed_fluxes, label='Original SED', c='k')   
plt.loglog(model_wavelengths, scaled_sed_fluxes, label='Scaled SED', c= 'r')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (Fnu)')
plt.ylim(1e-30, 1e-2)
plt.title("SED of Galaxy: " + test_name + " at z = " + str(test_redshift))
plt.legend()
plt.show()


# %%
# use astSED to calculate the colours

model_sed = astSED.SED(model_wavelengths, model_sed_fluxes, z=0)
m_U = model_sed.calcMag(filter_set['U'])
m_V = model_sed.calcMag(filter_set['V'])
m_J = model_sed.calcMag(filter_set['J'])    

# print the output
print('Model U:', m_U)
print('Model V:', m_V)
print('\n\n')


scaled_sed = astSED.SED(model_wavelengths, scaled_sed_fluxes, z=0)
s_U = scaled_sed.calcMag(filter_set['U'])
s_V = scaled_sed.calcMag(filter_set['V'])
s_J = scaled_sed.calcMag(filter_set['J'])

# print the output
print('Scaled U:', s_U)
print('Scaled V:', s_V)

# Create the model and the scaled SED colours
model_colours_UV = m_U - m_V
scaled_colours_UV = s_U - s_V

model_colours_VJ = m_V - m_J
scaled_colours_VJ = s_V - s_J





# plot the colours
plt.scatter(model_colours_UV, model_colours_VJ, label='Original SED', c='k')
plt.scatter(scaled_colours_UV, scaled_colours_VJ, label='Scaled SED', c='r')
plt.xlabel('U - V')
plt.ylabel('V - J')
plt.title('UVJ colours')
plt.legend()
plt.show()

# Find the percentage uncertainitiy in the colours
percentage_error_UV = (abs(m_U - s_U)/m_U)*100
print('Percentage error in UV:', percentage_error_UV)


# %%
# We can plot the SED with the redshift information
plt.loglog(test_sed['lambda (Angstroms)'], test_sed['Total Flux (erg/s/cm^2/Angstrom)'])
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (Fnu)')
plt.ylim(1e-30, 1e-2)
plt.title("SED of Galaxy: " + test_name + " at z = " + str(test_redshift))
plt.legend()


# get the effective U filter
U_eff = pb_U.effectiveWavelength()

# find the flux corresponding to the effective wavelength
U_flux = test_sed[test_sed['lambda (Angstroms)'] == U_eff]['Total Flux (erg/s/cm^2/Angstrom)']

# Find the flux of the galaxy in the U band, that is closest
# to the effective wavelength of the U band filter
U_flux = test_sed.iloc[(test_sed['lambda (Angstroms)']-U_eff).abs().argsort()[:1]]['Total Flux (erg/s/cm^2/Angstrom)'].values[0]    

# Plot the flux flux of the galaxy in the U band filter
plt.scatter(U_eff, U_flux, color='red')


print(f"U_eff: {U_eff}, the flux is {U_flux}")

# Find the error in the U band filter from the zfourge dataset



plt.show()

# Plot 

# %%


# %%
# Print the error from the cdfs_df
# Check 
e_mag_error = cdfs_df['e_mag_U'][cdfs_df['id'] == test_name].values[0]
e_mag = cdfs_df['mag_U'][cdfs_df['id'] == test_name].values[0]


# Print the magnitudes
print(f"e_mag: {e_mag}, e_mag_error: {e_mag_error}")


# find relative error
rel_error = e_mag_error/e_mag
print(f"relative error in U band filter: {rel_error}")
# the relative error in our measurement is 1.14%
# flux at the effective wavelength of the U band filter
# We will have to use the U flux, and pertub that
# by the error, and then calculate the relative error

print("The flux is ", U_flux)

# Calculate the absolute error in the flux
abs_error = U_flux*rel_error

print(f"The absolute error in the flux is {abs_error}")

# convert both of these to magnitudes
U_flux_mag = -2.5*np.log10(U_flux) -25
abs_error_mag = -2.5*np.log10(U_flux + abs_error) - U_flux_mag

# Print
print(f"magnitudes is {U_flux_mag} with an error of {abs_error_mag}")



# print


# %%
# Try a completely different approach
e_mag_error = cdfs_df['eU'][cdfs_df['id'] == test_name].values[0]
e_mag = cdfs_df['U'][cdfs_df['id'] == test_name].values[0]


# Print the magnitudes
print(f"e_mag: {e_mag}, e_mag_error: {e_mag_error}")


# find relative error
rel_error = e_mag_error/e_mag
print(f"relative error in U band filter: {rel_error}")
# the relative error in our measurement is 1.14%
# flux at the effective wavelength of the U band filter
# We will have to use the U flux, and pertub that
# by the error, and then calculate the relative error

print("The flux is ", U_flux)

# Calculate the absolute error in the flux
abs_error = U_flux*rel_error

print(f"The absolute error in the flux is {abs_error}")

# convert both of these to magnitudes
U_flux_mag = -2.5*np.log10(U_flux) -25
abs_error_mag = -2.5*np.log10(U_flux + abs_error) - U_flux_mag

# Print
print(f"magnitudes is {U_flux_mag} with an error of {abs_error_mag}")



# print

# %%
# Convert flux to magnitude
U_mag = -2.5*np.log10(U_flux) -25

# magnitude to flux
U_flux = 10**(-0.4*(U_mag + 25))


def mag_to_flux(mag, error):
    flux = 10**(-0.4*(mag + 25))
    flux_error = flux * np.log(10) * error
    return flux, flux_error


# flux to mag function
def flux_to_mag(flux, error):
    mag = -2.5*np.log10(flux) -25
    mag_error = np.log(10) * error / flux
    return mag, mag_error

# %%
U_mag

# %%
# convert the stuff from zfourge
U_mag_zfourge = cdfs_df['mag_U'][cdfs_df['id'] == test_name].values[0]
U_mag_zfourge_error = cdfs_df['e_mag_U'][cdfs_df['id'] == test_name].values[0]

U_flux_zfourge, U_flux_zfourge_error = mag_to_flux(U_mag_zfourge, U_mag_zfourge_error)

# %%


# %%
# # convert the e_mag_U to an error in flux
# # inspect the error

# print("The flux error in the U band filter is: ", U_flux_zfourge_error)
# print("The associated flux is: ", U_flux_zfourge)

# # The percentage of the flux that is the error is
# print("The error in the U band filter is: ", round((U_flux_zfourge_error/U_flux_zfourge)*100, 2), "%")

# # rel error
# rel_error = U_flux_zfourge_error/U_flux_zfourge



# # Instead of using the flux from zfourge, we can us the flux from the SED
# # get the flux at the u filter
# U_flux = test_sed.iloc[(test_sed['lambda (Angstroms)']-U_eff).abs().argsort()[:1]]['Total Flux (erg/s/cm^2/Angstrom)'].values[0]
# # print 
# print(f"Flux from SED: {U_flux}")

# # Perturb the flux
# perturbed_U_flux = perturb_flux(U_flux, rel_error)

# # the perturbed flux should be 


# # Plot the perturbed flux
# plt.scatter(U_eff, perturbed_U_flux, color='red', s=15)
# plt.scatter(U_eff, U_flux, color='blue', s=15)


# # Print the flux and the pertubed flux
# print(f"Flux: {U_flux_zfourge}, Perturbed Flux: {perturbed_U_flux}")


# # plot the sed
# plt.loglog(test_sed['lambda (Angstroms)'], test_sed['Total Flux (erg/s/cm^2/Angstrom)'])
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Flux (Fnu)')
# plt.ylim(1e-30, 1e-2)
# plt.title("SED of Galaxy: " + test_name + " at z = " + str(test_redshift))
# plt.legend()



# # Scale the scale the sed to the pertubed flux
# # scale the flux
# test_sed['Total Flux (erg/s/cm^2/Angstrom)'] = test_sed['Total Flux (erg/s/cm^2/Angstrom)']*(perturbed_U_flux/U_flux)

# # Calculate the 


# # Check the new flux after scaling
# U_flux_scaled = test_sed.iloc[(test_sed['lambda (Angstroms)']-U_eff).abs().argsort()[:1]]['Total Flux (erg/s/cm^2/Angstrom)'].values[0]

# # Print the new flux
# print(f"New flux: {U_flux_scaled}")


# # Plot the scaled SED in a new colour
# plt.loglog(test_sed['lambda (Angstroms)'], test_sed['Total Flux (erg/s/cm^2/Angstrom)'], c='green')
# plt.xlabel('Wavelength (Angstroms)')
# plt.ylabel('Flux (Fnu)')
# plt.ylim(1e-30, 1e-2)
# plt.title("SED of Scaled Galaxy: " + test_name + " at z = " + str(test_redshift))
# plt.legend()



# plt.show()



# %%


# %%
# Now we can plot the SED again, and look at the error
plt.loglog(test_sed['lambda (Angstroms)'], test_sed['Total Flux (erg/s/cm^2/Angstrom)'])
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (Fnu)')
plt.ylim(1e-30, 1e-2)
plt.title("SED of Galaxy: " + test_name + " at z = " + str(test_redshift))
plt.legend()

# Plot the flux flux of the galaxy in the U band filter
#plt.scatter(U_eff, U_flux, color='red')
#plt.errorbar(U_eff, U_flux, yerr=U_flux_zfourge_error, fmt='+', color='blue')



print("The relative error in the U band filter is: ", rel_error)


# Plot the error in the U band filter 
plt.scatter(U_eff, U_flux_zfourge_error, color='k')


plt.show()

# %%



