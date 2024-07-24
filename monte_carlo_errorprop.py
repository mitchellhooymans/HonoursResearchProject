# Monte Carlo Simulation for Error Proagation 
# This script will be intended to be used for the error propagation of the
# the uncertainities for the SEDs, as no uncertainities are provided by CIGALE the best way to model this
# will be by getting the error propogation from the Monte Carlo simulation modelling the errors as a gaussian distribution

# Begin by importing the relevant packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from carf import *


# Begin my importing the cdfs data from zfourge


# create a rnadom noise array
def random_noise(n):
    return np.random.normal(0, 0.5, n)



# create a array of equally spaced numbers from 0 to 1
def create_numbers_line(N, n):
    return np.linspace(0, N, n)

n = 1000 
rn_1 = random_noise(n)

print(rn_1)




nums = create_numbers_line(10, n)

# plot nums
plt.plot(nums)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Index vs Value')
plt.show()



# Now create a function that will add the random noise to the numbers line
def add_noise(nums, rn):
    return nums + rn

# Plot
plt.plot(add_noise(nums, rn_1))
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Index vs Value with Noise')
plt.show()

