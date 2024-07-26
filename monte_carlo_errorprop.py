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

