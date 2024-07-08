# This is the master SED Proccessing file that will do the majority of the SED Proccessing,
# It will condense the logic from most of the Preliminary Project Work notebooks into one notebook.


# Import the necessary packages
import sys
sys.path.append('../') 
import numpy as np
import pandas as pd
import os
from astLib import astSED
from astropy.io import fits
import matplotlib.pyplot as plt
import qarg_tools as qt


# Parameters


# Load Data
#filters = qt.load_passbands('UVJ')
#print(filters)

# Code logic
