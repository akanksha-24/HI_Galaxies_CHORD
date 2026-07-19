import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Estimate import *

def Gaussian(A, x, sigma, x0=0):
    return A*np.exp(-(x - x0)**2 / (2*sigma**2)) 

