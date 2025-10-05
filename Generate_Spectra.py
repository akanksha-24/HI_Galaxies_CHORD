import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import numpy as np
from numpy import diff
from random import choices
from scipy import special
from scipy.interpolate import UnivariateSpline
from scipy.signal import fftconvolve
import time
from mpi4py import MPI
import numba as nb
from math import erf

# for parallelization 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  
size_mpi = comm.Get_size()  

def Busy_general(x, a, b1, b2, xe, xp, c, w):
    ''' This is the functional definition for a General Busy Function:
        Reference: https://ui.adsabs.harvard.edu/abs/2014MNRAS.438.1176W/abstract (Section 4.1, Equation 4)'''
    
    err_p = special.erf(b1*(w+x-xe)) + 1
    err_m = special.erf(b2*(w-x+xe)) + 1
    pbola = (c*((x-xp)**2)) + 1
    
    return (a/4)*err_p*err_m*pbola

def integrate_profile(V, S):
    ''' Helper function which integrates an HI profile. The 'V' parameter is the velocity axis in km/s. The 'S' paramater is the flux in mJy'''

    return np.trapz(S, x=V)  # trapz is a numeric integrator in python using trapezoid rule, diff(V) if dV

def get_MHI(V, S, D):
    ''' Helper function which converts the integrated HI profile to HI Mass using equation: M_HI = 2.356x10^5 * (D^2) * S_tot '''

    int_S = integrate_profile(V, S)
    return 2.356e5 * (D**2) * int_S

def find_FWHM(x, y, level=0.5):
    ''' Helper function which finds the roots of the HI profile at the FWHM, to set to W50 width. 
    Returns an array of all roots r[] found where the FWHM width is r[-1] - r[0] '''

    spline = UnivariateSpline(x, y-(np.max(y)*level), s=0)
    roots = spline.roots() 
    FWHM = roots[-1] - roots[0]
    if len(roots) < 2:
        return np.nan, roots  # no valid FWHM found
    return FWHM, roots
    
def assign_units(x, B, W50, D, MHI_desired):
    ''' Helper function which scales the unitless axes of the generalized busy function into meaningful km/s and mJy Spectrum units '''

    # scale x-axis
    S_norm = (B/np.max(B))                  # Peak Flux set to 1 Jy if not specified
    FWHM, roots = find_FWHM(x, B)
    scale = W50 / FWHM                      # scaling factor from unitless → km/s
    V = x * scale                           # velocity axis in km/s
    #W_ = roots * scale                     # FWHM roots in km/s       

    # Scale y-axis
    MHI_initial = get_MHI(V, S_norm, D) 
    y_scale = MHI_desired / MHI_initial   
    S = S_norm * y_scale  

    return V, S                            # Return V and S axes and W_ roots (for checking)

# Gaussian Function
def normalDist(x, sigma, x0=0):
    return np.exp(-(x - x0)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def gaussian_kernel(V, FWHM):
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    G = normalDist(V - np.mean(V), sigma)   # center kernel at 0
    return G / G.sum()                      # normalize discrete sum = 1

def convolve_spectrum(S, V, FWHM):
    G = gaussian_kernel(V, FWHM)
    S_broad = fftconvolve(S, G, mode='same')
    return S_broad

# def convolve_spectra(B, V, FWHM):
#     G = gaussian_kernel(V, FWHM)
#     return fftconvolve(B, G[None, :], mode='same', axes=1)


def Generate_Spectra(size, MHI, W50, D, a=1, w=1, b1=None, b2=None, c=None, xe=None, xp=None, chunk_size=10000):
    ''' Main function which generates an HI Spectrum. 
    The shape of the busy function (specified by a, b1, b2, c) is randomly generated (unless otherwise specified). 
    The area under the profile is set by MHI. 
    The FWHM width (W50) is set by VHI and inclination using W50 = VHI*2sin(i). 
    The profile is centered around 0 so xe=0 and xp=0. We use a second order n=2 general busy function'''

    print("Generating Spectra....")
    start = time.time()

    #W = VHI*2*np.sin(i) # W_50       
    x = np.linspace(-10, 10, 1000).astype(np.float32)  # unitless axes used in generalized busy function definition

    if b1==None: b1 = np.random.uniform(1, 3, size=size)
    if b2==None: b2 = np.random.uniform(1, 3, size=size)
    if c==None: c = np.random.uniform(0, 1, size=size)
    if xe==None: xe = np.random.uniform(-0.5, 0.5, size=size)
    if xp==None: xp = np.random.uniform(-0.5, 0.5, size=size)

    for i in np.arange(size):
        B = Busy_general(x, a, b1[i], b2[i], xe[i], xp[i], c[i], w)  
        V, S = assign_units(x, B, W50[i], D[i], MHI_desired=MHI[i])

        # Thermal broadening
        FWHM_thermal = 10 # km/s 
        S_broad = convolve_spectrum(S, V, FWHM_thermal)
        final_M = get_MHI(V, S_broad, D)

    end = time.time()
    print(f"Runtime: {end - start:.3f} seconds")

    return final_M, V, S_broad, 

def Check_Spectra(catalog_fl):
    catalog = np.load(catalog_fl)
    size = catalog.shape[0]
    print(size)
    #size = 10
    MHI = 10**catalog[:,0]
    W50 = catalog[:,3]
    D = catalog[:,6]
    final_M, V, S_broad, = Generate_Spectra(size, MHI, W50, D)
    np.save("VolLim_20to60deg_Dmax100_spectra.npy", np.asarray([V, S_broad]))
    # MHI_res = np.log10(final_M) - np.log10(MHI)
    # plt.figure()
    # plt.scatter(MHI, MHI_res)
    # plt.show()

Check_Spectra('catalogs_output/VolLim_20to60deg_Dmax100.npy')