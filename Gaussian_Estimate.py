import numpy as np
import Galaxy_Functions as gf
import astropy.units as u
import Upchannelize as upchan

def sigma2FWHM(sigma):
    return sigma * 2 * np.sqrt(2*np.log(2)) 

def FWHM2sigma(FWHM):
    return FWHM / (2*np.sqrt(2*np.log(2)))

def W50_broadened(W50, sigma_disp=10):
    sigma_rot = FWHM2sigma(FWHM=W50)
    sigma_broadened = np.sqrt(sigma_rot**2 + sigma_disp**2) 
    W50_broad = sigma2FWHM(sigma_broadened)
    return W50_broad

# def S21_Gaussian(S_peak, W50):
#     a = S_peak
#     c = sigma_broaded(W50)
#     return a*c*np.sqrt(2*np.pi)

# def local_velores(fres=None):
#     if fres==None:
#         upchan.chord_bandwidth(range=[0,1500])
#     delta_v = gf.df_dv()





