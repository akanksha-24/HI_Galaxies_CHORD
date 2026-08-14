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
#from mpi4py import MPI
import numba as nb
from math import erf
import Galaxy_Functions as gf
from Gaussian_Estimate import *
from CHORD_Sensitivity import *
from matplotlib.backends.backend_pdf import PdfPages

# for parallelization 
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()  
# size_mpi = comm.Get_size()  

upchan_res = gf.width_vel2freq(del_Vrest=5) # for 5 km/s wide channels

def Busy_general(x, a, b1, b2, xe, xp, c, w, n):
    ''' This is the functional definition for a General Busy Function:
        Reference: https://ui.adsabs.harvard.edu/abs/2014MNRAS.438.1176W/abstract (Section 4.1, Equation 4)'''
    
    err_p = special.erf(b1*(w+x-xe)) + 1
    err_m = special.erf(b2*(w-x+xe)) + 1
    pbola = (c*(np.abs(x-xp)**n)) + 1
    return (a/4)*err_p*err_m*pbola

def integrate_profile(V, S):
    ''' Helper function which integrates an HI profile. The 'V' parameter is the velocity axis in km/s. The 'S' paramater is the flux in mJy'''
    return np.trapz(S, x=V)  

def MHI_to_Sint(MHI, z):    
    D_L = gf.Luminosity_Dist(z) # in Mpc
    Sint = MHI*(1+z) / (2.356e5 * (D_L**2))
    return Sint

def get_MHI(V, S, z):
    ''' Helper function which converts the integrated HI profile to HI Mass 
    From Equation 47 of Meyer 2017'''
    D_L = gf.Luminosity_Dist(z) # in Mpc
    int_S = integrate_profile(V, S)
    return 2.356e5 * (D_L**2) * int_S / (1+z)

def get_MHI_freq(f, S, z):
    ''' Helper function which converts the integrated HI profile in Hz (frequency axis) to HI Mass
    From Equation 45 of Meyer 2017'''
    D_L = gf.Luminosity_Dist(z) # in Mpc
    int_S = integrate_profile(f, S) # in Hz*Jy
    return 49.7 * (D_L**2) * int_S  # in Hz*Jy

def MHI_to_Sint_freq(MHI, z):    
    D_L = gf.Luminosity_Dist(z) # in Mpc
    return MHI / (49.7 * (D_L**2))

# Gaussian Function
def normalDist(x, sigma=10, x0=0):
    G =  np.exp(-(x - x0)**2 / (2*sigma**2)) 
    return G / np.trapz(G, x)                 # normalize computationally 

def convolve_spectrum(S, V, sigma=10):
    dv = V[1] - V[0]
    G = normalDist(V, sigma=sigma)
    S_broad = fftconvolve(S, G, mode='same') * dv  # since fftconvolve is a sum, must include dv
    return S_broad, G


def find_FWHM(x,y, level=0.5):
    profile_idx = np.argwhere(y > level*np.max(y))[:,0]
    width = x[np.max(profile_idx)] - x[np.min(profile_idx)]
    roots = [x[np.min(profile_idx)],x[np.max(profile_idx)]]
    return width

def units_check(V, B, Vaxis, B_resamp, S, S_broad, G, W50, z, MHI_desired, thres=1e-3):
    channels = S_broad > np.max(S_broad)*thres
    W50_broad = W50_broadened(W50)
    MHI_measured = get_MHI(Vaxis[channels], S_broad[channels], z)
    W50_measured = find_FWHM(Vaxis, S_broad)
    W50_f_measured = gf.width_vel2freq(W50_measured,z) / 1e3
    W50_f_expected = gf.width_vel2freq(W50_broad,z) / 1e3
    S21_measured = integrate_profile(Vaxis[channels], S_broad[channels])
    S21_expected = MHI_to_Sint(MHI_desired, z)
    spectra_extent_kms = np.max(Vaxis[channels]) - np.min(Vaxis[channels])
    spectra_extent_kHz = gf.width_vel2freq(spectra_extent_kms,z) / 1e3
    Nchans = spectra_extent_kHz / (upchan_res/1e3)
    
    plt.figure()
    plt.plot(V,B, label='scaled to width')
    plt.plot(Vaxis,B_resamp, label='padded')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=[10,8])
    plt.plot(Vaxis, S, label='scaled to height')
    plt.plot(Vaxis, S_broad, label='thermal broadened')
    #plt.xlim(-W50_broad, W50_broad)
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(Vaxis, S_broad, label='thermal broadened')
    plt.axvline(np.min(Vaxis[channels]))
    plt.axvline(np.max(Vaxis[channels]))
    plt.legend()
    plt.show()
    
    print(f"MHI measured is {np.log10(MHI_measured):.2f}")
    print(f"MHI expected is {np.log10(MHI_desired):.2f}")
    
    print(f"W50 measured is {W50_measured:.2f} km/s")
    print(f"W50 expected is {W50_broad:.2f} km/s")
    
    print(f"W50_f measured is {W50_f_measured:.2f} kHz")
    print(f"W50_f expected is {W50_f_expected:.2f} kHz")
    
    print(f"S21 measured is {S21_measured} Jy*km/s", )
    print(f"S21 expected is {S21_expected} Jy*km/s", )
    
    print(f"S21_peak measured is {np.max(S_broad)} Jy")
    print(f"S21_peak expected is {S21_expected/W50_broad} Jy")
    
    print(f"Velocity spectral extent {spectra_extent_kms} km/s")
    print(f"Frequency spectral extent {spectra_extent_kHz} kHz")
    print(f"Frequency channels {Nchans}")
    

def assign_units(x, B, W50, z, MHI_desired, faxis=None, thermal_broaden=True, check=True):
    '''assigns units of velocity (km/s) vs. Flux density (Jy)'''

    if faxis is None:
        Vres = width_freq2vel(5, z=0)
    else:
        fres = faxis[1] - faxis[0]
        Vres = width_freq2vel(fres, z=0)

    FWHM = find_FWHM(x, B)
    # don't allow spectra to be arbitarily narrow 
    if W50 < Vres:
        W50=Vres
    scale = W50 / FWHM                      # scaling factor from unitless → km/s    
    V = x * scale                           # velocity axis in km/s 
    dV = V[1] - V[0]
    W50_broad = W50_broadened(W50)

    # pad velocity range to allow for narrow spectra convolution and high mass spectra
    pad = W50_broad*2
    Vaxis = np.arange(-pad, pad+dV, Vres)
    B_resamp = np.interp(x=Vaxis, xp=V, fp=B)
    
    # Scale y-axis
    MHI_initial = get_MHI(Vaxis, B_resamp, z) 
    y_scale = MHI_desired / MHI_initial
    S = B_resamp * y_scale 
    
    # convolve with Gaussian
    if thermal_broaden:
        S_broad, G = convolve_spectrum(S, Vaxis)
    else:
        S_broad = S
        
    if check:
        units_check(V, B, Vaxis, B_resamp, S, S_broad, G, W50, z, MHI_desired)

    return Vaxis, S, S_broad 

def conversion_check(f, S, S_broad, faxis, S_resamp, W50, z, MHI_desired, thres=1e-3):
    channels = S_resamp > np.max(S_resamp)*thres
    MHI_measured = get_MHI_freq(faxis[channels], S_resamp[channels], z)
    W50_f_measured = find_FWHM(faxis, S_resamp) / 1e3
    S21_measured = integrate_profile(faxis[channels], S_resamp[channels])
    spectra_extent_kHz = (np.max(faxis[channels]) - np.min(faxis[channels])) / 1e3
    W50_expected = W50_broadened(W50)
    W50_f_expected = gf.width_vel2freq(W50_expected, z) / 1e3
    W50_measured = gf.width_freq2vel(W50_f_measured/1e3, z) 
    S21_expected = MHI_to_Sint_freq(MHI_desired, z)
    spectra_extent_kms = gf.width_freq2vel(spectra_extent_kHz/1e3, z)
    Nchans = spectra_extent_kHz / (upchan_res/1e3)
    freq_obs = gf.get_fobs(z) * 1e6 
    
    print(f"MHI measured is {np.log10(MHI_measured):.2f}")
    print(f"MHI expected is {np.log10(MHI_desired):.2f}")
    
    print(f"W50 measured is {W50_measured:.2f} km/s")
    print(f"W50 expected is {W50_expected:.2f} km/s")
    
    print(f"S21 measured is {S21_measured} Jy*Hz", )
    print(f"S21 expected is {S21_expected} Jy*Hz", )
    
    print(f"W50_f measured is {W50_f_measured:.2f} kHz")
    print(f"W50_f expected is {W50_f_expected:.2f} kHz")
    
    print(f"S21_peak measured is {np.max(S_broad)} Jy")
    print(f"S21_peak expected is {S21_expected/(W50_f_expected*1e3)} Jy")
    
    print(f"Velocity spectral extent {spectra_extent_kms} km/s")
    print(f"Frequency spectral extent {spectra_extent_kHz} kHz")
    print(f"Frequency channels expected {Nchans}")
    print(f"Frequency channels measured", np.sum(channels))
    #print(S_broad)
    
    plt.figure(figsize=[10,8])
    plt.plot(f, S)
    plt.plot(f, S_broad, linestyle='--')
    plt.plot(faxis, S_resamp, linestyle=':')
    #plt.xlim(-W50_f_expected*1e3+freq_obs, W50_f_expected*1e3+freq_obs)
    plt.show()
    
    plt.figure(figsize=[10,8])
    plt.plot(faxis, S_resamp, linestyle=':')
    plt.axvline(np.min(faxis[channels]))
    plt.axvline(np.max(faxis[channels]))
    #plt.xlim(-W50_f_expected*1e3+freq_obs, W50_f_expected*1e3+freq_obs)
    plt.show()
    

def assign_freqUnits_convert(x, B, W50, z, MHI_desired, faxis=None, check=True):
    V, S, S_broad = assign_units(x, B, W50, z, MHI_desired, check=check, faxis=faxis)
    f = gf.convert_f(V, z) * 1e6 #MHz to Hz
    f = f[::-1]

    W50_f_expected = gf.width_vel2freq(W50_broadened(W50), z)

    #put the spectra on the CHORD telescope resolution - upchannelized
    if faxis is None:
        pad = W50_f_expected*2
        freq_obs = gf.get_fobs(z) * 1e6 # in Hz
        faxis = np.arange(-pad, pad+upchan_res, upchan_res) + freq_obs

    S_resamp = np.interp(x=faxis, xp=f, fp=S_broad)
    
    if check:
        conversion_check(f, S, S_broad, faxis, S_resamp, W50, z, MHI_desired)
    return faxis, S_resamp # in Hz and Jy

def assign_freqUnits(x, B, W50, D, z, MHI_desired, coarseRes=False, Vel_res=5):
    #f_con, S_con = assign_freqUnits_convert(x, B, W50, D, z, MHI_desired, coarseRes=False, Vel_res=5)
    W50broad = W50_broadened(W50)
    W_freq = gf.width_vel2freq(del_Vrest=W50, z=z) # in Hz
    #print("Wf expected is ", gf.width_vel2freq(del_Vrest=W50broad, z=z))

    if coarseRes:
        chan_res = gf.chan_width # in Hz
    else:
        chan_res = upchan_res
        
    if np.log10(MHI_desired) < 7: 
        max_Wf = gf.width_vel2freq(del_Vrest=200)
    elif (np.log10(MHI_desired) > 7) and (np.log10(MHI_desired) < 10):
        max_Wf = gf.width_vel2freq(del_Vrest=900)
    else:
        max_Wf = gf.width_vel2freq(del_Vrest=2500)
    Nchans = np.max(max_Wf/chan_res)
    freq_axis = np.arange(-Nchans*chan_res/2, Nchans*chan_res/2+1, chan_res)

    # scale x-axis width by frequency width
    S_norm = (B/np.max(B))                  # Peak Flux set to 1 Jy if not specified
    FWHM, roots = find_FWHM(x, B)
    scale = W_freq / FWHM                 # scaling factor from unitless → km/s    
    f_highres = x * scale                   # velocity axis in km/s

    # convert to MHz for integration
    f_highres = f_highres #/ 1e6 # in Jy
    freq_axis = freq_axis #/ 1e6 # in Jy
    S_resamp = np.interp(x=freq_axis, xp=f_highres, fp=S_norm)

    # Scale y-axis
    MHI_initial = get_MHI_freq(freq_axis, S_resamp, z) 
    y_scale = MHI_desired / MHI_initial   
    S_freq = S_resamp * y_scale 
    
    freq_obs = gf.get_fobs(z) * 1e6 # in Hz

    #Sf_convolve = S_freq
    sigma_f = gf.width_vel2freq(del_Vrest=10, z=z) # in Hz
    Sf_convolve = convolve_spectrum(S_freq, freq_axis, sigma=sigma_f)
    #df, _ = find_FWHM(freq_axis, Sf_convolve)
    #print("Wf actual is", df)
    #MHI_final = get_MHI_freq(freq_axis, Sf_convolve, z)

    #print("intial mass is ", MHI_initial)
    #print("desired mass is ", np.log10(MHI_desired))
    #print("final mass is ", np.log10(MHI_final))

    S_int, _ = gf.int_S21(MHI=MHI_desired, z=z)
    #print("S peak in Jy from velocity", S_int/W50broad)
    #Sf_int, _ = gf.int_S21Hz(MHI=MHI_desired, z=z) 
    #print("S peak in Jy from freq", Sf_int/gf.width_vel2freq(W50broad, z=z)) 

    freq_final = freq_axis + freq_obs
    Speak = S_int/W50broad
    # plt.figure()
    # #plt.plot(f_highres, S_norm)
    # #plt.plot(freq_axis, S_resamp)
    # plt.plot(freq_final, Sf_convolve)
    # plt.title(f'log(MHI)={np.log10(MHI_desired):.1f}, Speak={S_int/W50broad:.3f} Jy')
    # plt.show()

    # W50_broad = W50_broadened(W50)
    # Wf = gf.width_vel2freq(del_Vrest=W50_broadened(W50_broad), z=z)
    # print("W50_f in frequency expected is ", Wf)
    # df = np.max(freq_axis[Sf_convolve > 0]) - np.min(freq_axis[Sf_convolve > 0])
    # print("W50_f in frequency expected is ", Wf)

    return freq_final, Sf_convolve, Speak

def SNR_check(MHI, W50, z, RMS_Jy, SNRint):
    W50_broad = W50_broadened(W50)
    SNR_W50 = gf.SNR_int(z=z, MHI=MHI, DeltaV=W50_broad, RMS_chan=RMS_Jy, chan_width=upchan_res)
    SNR_spectra = SNRint
    print(f"SNRint from W50 estimate is {SNR_W50:.2f}")
    print(f"SNRint from spectra is {SNR_spectra:.2f}")
    print(f"the ratio is {SNR_W50/SNR_spectra:.2f}")
    

def SNRint(f, Sf, z, W50, MHI, RMS_Jy, window=1e-3, check=True):
    if z==0:
        z=1e-4

    channels = Sf > np.max(Sf)*window 
    S_int = integrate_profile(f[channels], Sf[channels]) # in Jy*Hz
    DL = gf.Luminosity_Dist(z)
    N_chans = np.sum(channels)
    
    if N_chans!=0: # avoid divide by 0
        SNRint = S_int / (RMS_Jy*upchan_res*np.sqrt(N_chans))
    else:
        SNRint=0
    
    if check:
        SNR_check(MHI, W50, z, RMS_Jy, SNRint)
               
    return SNRint

def Generate_Spectra(MHI, W50, z, RMS_Jy, faxis=None, a=1, w=1, b1=None, b2=None, c=None, xe=None, xp=None, n=None):
    ''' Main function which generates an HI Spectrum. 
    The shape of the busy function (specified by a, b1, b2, c) is randomly generated (unless otherwise specified). 
    The area under the profile is set by MHI. 
    The FWHM width (W50) is set by VHI and inclination using W50 = VHI*2sin(i). 
    The profile is centered around 0 so xe=0 and xp=0. We use a second order n=2 general busy function'''

    #print("Generating Spectra....")
    #start = time.time()
    
    if z==0:
        z=1e-4 # avoid multiply/divide by 0
      
    x = np.linspace(-20, 20, 10000).astype(np.float32)  # unitless axes used in generalized busy function definition

    if b1==None: b1 = np.random.uniform(1, 5)
    if b2==None: b2 = np.random.uniform(1, 5)
    if c==None: c = np.random.uniform(0, 4)
    if xe==None: xe = 0 #np.random.uniform(-0.1, 0.1, size=size)
    if xp==None: xp =  0 #np.random.uniform(-0.1, 0.1, size=size)
    if n==None: n = np.random.uniform(1, 4) # n=odd number results in negative values, n=4 is too broad

    B = Busy_general(x, a, b1, b2, xe, xp, c, w, n)
    f, Sf = assign_freqUnits_convert(x, B, W50, z=z, MHI_desired=MHI, check=False, faxis=faxis)
    SNR_int = SNRint(f, Sf, z, W50, MHI, RMS_Jy, check=False)
                                                   
    #end = time.time()
    #return f, Sf, Sf_broad, V, S, S_broad, f_v, Sf_broad_v, SNR_int
    return SNR_int, f, Sf

def Save_Spectra(catalog_fl, size=None):
    catalog = np.load(catalog_fl)
    if size is None:
        size = catalog.shape[1]

    MHI = catalog[0]
    W50 = catalog[3]
    D = catalog[6]
    z = catalog[8]

    SNR_int = Generate_Spectra(size, MHI, W50, D, z)
    np.save("catalogs_output/Spectra_SNR_int_D200_sigma1.npy", SNR_int)
    #np.save("VolLim_20to60deg_Dmax100_spectra.npy", np.asarray([V, S_broad]))
    #MHI_res = np.log10(final_M) - np.log10(MHI)
    #plt.figure()
    #plt.scatter(MHI, MHI_res)
    #plt.show()

#if __name__ == "__main__":
#    Save_Spectra(catalog_fl='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', size=None)
