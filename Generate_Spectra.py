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
# need to fix
#RMS_mJy = time2RMS(days=5*365/24, decl=np.deg2rad(20), nu=upchan_res*u.Hz).value

def Busy_general(x, a, b1, b2, xe, xp, c, w, n):
    ''' This is the functional definition for a General Busy Function:
        Reference: https://ui.adsabs.harvard.edu/abs/2014MNRAS.438.1176W/abstract (Section 4.1, Equation 4)'''
    
    err_p = special.erf(b1*(w+x-xe)) + 1
    err_m = special.erf(b2*(w-x+xe)) + 1
    pbola = (c*(np.abs(x-xp)**n)) + 1

    # plt.figure()
    # plt.plot(x, (a/4)*err_p*err_m*pbola)
    # plt.show()
    
    return (a/4)*err_p*err_m*pbola

def integrate_profile(V, S):
    ''' Helper function which integrates an HI profile. The 'V' parameter is the velocity axis in km/s. The 'S' paramater is the flux in mJy'''

    return np.trapz(S, x=V)  # trapz is a numeric integrator in python using trapezoid rule, diff(V) if dV

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
    int_S = integrate_profile(f, S) # in MHz*Jy
    return 49.7 * (D_L**2) * int_S * 1e6 # in Hz*Jy


# def find_FWHM(x, y, level=0.5):
#     ''' Helper function which finds the roots of the HI profile at the FWHM, to set to W50 width. 
#     Returns an array of all roots r[] found where the FWHM width is r[-1] - r[0] '''

#     spline = UnivariateSpline(x, y-(np.max(y)*level), s=0)
#     roots = spline.roots() 
#     FWHM = roots[-1] - roots[0]
#     if len(roots) < 2:
#         return np.nan, roots  # no valid FWHM found
#     return FWHM, roots

def find_FWHM(x,y, level=0.5):
    profile_idx = np.argwhere(y > level*np.max(y))[:,0]
    width = x[np.max(profile_idx)] - x[np.min(profile_idx)]
    roots = [x[np.min(profile_idx)],x[np.max(profile_idx)]]
    return width, roots

def assign_units(x, B, W50, D, z, MHI_desired, Vlim=50, thermal_broaden=True):
    '''assigns units of velocity (km/s) vs. Flux density (Jy)'''

    # scale x-axis
    S_norm = (B/np.max(B))                  # Peak Flux set to 1 Jy if not specified
    FWHM, roots = find_FWHM(x, B)
    scale = W50 / FWHM                      # scaling factor from unitless → km/s    
    V = x * scale                           # velocity axis in km/s

    # for very narrow spectra, pad 0's to be able to convolve with gaussian later
    if np.max(V) <= Vlim:
        dV = V[1] - V[0]
        if dV < 1e-6: dV = 1e-6 # avoid divide by 0
        n_pad = int((Vlim - np.max(V)) / dV)  # extend array to -20 to 20 km/s
        left_pad  = V[0] - dV * np.arange(n_pad, 0, -1)
        right_pad = V[-1] + dV * np.arange(1, n_pad + 1)
        V = np.concatenate([left_pad, V, right_pad])
        S_norm = np.pad(S_norm, pad_width=(n_pad, n_pad), mode='constant', constant_values=0.0)   

    # Scale y-axis
    MHI_initial = get_MHI(V, S_norm, z) 
    y_scale = MHI_desired / MHI_initial   
    S = S_norm * y_scale  
    if thermal_broaden:
        S_broad = convolve_spectrum(S, V)
    else:
        S_broad = S

    return V, S_broad     # Return V and S axes and W_ roots (for checking)

def assign_freqUnits(x, B, W50, D, z, MHI_desired, coarseRes=False, Vel_res=5):
    W50broad = W50_broadened(W50)
    W_freq = gf.width_vel2freq(del_Vrest=W50, z=z) # in Hz

    if coarseRes:
        chan_res = gf.chan_width # in Hz
    else:
        chan_res = upchan_res
        #print("chan res in Hz", chan_res)

    if np.log10(MHI_desired) < 7: 
        max_Wf = gf.width_vel2freq(del_Vrest=200)
    elif (np.log10(MHI_desired) > 7) and (np.log10(MHI_desired) < 10):
        max_Wf = gf.width_vel2freq(del_Vrest=900)
    else:
        max_Wf = gf.width_vel2freq(del_Vrest=2500)
    Nchans = np.max(max_Wf/chan_res)
    chan_res = chan_res 
    freq_axis = np.arange(-Nchans*chan_res/2, Nchans*chan_res/2+1, chan_res)

    # scale x-axis width by frequency width
    S_norm = (B/np.max(B))                  # Peak Flux set to 1 Jy if not specified
    FWHM, roots = find_FWHM(x, B)
    scale = W_freq / FWHM                 # scaling factor from unitless → km/s    
    f_highres = x * scale                   # velocity axis in km/s

    # convert to MHz to integration
    f_highres = f_highres / 1e6
    freq_axis = freq_axis / 1e6
    S_resamp = np.interp(x=freq_axis, xp=f_highres, fp=S_norm)

    # Scale y-axis
    MHI_initial = get_MHI_freq(freq_axis, S_resamp, z) 
    y_scale = MHI_desired / MHI_initial   
    S_freq = S_resamp * y_scale 
    
    freq_obs = gf.get_fobs(z) # in MHz

    sigma_f = gf.width_vel2freq(del_Vrest=10) / 1e6 # in MHz
    Sf_convolve = convolve_spectrum(S_freq, freq_axis, sigma=sigma_f)
    MHI_final = get_MHI_freq(freq_axis, Sf_convolve, z)

    #print("intial mass is ", MHI_initial)
    #print("desired mass is ", MHI_desired) 
    #print("final mass is ", MHI_final)

    S_int, _ = gf.int_S21(MHI=MHI_desired, z=z)
    #print("S peak in Jy from velocity", S_int/W50broad)
    Sf_int, _ = gf.int_S21Hz(MHI=MHI_desired, z=z) 
    #print("S peak in Jy from freq", Sf_int/gf.width_vel2freq(W50broad, z=z)) 

    freq_final = freq_axis + freq_obs
    Speak = S_int/W50broad
    # plt.figure()
    # #plt.plot(f_highres, S_norm)
    # #plt.plot(freq_axis, S_resamp)
    # plt.plot(freq_final, Sf_convolve)
    # plt.title(f'log(MHI)={np.log10(MHI_desired):.1f}, Speak={S_int/W50broad:.3f} Jy')
    # plt.show()

    return freq_final, Sf_convolve, Speak


# Gaussian Function
def normalDist(x, sigma=10, x0=0):
    G =  np.exp(-(x - x0)**2 / (2*sigma**2)) 
    return G / np.trapz(G, x)                 # normalize computationally 

def convolve_spectrum(S, V, sigma=10):
    #G = gaussian_kernel(V)
    dv = V[1] - V[0]
    G = normalDist(V, sigma=sigma)
    # print("AREA is ", np.trapz(G, V))
    # plt.figure()
    # plt.plot(V, G)
    # plt.show()
    S_broad = fftconvolve(S, G, mode='same') * dv  # since fftconvolve is a sum, must include dv
    return S_broad

# def gaussian_kernel(V, sigma=10):
#     #sigma = FWHM / (2*np.sqrt(2*np.log(2)))
#     G = normalDist(V, sigma)                # center kernel at 0
#     return G / G.sum()                      # normalize discrete sum = 1

# def convolve_spectra(B, V, FWHM):
#     G = gaussian_kernel(V, FWHM)
#     return fftconvolve(B, G[None, :], mode='same', axes=1)

def SNRint(f, Sf, z, W50, MHI, D_C, RMS_mJy, obs_yr=5, prevSNR=None):
    W50_broad = W50_broadened(W50)
    Wf = gf.width_vel2freq(del_Vrest=W50_broadened(W50_broad))

    #chan_mask = Sf > RMS_mJy*1e-3
    chan_mask = Sf > RMS_mJy*1e-8
    S_int = integrate_profile(f[chan_mask], Sf[chan_mask]) # in Jy*MHz
    #N_chans = np.sum(Sf > RMS_mJy*1e-3)
    N_chans = np.sum(Sf > 1e-8)
    #print("N_chans is ", N_chans)
    if N_chans!=0:
        SNRint = S_int / (RMS_mJy*upchan_res*1e-9*np.sqrt(N_chans))
        #print("Spectra SNR", SNRint)
        #print("previous SNR", prevSNR)
    else:
        SNRint=0
        #if (prevSNR is None)==0:
        #    print("previous 0 SNR was ", prevSNR)
    return SNRint

    #SNRint2 = gf.SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=183000)#=upchan_res)
    #print("integrated SNR old way is ", SNRint2)
    #v_ch = 5
    #f_smo = np.minimum(W50_broad/(v_ch),(10.**2.5)/v_ch)
    #SNR_MJ = MHI*np.sqrt(f_smo)/(RMS_mJy*W50_broad*235.6*D_C**2)
    #print("integrated singal to noise by MJ is ", SNR_MJ)



def Generate_Spectra(size, MHI, W50, D_C, z, RMS, a=1, w=1, b1=None, b2=None, c=None, xe=None, xp=None, n=None, 
                     thermal_broaden=True, fname='Plots/spectra_check_z0p1.pdf', plotSpectra=False, prevSNR=None):
    ''' Main function which generates an HI Spectrum. 
    The shape of the busy function (specified by a, b1, b2, c) is randomly generated (unless otherwise specified). 
    The area under the profile is set by MHI. 
    The FWHM width (W50) is set by VHI and inclination using W50 = VHI*2sin(i). 
    The profile is centered around 0 so xe=0 and xp=0. We use a second order n=2 general busy function'''

    #print("Generating Spectra....")
    start = time.time()

    #W = VHI*2*np.sin(i) # W_50       
    x = np.linspace(-20, 20, 10000).astype(np.float32)  # unitless axes used in generalized busy function definition

    #D_L1 = (1+z)*D_C
    D_L = cosmo.luminosity_distance(z).to_value(u.Mpc)

    if b1==None: b1 = np.random.uniform(1, 5, size=size)
    if b2==None: b2 = np.random.uniform(1, 5, size=size)
    if c==None: c = np.random.uniform(0, 4, size=size)
    if xe==None: xe = np.zeros(size) #np.random.uniform(-0.1, 0.1, size=size)
    if xp==None: xp =  np.zeros(size) #np.random.uniform(-0.1, 0.1, size=size)
    if n==None: n = np.random.uniform(1, 4, size=size) # n=odd number results in negative values, n=4 is too broad

    SNR_int = []
    with PdfPages(fname) as pdf_:
        for i in np.arange(size):
            #print(i)
            B = Busy_general(x, a, b1[i], b2[i], xe[i], xp[i], c[i], w, n[i])
            #V, S = assign_units(x, B, 1, D_L[i], z[i], MHI_desired=MHI[i])
            V, S = assign_units(x, B, W50[i], D_L[i], z[i], MHI_desired=MHI[i])
            f, Sf, Speak = assign_freqUnits(x, B, W50[i], D_L[i], z[i], MHI_desired=MHI[i])
            f_V = gf.convert_f(V, z[i])
            final_M = get_MHI_freq(f, Sf, z[i])
            #print("Final  M is ", final_M)
            SNR = SNRint(f, Sf, z[i], W50[i], MHI[i], D_C=D_C[i], RMS_mJy=RMS[i], prevSNR=prevSNR[i])
            SNR_int.append(SNR)
            if plotSpectra:
                fig = plt.figure()
                #plt.plot(f_V, S)
                plt.plot(f, Sf)
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('Flux Density (Jy)')
                plt.title(f'log(MHI)={np.log10(MHI[i]):.1f}, D={D_C[i]:.0f} Mpc, z={z[i]:.2f}, SNR={SNR:.0f}, Speak={Speak*1000:.3f} mJy')
                pdf_.savefig(fig)
                plt.show()
                plt.close(fig)

        end = time.time()
    #return x, B
    #print(f"Runtime: {end - start:.3f} seconds")
    return SNR_int
    #return final_M, V, S, SNR_int#, freq

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

if __name__ == "__main__":
    Save_Spectra(catalog_fl='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', size=None)
