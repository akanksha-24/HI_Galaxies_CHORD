import numpy as np
import astropy.units as u
import astropy.constants as c
import Galaxy_Functions as gf
import time
import matplotlib.pyplot as plt
from Galaxy_Functions import *

D=6
Tsys=30*u.K
#nu=183*u.kHz
eff=0.5
Aeff=eff*(np.pi*(D/2)**2) * u.m**2
#SEFD=9*u.Jy 

def RMS_fromDays(days, decl, nu, z=0, N=512, PB=True):
    omega = 360 / (23.9345 * 3600) * u.deg / u.s
    FWHM_PB = np.rad2deg(1.2*0.21*(1+z)/D) * u.deg
    #print("FWHM PB is", FWHM_PB)
    sigma = ((2*c.k_B*Tsys/Aeff)*np.sqrt(omega*np.cos(decl)/(N*(N-1)*nu*FWHM_PB*days))).to_value(u.mJy)
    return sigma

def native_freso(f_low=0, f_high=1500):
    resolution = 1500 / 8192
    nchans = int((f_high - f_low) / resolution)
    return np.linspace(f_low, f_high, nchans)

def beam_size(waveln=0.21, Bmax=(8.5*22), z=0):
    return 1.029*waveln*(1+z)/Bmax

def getRMS(tau=3600, nu=183*u.kHz, N=512):
    tau=tau *u.s
    #print("chan_res is", nu)
    RMS = (2*c.k_B*Tsys) / (Aeff*np.sqrt(N*(N-1)*nu*tau))
    return RMS.to(u.mJy)

def RMS_intTime(RMS, nu=183*u.kHz, N=512):
    RMS = (RMS * u.mJy).to(u.Jy)
    tau = ((2*c.k_B*Tsys)**2) / (N*(N-1)*nu*(Aeff*RMS)**2)
    return tau.to(u.s)

def dwell_time(decl, PB=True):
    sidereel_rate = 360 / (23.9345 * 3600) # deg/s
    if PB:
        beam = beam_size(Bmax=6)
    else:
        beam = beam_size(Bmax=8.5*22)
    theta = np.rad2deg(beam)
    #print("theta is ", theta)
    return theta / (np.cos(decl) * sidereel_rate) # in units s/day

def obs_days(RMS, decl, PB=True, nu=183*u.kHz, N=512):
    tau = RMS_intTime(RMS, nu, N)
    dwell = dwell_time(decl, PB)
    #print("dwell time is ", dwell)
    days = tau / dwell
    return days.value

def time2RMS(days, decl, PB=True, nu=183*u.kHz, N=512):
    dwell = dwell_time(decl, PB) # u.s/day
    #print('dwell time is ', dwell)
    tau = dwell * days # u.s
    #print("tau is ", tau)
    return getRMS(tau, nu, N)

#def Equal_Depth_Survey(catalog, RMS)



def noise_cuts(catalog_fl, bandwidth, obs_length, nstrips, sigma=5, flname='', figname=''):
    strip_obs = obs_length.to(u.day) / nstrips
    noise = time2RMS(strip_obs.value, decl=np.deg2rad(45)).to(u.Jy)

    catalog = np.load(catalog_fl)
    MHI = catalog[:,0]
    W50 = catalog[:,1]
    D = catalog[:,2]
    S_flux = gf.MHI_toS(MHI=MHI, delV=W50, D=D, unitless=True) # units of Jy
    print(S_flux)
    #int_noise = noise.value*np.sqrt(bandwidth/W50)
    Slimit_int = noise.value*sigma
    np.save(flname, np.asarray(S_flux))
    
    #S_flux = np.load(flname)
    mask = S_flux > Slimit_int   

    print(len(MHI))              
    print(mask.sum())            

    MHI_masked = MHI[mask]       
    np.save('MHI_masked.npy', MHI_masked)
    #MHI_masked = np.load('MHI_masked.npy')

    plt.figure(figsize=[4,3], dpi=300)
    plt.hist(np.log10(MHI_masked), bins=30, histtype='step')
    plt.yscale('log')
    plt.xlabel('log(M$_{HI}$) [M$_{\odot}$]')
    plt.ylabel('log(Counts)')
    plt.tight_layout()
    plt.savefig('figname')
    plt.show()

    return np.log10(MHI_masked) 

def build_survey(obs_years, z, switch_int=7, beam_sep=2.5, start=20, end=80, beam_centers=None):
    if beam_centers is None:
        beam_centers = np.deg2rad(np.arange(start,end+beam_sep,beam_sep)) # radians
    #print("beam_centers is ", np.rad2deg(beam_centers))
    nstrips = len(beam_centers)
    #print("nstrips is ", nstrips)
    switch_time = switch_int*nstrips
    total_time = obs_years*365-switch_time
    obs_strip = total_time/nstrips
    time_strip = obs_strip * np.cos(beam_centers)
    timespent = np.sum(time_strip)
    while np.ceil(timespent) < total_time:
        timeleft = (total_time - timespent)/nstrips
        left_strip = timeleft * np.cos(beam_centers)
        time_strip = time_strip + left_strip
        timespent = np.sum(time_strip)
    #constant = np.cos(beam_centers)/time_strip
    #print(constant)
    nu = width_vel2freq(5) * u.Hz
    #RMS = RMS_fromDays(days=time_strip, decl=beam_centers, nu=nu)
    #print("RMS is ", RMS) # in mJy
    RMS_z = RMS_fromDays(days=time_strip[0], decl=beam_centers[0], nu=nu, z=z)
    return time_strip, RMS_z, beam_centers

#noise_cuts(catalog_fl="catalogs_output/VolLim_20to60deg_Dmax200.npy", bandwidth=38, obs_length=5*u.year, nstrips=20,
#           flname="catalogs_output/Sflux_20to60deg_Dmax200.npy")

# noise_cuts(catalog_fl="catalogs_output/VolLim_20to60deg_Dmax500.npy", bandwidth=38, obs_length=5*u.year, nstrips=20,
#            flname="catalogs_output/Sflux_20to60deg_Dmax500.npy")

# noise_cuts(catalog_fl="catalogs_output/VolLim_20to60deg_Dmax500.npy", bandwidth=38, obs_length=5*u.year, nstrips=20,
#             flname="catalogs_output/Sflux_VolLim_20to60deg_Dmax500.npy", figname='Plots/VolLim_20to60deg_Dmax200_counts.png')
