import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from scipy.special import gammaincc,expi,gamma
from astropy.cosmology import Planck18 as cosmo_P2018
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

# HIMF Schechter Parameters from Jones 2018:
phi_s=4.5e-3
M_s=10**9.94 
alpha=-1.25
MHI_grid = np.logspace(5,12,10000)

#Constant for MHI calculation:
C21 = (2.356 * 10**5) * u.solMass * u.Mpc**-2 * (u.Jy*u.km/u.s)**-1

def mid_bin(var):
    return 0.5*(var[1:] + var[:-1])

############################# Setting Cosmology distances and volume  ######################

def set_cosmology(h=0.7, om=0.315):
    H0 = h * 100.0  # km/s/Mpc
    cosmo = FlatLambdaCDM(H0=H0, Om0=om)
    return cosmo

def comoving_volume(zmax, npt=20000, solidang=4*np.pi, cosmo=cosmo_P2018):
    '''Get comoving volume, dV shell volume and distance from redshift'''
    z_arr = np.linspace(0.0, zmax, npt)
    dz = np.diff(z_arr)

    # find mid-points
    z = mid_bin(z_arr)
    D = cosmo.comoving_distance(z).to(u.Mpc)
    
    # Volume shells:
    dV_dz = solidang * (D**2) * (c.c.to(u.km/u.s) / cosmo.H(z).to(u.km/u.s/u.Mpc))
    dV = dV_dz * dz
    # Cumulative volume:
    V=np.cumsum(dV)
    return z, D, V, dV

def Comoving_Dist(z, cosmo=cosmo_P2018):
    return cosmo.comoving_distance(z).to(u.Mpc)

def build_z_interp(zmax=5.0, npt=10000, cosmo=cosmo_P2018):
    """Build fast interp to invert comoving_distance to get redshift z."""
    z_grid = np.linspace(0.0, zmax, npt)
    D = cosmo.comoving_distance(z_grid).to(u.Mpc)  # array in Mpc
    return interp1d(D, z_grid, kind="linear", bounds_error=False, fill_value=(0.0, zmax))

def VolumeFromDist(D, solidang):
    return (1/3)*solidang*(D**3)

############################### RA and Dec Spherical coords #############################

def solid_angle(dec1, dec2, ra1, ra2):
    """RA/Dec in degrees, returns solid angle in steradians for a rectangular patch."""
    dec1, dec2 = np.deg2rad(dec1), np.deg2rad(dec2)
    ra1, ra2 = np.deg2rad(ra1), np.deg2rad(ra2)
    
    # Compute delta RA, handling wrap-around and full circle
    delta_ra = (ra2 - ra1) % (2*np.pi)
    delta_ra = delta_ra if delta_ra > 0 else 2*np.pi
    
    return delta_ra * (np.sin(dec2) - np.sin(dec1))


################################   HIMF Schechter Functions  ############################

def schechter_fit_lg(MHI=MHI_grid, phi_s=phi_s, M_s=M_s, alpha=alpha):
    return np.log(10)*(phi_s) * (MHI/M_s)**(alpha+1) * np.exp(-1*(MHI/M_s)) 

def schechter_fit(MHI, phi_s=phi_s, M_s=M_s, alpha=alpha):
    return (phi_s)*(MHI/M_s)**(alpha) * np.exp(-1*(MHI/M_s)) / M_s

def my_gammainc(x0, alpha):
    if alpha==-1:
        return -expi(-x0)
    if alpha>-1:
        return gammaincc(alpha+1,x0)*gamma(alpha+1)
    else:
        tmp=my_gammainc(alpha+1,x0)
        tmp2=-(x0**(alpha+1))*np.exp(-x0)
        return (tmp+tmp2)/(alpha+1)
    return None #never get here

def schech_int_Sievers(L,alpha=alpha,L_star=M_s,phi_star=phi_s):
    x=L/L_star
    y=my_gammainc(alpha,x)
    return y*phi_star

def galaxy_density(MHI=MHI_grid, phi_s=phi_s, M_s=M_s, alpha=alpha):
    HIMF = schechter_fit_lg(MHI, phi_s=phi_s, M_s=M_s, alpha=alpha)
    n = np.trapz(HIMF, x=np.log10(MHI))
    return n

################################   Galaxy Relations   #######################################

def S_toMHI(S_peak, delV, D):
    '''Approximation of HI mass based on S21 integrated flux without spectra'''
    return ((C21*S_peak*delV*D**2)).to(u.solMass)

def MHI_toS(MHI, delV, D, unitless=True):
    '''Approximation of S21 flux based on HI mass without spectra'''
    if unitless:
        return MHI/(C21.value*delV*D**2)
    else:
        return ((MHI)/(C21*delV*D**2)).to(u.Jy)

def Vmax_correct(D, S21, S21lim, solidang):
    Dmax = D*np.sqrt(S21/S21lim)
    Vmax = VolumeFromDist(Dmax, solidang)
    return Vmax

def VHI_polyFit(MHI):
    '''Polynomial Fit for rotational velocity from Spekkens&Lewis: https://www.overleaf.com/project/5e378eb163ee6f0001cc9a7f'''
    x = np.log10(MHI)
    lg_VHI = 0.0345*(x**3) - 0.955*(x**2) + 9.134*x - 27.99
    return 10**lg_VHI

def estimate_W50(Vrot, i, broaden=True, thermal_FWHM=10):
    '''Get W50 from rotational velocity'''
    W50 = Vrot*2*np.sin(i)
    # Apply thermal broadening - estimating convolution by Gaussian, add in quadrature
    if broaden==True: 
        W50 = np.sqrt(W50**2 + thermal_FWHM**2)
    return W50



