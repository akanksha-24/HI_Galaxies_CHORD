import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from scipy.special import gammaincc,expi,gamma
from astropy.cosmology import Planck18 as cosmo_P2018
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
import Gaussian_Estimate as gauss
from astropy.coordinates import Angle

# Default is Jones 2018
phi_s=4.5e-3
M_s=10**9.94 
alpha=-1.25

del_alpha = 0.1 
del_M_s=10**9.94*np.log(10)*0.051
del_phi_s=np.sqrt(0.2**2 + 0.8**2)*1e-3

MHI_grid = np.logspace(5,12,100000)

#Constant for MHI calculation:
C21 = (2.356 * 10**5) * u.solMass * u.Mpc**-2 * (u.Jy*u.km/u.s)**-1

# coarse chan wdith 
chan_width = ((1600/8192)*u.MHz).to_value(u.Hz) # or 1200/6144

def mid_bin(var):
    return (var[1:] + var[:-1])/2

############################# HIMFs #######################################################
def HIMF_Jones2018(MHI=MHI_grid):
    #HIMF Schechter Parameters from Jones+ 2018:
    phi_s=4.5e-3
    M_s=10**9.94 
    alpha=-1.25
    HIMF = schechter_fit_lg(MHI, phi_s=phi_s, M_s=M_s, alpha=alpha)
    return HIMF

#def Draw_Jones2018(MHI_MHI_grid):

def HIMF_Oman2021(MHI=MHI_grid):
    # HIMF & HIWF Schechter Parameters from Oman 2021 alpha.100 + all uncertainties:
    phi_s=10**(-2.26)
    M_s=10**9.92 
    alpha=-1.29

    HIMF = schechter_fit_lg(MHI, phi_s=phi_s, M_s=M_s, alpha=alpha)
    return HIMF

def HIMF_Ma2024(MHI=MHI_grid):
    #HIMF Schechter Parameters from FASHI Wenlin Ma+ 2024
    phi_s=6.58e-3
    M_s=10**9.86 
    alpha=-1.30
    HIMF = schechter_fit_lg(MHI, phi_s=phi_s, M_s=M_s, alpha=alpha)
    return HIMF

def Oman2021_HIWF():
    phi_sW=10**(-1.67)
    W_s=307
    alpha_W=-0.63
    beta_W=2.0

    W50 = np.logspace(1,3,10000)
    HIWF  = HIWF_fit(W50=W50, phi_s=phi_sW, W_s=W_s, alpha=alpha_W, Beta=beta_W)
    HIWF_Schec = HIWF_Schecter(W50=W50, phi_s=phi_sW, W_s=W_s, alpha=alpha_W)
    return W50, HIWF, HIWF_Schec

############################# Setting Cosmology distances and volume  ######################

def set_cosmology(h=0.7, om=0.315):
    H0 = h * 100.0  # km/s/Mpc
    cosmo = FlatLambdaCDM(H0=H0, Om0=om)
    return cosmo

cosmo=cosmo_Jones2018 = set_cosmology(h=0.7)
# cosmo_P2018

def comoving_volume(zmax, zmin=0, zstep=1e-3, solidang=4*np.pi):
    '''Get comoving volume, dV shell volume and distance from redshift'''
    if zmin==0:
        zmin=zstep/2
    z_arr = np.arange(zmin, zmax, zstep)
    dz = np.diff(z_arr)

    # find mid-points
    z = mid_bin(z_arr)
    H = cosmo.H(z).value  # km/s/Mpc
    D = cosmo.comoving_distance(z).to_value(u.Mpc)
    c_kms = c.c.to_value(u.km/u.s)
    
    # Volume shells:
    dV_dz = solidang * (D**2) * (c_kms / H)
    dV = dV_dz * dz
    # Cumulative volume:
    V=np.cumsum(dV)
    return z, D, V, dV

def freq_fromz(z, f0=1420*u.MHz):
    return f0 / (1+z)

def Comoving_Dist(z):
    return cosmo.comoving_distance(z).to(u.Mpc)

def Luminosity_Dist(z):
    return cosmo.luminosity_distance(z).to_value(u.Mpc)

def build_z_interp(zmin=0, zmax=5.0, zstep=1e-3, dtype=np.float32):
    """Build fast interp to invert comoving_distance to get redshift z."""
    if zmin==0:
        zmin=zstep/2
    z_grid = np.arange(zmin, zmax, zstep, dtype=dtype)
    D = cosmo.comoving_distance(z_grid.astype(np.float64)).to(u.Mpc).value.astype(dtype)
    return interp1d(D, z_grid, kind="linear", bounds_error=False, fill_value=(0.0, zmax))

def VolumeFromDist(D, solidang=4*np.pi, Dmin=0):
    return (1/3)*solidang*(D**3 - Dmin**3)

def Hubble_redshift(D, H0=(67.77*u.km/u.s/u.Mpc).value):
    '''Use for local universe'''
    c_kms = c.c.to_value(u.km/u.s)
    return D*H0/c_kms


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

def HIWF_fit(W50, phi_s, W_s, alpha, Beta):
    return np.log10(10)*(phi_s)*((W50/W_s)**alpha)*(np.exp(-W50/W_s)**Beta)

def HIWF_Schecter(W50, phi_s, W_s, alpha):
    return np.log10(10)*(phi_s)*((W50/W_s)**alpha)*(np.exp(-1*W50/W_s))

def schechter_fit_lg(MHI=MHI_grid, phi_s=phi_s, M_s=M_s, alpha=alpha):
    return np.log(10)*(phi_s) * (MHI/M_s)**(alpha+1) * np.exp(-1*(MHI/M_s)) 

def schechter_fit(MHI, phi_s=phi_s, M_s=M_s, alpha=alpha):
    return (phi_s)*(MHI/M_s)**(alpha) * np.exp(-1*(MHI/M_s)) / M_s43

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

def S_toMHI(S_peak, delV, D, unitless=True):
    '''Approximation of HI mass based on S peak flux without spectra'''
    if unitless:
        return C21.value*S_peak*delV*D**2
    else:
        return ((C21*S_peak*delV*D**2)).to(u.solMass)

def MHI_toS(MHI, delV, D, unitless=True):
    '''Approximation of S peak flux based on HI mass without spectra'''
    if unitless:
        return MHI/(C21.value*delV*D**2)
    else:
        return ((MHI)/(C21*delV*D**2)).to(u.Jy)
    
def int_S21(MHI, z):
    D_L = cosmo.luminosity_distance(z).to_value(u.Mpc)
    S21 = MHI*(1+z)/(D_L**2 * C21.value) 
    return S21, D_L

def int_S21Hz(MHI, z):
    '''returns integrated flux in units of Jy*Hz'''
    D_L = cosmo.luminosity_distance(z).to_value(u.Mpc)
    S21 = MHI/(D_L**2 * 49.7)
    return S21, D_L 

def SNR_int(z, MHI, DeltaV, RMS_chan, chan_width=chan_width):
    '''The integrated signal-to-noise from rest-frame velocity widths
       Using Equation 156 from https://arxiv.org/pdf/1705.04210'''
    
    D_L = cosmo.luminosity_distance(z).to_value(u.Mpc)
    C = 2.92*10**-4
    return C*(MHI/(RMS_chan*D_L**2))*np.sqrt((1+z)/(chan_width*DeltaV))

def SNR_int_check(z, MHI, df, RMS_chan, chan_width):
    D_L = cosmo.luminosity_distance(z).to_value(u.Mpc)
    return (MHI/(49.7*RMS_chan*D_L**2))/np.sqrt(chan_width*df) 

    
def estimate_DLmax(MHI, z, sigma, RMS_chan, DeltaV, chan_width=chan_width):
    C = 2.92*10**-4
    RMS_chan = (RMS_chan*u.mJy).to_value(u.Jy)
    DLsquared = C*(MHI/(RMS_chan*sigma))*np.sqrt((1+z)/(chan_width*DeltaV))
    return np.sqrt(DLsquared)

def estimate_MHImax(z, sigma, RMS_chan, DeltaV, chan_width=chan_width):
    C = 2.92*10**-4
    RMS_chan = (RMS_chan*u.mJy).to_value(u.Jy)
    D_L = cosmo.luminosity_distance(z).to_value(u.Mpc)
    MHImax = (sigma*RMS_chan*np.sqrt(chan_width*DeltaV)*D_L**2)/(np.sqrt(1+z)*C)
    #print(MHImax)
    MHImax2 = (2.35*10**5)*(sigma*RMS_chan)*(D_L**2)*(np.sqrt(c.c.to_value(u.km/u.s)*chan_width*DeltaV/((1+z)*1420.405751768*1e6)))
    #print(MHImax2)
    return MHImax2
    
def Vmax_correct(catalog_file, sigma=6, RMS=0.1, fromD=True, mockAlf=False, solidang=None):
    catalog = np.load(catalog_file)
    #Dsurvey = Comoving_Dist(z=1).value #np.nanmax(catalog[6]) # max survey distance
    #Dmin = Comoving_Dist(z=0.4).value
    Dsurvey = np.max(catalog[6])
    print("Dsurvey is ", Dsurvey)
    Dmin = np.nanmin(catalog[6])
    #print("Dmin is ", Dmin)
    W50_broad = gauss.W50_broadened(W50=catalog[3])
    if solidang is None:
        solidang = solid_angle(dec1=np.min(catalog[5]), dec2=np.max(catalog[5]), ra1=np.min(catalog[4]), ra2=np.max(catalog[4]))
        print("solid ang is ", solidang)
    # zmin = np.nanmin(catalog[8])
    # print("zmin is ", zmin)
    if fromD:
        #chan_width = (1500*u.MHz/8192).to_value(u.Hz)
        chan_width = width_vel2freq(5)
        Dmax = estimate_DLmax(MHI=catalog[0], z=catalog[8], sigma=sigma, RMS_chan=RMS, chan_width=chan_width, DeltaV=W50_broad)
    if fromD==False or mockAlf:
        S21, D = int_S21(MHI=catalog[0], z=catalog[8])
        if mockAlf:
            S21lim = S21th_ALFALFA(W50=W50_broad, SNR=6.5)
        else:
            RMS = (RMS*u.mJy).to_value(u.Jy)
            S21lim = S21_th(W50=W50_broad, RMS=RMS, sigma=sigma)
        Dmax = D*np.sqrt(S21/S21lim)
    Dmax_comov = Dmax / (1 + catalog[8])  # convert from luminosity distance to comoving
    #print(Dmax_comov)
    Dmax_comov = np.minimum(Dmax_comov, Dsurvey)  
    print("Dmax from plotting ", Dmax_comov)
    Vmax = VolumeFromDist(Dmax_comov, solidang=solidang, Dmin=Dmin)
    print("Vmax from Plotting ", Vmax)
    #print(Vmax)
    return Vmax, catalog[0], W50_broad

def VHI_polyFit(MHI, dtype=np.float32):
    '''Polynomial Fit for rotational velocity from Spekkens&Lewis: https://www.overleaf.com/project/5e378eb163ee6f0001cc9a7f'''
    coeff = np.array([0.0345, -0.955, 9.134, -27.99], dtype=dtype)
    x = np.log10(MHI)
    lg_VHI = coeff[0]*x**3 + coeff[1]*x**2 + coeff[2]*x + coeff[3]
    return 10**lg_VHI

def estimate_W50(Vrot, i, broaden=True, thermal_FWHM=10, dtype=np.float32):
    '''Get W50 from rotational velocity'''
    W50 = Vrot*2*np.sin(i)
    # thermal_FWHM = np.array(thermal_FWHM, dtype=dtype)
    # # Apply thermal broadening - estimating convolution by Gaussian, add in quadrature
    # if broaden==True: 
    #     W50 = np.sqrt(W50**2 + thermal_FWHM**2) # 10 u.km/u.s
    return W50

# def S21th_ALFALFA(W50, SNR):
#     S21_th = np.zeros(len(W50))
#     mask = W50 < 200
#     # S21_th[mask] =  0.15 * SNR[mask] * np.sqrt(W50[mask]/200) 
#     # S21_th[~mask] = 0.15 * SNR[~mask] * (W50[~mask]/200)
#     S21_th[mask] =  0.15 * SNR * np.sqrt(W50[mask]/200) 
#     S21_th[~mask] = 0.15 * SNR * (W50[~mask]/200)
#     return S21_th

def S21_th(W50, RMS, sigma, chan_kms=48): 
    S21_th = sigma * RMS * np.sqrt(W50 * chan_kms)
    # S21_th = np.zeros(len(W50))
    # mask = W50 < 48
    # S21_th[mask] = sigma * RMS * np.sqrt(W50[mask] * chan_kms)
    # S21_th[~mask] = sigma * RMS * (W50[~mask] * chan_kms)
    return S21_th

########################## ALFALFA Functions ###########################

def ALF_boundaries(ra_deg, dec_deg):
    ra_hour = Angle(ra_deg, unit=u.deg).hour

    # Handle RA wrap-around (22h → 24h and 0h → 3h)
    def in_ra_range(ra, ra_min, ra_max):
        if ra_min < ra_max:
            return (ra >= ra_min) & (ra <= ra_max)
        else:
            # Wraps around 0h
            return (ra >= ra_min) | (ra <= ra_max)

    # Apply Dec + RA cuts for each region (from Jones+2018 Table D1 and D2)
    mask = (
        # Spring
        ((dec_deg >= 0) & (dec_deg < 16)  & in_ra_range(ra_hour, 7.7, 16.5)) |
        ((dec_deg >= 16) & (dec_deg < 18)  & in_ra_range(ra_hour, 7.7, 16.0)) |
        ((dec_deg >= 18) & (dec_deg < 20) & in_ra_range(ra_hour, 8.7, 15.4)) |
        ((dec_deg >= 20) & (dec_deg < 24) & in_ra_range(ra_hour, 9.4, 15.4)) |
        ((dec_deg >= 24) & (dec_deg <= 30) & in_ra_range(ra_hour, 7.6, 16.5)) | 
        ((dec_deg >= 30) & (dec_deg <= 32) & in_ra_range(ra_hour, 8.5, 16.0)) | 
        ((dec_deg >= 32) & (dec_deg <= 36) & in_ra_range(ra_hour, 9.5, 15.5)) | 
        # Fall
        ((dec_deg >= 0) & (dec_deg < 2)  & in_ra_range(ra_hour, 22.0, 3.0)) |
        ((dec_deg >= 2) & (dec_deg < 6)  & in_ra_range(ra_hour, 22.5, 3.0)) |
        ((dec_deg >= 6) & (dec_deg < 10) & in_ra_range(ra_hour, 22.0, 3.0)) |
        ((dec_deg >= 10) & (dec_deg < 14) & in_ra_range(ra_hour, 22.0, 2.5)) |
        ((dec_deg >= 14) & (dec_deg <= 36) & in_ra_range(ra_hour, 22.0, 3.0))
    )

    return mask

def ALF_completeness(S21, W50, C=50):
    logS21 = np.log10(S21)
    logW = np.log10(W50)

    mask = np.zeros_like(S21, dtype=bool)
    low = logW < 2.5
    high = ~low

    if C==50:
        mask[low]  = logS21[low]  > (0.5 * logW[low]  - 1.170)
        mask[high] = logS21[high] > (1.0 * logW[high] - 2.420)

    if C==90:
        mask[low]  = logS21[low]  > (0.5 * logW[low]  - 1.14)
        mask[high] = logS21[high] > (1.0 * logW[high] - 2.39)
    return mask

def S21th_C90(W50, SNR):
    logW = np.log10(W50)
    logS21 = np.zeros(len(W50))
    low = logW < 2.5
    high = ~low
    logS21[low]  = (0.5 * logW[low]  - 1.170)
    logS21[high] = (1.0 * logW[high] - 2.39)
    S21_th = 10**(logS21)*SNR
    return S21_th

def S21th_ALFALFA(W50, SNR):
    S21_th = np.zeros(len(W50))
    mask = W50 < 200
    # S21_th[mask] =  0.15 * SNR[mask] * np.sqrt(W50[mask]/200) 
    # S21_th[~mask] = 0.15 * SNR[~mask] * (W50[~mask]/200)
    S21_th[mask] =  0.15 * SNR * np.sqrt(W50[mask]/200) 
    S21_th[~mask] = 0.15 * SNR * (W50[~mask]/200)
    return S21_th
    
def Vmax_ALF(alf_fl, solidang=(7000*u.deg**2).to_value(u.sr)):
    alf_cat = np.load(alf_fl)
    S21 = alf_cat[7] #Jy*km*s-1
    SNR = alf_cat[2]
    W50 = alf_cat[3]
    MHI = alf_cat[0]
    D = alf_cat[6]
    S21lim = S21th_ALFALFA(W50, SNR=6.5)
    #S21lim = S21th_C90(W50, SNR=6.5)
    Dmax = D*np.sqrt(S21/S21lim)
    Vmax = VolumeFromDist(Dmax, solidang, Dmin=0)
    return Vmax, MHI, W50

########################## Unit conversions ############################

def get_fobs(z, f_rest=1420.40575177):
    return f_rest / (1 + z)

def convert_f(v, z, f_rest=1420.40575177):
    '''Convert velocities in km/s to observed frequencies in MHz'''
    v_c = c.c.to(u.km/u.s).value 
    f_obs = get_fobs(z, f_rest=f_rest)
    freqs = f_obs*np.sqrt((1 - v/v_c) / (1 + v/v_c))
    return freqs

def df_dv(v, z, f_rest=1420.40575177):
    '''frequency/velocity Jacobian |df/dv| (absolute) for converting velocity widths and spectral integrations'''
    v_c = c.c.to(u.km/u.s).value 
    f_obs = get_fobs(z, f_rest=f_rest) # In MHz
    df_dv = f_obs / (v_c * (1 + v/v_c) * np.sqrt(1 - (v/v_c)**2))
    return df_dv # In units MHz / (km/s)

def width_vel2freq(del_Vrest, z=0, f_rest=1420.40575177):
    '''converts a velocity width to frequency width'''
    v_c = c.c.to(u.km/u.s).value 
    del_fobs = del_Vrest*f_rest/(v_c*(1+z)) # in units MHz
    return del_fobs*10**6 # in units Hz

def width_freq2vel(del_fobs, z=0, f_rest=1420.40575177):
    '''converts frequency width to velocity width'''
    v_c = c.c.to(u.km/u.s).value 
    del_Vrest = del_fobs*v_c*(1+z)/f_rest
    return del_Vrest # in units km/s

def convert_T(obs_freq, S, b_max=22*8.5):
    '''Convert spectral flux densitities in mJy to temperatures in K'''
    v_c = (c.c.to(u.km/u.s))
    f = (obs_freq*u.MHz).to(1/u.s)
    wavelength = (v_c/f).to(u.m)
    ang_res = (wavelength) / (b_max*u.m)
    T = (S*u.mJy * wavelength**2 / (2*c.k_B*ang_res**2)).to(u.K) 
    
    return T.value



