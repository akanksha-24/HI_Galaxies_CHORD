import Galaxy_Functions as gf
import astropy.units as u
import astropy.constants as c
import numpy as np
import time
import pandas as pd
import Plotting as plot

def sample_HIMF(N, M_HI, phi_s=gf.phi_s, M_s=gf.M_s, alpha=gf.alpha): 
    # Compute Schechter PDF in log-space
    pdf = gf.schechter_fit_lg(M_HI, phi_s, M_s, alpha)
    
    # Normalize PDF
    dlogM = np.diff(np.log10(M_HI))
    # trap method: average adjacent PDF values
    pdf_mid = gf.mid_bin(pdf)
    pdf_norm = pdf_mid / np.sum(pdf_mid * dlogM)  # normalized over log-space

    # Build cumulative distribution function (CDF) -> this method is very fast
    cdf = np.cumsum(pdf_norm * dlogM)
    cdf /= cdf[-1]  #normalize to 1

    # Draw uniform random numbers
    u = np.random.rand(N)

    # Map uniform randoms to mass using inverse CDF
    idx = np.searchsorted(cdf, u)
    samples = M_HI[idx]

    return samples

def sample_radec(ra1, ra2, dec1, dec2, N):
    """
    Sample N random RA/Dec points uniformly in a rectangular patch on the sphere.
    Input: ra1, ra2, dec1, dec2 in degrees
    Output: ra, dec arrays in degrees
    """
    ra1, ra2 = np.deg2rad(ra1), np.deg2rad(ra2)
    dec1, dec2 = np.deg2rad(dec1), np.deg2rad(dec2)
 
    ra = np.random.uniform(ra1, ra2, N) # Uniform RA
    sin_dec1, sin_dec2 = np.sin(dec1), np.sin(dec2)
    sin_dec = np.random.uniform(sin_dec1, sin_dec2, N) # Uniform in sin(dec)
    dec = np.arcsin(sin_dec)

    return np.rad2deg(ra), np.rad2deg(dec)

def sample_in_shell(V1, V2, N, solidang):
    """Sample N comoving distances uniformly in volume between V1 and V2"""
    u = np.random.random(N)
    V = V1 + u*(V2-V1)
    D = (3*V/solidang)**(1/3)
    return D.value, V.value

def Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, V1, V2, zinterp, solidang):
    MHI_sample = sample_HIMF(N, MHI) # draw from HIMF
    VHI_sample = gf.VHI_polyFit(MHI_sample) # estimate from abundance matching
    cos_i = np.random.random(N)   # uniform in [0,1]
    i_sample = np.arccos(cos_i)   # inclination angles in [0, π/2] following a sin(i) probablity distribution function
    W50_sample = gf.estimate_W50(VHI_sample, i_sample, broaden=True)
    #ra_sample, dec_sample = sample_radec(ra1, ra2, dec1, dec2, N)
    D_sample, Vol_sample = sample_in_shell(V1, V2, N, solidang)
    #z_sample = zinterp(D_sample)
    #Vol_drawn = np.full(N, V2.value)
    #samples = np.column_stack([MHI_sample, VHI_sample, i_sample, W50_sample, ra_sample, dec_sample, D_sample, Vol_sample, z_sample, Vol_drawn])
    samples = np.column_stack([MHI_sample, W50_sample, D_sample])
    return samples

def Save_Catalog_npz(samples, zmax, solidang, flname, z_arr, N_arr):
    np.savez(flname, zmax=zmax, solidang=solidang, z_arr=z_arr, N_arr=N_arr,
                        MHI_lg=samples[:,0],
                        VHI=samples[:,1], 
                        Incl=samples[:,2], 
                        W50=samples[:,3],
                        RA=samples[:,4],
                        Dec=samples[:,5],
                        Distace=samples[:,6],
                        Volume=samples[:,7],
                        Redshift=samples[:,8])
    
def Save_large(samples, flname):
    extn = ['MHI_lg', 'VHI', 'Incl', 'W50', 'RA', 'Dec', 'Distace', 'Volume', 'Redshift']
    for i in samples.shape[1]:
        np.save(flname+'_'+extn+'.npy',samples[:,i])


def Gen_Catalog(zmax, npt, dec1, dec2, ra1=0, ra2=360, MHImin=5, MHImax=12, Dmax=None, footprint=None,
                Fluxlim=False, sigma=1, noise=0.01*u.mJy, vel_width=10*u.km/u.s, MHIres=10000, draw=True, fltype='npy', flname='catalog.npy', savelarge=True):
    
    start = time.time()
    if footprint==None:
        solidang = gf.solid_angle(dec1, dec2, ra1, ra2)
    else:
        solidang = footprint.to(u.sr)
    
    # Split into dV shells
    z_interp = gf.build_z_interp(zmax, npt=npt*10) # interpolation inversion to make it fast to get redshift from distance
    if Dmax!=None:
        zmax = z_interp(Dmax) # use for nearby universe i.e D=200 Mpc
    print("max redshift is ", zmax)
    z, D, V, dV = gf.comoving_volume(zmax=zmax, npt=npt, solidang=solidang)
    
    # Set MHI and density
    MHI = np.logspace(MHImin, MHImax, MHIres)
    n = gf.galaxy_density(MHI)
    N_arr = [] ; samples_list = [] ; 

    # iterate through dV shells
    for i in np.arange(len(dV)):
        if Fluxlim: # if Flux-limited, re-calculate n for each dV. If volume-limited, skip this.
            F_lim = sigma*noise
            MHImin = np.log10(gf.S_toMHI(F_lim, vel_width, D[i]).value)
            MHImin = max(5, MHImin) # apply lowerlimit of MHI=10**5
            MHI = np.logspace(MHImin, MHImax, MHIres)
            n = gf.galaxy_density(MHI)

        N = n*dV[i].value  # number of samples to draw
        N = 0 if np.isinf(N) else int(N)
        N_arr.append(N)
        
        if draw:
            if i==0:
                samples_ = Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, 0, V[i], z_interp, solidang)
            else:
                samples_ = Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, V[i-1], V[i], z_interp, solidang)
            samples_list.append(samples_)
    end = time.time()
    print(f"Runtime:    {end - start:.3f} seconds")
    if draw:
        samples = np.vstack(samples_list)
        if savelarge==True:
            Save_large(samples, flname)
        # if fltype=='npz':
        #     Save_Catalog_npz(samples, zmax, solidang, flname, z, N_arr)
        else:
           np.save(flname, samples)
    return N_arr, z


def LoadALFALFA(alftable, flsave):
    alf = pd.read_table(alftable, delimiter=',')
    # Select relevant source:
    alf = alf[(alf['HIcode'])==1]
    alf = alf[(alf['Vhelio'])<15000]
    alf = alf[(alf['Vhelio'])>0]

    lg_MHI = (alf['logMH']) # log(solMass)
    W50 = (alf['W50']).astype(float) # u.km/u.s
    S21 = (alf['HIflux']).astype(float) # u.Jy * u.km / u.s
    SNR = (alf['SNR']).astype(float)
    Dist = (alf['Dist']).astype(float) #* u.Mpc
    Vhelio = (alf['Vhelio']).astype(float)
    RA = (alf['RAdeg_HI']).astype(float)
    Dec = (alf['DECdeg_HI']).astype(float)





        







