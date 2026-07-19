import Galaxy_Functions as gf
import astropy.units as u
import astropy.constants as c
import numpy as np
import time
import pandas as pd
import Plotting as plot
from mpi4py import MPI
import glob
import os
import re
import pandas as pd
import Forecasting as fore
from CHORD_Sensitivity import *

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
    idx[idx >= len(M_HI)] = len(M_HI)-1 
    samples = M_HI[idx]

    return samples

def sample_radec(ra1, ra2, dec1, dec2, N, dtype=np.float32):
    """
    Sample N random RA/Dec points uniformly in a rectangular patch on the sphere.
    Input: ra1, ra2, dec1, dec2 in degrees
    Output: ra, dec arrays in degrees
    """
    ra1, ra2 = np.deg2rad(ra1), np.deg2rad(ra2)
    dec1, dec2 = np.deg2rad(dec1), np.deg2rad(dec2)
 

    ra = np.random.uniform(ra1, ra2, N).astype(dtype) # Uniform RA
    sin_dec1, sin_dec2 = np.sin(dec1), np.sin(dec2)
    sin_dec = np.random.uniform(sin_dec1, sin_dec2, N).astype(dtype) # Uniform in sin(dec)
    dec = np.arcsin(sin_dec)

    return np.rad2deg(ra), np.rad2deg(dec)

def sample_in_shell(D1, D2, N, solidang=4*np.pi, dtype=np.float32):
    """
    Sample N comoving distances uniformly in volume between radii D1 and D2.
    Returns distances D and corresponding partial volumes V.
    """
    u = np.random.random(N).astype(dtype)
    # Uniform in D^3 ensures uniform in volume
    D = ((D1**3) + u * (D2**3 - D1**3))**(1/3)
    V = (solidang / 3.0) * (D**3)
    return D, V

def Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, D1, D2, zinterp, solidang,  
                 dtype=np.float32, SNRint=False, RMS=0.2, sigma=6,
                 phi_s=gf.phi_s, M_s=gf.M_s, alpha=gf.alpha):
    MHI_sample = sample_HIMF(N, MHI, phi_s=phi_s, M_s=M_s, alpha=alpha).astype(dtype) # draw from HIMF
    VHI_sample = gf.VHI_polyFit(MHI_sample, dtype=dtype).astype(dtype) # estimate from abundance matching
    cos_i = np.random.random(N).astype(dtype)   # uniform in [0,1]
    i_sample = np.arccos(cos_i)   # inclination angles in [0, π/2] following a sin(i) probablity distribution function
    W50_sample = gf.estimate_W50(VHI_sample, i_sample, broaden=True, dtype=dtype)
    ra_sample, dec_sample = sample_radec(ra1, ra2, dec1, dec2, N, dtype=dtype)
    D_sample, Vol_sample = sample_in_shell(D1, D2, N, solidang, dtype=dtype)
    z_sample = zinterp(D_sample)
    #print(np.min(z_sample))
    #Vol_drawn = np.full(N, V2.value)
    if SNRint:
        mask, _ = fore.SNRint_detections(MHI=MHI_sample, W50=W50_sample, z=z_sample, RMS=RMS, sigma=sigma)
        samples = np.column_stack([MHI_sample[mask], VHI_sample[mask], i_sample[mask], W50_sample[mask], 
                                   ra_sample[mask], dec_sample[mask], D_sample[mask], Vol_sample[mask], z_sample[mask]])
    else:
        samples = np.column_stack([MHI_sample, VHI_sample, i_sample, W50_sample, ra_sample, dec_sample, D_sample, Vol_sample, z_sample])
    return samples

def Save_Catalog_npz(samples, zmax, solidang, flname, z_arr, N_arr):
    np.savez(flname, zmax=zmax, solidang=solidang, z_arr=z_arr, N_arr=N_arr,
                        MHI=samples[:,0],
                        VHI=samples[:,1], 
                        Incl=samples[:,2], 
                        W50=samples[:,3],
                        RA=samples[:,4],
                        Dec=samples[:,5],
                        Distance=samples[:,6],
                        Volume=samples[:,7],
                        Redshift=samples[:,8])
    
# def Save_large(samples, flname):
#     extn = ['MHI', 'VHI', 'Incl', 'W50', 'RA', 'Dec', 'Distace', 'Volume', 'Redshift']
#     for i in samples.shape[1]:
#         np.save(flname+'_'+extn+'.npy',samples[:,i])


def Gen_Catalog(zmax, dec1, dec2, zmin=0, ra1=0, ra2=360, MHImin=5, MHImax=12,
                    footprint=None, Fluxlim=False, sigma=1,
                    noise=1e-4, vel_width=10, MHIres=10000,
                    draw=True, fltype='npy', flname='catalog.npy',
                    savelarge=True, dtype=np.float32, SNRint=False, sigma_int=6, RMS=0.2, save=True,
                    phi_s=gf.phi_s, M_s=gf.M_s, alpha=gf.alpha, obs_year=5):
    
    print("starting Job...")
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start = time.time()

    z_step = 1e-4
    npt = zmax/z_step
    z_interp = gf.build_z_interp(zmin, zmax, zstep=z_step, dtype=dtype)

    if footprint is None:
        solidang = gf.solid_angle(dec1, dec2, ra1, ra2)
    else:
        solidang = footprint.to(u.sr)
    
    if rank == 0:
        print(f"max redshift is {zmax}")
    z, D, V, dV = gf.comoving_volume(zmin=zmin, zmax=zmax, zstep=z_step, solidang=solidang)
    print("Max distance is ", np.max(D))

    #Split work among ranks
    all_indices = np.arange(len(dV))
    rank_indices = np.array_split(all_indices, size)[rank]

    # Set up HIMF grid
    MHI = np.logspace(MHImin, MHImax, MHIres)
    n = gf.galaxy_density(MHI, phi_s=phi_s, M_s=M_s, alpha=alpha)
    print("the number density is ", n)

    local_samples = []
    local_Narr = []

    for i in rank_indices:
        if Fluxlim:
            #F_lim = sigma * noise
            #MHImin_i = np.log10(gf.S_toMHI(F_lim, vel_width, D[i], unitless=True))
            _, RMS_, _ = build_survey(obs_years=obs_year, z=z[i], start=dec1, end=dec2)
            MHI_lim = gf.estimate_MHImax(z=z[i], sigma=6, RMS_chan=RMS_,  DeltaV=20, chan_width=gf.width_vel2freq(5))
            MHImin_i = np.log10(MHI_lim) #- 0.5
            MHImin_i = max(5, MHImin_i)
            #print("MHImin is ", MHImin_i)
            MHI = np.logspace(MHImin_i, MHImax, MHIres)
            n = gf.galaxy_density(MHI, phi_s=phi_s, M_s=M_s, alpha=alpha)

        N = n * dV[i]
        N = 0 if np.isinf(N) else int(N)
        local_Narr.append(N)

        if draw:
            if i == 0:
                if zmin==0:
                    samples_ = Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, 0, D[i], z_interp, 
                                            solidang, dtype=dtype, SNRint=SNRint, sigma=sigma_int, RMS=RMS,
                                            phi_s=phi_s, M_s=M_s, alpha=alpha)
                else:
                    continue
            else:
                samples_ = Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, D[i-1], D[i], z_interp, 
                                        solidang, dtype=dtype, SNRint=SNRint, sigma=sigma_int, RMS=RMS,
                                        phi_s=phi_s, M_s=M_s, alpha=alpha)
            local_samples.append(samples_)
    print("Saving samples...")
    if draw:
        rank_filename = f"{flname}_rank{rank}.npy"
        local_arrays = (np.vstack(local_samples) if len(local_samples) > 0 else np.empty((0,9), dtype=dtype)).T
        #print("minimum z drawn is ", np.min(local_arrays.T[8]))
        if save:
            np.save(rank_filename, local_arrays)

    all_Narr = comm.gather(local_Narr, root=0)

    end = time.time()
    if rank == 0:
        print(f"[Rank 0] Total runtime: {end - start:.2f} sec")

        N_arr = np.concatenate(all_Narr)
        if draw:
            return local_arrays
    
def merge_rankfiles(flname, delete_rank_files=True, dtype=np.float32):
    files = glob.glob(f"{flname}_*")
    if not files:
        raise FileNotFoundError(f"No files found matching {flname}_*")
    
    def extract_rank(filename):
        # Extracts the integer after the last underscore in the filename
        match = re.search(r"_(\d+)\.npy$", os.path.basename(filename))
        return int(match.group(1)) if match else -1

    files.sort(key=extract_rank)

    arrays = []
    for f in files:
        arr = np.load(f, mmap_mode='r')
        arrays.append(arr)

    merged = np.concatenate(arrays, axis=1).astype(dtype, copy=False)

    outname = f"{flname}_merged.npy"
    np.save(outname, merged)
    
    if delete_rank_files:
        for f in files:
            os.remove(f)

    print(f"Merged {len(files)} rank files → {outname} (shape={merged.shape})")
    return merged

def LoadMockALFALFA(datafile, outfile, changeVelocity=False, Dist_range=None, RA_range=None, Dec_range=None):
    fulldata = pd.read_pickle(datafile)
    D = fulldata['d_mw']
    ra = fulldata['ra_deg']
    dec = fulldata['dec_deg']

    # select a portion of the catalog:
    mask = np.ones(len(fulldata), dtype=bool)
    if Dist_range is not None:
        mask &= (D >= Dist_range[0]) & (D <= Dist_range[1])
    if RA_range is not None:
        mask &= (ra >= RA_range[0]) & (ra <= RA_range[1])
    if Dec_range is not None:
        mask &= (dec >= Dec_range[0]) & (dec <= Dec_range[1])
    simdata = fulldata[mask].copy()

    M_HI = simdata['mcold_atom']
    i = simdata['random_incl_rad']
    if changeVelocity:
        Vrot = gf.VHI_polyFit(M_HI)
        W_50 = gf.estimate_W50(Vrot=Vrot, i=i, broaden=False)
    else:
        W_50 = simdata['w50']
        Vrot = W_50 / (2*np.sin(i))
    D = simdata['d_mw']
    Vol = gf.VolumeFromDist(D) # full sky volume which can be multiplied by footprint, when selected
    z = gf.Hubble_redshift(D)
    ra = simdata['ra_deg']
    dec = simdata['dec_deg']

    catalog = np.asarray([M_HI, Vrot, i, W_50, ra, dec,  D, Vol,  z])
    np.save(outfile, catalog)


def LoadALFALFA(alftable='/Users/akankshabij/Documents/PhD/Data/ALFALFA/a100.code12.table2.190808.csv',
                 flsave='catalogs_output/ALFALFA_a100_Dmax200_ALFboundaries_Alfthreshold.npy', C=90):
    alf = pd.read_table(alftable, delimiter=',')
    # Select relevant source:
    alf = alf[(alf['HIcode'])==1]
    alf = alf[(alf['Vhelio'])<=15000]
    alf = alf[(alf['Vhelio'])>0]
    alf = alf[np.log10(alf['W50'])>=1.2]
    alf = alf[alf['SNR']>=6.5]
    alf = alf[gf.ALF_boundaries(ra_deg=alf['RAdeg_HI'], dec_deg=alf['DECdeg_HI'])]
    alf = alf[(alf['Dist'])<=200]
    #alf = alf[gf.ALF_completeness(S21=alf['HIflux'], W50=alf['W50'], C=C)]
    S21_thalf = gf.S21th_ALFALFA(alf['W50'], SNR=6.5)
    alf = alf[alf['SNR'] > S21_thalf]
    print(len(alf))

    lg_MHI = (alf['logMH']) # log(solMass)
    W50 = (alf['W50']).astype(float) # u.km/u.s
    S21 = (alf['HIflux']).astype(float) # u.Jy * u.km / u.s
    SNR = (alf['SNR']).astype(float)
    Dist = (alf['Dist']).astype(float) #* u.Mpc
    Vhelio = (alf['Vhelio']).astype(float)
    RA = (alf['RAdeg_HI']).astype(float)
    Dec = (alf['DECdeg_HI']).astype(float)
    zinterp = gf.build_z_interp(0,0.1,10000)
    z = zinterp(Dist)

    samples = np.asarray([10**lg_MHI, Vhelio, SNR, W50, RA, Dec, Dist, S21, z])
    np.save(flsave, samples)

def load_catalogParams(catalog_file):
    cat = np.load(catalog_file)
    MHI = cat[0]
    Vrot = cat[1]
    i = cat[2] 
    W_50 = cat[3] 
    ra = cat[4] 
    dec = cat[5]  
    D = cat[6]
    Vol = cat[7]  
    z = cat[8]
    return MHI, Vrot, i, W_50, ra, dec, D, Vol, z 

if __name__ == "__main__":
    Gen_Catalog(zmax=1, dec1=20, dec2=80, Fluxlim=True, obs_year=5,
                flname='catalogs_output/FluxLim_20to80deg_zmax1_obsyear5.npy', dtype=np.float64)
    Gen_Catalog(zmax=1, dec1=20, dec2=50, Fluxlim=True, obs_year=1,
                flname='catalogs_output/FluxLim_20to50deg_zmax1_obsyear1.npy', dtype=np.float64)
    # Gen_Catalog(zmax=0.0475, dec1=20, dec2=80, Fluxlim=False, obs_year=5,
    #             flname='catalogs_output/VolLim_20to80deg_Dmax200_MHIlim.npy', dtype=np.float64)
    #LoadALFALFA()
    #Gen_Catalog(zmax=0.8, npt=100000, dec1=20, dec2=80, Fluxlim=False, flname='catalogs_output/VolLim_20to80deg_zmax0p8', dtype=np.float64)
    #Gen_Catalog(zmin=0.4, zmax=1, npt=10000, dec1=20, dec2=80, Fluxlim=False, MHImin=9, MHImax=12,
    #            flname='catalogs_output/VolLim_20to60deg_zmin0p4_zmax1_MHI9to12.npy')
    # LoadMockALFALFA('../ALFALFA_Mock_Brooks/mock_whole_sky_df', changeVelocity=True, 
    #                 Dist_range=[0,200], Dec_range=[20,80], 
    #                 outfile='catalogs_output/MockAlf_D200_Dec20to80_changeVelocity.npy')
    #merge_rankfiles(flname='catalogs_output/VolLim_20to60deg_zmin0p4_zmax1', delete_rank_files=False)

    





        







