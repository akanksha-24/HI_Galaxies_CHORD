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

def sample_in_shell(V1, V2, N, solidang, dtype=np.float32):
    """Sample N comoving distances uniformly in volume between V1 and V2"""
    u = np.random.random(N).astype(dtype)
    V = V1 + u*(V2-V1)
    D = (3*V/solidang)**(1/3)
    return D, V

def Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, V1, V2, zinterp, solidang,  dtype=np.float32):
    MHI_sample = sample_HIMF(N, MHI).astype(dtype) # draw from HIMF
    VHI_sample = gf.VHI_polyFit(MHI_sample, dtype=dtype).astype(dtype) # estimate from abundance matching
    cos_i = np.random.random(N).astype(dtype)   # uniform in [0,1]
    i_sample = np.arccos(cos_i)   # inclination angles in [0, π/2] following a sin(i) probablity distribution function
    W50_sample = gf.estimate_W50(VHI_sample, i_sample, broaden=True, dtype=dtype)
    ra_sample, dec_sample = sample_radec(ra1, ra2, dec1, dec2, N, dtype=dtype)
    D_sample, Vol_sample = sample_in_shell(V1, V2, N, solidang, dtype=dtype)
    z_sample = zinterp(D_sample)
    #Vol_drawn = np.full(N, V2.value)
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


def Gen_Catalog(zmax, npt, dec1, dec2, ra1=0, ra2=360, MHImin=5, MHImax=12,
                    Dmax=None, footprint=None, Fluxlim=False, sigma=1,
                    noise=1e-4, vel_width=10, MHIres=10000,
                    draw=True, fltype='npy', flname='catalog.npy',
                    savelarge=True, dtype=np.float32):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start = time.time()

    if footprint is None:
        solidang = gf.solid_angle(dec1, dec2, ra1, ra2)
    else:
        solidang = footprint.to(u.sr)
    
    z_interp = gf.build_z_interp(zmax, npt=npt*10, dtype=dtype)
    if Dmax is not None:
        zmax = z_interp(Dmax)

    if rank == 0:
        print(f"max redshift is {zmax}")
    z, D, V, dV = gf.comoving_volume(zmax=zmax, npt=npt, solidang=solidang)

    # Split work among ranks
    all_indices = np.arange(len(dV))
    rank_indices = np.array_split(all_indices, size)[rank]

    # Set up HIMF grid
    MHI = np.logspace(MHImin, MHImax, MHIres)
    n = gf.galaxy_density(MHI)

    local_samples = []
    local_Narr = []

    for i in rank_indices:
        if Fluxlim:
            F_lim = sigma * noise
            MHImin_i = np.log10(gf.S_toMHI(F_lim, vel_width, D[i], unitless=True))
            MHImin_i = max(5, MHImin_i)
            MHI = np.logspace(MHImin_i, MHImax, MHIres)
            n = gf.galaxy_density(MHI)

        N = n * dV[i]
        N = 0 if np.isinf(N) else int(N)
        local_Narr.append(N)

        if draw:
            if i == 0:
                samples_ = Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, 0, V[i], z_interp, solidang, dtype=dtype)
            else:
                samples_ = Draw_Samples(N, MHI, ra1, ra2, dec1, dec2, V[i-1], V[i], z_interp, solidang, dtype=dtype)
            local_samples.append(samples_)

    if draw:
        rank_filename = f"{flname}_rank{rank}.npy"
        local_arrays = np.vstack(local_samples) if len(local_samples) > 0 else np.empty((0,9), dtype=dtype)
        np.save(rank_filename, local_arrays)

    all_Narr = comm.gather(local_Narr, root=0)

    end = time.time()
    if rank == 0:
        print(f"[Rank 0] Total runtime: {end - start:.2f} sec")

        N_arr = np.concatenate(all_Narr)
        return N_arr, z
    else:
        return None, None
    
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

    merged = np.concatenate(arrays, axis=0).astype(dtype, copy=False)

    outname = f"{flname}_merged.npy"
    np.save(outname, merged)
    
    if delete_rank_files:
        for f in files:
            os.remove(f)

    print(f"Merged {len(files)} rank files → {outname} (shape={merged.shape})")
    return merged


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

    samples = np.column_stack([10**lg_MHI, Vhelio, SNR, W50, RA, Dec, Dist, S21])
    np.save(flsave, samples)

if __name__ == "__main__":
    Gen_Catalog(zmax=0.5, Dmax=500, npt=1000, dec1=20, dec2=80, Fluxlim=False, flname='catalogs_output/VolLim_20to60deg_Dmax500')


    





        







