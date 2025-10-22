import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import numpy as np
from random import choices
from scipy.interpolate import UnivariateSpline
from scipy.signal import fftconvolve
import time
from mpi4py import MPI
import numba as nb
from math import erf
import Galaxy_Functions as gf
#import cupy as cp

# for parallelization 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  
size_mpi = comm.Get_size() 

def erf_fast(x):
    """Fast approximate erf(x), max error ~1e-4."""
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)

    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y

def Busy_general(x, a, b1, b2, xe, xp, c, w, dtype=np.float32):
    """
    GPU vectorized Busy function.
    Inputs:
      x: 1D array (n_x,)
      a: scalar
      b1,b2,xe,xp,c: 1D arrays (n_profiles,)
    Returns:
      B: (n_profiles, n_x) cupy array (float32)
    """
    # Broadcast shapes: (n_p, 1) * (1, n_x) -> (n_p, n_x)
    # err_p = erf(b1*(w + x - xe)) + 1.0
    err_p = erf_fast(b1[:, None] * (w + x[None, :] - xe[:, None])) + 1.0
    err_m = erf_fast(b2[:, None] * (w - x[None, :] + xe[:, None])) + 1.0
    pbola = c[:, None] * ((x[None, :] - xp[:, None])**2) + 1.0

    B = (a / 4.0) * err_p * err_m * pbola
    return B.astype(dtype)

def integrate_profile(X, Y):
    return np.trapz(Y, x=X, axis=1)

def get_MHI(V, S, D):
    int_S = integrate_profile(V, S)
    return 2.356e5 * (D**2) * int_S

def find_FWHM(x, y, level=0.5):
    spline = UnivariateSpline(x, y - (np.max(y)*level), s=0)
    roots = spline.roots()
    if len(roots) < 2:
        return np.nan, roots
    return roots[-1] - roots[0], roots

def convert_f(v, z, f_rest=1420.40575177):
    '''Convert velocities in km/s to observed frequencies in MHz'''
    v_c = c.c.to(u.km/u.s).value 
    f_obs = np.asarray(gf.get_fobs(z, f_rest=f_rest))[:,None] # handles array broadcasting
    freqs = f_obs*np.sqrt((1 - v/v_c) / (1 + v/v_c))
    return freqs

def assign_units(x, B, W50, D, z, MHI_desired, FWHM_thermal=10):
    '''assigns units of velocity (km/s) vs. Flux density (Jy)'''
    # Scale x-axis
    S_norm = B / B.max(axis=1, keepdims=True)
    FWHM = 2.1 # approx for speed

    # for very narrow spectra:
    mask = W50 < FWHM_thermal
    W50 = W50.copy()  # avoid modifying array
    if mask.any():
        W50[mask] = np.sqrt(FWHM_thermal**2 + W50[mask]**2)
    
    scale = W50 / FWHM
    V = scale[:, None] * x[None, :] 
    f = convert_f(V, z) # convert to frequency axis -> Note: df will not be constant

    # Scale y-axis
    MHI_initial = get_MHI(V, S_norm, D)
    y_scale = MHI_desired / MHI_initial
    S = S_norm * y_scale[:, None] 
    MHI_final = get_MHI(V, S, D)
    return V, f, S, MHI_final

def normalDist(x, sigma, x0=0):
    return np.exp(-(x - x0)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def gaussian_kernel(V, FWHM):
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    G = normalDist(V - np.mean(V), sigma)
    return G / G.sum()

def convolve_spectrum(S, V, FWHM):
    G = gaussian_kernel(V, FWHM)
    return fftconvolve(S, G, mode='same')

def define_xaxis(f50, freq_res=0.001, x0=-5, x1=5, dtype=np.float32):
    fwhm_approx = 2.2 # for w=1, FWHM in x is ~2.2
    N = f50*(x1-x0) / (freq_res*fwhm_approx) # to select the right amount of points for 
    x = np.linspace(x0, x1, int(N)).astype(np.float32) 
    return x

def Generate_Spectra(size, MHI, W50, D_C, z, a=1.0, w=1.0,
                     b1=None, b2=None, c=None, xe=None, xp=None,
                     dtype=np.float32, freq_res=0.001):
    """
    Generates spectra using the Numba-parallel Busy function.
    - chunk_size controls memory usage (avoid allocating > few GB)
    """

    print(f"[Rank {rank}] Generating Spectra (size={size})...")
    start = time.time()

    #f50 = W50 * gf.df_dv(W50, z=z)
    N = 100000
    x = np.linspace(-5,5,N)
    
    # Get luminosity distance D_L from co-mocing distance D_C:
    D_L = (1+z)*D_C

    if b1 is None: b1 = np.random.uniform(1, 3, size=size).astype(dtype)
    if b2 is None: b2 = np.random.uniform(1, 3, size=size).astype(dtype)
    if c  is None: c  = np.random.uniform(0, 1, size=size).astype(dtype)
    if xe is None: xe = np.random.uniform(-0.1, 0.1, size=size).astype(dtype)
    if xp is None: xp = np.random.uniform(-0.1, 0.1, size=size).astype(dtype)

    B = Busy_general(x, a, b1, b2, xe, xp, c, w=1)
    Vel, freq, S_flux, finalM = assign_units(x, B, W50, D_L, z, MHI)

    end = time.time()
    print(f"[Rank {rank}] Runtime: {end - start:.2f} sec for {size} spectra")

    return Vel, S_flux, freq, finalM




# def Generate_Spectra_chunked(size, MHI, W50, D_C, z, a=1.0, w=1.0,
#                      b1=None, b2=None, c=None, xe=None, xp=None,
#                      chunk_size=100000, dtype=np.float32, freq_res=0.001):
#     """
#     Generates spectra using the Numba-parallel Busy function.
#     - chunk_size controls memory usage (avoid allocating > few GB)
#     """

#     print(f"[Rank {rank}] Generating Spectra (size={size})...")
#     start = time.time()

#     #f50 = W50 * gf.df_dv(W50, z=z)
#     N = 100000
#     x = np.linspace(-5,5,N)
    
#     # Get luminosity distance D_L from co-mocing distance D_C:
#     D_L = (1+z)*D_C

#     if b1 is None: b1 = np.random.uniform(1, 3, size=size)
#     if b2 is None: b2 = np.random.uniform(1, 3, size=size)
#     if c  is None: c  = np.random.uniform(0, 1, size=size)
#     if xe is None: xe = np.random.uniform(-0.1, 0.1, size=size)
#     if xp is None: xp = np.random.uniform(-0.1, 0.1, size=size)

#     # Process in manageable chunks
#     for start_idx in range(0, size, chunk_size):
#         end_idx = min(start_idx + chunk_size, size)
#         n_chunk = end_idx - start_idx

#         b1_chunk = b1[start_idx:end_idx]
#         b2_chunk = b2[start_idx:end_idx]
#         c_chunk  = c[start_idx:end_idx]
#         xe_chunk = xe[start_idx:end_idx]
#         xp_chunk = xp[start_idx:end_idx]

#         # ⚡ Call Numba-parallel Busy function
#         B_chunk = Busy_general(x, a, b1_chunk, b2_chunk, xe_chunk, xp_chunk, c_chunk, w=1)
#         Vchunk, fchunk, Schunk = assign_units(x, B_chunk, W50[start_idx], D_L[start_idx], z[start_idx], MHI[start_idx]

#         # for i in range(n_chunk):
#         #     Vel[start_idx+i], freq[start_idx+i], S = assign_units(x, B_chunk[i], W50[start_idx+i], D_L[start_idx+i], z[start_idx+i], MHI[start_idx+i])
#         #     if Vel[start_idx+i] is None:
#         #         final_M[start_idx+i] = np.nan
#         #         continue

#             #FWHM_thermal = 10.0  # km/s
#             #S_flux[start_idx+i] = convolve_spectrum(S, Vel[start_idx+i], FWHM_thermal)
#             #final_M[start_idx+i] = get_MHI(Vel[start_idx+i], S_flux[start_idx+i], D_L[start_idx+i])

#     end = time.time()
#     print(f"[Rank {rank}] Runtime: {end - start:.2f} sec for {size} spectra")

#     return final_M, Vel, S_flux, freq

@nb.njit(parallel=True, fastmath=True)
def interpolate_faxis(f_arr, Sarr, f_full, fres=0.001):
    Nspec = f_arr.shape[0]
    S_full = np.zeros((Nspec, len(f_full)))

    for i in nb.prange(Nspec):
        f_cutout = f_arr[i, ::-1]
        Sflux = Sarr[i, ::-1]
        freq_new = np.arange(f_cutout[0], f_cutout[-1] + fres/2, fres)

        # Numba supports np.interp
        S_new = np.interp(freq_new, f_cutout, Sflux)

        start_idx = np.searchsorted(f_full, freq_new[0])
        end_idx   = start_idx + len(freq_new)
        if end_idx > len(f_full):
            end_idx = len(f_full)
        n_insert = end_idx - start_idx
        S_full[i, start_idx:end_idx] = S_new[:n_insert]

    return S_full

def Run_Spectra(catalog_fl, zmax, size=None, plot=False, gpu=False, fmax=1421, fres=0.001, dtype=np.float32, interpolate=True):
    catalog = np.load(catalog_fl)
    if size==None:
        size = catalog.shape[0]
        print("catalog size ", catalog.shape[0])

    # Split work across MPI ranks
    chunk_size = size // size_mpi
    start_i = rank * chunk_size
    end_i = size if rank == size_mpi-1 else (rank + 1) * chunk_size

    local_cat = catalog[start_i:end_i]
    MHI = local_cat[:,0]
    W50 = local_cat[:,3]
    D = local_cat[:,6]
    z = local_cat[:,8]

    print(f"Generatiing Spectra [Rank {rank}]")
    t1 = time.time()

    Vel, S_flux, freq, final_M = Generate_Spectra(local_cat.shape[0], MHI, W50, D, z, dtype=dtype)
    np.save(f"spectra_rank{rank}.npy", np.asarray([Vel, freq, S_flux], dtype=dtype))
    np.save(f"intMHI_rank{rank}.npy", np.asarray([final_M], dtype=dtype))
    t2 = time.time()
    print(f"[Rank {rank}] Done and saved spectra — took {t2 - t1:.2f} seconds.")
    if plot:
        print("Plotting...")
        import Plotting as plot
        plot.check_Spectra(MHI, final_M, W50, Vel, S_flux, freq, z, D, "Plots/Test_spectra_Erfapprox.pdf")

    if interpolate:
        t3 = time.time()
        print(f"[Rank {rank}] Starting Interpolation...")
        fmin = gf.get_fobs(zmax)
        f_full = np.arange(fmin, fmax + fres/2, fres)
        print(len(f_full))
        S_full = interpolate_faxis(freq, S_flux, f_full, fres)
        np.save(f"Sfull_rank{rank}.npy", S_full)
        t4 = time.time()
        print(f"[Rank {rank}] Done and saved full arra — took {t4 - t3:.2f} seconds.")


if __name__ == "__main__":
    #Run_Spectra(catalog_fl='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', size=20, zmax=0.04, plot=True, interpolate=False)
    Run_Spectra(catalog_fl='../catalogs_output/VolLim_20to60deg_Dmax500.npy', zmax=0.117, plot=False)

