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
from mpi4py import MPI
import numba as nb
from math import erf
import Galaxy_Functions as gf
from scipy.interpolate import PchipInterpolator
#import cupy as cp

# for parallelization 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  
size_mpi = comm.Get_size() 

# def Busy_general_batch_gpu(x, a, b1, b2, xe, xp, c, w, dtype=cp.float32):
#     """
#     Fully vectorized Busy function on GPU using CuPy.
#     Arrays must be CuPy arrays (on GPU).
#     Returns shape (n_profiles, len(x)).
#     """
#     x = x.astype(dtype)
#     a, b1, b2, xe, xp, c = [arr.astype(dtype) for arr in (a, b1, b2, xe, xp, c)]
#     w = dtype(w)

#     # Broadcast to (n_profiles, n_x)
#     X = x[cp.newaxis, :]
#     B1, B2, XE, XP, C = [arr[:, cp.newaxis] for arr in (b1, b2, xe, xp, c)]

#     err_p = erf(B1 * (w + X - XE)) + 1.0
#     err_m = erf(B2 * (w - X + XE)) + 1.0
#     pbola = C * (X - XP) ** 2 + 1.0

#     out = (a[:, cp.newaxis] / 4.0) * err_p * err_m * pbola
#     return out

@nb.njit(parallel=True, fastmath=True)
def Busy_general_batch(x, a, b1, b2, xe, xp, c, w, dtype=np.float32):
    """
    Vectorized Busy Function for all spectra.
    Returns array of shape (n_profiles, len(x)).
    """
    n_profiles = b1.size
    n_x = x.size
    out = np.empty((n_profiles, n_x), dtype=dtype)

    for i in nb.prange(n_profiles):  # parallel loop
        for j in range(n_x):
            err_p = erf(b1[i]*(w + x[j] - xe[i])) + 1.0
            err_m = erf(b2[i]*(w - x[j] + xe[i])) + 1.0
            pbola = c[i]*((x[j] - xp[i])**2) + 1.0
            out[i, j] = (a / 4.0) * err_p * err_m * pbola

    return out

def integrate_profile(X, Y):
    return np.trapz(Y, x=X)

def get_MHI(V, S, D):
    int_S = integrate_profile(V, S)
    return 2.356e5 * (D**2) * int_S

def get_MHI_Hz(f, S, D):
    dfdv = gf.df_dv()
    int_S = integrate_profile(f, S)
    return 2.356e5 * (D**2) * int_S

def find_FWHM(x, y, level=0.5):
    spline = UnivariateSpline(x, y - (np.max(y)*level), s=0)
    roots = spline.roots()
    if len(roots) < 2:
        return np.nan, roots
    return roots[-1] - roots[0], roots

def assign_units(x, B, W50, D, z, MHI_desired, FWHM_thermal=10):
    '''assigns units of velocity (km/s) vs. Flux density (Jy)'''
    # Scale x-axis
    S_norm = (B / np.max(B))
    FWHM, roots = find_FWHM(x, B)
    # if root finding fails
    if np.isnan(FWHM) or FWHM == 0: 
        return None, None
    # for very narrow spectra:
    if W50 <= FWHM_thermal:
        scale = FWHM_thermal / FWHM
    else:
        scale = W50 / FWHM
    V = x * scale
    f = gf.convert_f(V, z) # convert to frequency axis -> Note: df will not be constant

    # Scale y-axis
    MHI_initial = get_MHI(V, S_norm, D)
    y_scale = MHI_desired / MHI_initial
    S = S_norm * y_scale
    return V, f, S

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
                     chunk_size=100000, dtype=np.float32, freq_res=0.001):
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

    if b1 is None: b1 = np.random.uniform(1, 3, size=size)
    if b2 is None: b2 = np.random.uniform(1, 3, size=size)
    if c  is None: c  = np.random.uniform(0, 1, size=size)
    if xe is None: xe = np.random.uniform(-0.1, 0.1, size=size)
    if xp is None: xp = np.random.uniform(-0.1, 0.1, size=size)

    final_M = np.empty(size, dtype=dtype)
    Vel = np.empty((size, N), dtype=dtype)
    S_flux = np.empty((size, N), dtype=dtype)
    freq = np.empty((size, N), dtype=dtype)

    # Process in manageable chunks
    for start_idx in range(0, size, chunk_size):
        end_idx = min(start_idx + chunk_size, size)
        n_chunk = end_idx - start_idx

        b1_chunk = b1[start_idx:end_idx]
        b2_chunk = b2[start_idx:end_idx]
        c_chunk  = c[start_idx:end_idx]
        xe_chunk = xe[start_idx:end_idx]
        xp_chunk = xp[start_idx:end_idx]

        # ⚡ Call Numba-parallel Busy function
        B_chunk = Busy_general_batch(x, a, b1_chunk, b2_chunk, xe_chunk, xp_chunk, c_chunk, w=1)

        for i in range(n_chunk):
            Vel[start_idx+i], freq[start_idx+i], S = assign_units(x, B_chunk[i], W50[start_idx+i], D_L[start_idx+i], z[start_idx+i], MHI[start_idx+i])
            if Vel[start_idx+i] is None:
                final_M[start_idx+i] = np.nan
                continue

            FWHM_thermal = 10.0  # km/s
            S_flux[start_idx+i] = convolve_spectrum(S, Vel[start_idx+i], FWHM_thermal)
            final_M[start_idx+i] = get_MHI(Vel[start_idx+i], S_flux[start_idx+i], D_L[start_idx+i])

    end = time.time()
    print(f"[Rank {rank}] Runtime: {end - start:.2f} sec for {size} spectra")

    return final_M, Vel, S_flux, freq

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
        size = catalog[0].shape[0]

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

    final_M, Vel, S_flux, freq = Generate_Spectra(size, MHI, W50, D, z, dtype=dtype)
    np.save(f"spectra_rank{rank}.npy", np.asarray([Vel, freq, S_flux], dtype=dtype))
    np.save(f"intMHI_rank{rank}.npy", np.asarray([final_M], dtype=dtype))
    t2 = time.time()
    print(f"[Rank {rank}] Done and saved spectra — took {t2 - t1:.2f} seconds.")
    if plot:
        print("Plotting...")
        import Plotting as plot
        plot.check_Spectra(MHI, final_M, W50, Vel, S_flux, freq, z, D, "Plots/Test_spectra.pdf")

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
    Run_Spectra(catalog_fl='/scratch/akanksha/catalogs/VolLim_20to60deg_z0p5_merged.npy', zmax=0.5, plot=False)

