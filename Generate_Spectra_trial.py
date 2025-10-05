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

# for parallelization 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  
size_mpi = comm.Get_size()  

@nb.njit(parallel=True, fastmath=True)
def Busy_general_batch(x, a, b1, b2, xe, xp, c, w):
    """
    Vectorized Busy Function for all spectra.
    Returns array of shape (n_profiles, len(x)).
    """
    n_profiles = b1.size
    n_x = x.size
    out = np.empty((n_profiles, n_x), dtype=np.float32)

    for i in nb.prange(n_profiles):  # parallel loop
        for j in range(n_x):
            err_p = erf(b1[i]*(w + x[j] - xe[i])) + 1.0
            err_m = erf(b2[i]*(w - x[j] + xe[i])) + 1.0
            pbola = c[i]*((x[j] - xp[i])**2) + 1.0
            out[i, j] = (a / 4.0) * err_p * err_m * pbola

    return out

def integrate_profile(V, S):
    return np.trapz(S, x=V)

def get_MHI(V, S, D):
    int_S = integrate_profile(V, S)
    return 2.356e5 * (D**2) * int_S

def find_FWHM(x, y, level=0.5):
    spline = UnivariateSpline(x, y - (np.max(y)*level), s=0)
    roots = spline.roots()
    if len(roots) < 2:
        return np.nan, roots
    return roots[-1] - roots[0], roots

def assign_units(x, B, W50, D, MHI_desired):
    S_norm = (B / np.max(B))
    FWHM, roots = find_FWHM(x, B)
    if np.isnan(FWHM) or FWHM == 0:
        return None, None
    scale = W50 / FWHM
    V = x * scale
    MHI_initial = get_MHI(V, S_norm, D)
    y_scale = MHI_desired / MHI_initial
    S = S_norm * y_scale
    return V, S

def normalDist(x, sigma, x0=0):
    return np.exp(-(x - x0)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def gaussian_kernel(V, FWHM):
    sigma = FWHM / (2*np.sqrt(2*np.log(2)))
    G = normalDist(V - np.mean(V), sigma)
    return G / G.sum()

def convolve_spectrum(S, V, FWHM):
    G = gaussian_kernel(V, FWHM)
    return fftconvolve(S, G, mode='same')

def Generate_Spectra(size, MHI, W50, D, a=1.0, w=1.0,
                     b1=None, b2=None, c=None, xe=None, xp=None,
                     chunk_size=100000):
    """
    Generates spectra using the Numba-parallel Busy function.
    - chunk_size controls memory usage (avoid allocating > few GB)
    """

    print(f"[Rank {rank}] Generating Spectra (size={size})...")
    start = time.time()

    x = np.linspace(-10, 10, 1000).astype(np.float32)

    if b1 is None: b1 = np.random.uniform(1, 3, size=size)
    if b2 is None: b2 = np.random.uniform(1, 3, size=size)
    if c  is None: c  = np.random.uniform(0, 1, size=size)
    if xe is None: xe = np.random.uniform(-0.5, 0.5, size=size)
    if xp is None: xp = np.random.uniform(-0.5, 0.5, size=size)

    final_M = np.empty(size, dtype=np.float64)
    last_V, last_Sb = None, None

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
        B_chunk = Busy_general_batch(x, a, b1_chunk, b2_chunk, xe_chunk, xp_chunk, c_chunk, w)

        for i in range(n_chunk):
            V, S = assign_units(x, B_chunk[i], W50[start_idx+i], D[start_idx+i], MHI[start_idx+i])
            if V is None:
                final_M[start_idx+i] = np.nan
                continue

            FWHM_thermal = 10.0  # km/s
            S_broad = convolve_spectrum(S, V, FWHM_thermal)
            final_M[start_idx+i] = get_MHI(V, S_broad, D[start_idx+i])

            # store last for sanity check output
            last_V, last_Sb = V, S_broad

    end = time.time()
    print(f"[Rank {rank}] Runtime: {end - start:.2f} sec for {size} spectra")

    return final_M, last_V, last_Sb

def Check_Spectra(catalog_fl):
    catalog = np.load(catalog_fl)
    size = catalog.shape[0]

    # Split work across MPI ranks
    chunk_size = size // size_mpi
    start_i = rank * chunk_size
    end_i = size if rank == size_mpi-1 else (rank + 1) * chunk_size

    local_cat = catalog[start_i:end_i]
    MHI = 10**local_cat[:,0]
    W50 = local_cat[:,3]
    D = local_cat[:,6]

    print(f"Generatiing Spectra [Rank {rank}]")
    t1 = time.time()

    final_M, V, S_broad = Generate_Spectra(len(MHI), MHI, W50, D)
    np.save(f"spectra_rank{rank}.npy", np.asarray([V, S_broad], dtype=object))

    t2 = time.time()
    print(f"[Rank {rank}] Done and saved — took {t2 - t1:.2f} seconds.")


if __name__ == "__main__":
    ti = time.time()
    Check_Spectra('catalogs_output/VolLim_20to60deg_Dmax100.npy')
    tf = time.time()
    print(f"[Rank {rank}] Total runtime: {tf - ti:.2f} seconds.")
