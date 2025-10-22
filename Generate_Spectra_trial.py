# run_gpu.py
import time
from mpi4py import MPI
import sys
from cupyx.scipy import special
import astropy.units as u
import astropy.constants as c
import cupy as cp

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_mpi = comm.Get_size()


def Busy_general_batch_cupy(x, a, b1, b2, xe, xp, c, w):
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
    err_p = special.erf(b1[:, None] * (w + x[None, :] - xe[:, None])) + 1.0
    err_m = special.erf(b2[:, None] * (w - x[None, :] + xe[:, None])) + 1.0
    pbola = c[:, None] * ((x[None, :] - xp[:, None])**2) + 1.0

    B = (a / 4.0) * err_p * err_m * pbola
    return B.astype(cp.float32)

def integrate_profile(X, Y):
    return cp.trapz(Y, x=X)

def get_MHI(V, S, D):
    int_S = integrate_profile(V, S)
    return 2.356e5 * (D**2) * int_S

def get_fobs(z, f_rest=1420.40575177):
    return f_rest / (1 + z)

def convert_f(v, z, f_rest=1420.40575177):
    '''Convert velocities in km/s to observed frequencies in MHz'''
    v_c = c.c.to(u.km/u.s).value 
    f_obs = get_fobs(z, f_rest=f_rest)
    freqs = f_obs*cp.sqrt((1 - v/v_c) / (1 + v/v_c))
    return freqs

def find_FWHM(x, B):
    return 2.1

def assign_units(x, B, W50, D, z, MHI_desired, FWHM_thermal=10):
    '''assigns units of velocity (km/s) vs. Flux density (Jy)'''
    # Scale x-axis
    S_norm = B / B.max(axis=1, keepdims=True)
    FWHM = find_FWHM(x, B)

    mask = W50 < FWHM_thermal
    W50[mask] = cp.sqrt(FWHM_thermal**2 + W50[mask]**2)

    scale = W50 / FWHM
    V = x * scale
    f = convert_f(V, z) # convert to frequency axis -> Note: df will not be constant

    # Scale y-axis
    MHI_initial = get_MHI(V, S_norm, D)
    y_scale = MHI_desired / MHI_initial
    S = S_norm * y_scale
    return V, f, S

# def convolve_profile_gpu(S_gpu, V_gpu, FWHM_thermal=10.0):
#     """
#     Convolve a single profile S_gpu defined on velocity grid V_gpu with a Gaussian of FWHM_thermal (km/s).
#     Returns convolved S_gpu (same shape).
#     """
#     # produce gaussian kernel in velocity space
#     # sigma = FWHM / (2*sqrt(2*ln2))
#     sigma = FWHM_thermal / (2.0 * cp.sqrt(2.0 * cp.log(2.0)))
#     # build Gaussian on same V grid, centered at mean(V)
#     vmean = cp.mean(V_gpu)
#     G = cp.exp(-((V_gpu - vmean)**2) / (2.0 * sigma**2))
#     G = G / G.sum()
#     # convolution using FFT (cupyx)
#     try:
#         conv = cp_fftconvolve(S_gpu, G, mode='same')
#     except Exception:
#         # fallback to direct convolution (slower)
#         conv = cp.convolve(S_gpu, G, mode='same')
#     return conv

def Generate_Spectra_GPU(size, MHI, W50, D_C, z, a=1.0, w=1.0,
                         b1=None, b2=None, c=None, xe=None, xp=None,
                         chunk_size=None, dtype=cp.float32):
    print(f"[Rank {rank}] Generating Spectra (size={size}) on GPU...")
    t0 = time.time()

    # default random params
    if b1 is None: b1 = cp.random.uniform(1, 3, size=size).astype(dtype)
    if b2 is None: b2 = cp.random.uniform(1, 3, size=size).astype(dtype)
    if c  is None: c  = cp.random.uniform(0, 1, size=size).astype(dtype)
    if xe is None: xe = cp.random.uniform(-0.5, 0.5, size=size).astype(dtype)
    if xp is None: xp = cp.random.uniform(-0.5, 0.5, size=size).astype(dtype)

    # grid x (same as your code)
    N = 100000
    x = cp.linspace(-5, 5, N).astype(dtype)

    # Get luminosity distance D_L from co-mocing distance D_C:
    D_L = (1+z)*D_C

    for chunk_idx, start_idx in enumerate(range(0, size, chunk_size)):
        final_M = cp.empty(chunk_size, dtype=dtype)
        Vel = cp.empty((chunk_size, N), dtype=dtype)
        S_flux = cp.empty((chunk_size, N), dtype=dtype)
        freq = cp.empty((chunk_size, N), dtype=dtype)

        end_idx = min(start_idx + chunk_size, size)

        # slice host arrays
        b1_chunk = b1[start_idx:end_idx]
        b2_chunk = b2[start_idx:end_idx]
        c_chunk  = c[start_idx:end_idx]
        xe_chunk = xe[start_idx:end_idx]
        xp_chunk = xp[start_idx:end_idx]
        W50_chunk = W50[start_idx:end_idx]
        D_chunk = D_L[start_idx:end_idx]
        MHI_chunk = MHI[start_idx:end_idx]

        # Move parameters to GPU (small arrays)
        B_chunk = Busy_general_batch_cupy(x, a, b1_chunk, b2_chunk,
                                          xe_chunk, xp_chunk, c_chunk, w)
        # B_chunk is (n_chunk, n_x) on GPU (float32)

        # Assign units & scaling on GPU
        Vel[chunk_idx], freq[chunk_idx], S_flux[chunk_idx] = assign_units(x, B_chunk,
                                                      W50_chunk, D_chunk, MHI_chunk)

        cp.save(f"spectra_gpu_rank{rank}_chunk{chunk_idx}.npy", cp.asarray([Vel, freq, S_flux], dtype=object))

        t1 = time.time()
        print(f"[Chunk {chunk_idx}] Finished in {t1 - t0:.2f} sec "
            f"for {size} spectra")
    return Vel, freq, S_flux


# -----------------------------------------------------------------------------
# High-level wrapper that each MPI rank calls (similar to your Run_Spectra)
# -----------------------------------------------------------------------------
def Run_Spectra_GPU(catalog_fl, dtype=cp.float32):
    catalog = cp.load(catalog_fl)
    size = catalog.shape[0]

    # Split work across MPI ranks
    chunk_size = size // size_mpi
    start_i = rank * chunk_size
    end_i = size if rank == size_mpi-1 else (rank + 1) * chunk_size

    local_cat = catalog[start_i:end_i]
    # adapt column indices per your catalogue format
    MHI = local_cat[:,0].astype(dtype)
    W50 = local_cat[:,3].astype(dtype)
    D   = local_cat[:,6].astype(dtype)
    z = local_cat[:,8].astype(dtype)

    print(f"[Rank {rank}] Generatiing Spectra (local size={size})")
    t1 = time.time()

    Vel, freq, S_flux = Generate_Spectra_GPU(len(MHI), MHI, W50, D)

    if plot:
        print("Plotting...")
        import Plotting as plot
        plot.check_Spectra(MHI, MHI, W50, Vel, S_flux, freq, z, D, "Plots/Test_spectra_GPU.pdf")

    t2 = time.time()
    print(f"[Rank {rank}] Done and saved — took {t2 - t1:.2f} seconds.")


if __name__ == "__main__":
    Run_Spectra_GPU(catalog_fl='../catalogs_output/VolLim_20to60deg_Dmax500.npy', size=20, plot=True)
