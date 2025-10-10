# run_gpu.py
import time
import numpy as np
from mpi4py import MPI
import sys

# GPU libs
try:
    import cupy as cp
    import cupyx
    from cupyx.scipy.signal import fftconvolve as cp_fftconvolve
except Exception as e:
    raise RuntimeError("CuPy and cupyx are required on GPU nodes. Install proper cupy for your CUDA toolkit.") from e

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_mpi = comm.Get_size()

# -----------------------------------------------------------------------------
# GPU utility helpers
# -----------------------------------------------------------------------------
def get_device_for_rank():
    """Choose a GPU device for this MPI rank (round-robin)."""
    ngpu = cp.cuda.runtime.getDeviceCount()
    if ngpu == 0:
        raise RuntimeError("No GPUs found on node.")
    dev_id = rank % ngpu
    cp.cuda.Device(dev_id).use()
    return dev_id

def estimate_chunk_size_from_gpu(mem_fraction=0.6, bytes_per_spectrum=1000*4):
    """
    Estimate a safe chunk size given an approximate bytes_per_spectrum
    (default: 1000 float32 points ~ 4 KB). This is heuristic.
    """
    dev = cp.cuda.Device()
    total_mem = dev.total_memory()
    usable = int(total_mem * mem_fraction)
    chunk_bytes = usable // bytes_per_spectrum
    # leave some margin
    return max(1024, int(chunk_bytes // 2))

# -----------------------------------------------------------------------------
# GPU Busy + helpers (vectorized)
# -----------------------------------------------------------------------------
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
    # ensure float32 for memory savings and speed
    x_gpu = cp.asarray(x, dtype=cp.float32)               # shape (n_x,)
    b1_g = cp.asarray(b1, dtype=cp.float32)               # shape (n_p,)
    b2_g = cp.asarray(b2, dtype=cp.float32)
    xe_g = cp.asarray(xe, dtype=cp.float32)
    xp_g = cp.asarray(xp, dtype=cp.float32)
    c_g  = cp.asarray(c, dtype=cp.float32)

    # Broadcast shapes: (n_p, 1) * (1, n_x) -> (n_p, n_x)
    # err_p = erf(b1*(w + x - xe)) + 1.0
    err_p = cp.erf(b1_g[:, None] * (w + x_gpu[None, :] - xe_g[:, None])) + 1.0
    err_m = cp.erf(b2_g[:, None] * (w - x_gpu[None, :] + xe_g[:, None])) + 1.0
    pbola = c_g[:, None] * ((x_gpu[None, :] - xp_g[:, None])**2) + 1.0

    B = (a / 4.0) * err_p * err_m * pbola
    # Keep single precision
    return B.astype(cp.float32)

# -----------------------------------------------------------------------------
# GPU-friendly FWHM finder and unit assignment
# -----------------------------------------------------------------------------
def assign_units_gpu(x, B, W50_arr, D_arr, MHI_desired_arr):
    """
    Vectorized (per chunk) approximation of assign_units on GPU.
    Inputs:
      x: 1D array (n_x,) on CPU or GPU (we convert to GPU)
      B: (n_profiles, n_x) cupy array
      W50_arr, D_arr, MHI_desired_arr: 1D arrays length n_profiles (numpy)
    Returns:
      V_gpu_list: list of 1D cupy arrays (velocity grid per profile)
      S_gpu_list: list of 1D cupy arrays (profile per profile, same length n_x)
      mask_valid: boolean mask of valid profiles
    NOTE: For memory, this function returns python lists of cupy arrays for per-profile further ops.
    """
    x_gpu = cp.asarray(x, dtype=cp.float32)
    n_profiles, n_x = B.shape

    W50_g  = cp.asarray(W50_arr, dtype=cp.float32)
    D_g    = cp.asarray(D_arr, dtype=cp.float32)
    MHI_g  = cp.asarray(MHI_desired_arr, dtype=cp.float32)

    dx = float(x[1] - x[0])

    # max per profile
    Bmax = B.max(axis=1)   # shape (n_profiles,)
    half = 0.5 * Bmax

    # threshold mask (n_profiles, n_x)
    mask = B >= half[:, None]

    # find left index (first True) and right index (last True) per profile
    # left: use argmax on mask along axis=1 because argmax returns first index of max,
    # but if all False we need to detect that
    left_idx = cp.argmax(mask, axis=1)
    # right: argmax on reversed
    rev_mask = mask[:, ::-1]
    right_rev = cp.argmax(rev_mask, axis=1)
    right_idx = (n_x - 1) - right_rev

    # detect profiles where no True exists: Bmax == 0 or mask all False
    valid = Bmax > 0.0
    valid = valid & (right_idx > left_idx)

    # FWHM in x-units:
    FWHM_x = (right_idx - left_idx) * dx   # array-length n_profiles

    # scale = W50 / FWHM_x
    scale = cp.empty_like(FWHM_x)
    # guard division
    scale[valid] = W50_g[valid] / FWHM_x[valid]
    scale[~valid] = 0.0

    # compute V and rescale: V = x * scale_i -> per-profile
    V_list = []
    S_list = []
    valid_mask = cp.asnumpy(valid)  # we'll need this on host for indexing storage etc.

    # We'll produce lists of GPU arrays for each valid profile (to convolve individually)
    for i in range(n_profiles):
        if not bool(valid[i]):
            V_list.append(None)
            S_list.append(None)
            continue
        s_norm = B[i] / Bmax[i]  # normalized profile
        V_i = x_gpu * scale[i]   # velocity grid
        # integrate to get initial MHI (trapz on GPU)
        # get integrand in float32 -> cast to float64 for trapz accuracy then back
        MHI_initial = cp.trapz(s_norm.astype(cp.float64), x=V_i.astype(cp.float64)).astype(cp.float32)
        if MHI_initial == 0:
            V_list.append(None)
            S_list.append(None)
            valid_mask[i] = False
            continue
        y_scale = MHI_g[i] / MHI_initial
        S_i = s_norm * y_scale
        V_list.append(V_i)
        S_list.append(S_i)
    return V_list, S_list, valid_mask

# -----------------------------------------------------------------------------
# GPU convolution (per-profile)
# -----------------------------------------------------------------------------
def convolve_profile_gpu(S_gpu, V_gpu, FWHM_thermal=10.0):
    """
    Convolve a single profile S_gpu defined on velocity grid V_gpu with a Gaussian of FWHM_thermal (km/s).
    Returns convolved S_gpu (same shape).
    """
    # produce gaussian kernel in velocity space
    # sigma = FWHM / (2*sqrt(2*ln2))
    sigma = FWHM_thermal / (2.0 * cp.sqrt(2.0 * cp.log(2.0)))
    # build Gaussian on same V grid, centered at mean(V)
    vmean = cp.mean(V_gpu)
    G = cp.exp(-((V_gpu - vmean)**2) / (2.0 * sigma**2))
    G = G / G.sum()
    # convolution using FFT (cupyx)
    try:
        conv = cp_fftconvolve(S_gpu, G, mode='same')
    except Exception:
        # fallback to direct convolution (slower)
        conv = cp.convolve(S_gpu, G, mode='same')
    return conv

# -----------------------------------------------------------------------------
# Main GPU generate function - chunked processing
# -----------------------------------------------------------------------------
def Generate_Spectra_GPU(size, MHI, W50, D, a=1.0, w=1.0,
                         b1=None, b2=None, c=None, xe=None, xp=None,
                         chunk_size_gpu=None):
    """
    GPU-enabled spectra generation. Similar interface to your CPU function.
    Processes data in chunks (chunk_size_gpu) to fit in GPU memory.
    Returns final_M (numpy array), and last V and S for sanity check.
    """

    print(f"[Rank {rank}] Generating Spectra (size={size}) on GPU...")
    t0 = time.time()

    # choose a GPU device for this rank
    dev_id = get_device_for_rank()
    device = cp.cuda.Device(dev_id)

    # default random params
    if b1 is None: b1 = np.random.uniform(1, 3, size=size).astype(np.float32)
    if b2 is None: b2 = np.random.uniform(1, 3, size=size).astype(np.float32)
    if c  is None: c  = np.random.uniform(0, 1, size=size).astype(np.float32)
    if xe is None: xe = np.random.uniform(-0.5, 0.5, size=size).astype(np.float32)
    if xp is None: xp = np.random.uniform(-0.5, 0.5, size=size).astype(np.float32)

    # grid x (same as your code)
    x = np.linspace(-10, 10, 1000).astype(np.float32)
    n_x = x.size

    # estimate chunk size if not provided
    if chunk_size_gpu is None:
        # assume bytes_per_spectrum ~ n_x * 4 bytes + overhead; adjust if different
        chunk_size_gpu = estimate_chunk_size_from_gpu(mem_fraction=0.55,
                                                      bytes_per_spectrum=n_x*4)
    chunk_size_gpu = int(chunk_size_gpu)
    print(f"[Rank {rank}] Using chunk_size_gpu={chunk_size_gpu}")

    final_M = np.empty(size, dtype=np.float32)
    last_V = None
    last_Sb = None

    total_chunks = (size + chunk_size_gpu - 1) // chunk_size_gpu
    print_every = max(1, total_chunks // 10)   # print every 10%

    for chunk_idx, start_idx in enumerate(range(0, size, chunk_size_gpu)):
        end_idx = min(start_idx + chunk_size_gpu, size)
        n_chunk = end_idx - start_idx

        # slice host arrays
        b1_chunk = b1[start_idx:end_idx]
        b2_chunk = b2[start_idx:end_idx]
        c_chunk  = c[start_idx:end_idx]
        xe_chunk = xe[start_idx:end_idx]
        xp_chunk = xp[start_idx:end_idx]
        W50_chunk = W50[start_idx:end_idx]
        D_chunk = D[start_idx:end_idx]
        MHI_chunk = MHI[start_idx:end_idx]

        # Move parameters to GPU (small arrays)
        B_chunk = Busy_general_batch_cupy(x, a, b1_chunk, b2_chunk,
                                          xe_chunk, xp_chunk, c_chunk, w)
        # B_chunk is (n_chunk, n_x) on GPU (float32)

        # Assign units & scaling on GPU
        V_list, S_list, valid_mask = assign_units_gpu(x, B_chunk,
                                                      W50_chunk, D_chunk, MHI_chunk)

        # For each profile in chunk do convolution on GPU and compute final MHI
        for i in range(n_chunk):
            if not valid_mask[i]:
                final_M[start_idx + i] = np.nan
                continue

            S_gpu = S_list[i]
            V_gpu = V_list[i]

            # convolve on GPU
            S_broad_gpu = convolve_profile_gpu(S_gpu, V_gpu, FWHM_thermal=10.0)

            # integrate
            MHI_val = cp.trapz(S_broad_gpu.astype(cp.float64),
                               x=V_gpu.astype(cp.float64)).astype(cp.float32)
            final_M[start_idx + i] = float(MHI_val.get())

            # store last for sanity check
            if (end_idx == size) and (i == n_chunk - 1):
                last_V = cp.asnumpy(V_gpu)
                last_Sb = cp.asnumpy(S_broad_gpu)

        # free GPU memory explicitly
        del B_chunk, V_list, S_list
        cp._default_memory_pool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()

        # progress updates
        if (chunk_idx + 1) % print_every == 0 or end_idx == size:
            percent_done = (chunk_idx + 1) / total_chunks * 100
            elapsed = time.time() - t0
            rate = (chunk_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_chunks - (chunk_idx + 1)) / rate if rate > 0 else 0
            print(f"[Rank {rank}] {percent_done:5.1f}% complete — "
                  f"{elapsed/60:6.1f} min elapsed, ETA {eta/60:6.1f} min "
                  f"(GPU {dev_id})")

    t1 = time.time()
    print(f"[Rank {rank}] Finished in {t1 - t0:.2f} sec "
          f"for {size} spectra on GPU {dev_id}")
    return final_M, last_V, last_Sb


# -----------------------------------------------------------------------------
# High-level wrapper that each MPI rank calls (similar to your Run_Spectra)
# -----------------------------------------------------------------------------
def Run_Spectra_GPU(catalog_fl):
    catalog = np.load(catalog_fl)
    size = catalog.shape[0]

    # Split work across MPI ranks
    chunk_size = size // size_mpi
    start_i = rank * chunk_size
    end_i = size if rank == size_mpi-1 else (rank + 1) * chunk_size

    local_cat = catalog[start_i:end_i]
    # adapt column indices per your catalogue format
    MHI = local_cat[:,0].astype(np.float32)
    W50 = local_cat[:,3].astype(np.float32)
    D   = local_cat[:,6].astype(np.float32)

    print(f"[Rank {rank}] Generatiing Spectra (local size={len(MHI)})")
    t1 = time.time()

    final_M, V, S_broad = Generate_Spectra_GPU(len(MHI), MHI, W50, D)
    np.save(f"spectra_rank{rank}.npy", np.asarray([V, S_broad], dtype=object))

    t2 = time.time()
    print(f"[Rank {rank}] Done and saved — took {t2 - t1:.2f} seconds.")


if __name__ == "__main__":
    # Example usage: mpirun -n 48 python run_gpu.py <catalog.npy>
    
    if len(sys.argv) < 2:
        print("Usage: python run_gpu.py <catalog.npy>")
        sys.exit(1)
    catalog_fl = sys.argv[1]
    Run_Spectra_GPU(catalog_fl)
