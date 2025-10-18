import cupy as cp
import sys
import time
from mpi4py import MPI

# ------------------ Frequency Utilities ------------------ #

def chord_bandwidth(range=[300,1500], nchans=8192, sampling_rate=0.417):
    freq_range = (1 / (sampling_rate * 1e-3))  # convert sampling rate in ns to MHz
    bandwidth = freq_range / (nchans*2)
    return bandwidth

def coarsechans_index(fmin, fmax, range=[300,1500], nchans=8192, sampling_rate=0.417):
    bw = chord_bandwidth(range=range, nchans=nchans, sampling_rate=sampling_rate)
    max_index = float(cp.floor((fmax - range[0]) / bw))
    min_index = float(cp.ceil((fmin - range[0]) / bw))
    return cp.arange(min_index, max_index + 1)

def idealchans_index(fmin, fmax, ideal_res, range=[300,1500], nchans=8192, sampling_rate=0.417):
    f_ideal = cp.linspace(fmin, fmax, int((fmax-fmin)/ideal_res))
    bw = chord_bandwidth(range, nchans, sampling_rate)
    chan_index = f_ideal / bw
    return chan_index

# ------------------ Window Function ------------------ #

def window(index, taps=4, N=8192*2, dtype=cp.float32):
    index = cp.asarray(index, dtype=dtype)
    center = taps * N / 2
    scale = taps * N - 1
    W = (cp.cos(cp.pi * (index - center) / scale))**2 * cp.sinc((index - center)/N)
    return W.astype(dtype)

# ------------------ Exponentials ------------------ #

def exponential_chan(s, mtx, N=8192*2):
    s = cp.asarray(s, dtype=cp.float32).reshape(1, -1)
    mtx = cp.asarray(mtx, dtype=cp.float32).reshape(mtx.shape[0], mtx.shape[1], 1)
    return cp.exp(-2j * cp.pi * (mtx * s) / N)

def exponential_upchan(s, mtx):
    s = cp.asarray(s, dtype=cp.float32).reshape(1, -1)
    mtx = cp.asarray(mtx, dtype=cp.float32).reshape(mtx.shape[0], mtx.shape[1], 1)
    return cp.exp(1j * cp.pi * (mtx * s))

# ------------------ PFB Stages ------------------ #

def weight_chan(cf, taps=4, N=8192*2):
    j = cp.arange(taps * N, dtype=cp.float32).reshape(1, -1)
    return cp.sum(window(j, taps, N) * exponential_chan(j, cf, N), axis=2)

def weight_upchan(cfu, U, taps=4):
    k = cp.arange(taps*U, dtype=cp.float32).reshape(1, -1)
    return cp.sum(window(k, taps, U) * exponential_upchan(k, cfu), axis=2)

# ------------------ Scaling Factors ------------------ #

def scaling(U):
    factors = {1: 1.216e-10, 2: 7.84e-11, 4: 3.20e-11, 8: 1.51e-11,
               16: 7.44e-12, 32: 3.70e-12, 64: 1.85e-12}
    if U not in factors:
        raise ValueError("U must be a power of 2 between 1 and 64.")
    return factors[U]

# ------------------ Full Response Matrix ------------------ #

def response_mtx(c, f, U, taps=4, N=8192*2, dtype=cp.float32):
    c = cp.asarray(c, dtype=dtype)
    f = cp.asarray(f, dtype=dtype)
    ncoarse, nfreq = c.shape[0], f.shape[0]

    # Coarse channelization
    submtx_chan = weight_chan(c[:, None] - f[None, :], taps=taps, N=N)
    mtx_chan = cp.repeat(submtx_chan, U, axis=0)

    # Fine upchannelization
    u = cp.arange(U, dtype=dtype)
    c_exp = c[:, None, None]
    u_exp = u[None, :, None]
    f_exp = f[None, None, :]
    mtx_up_input = (c_exp * U + u_exp - f_exp) / U
    mtx_up_2d = mtx_up_input.reshape(-1, nfreq)
    mtx_upchan = weight_upchan(mtx_up_2d, U, taps=taps)

    return mtx_chan * mtx_upchan

# ------------------ MPI Parallel Execution ------------------ #

def run_parallel_mpi(fmin, fmax, U, coarse_chunk_size=128):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    coarse = coarsechans_index(fmin=fmin, fmax=fmax)
    f = idealchans_index(fmin, fmax, ideal_res=0.001)

    # split coarse channels across ranks
    coarse_per_rank = len(coarse) // size
    start = rank * coarse_per_rank
    end = (rank+1) * coarse_per_rank if rank != size-1 else len(coarse)
    coarse_chunk = coarse[start:end]

    # further chunk if needed to save memory
    for i in range(0, len(coarse_chunk), coarse_chunk_size):
        c_subchunk = coarse_chunk[i:i+coarse_chunk_size]
        R_chunk = response_mtx(c_subchunk, f, U)
        cp.save(f'R_rank{rank}_chunk{i}.npy', R_chunk)
        cp.save(f'c_rank{rank}_chunk{i}.npy', c_subchunk)
        cp.save(f'f_rank{rank}_chunk{i}.npy', f)

# ------------------ Main ------------------ #

if __name__ == "__main__":
    t1 = time.time()
    if len(sys.argv) < 4:
        print("Usage: python Upchannelize.py <fmin> <fmax> <U>")
        sys.exit(1)

    fmin = float(sys.argv[1])
    fmax = float(sys.argv[2])
    U = int(sys.argv[3])

    run_parallel_mpi(fmin, fmax, U)

    t2 = time.time()
    print(f"[Rank] Finished. Total Runtime {t2 - t1:.2f} seconds")
