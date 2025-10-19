import cupy as cp
import sys
import time
import glob
import numpy as np
import os
import re
#from mpi4py import MPI

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

# def run(fmin, fmax, U):
#     coarse = coarsechans_index(fmin=fmin, fmax=fmax)
#     f = idealchans_index(fmin, fmax, ideal_res=0.01)
#     R = response_mtx(coarse, f, U)

#     cp.save(f'R_{fmin}_{fmax}_{U}.npy', R)
#     cp.save(f'c_{fmin}_{fmax}_{U}.npy', coarse)
#     cp.save(f'f_{fmin}_{fmax}_{U}.npy', f)

def run_serial(fmin, fmax, U, coarse_chunk_size=64, fine_chunk_size=1000, outdir='/scratch/akanksha/upchan/', res=0.029):
    coarse = coarsechans_index(fmin=fmin, fmax=fmax)
    f_full = idealchans_index(fmin, fmax, ideal_res=res)

    for i in range(0, len(coarse), coarse_chunk_size):
        c_chunk = coarse[i:i+coarse_chunk_size]
        for j in range(0, len(f_full), fine_chunk_size):
            f_chunk = f_full[j:j+fine_chunk_size]
            R_chunk = response_mtx(c_chunk, f_chunk, U)
            cp.save(f'{outdir}/R_c{i}_f{j}.npy', R_chunk)
            cp.save(f'{outdir}/c_c{i}_f{j}.npy', c_chunk)
            cp.save(f'{outdir}/f_c{i}_f{j}.npy', f_chunk)

def numeric_sort(files, pattern=r'R_c(\d+)_f(\d+).npy'):
    return sorted(files, key=lambda x: [int(i) for i in re.findall(pattern, x)[0]])

def merge_chunks(outdir, delete_chunks=True):
    # Sort files numerically
    R_files = numeric_sort(glob.glob(f"{outdir}/R_c*.npy"))
    c_files = numeric_sort(glob.glob(f"{outdir}/c_c*.npy"), pattern=r'c_c(\d+)_f(\d+).npy')
    f_files = numeric_sort(glob.glob(f"{outdir}/f_c*.npy"), pattern=r'f_c(\d+)_f(\d+).npy')

    if not R_files:
        raise FileNotFoundError("No chunk files found in directory.")

    # Load first chunk to get shapes
    first_R = np.load(R_files[0], mmap_mode='r')
    first_c = np.load(c_files[0], mmap_mode='r')
    first_f = np.load(f_files[0], mmap_mode='r')

    ncoarse_total = sum(np.load(f, mmap_mode='r').shape[0] for f in c_files)
    nfreq_total = first_f.shape[0]
    U = first_R.shape[0] // first_c.shape[0]

    # Prepare output arrays
    R_merged = np.empty((ncoarse_total*U, nfreq_total), dtype=first_R.dtype)
    c_merged = np.empty(ncoarse_total, dtype=first_c.dtype)
    f_merged = np.empty(nfreq_total, dtype=first_f.dtype)

    # Fill f_merged (same for all chunks)
    f_merged[:] = first_f[:]

    coarse_offset = 0
    for R_file, c_file in zip(R_files, c_files):
        R_chunk = np.load(R_file)
        c_chunk = np.load(c_file)
        n_coarse = c_chunk.shape[0]

        R_merged[coarse_offset*U:(coarse_offset+n_coarse)*U, :] = R_chunk
        c_merged[coarse_offset:coarse_offset+n_coarse] = c_chunk
        coarse_offset += n_coarse

    # Save merged arrays
    np.save(os.path.join(outdir, 'R_merged.npy'), R_merged)
    np.save(os.path.join(outdir, 'c_merged.npy'), c_merged)
    np.save(os.path.join(outdir, 'f_merged.npy'), f_merged)

    if delete_chunks:
        for f in R_files + c_files + f_files:
            os.remove(f)

    print("Chunks merged successfully.")

# def run_parallel_mpi(fmin, fmax, U, coarse_chunk_size=128):
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     coarse = coarsechans_index(fmin=fmin, fmax=fmax)
#     f = idealchans_index(fmin, fmax, ideal_res=0.001)

#     # split coarse channels across ranks
#     coarse_per_rank = len(coarse) // size
#     start = rank * coarse_per_rank
#     end = (rank+1) * coarse_per_rank if rank != size-1 else len(coarse)
#     coarse_chunk = coarse[start:end]

#     # further chunk if needed to save memory
#     for i in range(0, len(coarse_chunk), coarse_chunk_size):
#         c_subchunk = coarse_chunk[i:i+coarse_chunk_size]
#         R_chunk = response_mtx(c_subchunk, f, U)
#         cp.save(f'R_rank{rank}_chunk{i}.npy', R_chunk)
#         cp.save(f'c_rank{rank}_chunk{i}.npy', c_subchunk)
#         cp.save(f'f_rank{rank}_chunk{i}.npy', f)

# ------------------ Main ------------------ #

if __name__ == "__main__":
    t1 = time.time()
    if len(sys.argv) < 4:
        print("Usage: python Upchannelize.py <fmin> <fmax> <U>")
        sys.exit(1)

    print(f"Generating Response Matrix")
    fmin = float(sys.argv[1])
    fmax = float(sys.argv[2])
    U = int(sys.argv[3])

    run_serial(fmin, fmax, U, outdir=f'/scratch/akanksha/upchan/{int(fmin)}_{int(fmax)}_U{U}', )

    t2 = time.time()
    print(f"Finished. Total Runtime {t2 - t1:.2f} seconds")
