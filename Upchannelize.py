#import cupy as cp
import numpy as cp
import sys
import time
import glob
#import numpy as np
import os
import re
#from mpi4py import MPI
import matplotlib.pyplot as plt

# ------------------ Frequency Utilities ------------------ #

def chord_bandwidth(range=[300,1500], nchans=8192, sampling_rate=0.417):
    freq_range = (1 / (sampling_rate * 1e-3))  # convert sampling rate in ns to MHz
    bandwidth = freq_range / (nchans*2)
    # bandwidth = (range[1] - range[0]) / nchans
    return bandwidth

def freqs2chans(f, range=[300,1500], nchans=8192, sampling_rate=0.417):
    bw = chord_bandwidth(range=range, nchans=nchans, sampling_rate=sampling_rate)
    return (f - range[0]) / bw

def chans2freq(chans, range=[300,1500], nchans=8192, sampling_rate=0.417):
    bw = chord_bandwidth(range=range, nchans=nchans, sampling_rate=sampling_rate)
    return chans*bw + range[0]

def coarsechans_index(fmin, fmax, range=[300,1500], nchans=8192, sampling_rate=0.417):
    min_index = float(cp.floor(freqs2chans(fmin, range, nchans, sampling_rate)))
    max_index = float(cp.ceil(freqs2chans(fmax, range, nchans, sampling_rate)))
    return cp.arange(min_index, max_index + 1)

def idealchans_index(fmin, fmax, ideal_res, range=[300,1500], nchans=8192, sampling_rate=0.417):
    f_ideal = cp.linspace(fmin, fmax, int((fmax-fmin)/ideal_res))
    chan_index = freqs2chans(f_ideal, range, nchans, sampling_rate)
    return chan_index

# For testing
def get_fine_freqs(coarse_frequencies):
    fmax = cp.max(coarse_frequencies)+2
    fmin = cp.min(coarse_frequencies)-2
    dc = coarse_frequencies[1] - coarse_frequencies[0] 
    return cp.arange(fmax, fmin, dc / 3) 

# ------------------ Window Function ------------------ #

def window(index, taps=4, N=8192*2):
    index = cp.asarray(index)
    center = taps * N / 2
    scale = taps * N - 1
    W = (cp.cos(cp.pi * (index - center) / scale))**2 * cp.sinc((index - center)/N)
    return W

# ------------------ Exponentials ------------------ #

def exponential_chan(s, mtx, N):
    # N = -2 in the unchannelization stage (to give cp.exp(1j * cp.pi * (mtx * s)))
    s = s.reshape(1, -1)
    mtx = mtx.reshape(mtx.shape[0], mtx.shape[1], 1)
    return cp.exp(-2j * cp.pi * (mtx * s) / N) # 


# def exponential_upchan(s, mtx):
#     s = s.reshape(1, -1)
#     mtx = mtx.reshape(mtx.shape[0], mtx.shape[1], 1)
#     return cp.exp(1j * cp.pi * (mtx * s))
    

# ------------------ PFB Stages ------------------ #

def weight_chan(cf, N, taps=4, upchan=False):
    # N here is either Number of channels -> 8192*2 for first round PFB 
    # or U the Upchannelization factor for second round PFB 

    j = cp.arange(taps * N).reshape(1, -1) # shape: (1, taps*N)
    if upchan:
        summation = window(j, taps, N) * exponential_chan(j, cf, -2)
    else:
        summation = window(j, taps, N) * exponential_chan(j, cf, N) # shape: (coarse chans, nfreq, taps*N)
    return cp.sum(summation, axis=2) # shape: (coarse chans, nfreq)

# def weight_upchan(cfu, U, taps=4):
#     k = cp.arange(taps*U).reshape(1, -1)
#     return cp.sum(window(k, taps, U) * exponential_chan(k, cfu), axis=2)

# ------------------ Scaling Factors ------------------ #

def scaling(U):
    '''pre-computed overal scaling factors
       for each up-channelization factor U'''
    if U == 1: k = 1.216103148777748e-10
    elif U == 2: k = 7.841991167761238e-11
    elif U == 4: k = 3.195692185478832e-11
    elif U == 8: k = 1.5098060514380606e-11
    elif U == 16: k = 7.437551472089143e-12
    elif U == 32: k = 3.701749876806638e-12
    elif U == 64: k = 1.847847543734494e-12
    else:
        raise('U can only be a power of 2 between [1, 64].')
    return k


def response_mtx(c, f, U, taps=4, N=8192*2):
    c = c.astype(int)
    ncoarse, nfreq = c.shape[0], f.shape[0]

    # Coarse channelization
    submtx_chan = weight_chan(c[:, None] - f[None, :], N=N, taps=taps, upchan=False)
    mtx_chan = cp.repeat(submtx_chan, U, axis=0) # reshaping the resulting matrix so that we get U identical rows per coarse channel

    # Fine upchannelization
    u = cp.arange(U)
    u_exp = u[None, :, None]
    f_exp = f[None, None, :] 
    mtx_up_input = (U - 1)/U - 2*u_exp/U + 2*f_exp
    mtx_up_2d = mtx_up_input.reshape(-1, nfreq)
    mtx_upchan = weight_chan(mtx_up_2d, N=U, taps=taps, upchan=True)
    mtx_upchan = cp.tile(mtx_upchan, (len(c), 1))

    return mtx_chan * mtx_upchan

# ------------------ MPI Parallel Execution ------------------ #

# def run(fmin, fmax, U):
#     coarse = coarsechans_index(fmin=fmin, fmax=fmax)
#     f = idealchans_index(fmin, fmax, ideal_res=0.01)
#     R = response_mtx(coarse, f, U)

#     cp.save(f'R_{fmin}_{fmax}_{U}.npy', R)
#     cp.save(f'c_{fmin}_{fmax}_{U}.npy', coarse)
#     cp.save(f'f_{fmin}_{fmax}_{U}.npy', f)

def run_serial(fmin, fmax, U, coarse_chunk_size=64, fine_chunk_size=1000, outdir='/scratch/akanksha/upchan/', res=0.0294):
    coarse = coarsechans_index(fmin=fmin, fmax=fmax)
    freqs = cp.linspace(1420, 1418, int(cp.floor((1420-1418)/(0.029*3))))
    fine_freqs = get_fine_freqs(freqs)
    fine_chans = freqs2chans(fine_freqs[::-1])
    R = response_mtx(coarse, fine_chans, U=U)
    print("Response is ", R)
    print(R.shape)

    # for i in range(0, len(coarse), coarse_chunk_size):
    #     c_chunk = coarse[i:i+coarse_chunk_size]
    #     for j in range(0, len(f_full), fine_chunk_size):
    #         f_chunk = f_full[j:j+fine_chunk_size]
    #         R_chunk = response_mtx(c_chunk, f_chunk, U)
    #         cp.save(f'{outdir}/R_c{i}_f{j}.npy', R_chunk)
    #         cp.save(f'{outdir}/c_c{i}_f{j}.npy', c_chunk)
    #         cp.save(f'{outdir}/f_c{i}_f{j}.npy', f_chunk)

def numeric_sort(files, pattern=r'R_c(\d+)_f(\d+).npy'):
    return sorted(files, key=lambda x: [int(i) for i in re.findall(pattern, x)[0]])

# def merge_chunks(outdir, delete_chunks=True):
#     # Sort files numerically
#     R_files = numeric_sort(glob.glob(f"{outdir}/R_c*.npy"))
#     c_files = numeric_sort(glob.glob(f"{outdir}/c_c*.npy"), pattern=r'c_c(\d+)_f(\d+).npy')
#     f_files = numeric_sort(glob.glob(f"{outdir}/f_c*.npy"), pattern=r'f_c(\d+)_f(\d+).npy')

#     if not R_files:
#         raise FileNotFoundError("No chunk files found in directory.")

#     # Load first chunk to get shapes
#     first_R = np.load(R_files[0], mmap_mode='r')
#     first_c = np.load(c_files[0], mmap_mode='r')
#     first_f = np.load(f_files[0], mmap_mode='r')

#     ncoarse_total = sum(np.load(f, mmap_mode='r').shape[0] for f in c_files)
#     nfreq_total = first_f.shape[0]
#     U = first_R.shape[0] // first_c.shape[0]

#     # Prepare output arrays
#     R_merged = np.empty((ncoarse_total*U, nfreq_total), dtype=first_R.dtype)
#     c_merged = np.empty(ncoarse_total, dtype=first_c.dtype)
#     f_merged = np.empty(nfreq_total, dtype=first_f.dtype)

#     # Fill f_merged (same for all chunks)
#     f_merged[:] = first_f[:]

#     coarse_offset = 0
#     for R_file, c_file in zip(R_files, c_files):
#         R_chunk = np.load(R_file)
#         c_chunk = np.load(c_file)
#         n_coarse = c_chunk.shape[0]

#         R_merged[coarse_offset*U:(coarse_offset+n_coarse)*U, :] = R_chunk
#         c_merged[coarse_offset:coarse_offset+n_coarse] = c_chunk
#         coarse_offset += n_coarse

#     # Save merged arrays
#     np.save(os.path.join(outdir, 'R_merged.npy'), R_merged)
#     np.save(os.path.join(outdir, 'c_merged.npy'), c_merged)
#     np.save(os.path.join(outdir, 'f_merged.npy'), f_merged)

#     if delete_chunks:
#         for f in R_files + c_files + f_files:
#             os.remove(f)

#     print("Chunks merged successfully.")

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

    run_serial(fmin, fmax, U, outdir=f'/scratch/akanksha/upchan/{int(fmin)}_{int(fmax)}_U{U}', res=0.029)

    t2 = time.time()
    print(f"Finished. Total Runtime {t2 - t1:.2f} seconds")
