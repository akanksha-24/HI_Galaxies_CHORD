import numpy as np
import sys
import time
import glob
import os
import re
#from mpi4py import MPI
import matplotlib.pyplot as plt

def spectra_freq(Dmax=None, zmax=None, fmax=None, fres=0.001):
    fmin = gf.get_fobs(zmax)
    f_full = np.arange(fmin, fmax + fres/2, fres)


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
    min_index = float(np.floor(freqs2chans(fmin, range, nchans, sampling_rate)))
    max_index = float(np.ceil(freqs2chans(fmax, range, nchans, sampling_rate)))
    return np.arange(min_index, max_index + 1)

def idealchans_index(fmin, fmax, ideal_res, range=[300,1500], nchans=8192, sampling_rate=0.417):
    f_ideal = np.linspace(fmin, fmax, int((fmax-fmin)/ideal_res))
    chan_index = freqs2chans(f_ideal, range, nchans, sampling_rate)
    return chan_index

# For testing
def get_fine_freqs(coarse_frequencies):
    fmax = np.max(coarse_frequencies)+2
    fmin = np.min(coarse_frequencies)-2
    dc = coarse_frequencies[1] - coarse_frequencies[0] 
    return np.arange(fmax, fmin, dc / 3) 


def window(index, taps=4, N=8192*2):
    index = np.asarray(index)
    center = taps * N / 2
    scale = taps * N - 1
    W = (np.cos(np.pi * (index - center) / scale))**2 * np.sinc((index - center)/N)
    return W

def exponential_chan(s, mtx, N):
    # N = -2 in the unchannelization stage (to give cp.exp(1j * cp.pi * (mtx * s)))
    s = s.reshape(1, -1)
    mtx = mtx.reshape(mtx.shape[0], mtx.shape[1], 1)
    return np.exp(-2j * np.pi * (mtx * s) / N) # 

def weight_chan(cf, N, taps=4, upchan=False):
    # N here is either Number of channels -> 8192*2 for first round PFB 
    # or U the Upchannelization factor for second round PFB 

    j = np.arange(taps * N).reshape(1, -1) # shape: (1, taps*N)
    if upchan:
        summation = window(j, taps, N) * exponential_chan(j, cf, -2)
    else:
        summation = window(j, taps, N) * exponential_chan(j, cf, N) # shape: (coarse chans, nfreq, taps*N)
    return np.sum(summation, axis=2) # shape: (coarse chans, nfreq)

# def weight_upchan(cfu, U, taps=4):
#     k = cp.arange(taps*U).reshape(1, -1)
#     return cp.sum(window(k, taps, U) * exponential_chan(k, cfu), axis=2)


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
    mtx_chan = np.repeat(submtx_chan, U, axis=0) # reshaping the resulting matrix so that we get U identical rows per coarse channel

    # Fine upchannelization
    u = np.arange(U)
    u_exp = u[None, :, None]
    f_exp = f[None, None, :] 
    mtx_up_input = (U - 1)/U - 2*u_exp/U + 2*f_exp
    mtx_up_2d = mtx_up_input.reshape(-1, nfreq)
    mtx_upchan = weight_chan(mtx_up_2d, N=U, taps=taps, upchan=True)
    mtx_upchan = np.tile(mtx_upchan, (len(c), 1))

    return mtx_chan * mtx_upchan

def save_response(fmin, fmax, U, outdir='/scratch/akanksha/upchan/', res=0.0294):
    coarse = coarsechans_index(fmin=fmin, fmax=fmax)
    freqs = chans2freq(coarse)
    fine_freqs = get_fine_freqs(freqs)
    fine_chans = freqs2chans(fine_freqs[::-1])

    k = scaling(U)
    R = response_mtx(coarse, fine_chans, U=U)

    # flat spectrum to get normalization vector
    flat = np.ones_like(freqs)
    norm_unscaled = np.matmul(np.abs(R)**2, flat[::-1])
    norm_scaled = norm_unscaled * k

    np.save(outdir + f"R_{fmin}_{fmax}_{U}.npy", R)
    np.save(outdir + f"norm_{fmin}_{fmax}_{U}.npy", norm_scaled)

    return R, norm_scaled

# def Upchannelize(R, norm):

 

# if __name__ == "__main__":
#     t1 = time.time()
#     if len(sys.argv) < 4:
#         print("Usage: python Upchannelize.py <fmin> <fmax> <U>")
#         sys.exit(1)

#     print(f"Generating Response Matrix")
#     fmin = float(sys.argv[1])
#     fmax = float(sys.argv[2])
#     U = int(sys.argv[3])

#     R, norm = save_response(fmin, fmax, U, outdir=f'/scratch/akanksha/upchan/', res=0.029)

#     t2 = time.time()
#     print(f"Finished. Total Runtime {t2 - t1:.2f} seconds")
