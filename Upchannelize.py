import cupy as cp
import sys
import time

def chord_bandwidth(range=[300,1500], nchans=8192, sampling_rate=0.417):
    freq_range = (1 / (sampling_rate * 1e-3)) # the 1e-3 converts a sampling rate in nano secs to micro secs, to give a frequency in MHz
    bandwidth = freq_range / (nchans*2)
    #bandwidth = (range[1] - range[0])/nchans 
    return bandwidth

def coarsechans_index(fmin, fmax, range=[300,1500], nchans=8192, sampling_rate=0.417):
    bw = chord_bandwidth(range=range, nchans=nchans, sampling_rate=sampling_rate)
    max_index = cp.floor((fmax - range[0]) / bw)
    min_index = cp.ceil((fmin - range[0]) / bw)
    return cp.arange(min_index, max_index+1)

def idealchans_index(fmin, fmax, ideal_res, range=[300,1500], nchans=8192, sampling_rate=0.417):
    f_ideal = cp.linspace(fmin, fmax, (fmax-fmin)/ideal_res)
    bw = chord_bandwidth(range, nchans, sampling_rate)
    chan_index = f_ideal / bw
    return chan_index

########################### Window Function (GPU) ###########################

def window(index, taps=4, N=8192*2, dtype=cp.float32):
    """
    Sinc-Hanning window 
    """
    index = cp.asarray(index, dtype=dtype)
    center = taps * N / 2
    scale = taps * N - 1
    W = (cp.cos(cp.pi * (index - center) / scale))**2 * cp.sinc((index - center)/N)
    return W.astype(dtype)


########################### Exponentials (GPU) ###########################

def exponential_chan(s, mtx, N=8192*2):
    """
    First-round PFB exponential: e^{-2 pi i (c-f) s / N}.
    """
    s = cp.asarray(s, dtype=cp.float32).reshape(1, -1)
    mtx = cp.asarray(mtx, dtype=cp.float32).reshape(mtx.shape[0], mtx.shape[1], 1)
    v = mtx * s
    return cp.exp(-2j * cp.pi * v / N)


def exponential_upchan(s, mtx):
    """
    Second-round PFB exponential: e^{i pi (c u - f) k}.
    """
    s = cp.asarray(s, dtype=cp.float32).reshape(1, -1)
    mtx = cp.asarray(mtx, dtype=cp.float32).reshape(mtx.shape[0], mtx.shape[1], 1)
    v = mtx * s
    return cp.exp(1j * cp.pi * v)


########################### First-round PFB ###########################

def weight_chan(cf, taps=4, N=8192*2):
    """
    First-round PFB channelization on GPU.
    """
    j = cp.arange(taps * N, dtype=cp.float32).reshape(1, -1)
    summation = window(j, taps, N) * exponential_chan(j, cf, N)
    return cp.sum(summation, axis=2)


########################### Second-round PFB ###########################

def weight_upchan(cfu, U, taps=4):
    """
    Second-round PFB channelization on GPU.
    """
    k = cp.arange(taps*U, dtype=cp.float32).reshape(1, -1)
    summation = window(k, taps, U) * exponential_upchan(k, cfu)
    return cp.sum(summation, axis=2)


########################### Scaling Factors ###########################

def scaling(U):
    factors = {1: 1.216103148777748e-10,
               2: 7.841991167761238e-11,
               4: 3.195692185478832e-11,
               8: 1.5098060514380606e-11,
               16: 7.437551472089143e-12,
               32: 3.701749876806638e-12,
               64: 1.847847543734494e-12}
    if U not in factors:
        raise ValueError("U must be a power of 2 between 1 and 64.")
    return factors[U]


########################### Full Response Matrix (GPU) ###########################

def response_mtx(c, f, U, taps=4, N=8192*2):
    """
    Computes full PFB response matrix (coarse + fine channels) on GPU.
    Returns a cupy array.
    """
    c = cp.asarray(c)
    f = cp.asarray(f)
    
    # Coarse channelization
    submtx_chan = c.T - f[:, 0]  # shape: (ncoarse, nfreq)
    submtx_chan = weight_chan(submtx_chan, taps, N)
    mtx_chan = cp.repeat(submtx_chan, U, axis=0)

    # Fine upchannelization
    submtx_upchan = cp.tile(cp.arange(U, dtype=cp.float32), (f.shape[0], 1)).T
    submtx_upchan = (U-1)/U - 2*submtx_upchan/U + 2*f[:,0]
    submtx_upchan = weight_upchan(submtx_upchan, U, taps)
    mtx_upchan = cp.tile(submtx_upchan, (len(c[0]), 1))

    response_mtx = mtx_chan * mtx_upchan

    return response_mtx


if __name__ == "__main__":
    # Example usage: mpirun -n 48 python run_gpu.py <catalog.npy>
    t1 = time.time()
    print("Generating Response Matrix...")

    if len(sys.argv) < 4:
        print("Usage: python Upchannelize.py <fmin> <fmax> <U>")
        sys.exit(1)
    fmin = sys.argv[1]
    fmax = sys.argv[2]
    U = sys.argv[3]
    c = coarsechans_index(fmin=fmin, fmax=fmax)
    f = idealchans_index(fmin, fmax, ideal_res=0.001)
    R = response_mtx(c, f, U, taps=4, N=8192*2)
    cp.save('R_{fmin}_{fmax}_{U}.npy', R)
    cp.save('f_{fmin}_{fmax}_{U}.npy', f)
    cp.save('c_{fmin}_{fmax}_{U}.npy', c)
    t2 = time.time()
    print(f"Finished. Total Runtime {t2-t1} seconds")


