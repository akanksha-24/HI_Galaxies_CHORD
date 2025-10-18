import sys
import time
import math
import numpy as np
from numba import cuda, float32, complex64

# ------------------ Frequency Utilities ------------------ #

def chord_bandwidth(range=[300,1500], nchans=8192, sampling_rate=0.417):
    freq_range = (1 / (sampling_rate * 1e-3))  # convert sampling rate in ns to MHz
    bandwidth = freq_range / (nchans*2)
    return bandwidth

def coarsechans_index(fmin, fmax, range=[300,1500], nchans=8192, sampling_rate=0.417):
    bw = chord_bandwidth(range=range, nchans=nchans, sampling_rate=sampling_rate)
    max_index = int(np.floor((fmax - range[0]) / bw))
    min_index = int(np.ceil((fmin - range[0]) / bw))
    return np.arange(min_index, max_index + 1)

def idealchans_index(fmin, fmax, ideal_res, range=[300,1500], nchans=8192, sampling_rate=0.417):
    f_ideal = np.linspace(fmin, fmax, int((fmax-fmin)/ideal_res))
    bw = chord_bandwidth(range, nchans, sampling_rate)
    chan_index = f_ideal / bw
    return chan_index

# ------------------ Numba CUDA Kernels ------------------ #

@cuda.jit
def weight_chan_kernel(cf, W_out, taps=4, N=8192*2):
    i, j = cuda.grid(2)
    if i < cf.shape[0] and j < cf.shape[1]:
        acc_real = 0.0
        acc_imag = 0.0
        for t in range(taps * N):
            x = t
            center = taps * N / 2
            scale = taps * N - 1
            # Cos^2 * sinc window
            if x - center == 0:
                sinc_val = 1.0
            else:
                sinc_val = math.sin(math.pi * (x - center)/N)/(math.pi * (x - center)/N)
            window_val = (math.cos(math.pi * (x - center)/scale) ** 2) * sinc_val
            exp_val = -2.0 * math.pi * (cf[i, j] * x) / N
            acc_real += window_val * math.cos(exp_val)
            acc_imag += window_val * math.sin(exp_val)
        W_out[i, j] = complex(acc_real, acc_imag)

@cuda.jit
def weight_upchan_kernel(cu, U, W_out, taps=4):
    i, j = cuda.grid(2)
    if i < cu.shape[0] and j < cu.shape[1]:
        acc_real = 0.0
        acc_imag = 0.0
        for t in range(taps * U):
            x = t
            center = taps * U / 2
            scale = taps * U - 1
            if x - center == 0:
                sinc_val = 1.0
            else:
                sinc_val = math.sin(math.pi * (x - center)/U)/(math.pi * (x - center)/U)
            window_val = (math.cos(math.pi * (x - center)/scale) ** 2) * sinc_val
            exp_val = math.pi * (cu[i, j] * x)
            acc_real += window_val * math.cos(exp_val)
            acc_imag += window_val * math.sin(exp_val)
        W_out[i, j] = complex(acc_real, acc_imag)

# ------------------ Full Response Matrix ------------------ #

def response_mtx(c, f, U, taps=4, N=8192*2):
    c = np.asarray(c, dtype=np.float32)
    f = np.asarray(f, dtype=np.float32)
    ncoarse = c.shape[0]
    nfreq = f.shape[0]

    cf = np.zeros((ncoarse, nfreq), dtype=np.float32)
    for i in range(ncoarse):
        for j in range(nfreq):
            cf[i,j] = c[i] - f[j]

    # Coarse channelization
    W_coarse = cuda.device_array((ncoarse, nfreq), dtype=complex64)
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(ncoarse / threadsperblock[0])
    blockspergrid_y = math.ceil(nfreq / threadsperblock[1])
    weight_chan_kernel[(blockspergrid_x, blockspergrid_y), threadsperblock](cf, W_coarse, taps, N)
    cuda.synchronize()
    W_coarse_host = W_coarse.copy_to_host()

    # Repeat for U upchannelization
    W_coarse_rep = np.repeat(W_coarse_host, U, axis=0)

    # Fine upchannelization
    cu = np.zeros((ncoarse*U, nfreq), dtype=np.float32)
    for i in range(ncoarse):
        for u in range(U):
            for j in range(nfreq):
                cu[i*U + u, j] = (c[i]*U + u - f[j]) / U

    W_up = cuda.device_array((ncoarse*U, nfreq), dtype=complex64)
    blockspergrid_x = math.ceil(cu.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(nfreq / threadsperblock[1])
    weight_upchan_kernel[(blockspergrid_x, blockspergrid_y), threadsperblock](cu, U, W_up, taps)
    cuda.synchronize()
    W_up_host = W_up.copy_to_host()

    # Element-wise multiplication
    R = W_coarse_rep * W_up_host
    return R

# ------------------ Serial Execution ------------------ #

def run_serial(fmin, fmax, U, coarse_chunk_size=64, fine_chunk_size=1000):
    coarse = coarsechans_index(fmin=fmin, fmax=fmax)
    f_full = idealchans_index(fmin, fmax, ideal_res=0.01)

    for i in range(0, len(coarse), coarse_chunk_size):
        c_chunk = coarse[i:i+coarse_chunk_size]
        for j in range(0, len(f_full), fine_chunk_size):
            f_chunk = f_full[j:j+fine_chunk_size]
            R_chunk = response_mtx(c_chunk, f_chunk, U)
            np.save(f'R_c{i}_f{j}.npy', R_chunk)
            np.save(f'c_c{i}_f{j}.npy', c_chunk)
            np.save(f'f_c{i}_f{j}.npy', f_chunk)

# ------------------ Main ------------------ #

if __name__ == "__main__":
    t1 = time.time()
    if len(sys.argv) < 4:
        print("Usage: python Upchannelize.py <fmin> <fmax> <U>")
        sys.exit(1)

    print(f"Generating Response Matrix with Numba CUDA kernels")
    fmin = float(sys.argv[1])
    fmax = float(sys.argv[2])
    U = int(sys.argv[3])

    run_serial(fmin, fmax, U)

    t2 = time.time()
    print(f"Finished. Total Runtime {t2 - t1:.2f} seconds")
