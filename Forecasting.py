import Generate_Catalog as gen
import Galaxy_Functions as gf
import Gaussian_Estimate as gauss
import Plotting as plot
import numpy as np
import astropy.units as u
import os
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
import CHORD_Sensitivity as chord

def SNR_detections(MHI, W50, z, RMS, sigma=6):
    chan_width = (1500*u.MHz/8192).to_value(u.Hz)
    RMS = (RMS*u.mJy).to_value(u.Jy)
    W50_broad = gauss.W50_broadened(W50)
    SNR = gf.SNR_int(z, MHI, W50_broad, chan_width=chan_width, RMS_chan=RMS)
    mask = SNR >= sigma
    return mask

def SNRint_fromFile(catalog_file, RMS, sigma=6, plt=True, figname='', title='', maskFl='', integrated=True):
    catalog = np.load(catalog_file)
    mask = SNR_detections(MHI=catalog[0], W50=catalog[3], z=catalog[8], RMS=RMS, sigma=sigma)
    if plt==True:
        plot.Detection_counts_MHI(MHI=catalog[0], mask=mask, RMS=RMS, figname=figname, title=title)
    if maskFl!='':
        np.save(maskFl, catalog[:,mask])

def detections_ALFALFA(catalog_file, maskFl=''):
    catalog = np.load(catalog_file)
    W50_broad = gauss.W50_broadened(W50=catalog[3])
    S21, _ = gf.int_S21(MHI=catalog[0], z=catalog[8])
    S21_thALF = gf.S21th_ALFALFA(W50=W50_broad, SNR=6.5)
    mask = S21 > S21_thALF
    if maskFl!='':
        np.save(maskFl, catalog[:,mask])
    return mask

def detections_fromRMS(catalog_file, maskFl='', RMS=0.1, sigma=6):
    catalog = np.load(catalog_file)
    W50_broad = gauss.W50_broadened(W50=catalog[3])
    RMS = (RMS*u.mJy).to_value(u.Jy)
    S21, _ = gf.int_S21(MHI=catalog[0], z=catalog[8])
    S21_th = gf.S21_th(W50=W50_broad, RMS=RMS, sigma=sigma, chan_kms=48)
    mask = S21 >= S21_th
    if maskFl!='':
        np.save(maskFl, catalog[:,mask])
    return mask, S21, W50_broad

def detections_fromObs(catalog_file, maskFl, obsyears, nstrips, sigma=6):
    obsdays = 365*obsyears/nstrips
    RMS = chord.time2RMS(days=obsdays, decl=20, PB=True, nu=183*u.kHz, N=512)
    print("RMS is ", RMS)
    detections_fromRMS(catalog_file=catalog_file, maskFl=maskFl, RMS=RMS.value, sigma=sigma)

# def SNRint_varyingRMS(catalog_file, RMS, sigma=6, plot=True, figname='', title=''):
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     MHI, _, _, W50, _, _, _, _, z = gen.load_catalogParams(catalog_file)
#     base, ext = os.path.splitext(catalog_file)
#     mask, _ = SNRint_detections(MHI, W50, z, RMS=RMS, sigma=sigma)

#     plt.figure()
#     for i in range(len(RMS)):
        
#         np.save(base+f'_mask{RMS[i]}.npy', mask)
#         counts = MHI[mask].shape[0]
#         print("counts is ", counts)
#         plt.hist(np.log10(MHI[mask]), bins=30, histtype='step', label=f'RMS={RMS[i]} mJy, total={counts}')
#     plt.yscale('log')
#     plt.title(title)
#     plt.xlabel('log(MHI)')
#     plt.ylabel('log(Counts)')
#     plt.legend()
#     plt.savefig(figname)


def SNRint_varyingRMS(catalog_file, RMS, sigma=6, plot=True, figname='', title=''):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 loads full catalog
    if rank == 0:
        MHI, _, _, W50, _, _, _, _, z = gen.load_catalogParams(catalog_file)
        N = len(MHI)
    else:
        MHI = None
        W50 = None
        z = None
        N = 0

    # Broadcast total size
    N = comm.bcast(N, root=0)

    # Compute counts and displacements for Scatterv
    counts = np.array([(N // size) + (1 if i < N % size else 0) for i in range(size)], dtype=np.int32)
    displs = np.insert(np.cumsum(counts[:-1]), 0, 0)

    # Allocate receive buffers for each array chunk
    MHI_local = np.empty(counts[rank], dtype=np.float32)
    W50_local = np.empty(counts[rank], dtype=np.float32)
    z_local = np.empty(counts[rank], dtype=np.float32)

    # Scatter data using Scatterv (buffer-based)
    if rank == 0:
        comm.Scatterv([np.ascontiguousarray(MHI), counts, displs, MPI.FLOAT], MHI_local, root=0)
        comm.Scatterv([np.ascontiguousarray(W50), counts, displs, MPI.FLOAT], W50_local, root=0)
        comm.Scatterv([np.ascontiguousarray(z), counts, displs, MPI.FLOAT], z_local, root=0)
    else:
        comm.Scatterv([None, counts, displs, MPI.FLOAT], MHI_local, root=0)
        comm.Scatterv([None, counts, displs, MPI.FLOAT], W50_local, root=0)
        comm.Scatterv([None, counts, displs, MPI.FLOAT], z_local, root=0)

    # Timing and mask calculation per rank
    print(f"Started masking on rank[{rank}]")
    t1 = time.time()
    mask_local, _ = SNR_detections(MHI_local, W50_local, z_local, RMS=RMS, sigma=sigma)
    print(f"Completed masking on rank[{rank}] in {time.time() - t1:.2f} seconds")

    # Prepare to gather masks (boolean arrays)
    mask_counts = counts  # same counts apply to masks
    mask_displs = displs

    if rank == 0:
        mask_full = np.empty(N, dtype=mask_local.dtype)
    else:
        mask_full = None

    # Gather masks from all ranks
    comm.Gatherv(mask_local, [mask_full, mask_counts, mask_displs, MPI.BOOL], root=0)

    # Rank 0 saves combined mask
    if rank == 0:
        base, _ = os.path.splitext(catalog_file)
        np.save(f"{base}_mask_{RMS}.npy", mask_full)
        print(f"Saved combined mask for RMS={RMS}")


def compareCatalogs(catalog1, catalog2, RMS, sigma=6, plt=True, figname='', title=''):
    MHI1, W501, Dec1, mask1 = SNRint_fromFile(catalog_file=catalog1, RMS=RMS, sigma=sigma, plt=False)
    MHI2, W502, Dec2, mask2 = SNRint_fromFile(catalog_file=catalog2, RMS=RMS, sigma=sigma, plt=False)
    if plt:
        #plot.Detection_compareCats(MHI1=MHI1[mask1], MHI2=MHI2[mask2], figname=figname, title=title)
        plot.Detection_compare_Decs(MHI1[mask1], MHI2[mask2], Dec1=Dec1[mask1], Dec2=Dec2[mask2], figname=figname, title=title)


if __name__ == "__main__":
#    detections_fromObs(catalog_file='catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy',
#                        obsyears=5, nstrips=20, maskFl='DetectionsVolLim_zmax0p1_5yearObs_20strips_20to80deg.npy')
#    detections_fromRMS(catalog_file='catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy', sigma=6, RMS=0.1, maskFl='DetectionsVolLim_zmax0p1_fromRMS0p1_20to80deg.npy')
#    detections_ALFALFA(catalog_file='catalogs_output/VolLim_20to80deg_Dmax200_rank0.npy', maskFl='DetectionsALFALFA_20to80deg_Dmax200.npy')
#    SNRint_fromFile('catalogs_output/MockAlf_FullSky.npy', RMS=1, plt=False, maskFl='catalogs_output/maskRMS1_sigma6_MockAlf_FullSky.npy')
#    SNRint_fromFile('catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy', RMS=0.1, plt=False, 
#                    maskFl='catalogs_output/Detected_RMS0p1_VolLim_20to80deg_Dmax200.npy')
    SNRint_fromFile('catalogs_output/VolLim_20to60deg_zmin0p4_zmax1_MHI9to12.npy', RMS=0.08, plt=False, 
                   maskFl='catalogs_output/Detected_VolLim_RMS0p08__20to80deg_z0p4to1_MHI9to12.npy')
#    SNRint_fromFile('catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy', RMS=0.1, plt=False, maskFl='catalogs_output/DetectedRMS0p1_sigma6_VolLim_20to60deg_zmax0p1.npy')
#     # compareCatalogs(catalog1='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', 
#     #                 catalog2='catalogs_output/MockAlf_FullSkyD200_Dec20to80_ChangeVelocity.npy',
#     #                 RMS=2, figname='Plots/Compare_VolFluxCats_RMS2_D200_MHIbins.png', title='Detections at RMS=2 mJy')
#     # SNRint_dectections(catalog_file='catalogs_output/MockAlf_FullSkyD200_Dec20to80_ChangeVelocity.npy', 
#     #                    RMS=2, figname='Mock_ALFALFA_RMS2.png', title='Detections from Mock ALFALFA at sensitivity of 2 mJy')
    
#     SNRint_fromFile(catalog_file='catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy', 
#                        RMS=0.1, figname='VolLim_z0p1_RMS0p1.png', title='Detections from Volume limited at sensitivity of 0.1 mJy')
#    SNRint_varyingRMS(catalog_file='catalogs_output/VolLim_20to60deg_zmin0p4_zmax1_merged.npy', RMS=0.5, figname='VaryingRMS_Vollim_z0p4_zmax1.png')
#    SNRint_varyingRMS(catalog_file='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', RMS=0.2, figname='test_parallel.png')





