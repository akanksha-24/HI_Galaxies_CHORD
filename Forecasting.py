import Generate_Catalog as gen
import Galaxy_Functions as gf
import Gaussian_Estimate as gauss
import Plotting as plot
import numpy as np
import astropy.units as u
import os
import matplotlib.pyplot as plt
import time

def SNRint_detections(MHI, W50, z, RMS, sigma=6):
    chan_width = (1500*u.MHz/8192).to_value(u.Hz)
    RMS = (RMS*u.mJy).to_value(u.Jy)
    W50_broad = gauss.W50_broadened(W50)
    SNRint = gf.SNR_int(z, MHI, W50_broad, chan_width=chan_width, RMS_chan=RMS)
    mask = SNRint >= sigma
    return mask, W50_broad

def SNRint_fromFile(catalog_file, RMS, sigma=6, plt=True, figname='', title=''):
    MHI, _, _, W50, _, Dec, _, _, z = gen.load_catalogParams(catalog_file)
    mask, W50_broad = SNRint_detections(MHI, W50, z)
    if plt==True:
        plot.Detection_counts_MHI(MHI, mask, RMS, figname=figname, title=title)
    return MHI, W50_broad, Dec, mask 

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
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 loads the full catalog
    if rank == 0:
        MHI, _, _, W50, _, _, _, _, z = gen.load_catalogParams(catalog_file)
        N = len(MHI)
    else:
        MHI = W50 = z = None
        N = 0

    # Broadcast total size to everyone
    N = comm.bcast(N, root=0)

    # Each rank determines its slice
    counts = N // size
    remainder = N % size
    start = rank * counts + min(rank, remainder)
    stop = start + counts + (1 if rank < remainder else 0)

    # Rank 0 sends out its data chunks manually
    if rank == 0:
        for r in range(1, size):
            r_start = r * counts + min(r, remainder)
            r_stop = r_start + counts + (1 if r < remainder else 0)
            comm.send(MHI[r_start:r_stop], dest=r, tag=0)
            comm.send(W50[r_start:r_stop], dest=r, tag=1)
            comm.send(z[r_start:r_stop], dest=r, tag=2)
        MHI = MHI[start:stop]
        W50 = W50[start:stop]
        z = z[start:stop]
    else:
        MHI = comm.recv(source=0, tag=0)
        W50 = comm.recv(source=0, tag=1)
        z = comm.recv(source=0, tag=2)

    # --- now each rank has its chunk ---
    print(f"Started masking on rank[{rank}]")
    t1 = time.time()
    mask, _ = SNRint_detections(MHI, W50, z, RMS=RMS, sigma=sigma)
    print(f"Completed masking on rank[{rank}] in {time.time()-t1} seconds]")

    # Gather masks if needed
    all_masks = comm.gather(mask, root=0)
    if rank == 0:
        mask_full = np.concatenate(all_masks)
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
#     # compareCatalogs(catalog1='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', 
#     #                 catalog2='catalogs_output/MockAlf_FullSkyD200_Dec20to80_ChangeVelocity.npy',
#     #                 RMS=2, figname='Plots/Compare_VolFluxCats_RMS2_D200_MHIbins.png', title='Detections at RMS=2 mJy')
#     # SNRint_dectections(catalog_file='catalogs_output/MockAlf_FullSkyD200_Dec20to80_ChangeVelocity.npy', 
#     #                    RMS=2, figname='Mock_ALFALFA_RMS2.png', title='Detections from Mock ALFALFA at sensitivity of 2 mJy')
    
#     SNRint_fromFile(catalog_file='catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy', 
#                        RMS=0.1, figname='VolLim_z0p1_RMS0p1.png', title='Detections from Volume limited at sensitivity of 0.1 mJy')
#    SNRint_varyingRMS(catalog_file='catalogs_output/VolLim_20to60deg_zmin0p4_zmax1_merged.npy', figname='VaryingRMS_Vollim_z0p4_zmax1.png')
    SNRint_varyingRMS(catalog_file='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', RMS=0.2, figname='test_parallel.png')





