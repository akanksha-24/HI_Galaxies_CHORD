import numpy as np
from Generate_Catalog import *
from Galaxy_Functions import *
import matplotlib.pyplot as plt 
from CHORD_Sensitivity import *
from Gaussian_Estimate import *
from Plotting import *
import sys
import signal
from Generate_Spectra import *
from mpi4py import MPI

# exit_now = False

# def handler(signum, frame):
#     global exit_now
#     print("SLURM time limit approaching — saving and exiting...")
#     exit_now = True

def choose_SchechParams(alpha_=-1.25, del_alpha=0.1, M_s_=10**9.94, del_M_s=10**9.94*np.log(10)*0.051,
                        phi_s_=4.5e-3, del_phi_s=np.sqrt(0.2**2 + 0.8**2)*1e-3):
    alpha = np.random.normal(loc=alpha_, scale=del_alpha, size=1)
    M_s = np.random.normal(loc=M_s_, scale=del_M_s, size=1)
    phi_s = np.random.normal(loc=phi_s_, scale=del_phi_s, size=1)
    return alpha, M_s, phi_s

def Vmax_corr(MHI, z, RMS_mJy, W50_broad, chan_width, sigma, solidang, spectra=False, SNR=None):
    D = cosmo.comoving_distance(z).to_value(u.Mpc)
    Dsurvey = np.max(D)
    Dmin = np.min(D)
    if spectra:
        Dmax = D*np.sqrt(SNR/6) # SNR = S21/(Slim/6)
    else:
        Dmax = estimate_DLmax(MHI, z, sigma=sigma, RMS_chan=RMS_mJy, chan_width=chan_width, DeltaV=W50_broad)
    Dmax_comov = Dmax / (1 + z)
    Dmax_comov = np.minimum(Dmax_comov, Dsurvey)
    Vmax = VolumeFromDist(Dmax_comov, solidang=solidang, Dmin=Dmin)
    return Vmax

def Spectra_SNRint(catalog, RMS, prevSNR=None, plotSpectra=True):
    #print("Detections shape is ", np.transpose(catalog).shape)
    # if plotSpectra:
    #     catalog_T = np.transpose(catalog)
    #     np.random.shuffle(catalog_T)
    #     catalog = np.transpose(catalog_T)
    #     print("final catalog shape is ", catalog.shape)
    print("catalog shape is ", catalog.shape)
    MHI = catalog[0]
    W50 = catalog[3]
    D = catalog[6]
    z = catalog[8]
    dec = catalog[5]
    SNR_int = Generate_Spectra(size=1, MHI=np.asarray(MHI), W50=np.asarray(W50), 
                               D_C=np.asarray(D), z=np.asarray(z), RMS=np.asarray(RMS), prevSNR=np.asarray(prevSNR), plotSpectra=plotSpectra)
    
    #SNR_int = Generate_Spectra(size=catalog.shape[1], MHI=MHI, W50=W50, D_C=D, z=z)
    return SNR_int

def MonteCarlo_HIMF(zmax=0.1, dec1=20, dec2=80, trials=1000, zmin=0, sigma=6, obs_year=5, Spectra=False):
    upchan_res = width_vel2freq(del_Vrest=5)
    #RMS_mJy = time2RMS(days=5*365/24, decl=np.deg2rad(20), nu=upchan_res*u.Hz).value
    solidang = solid_angle(dec1=dec1, dec2=dec2, ra1=0, ra2=360)
    print("solidang is ", solidang)
    bins = np.arange(4.9, 12.2, 0.2)
    binwidth = (bins[1:])-(bins[:-1])
    bin_centers = mid_bin(bins)

    z_step = 0.008
    z_bins = np.arange(-1*z_step/2, zmax+z_step, z_step)
    
    phi = np.zeros((trials, len(bins)-1))
    Counts = np.zeros((trials, len(bins)-1))
    z_Counts = np.zeros((trials, len(z_bins)-1))

    for i in np.arange(trials):
        print(i)
        # if exit_now:
        #     print("Saving checkpoint before timeout...")
        #     np.save('catalogs_output/phi_counts_z0p8_1yr_1000_until{i}.npy',
        #             np.asarray([phi, Counts]))
        #     np.save('catalogs_output/counts_z0p8_1yr_1000_unit{i}.npy',
        #             z_Counts)
        #     print("Done. Exiting.")
        #     sys.exit(0)

        alpha, M_s, phi_s = choose_SchechParams()
        HIMF = schechter_fit_lg(MHI_grid, phi_s=phi_s, M_s=M_s, alpha=alpha)
        # plt.figure()
        # plt.plot(np.log10(MHI_grid), np.log10(HIMF))
        #plt.plot(np.log10(MHI_grid), np.log10(HIMF_Jones2018(MHI=MHI_grid)), label='HIMF - drawn from, Jones2018')
        catalog = Gen_Catalog(zmax=zmax, dec1=dec1, dec2=dec2, zmin=zmin, save=False, 
                              phi_s=phi_s, alpha=alpha, M_s=M_s, Fluxlim=True, obs_year=obs_year)
        # cat2 = Gen_Catalog(zmax=1, dec1=dec1, dec2=dec2, zmin=0.3, save=False, 
        #                       phi_s=phi_s, alpha=alpha, M_s=M_s, Fluxlim=True, obs_year=obs_year)
        # print("cat 1 shape is ", cat1.shape)
        # print("cat 2 shape is ", cat2.shape)
        # catalog = np.concatenate((cat1, cat2), axis=1)
        print("catalog shape is ", catalog.shape)
        MHI = catalog[0]
        W50 = catalog[3]
        D = catalog[6]
        z = catalog[8]
        W50_broad = W50_broadened(W50)
        dec = catalog[5]
        #print("max dec is ", np.max(dec))
        #print("min dec is ", np.min(dec))
        #RMS_mJy = RMS_fromDays(days=5*365/24, decl=np.deg2rad(20), z=z, nu=upchan_res*u.Hz).value
        #print("RMS values are", RMS_mJy)
        _, RMS_mJy, _ = build_survey(switch_int=7, obs_years=obs_year, z=z, start=dec1, end=dec2)
        SNR = SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=upchan_res)
        SNR_spectra = SNR.copy()
        mask = SNR > 6
        print("number of detections without spectra", catalog[:,mask].shape[1])
        mask = (SNR > 6) & (SNR < 8)
        print("number of detections to generate spectra", catalog[:,mask].shape[1])
        #np.save('detected_test.npy', catalog[:,mask])
        for j in range(SNR.shape[0]):
            if (SNR[j] > 6) and SNR[j] < 8:
                #print("prevSNR is ", SNR[i])
                SNR_ = Generate_Spectra(size=1, MHI=np.asarray([MHI[j]]), W50=np.asarray([W50[j]]), 
                               D_C=np.asarray([D[j]]), z=np.asarray([z[j]]), RMS=np.asarray([RMS_mJy[j]]),
                                 prevSNR=np.asarray([SNR[j]]), plotSpectra=False)[0]
                #SNR_ = Spectra_SNRint(catalog=catalog[:,i], RMS=RMS_mJy[i], prevSNR=SNR[i])
                #print("factor difference", SNR[i]/SNR_)
                if SNR_==0:
                    print("0 SNR prev was ", SNR[j])
                if SNR_ < 6:
                    SNR_spectra[j] = SNR_
        mask = SNR_spectra > 6
        print("number of detections with spectra: ", catalog[:,mask].shape[1])
        Vmax = Vmax_corr(MHI[mask], z[mask], RMS_mJy[mask], W50_broad[mask], 
                         chan_width=upchan_res, sigma=sigma, solidang=solidang)
        counts_Vcorr, _ = np.histogram(np.log10(MHI[mask]), bins=bins, weights=1/Vmax)
        phi[i] = counts_Vcorr/binwidth
        counts, _ = np.histogram(np.log10(MHI[mask]), bins=bins)
        Counts[i] = counts
        z_Counts[i], _ = np.histogram(z[mask], bins=z_bins)

        np.save(f'catalogs_output/1run_phi_counts_checkpoint_wspectra_{obs_year}yr_z{zmax}_{i}.npy', np.asarray([phi, Counts]))
        np.save(f'catalogs_output/1run_z_counts_checkpoint_wspectra_{obs_year}yr_z{zmax}_{i}.npy', z_Counts)

        if i % 10 == 0:
            np.save(f'catalogs_output/phi_counts_checkpoint_wspectra_{obs_year}yr_z{zmax}_{i}.npy', np.asarray([phi, Counts]))
            np.save(f'catalogs_output/z_counts_checkpoint_wspectra_{obs_year}yr_z{zmax}_{i}.npy', z_Counts)
        #print("Counts is ", Counts[i])
        # plt.scatter(bin_centers, np.log10(phi[i]))
        # plt.savefig('Plots/trial1_phicounts.png')
        # plt.figure()
        # plt.plot(mid_bin(z_bins), z_Counts[i])
        # plt.yscale('log')
        # plt.savefig('Plots/trial1_counts_z.png')
        #recover_HIMF(catalog_fl='detected_test.npy', Vollim=False, RMS=RMS_mJy, sigma=6, bins=bins, figname='Plots/test_detections.png')
    #plt.savefig('Plots/HIMF_trials3.png')
    
    #np.save('catalogs_output/phi_counts_z0p8_1yr_1000.npy', np.asarray([phi, Counts]))
    #np.save('catalogs_output/counts_z0p8_1yr_1000.npy', z_Counts)
    #arr = np.load('catalogs_output/phi_counts.npy')
    #print(arr.shape)
    
def MonteCarlo_HIMF_parallel(zmax=0.1, dec1=20, dec2=80, trials=1000, zmin=0, sigma=6, obs_year=5, Spectra=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    upchan_res = width_vel2freq(del_Vrest=5)
    solidang = solid_angle(dec1=dec1, dec2=dec2, ra1=0, ra2=360)
    #print("solidang is ", solidang)
    bins = np.arange(4.9, 12.2, 0.2)
    binwidth = (bins[1:])-(bins[:-1])
    bin_centers = mid_bin(bins)

    z_step = 0.008
    z_bins = np.arange(-1*z_step/2, zmax+z_step, z_step)
    
    if rank == 0:
        phi = np.zeros((trials, len(bins) - 1))
        Counts = np.zeros((trials, len(bins) - 1))
        z_Counts = np.zeros((trials, len(z_bins) - 1))
    else:
        phi = None
        Counts = None
        z_Counts = None

    for i in np.arange(trials):   
        if rank==0:
            print(f"Trial {i}", flush=True)
            alpha, M_s, phi_s = choose_SchechParams()
        else:
            alpha = None
            M_s = None
            phi_s = None

        alpha, M_s, phi_s = comm.bcast((alpha, M_s, phi_s),root=0)

        seed_sequence = np.random.SeedSequence([12345, int(i), rank])
        rank_seed = seed_sequence.generate_state(1)[0]
        np.random.seed(rank_seed)

        catalog = Gen_Catalog(zmax=zmax, dec1=dec1, dec2=dec2, zmin=zmin, save=False, 
                            phi_s=phi_s, alpha=alpha, M_s=M_s, Fluxlim=True, obs_year=obs_year, 
                            comm=comm, gather=False)

        MHI = catalog[0]
        W50 = catalog[3]
        D = catalog[6]
        z = catalog[8]
        W50_broad = W50_broadened(W50)
        dec = catalog[5]
        _, RMS_mJy, _ = build_survey(switch_int=7, obs_years=obs_year, z=z, start=dec1, end=dec2)
        SNR = SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=upchan_res)
        #mask = SNR > 6
        #print("number of detections without spectra", catalog[:,mask].shape[1],flush=True)
        mask = (SNR > 6) & (SNR < 8)
        print("number of detections to generate spectra", catalog[:,mask].shape[1], flush=True)
        candidate_indices = np.where((SNR > 6) & (SNR < 8))[0]
        SNR_spectra = SNR.copy()

        if Spectra: 
            for j in candidate_indices:
                SNR_busy = Generate_Spectra(size=1, MHI=np.asarray([MHI[j]]), W50=np.asarray([W50[j]]), 
                                D_C=np.asarray([D[j]]), z=np.asarray([z[j]]), RMS=np.asarray([RMS_mJy[j]]),
                                    prevSNR=np.asarray([SNR[j]]), plotSpectra=False)[0]
                #SNR_ = Spectra_SNRint(catalog=catalog[:,i], RMS=RMS_mJy[i], prevSNR=SNR[i])
                #print("factor difference", SNR[i]/SNR_)
                if SNR_busy==0:
                    print("0 SNR prev was ", SNR[j], flush=True)

                #if SNR_busy < 6:
                SNR_spectra[j] = SNR_busy

        mask = SNR_spectra > sigma
        Vmax = Vmax_corr(MHI[mask], z[mask], RMS_mJy[mask], W50_broad[mask], SNR=SNR_spectra[mask],
                        chan_width=upchan_res, sigma=sigma, solidang=solidang, spectra=Spectra)
        #counts_Vcorr, _ = np.histogram(np.log10(MHI[mask]), bins=bins, weights=1/Vmax)
        #local_phi = counts_Vcorr/binwidth
        local_phi, _ = np.histogram(np.log10(MHI[mask]), bins=bins, weights=1/Vmax)
        local_Counts, _ = np.histogram(np.log10(MHI[mask]), bins=bins)
        local_zCounts, _ = np.histogram(z[mask], bins=z_bins)

        global_phi_sum = comm.reduce(local_phi,op=MPI.SUM,root=0)
        global_counts = comm.reduce(local_Counts,op=MPI.SUM,root=0)
        global_z_counts = comm.reduce(local_zCounts,op=MPI.SUM,root=0)

        if rank==0:
            phi[i] = global_phi_sum / binwidth
            Counts[i] = global_counts
            z_Counts[i] = global_z_counts

            #np.save(f'catalogs_output/1run_phi_counts_checkpoint_wspectra_{obs_year}yr_z{zmax}_{i}.npy', np.asarray([phi, Counts]))
            #np.save(f'catalogs_output/1run_z_counts_checkpoint_wspectra_{obs_year}yr_z{zmax}_{i}.npy', z_Counts)

            if i % 10 == 0:
                np.save(f'catalogs_output/phi_counts_checkpoint_wspectra_{obs_year}yr_z{zmax}_{i}.npy', np.asarray([phi, Counts]))
                np.save(f'catalogs_output/z_counts_checkpoint_wspectra_{obs_year}yr_z{zmax}_{i}.npy', z_Counts)
        #print("Counts is ", Counts[i])
            plt.figure()
            grid = np.logspace(5,11,100000)
            HIMF = schechter_fit_lg(grid, phi_s=phi_s, M_s=M_s, alpha=alpha)
            plt.plot(np.log10(grid), HIMF)
            plt.plot(np.log10(grid), HIMF_Jones2018(MHI=grid), color='blue')
            plt.scatter(bin_centers, phi[i])
            plt.yscale('log')
            plt.show()
            #plt.savefig('Plots/HIMF_check.png')
            #plt.close()
            # 
        #plt.plot(np.log10(MHI_grid), np.log10(HIMF_Jones2018(MHI=MHI_grid)), label='HIMF - drawn from, Jones2018')
        # plt.savefig('Plots/trial1_counts_z.png')
        #recover_HIMF(catalog_fl='detected_test.npy', Vollim=False, RMS=RMS_mJy, sigma=6, bins=bins, figname='Plots/test_detections.png')
    #plt.savefig('Plots/HIMF_trials3.png')
    
    #np.save('catalogs_output/phi_counts_z0p8_1yr_1000.npy', np.asarray([phi, Counts]))
    #np.save('catalogs_output/counts_z0p8_1yr_1000.npy', z_Counts)
    #arr = np.load('catalogs_output/phi_counts.npy')
    #print(arr.shape)

if __name__ == "__main__":
#    signal.signal(signal.SIGUSR1, handler)
    start = time.time()
    MonteCarlo_HIMF_parallel(trials=1001, zmax=1, zmin=0, dec1=20, dec2=80, obs_year=5, Spectra=True)
    end = time.time()
    print("Runtime is ", end-start, flush=True)
