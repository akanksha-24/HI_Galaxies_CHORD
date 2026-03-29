import numpy as np
from Generate_Catalog import *
from Galaxy_Functions import *
import matplotlib.pyplot as plt 
from CHORD_Sensitivity import *
from Gaussian_Estimate import *
from Plotting import *
import sys
import signal

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

def Vmax_corr(MHI, z, RMS_mJy, W50_broad, chan_width, sigma, solidang):
    D = cosmo.comoving_distance(z).to_value(u.Mpc)
    Dsurvey = np.max(D)
    Dmin = np.min(D)
    Dmax = estimate_DLmax(MHI, z, sigma=sigma, RMS_chan=RMS_mJy, chan_width=chan_width, DeltaV=W50_broad)
    Dmax_comov = Dmax / (1 + z)
    Dmax_comov = np.minimum(Dmax_comov, Dsurvey)
    Vmax = VolumeFromDist(Dmax_comov, solidang=solidang, Dmin=Dmin)
    return Vmax

def MonteCarlo_HIMF(zmax=0.1, dec1=20, dec2=80, trials=1000, zmin=0, sigma=6, obs_year=5):
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
        MHI = catalog[0]
        W50 = catalog[3]
        D = catalog[6]
        z = catalog[8]
        W50_broad = W50_broadened(W50)
        dec = catalog[5]
        print("max dec is ", np.max(dec))
        print("min dec is ", np.min(dec))
        #RMS_mJy = RMS_fromDays(days=5*365/24, decl=np.deg2rad(20), z=z, nu=upchan_res*u.Hz).value
        #print("RMS values are", RMS_mJy)
        _, RMS_mJy, _ = build_survey(switch_int=7, obs_years=obs_year, z=z, start=dec1, end=dec2)
        SNR = SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=upchan_res)
        mask = SNR > 6
        #np.save('detected_test.npy', catalog[:,mask])
        Vmax = Vmax_corr(MHI[mask], z[mask], RMS_mJy[mask], W50_broad[mask], chan_width=upchan_res, sigma=sigma, solidang=solidang)
        counts_Vcorr, _ = np.histogram(np.log10(MHI[mask]), bins=bins, weights=1/Vmax)
        phi[i] = counts_Vcorr/binwidth
        counts, _ = np.histogram(np.log10(MHI[mask]), bins=bins)
        Counts[i] = counts
        z_Counts[i], _ = np.histogram(z[mask], bins=z_bins)

        if i % 10 == 0:
            np.save(f'catalogs_output/phi_counts_checkpoint_{obs_year}yr_z{zmax}_{i}.npy', np.asarray([phi, Counts]))
            np.save(f'catalogs_output/z_counts_checkpoint_{obs_year}yr_z{zmax}_{i}.npy', z_Counts)
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
    
if __name__ == "__main__":
#    signal.signal(signal.SIGUSR1, handler)
    MonteCarlo_HIMF(trials=1001, zmax=1, zmin=0, dec1=20, dec2=50, obs_year=1)
