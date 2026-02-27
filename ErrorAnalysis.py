import numpy as np
from Generate_Catalog import *
from Galaxy_Functions import *
import matplotlib.pyplot as plt 
from CHORD_Sensitivity import *
from Gaussian_Estimate import *
from Plotting import *

def choose_SchechParams(alpha_=-1.25, del_alpha=0.1, M_s_=10**9.94, del_M_s=10**9.94*np.log(10)*0.051,
                        phi_s_=4.5e-3, del_phi_s=np.sqrt(0.2**2 + 0.8**2)*1e-3):
    alpha = np.random.normal(loc=alpha_, scale=del_alpha, size=1)
    M_s = np.random.normal(loc=M_s_, scale=del_M_s, size=1)
    phi_s = np.random.normal(loc=phi_s_, scale=del_phi_s, size=1)
    return alpha, M_s, phi_s

def Vmax_corr(MHI, z, RMS_mJy, W50_broad, chan_width, sigma, solidang):
    D = cosmo.comoving_distance(z).to_value(u.Mpc)
    Dsurvey = np.max(D)
    Dmax = estimate_DLmax(MHI, z, sigma=sigma, RMS_chan=RMS_mJy, chan_width=chan_width, DeltaV=W50_broad)
    Dmax_comov = Dmax / (1 + z)
    Dmax_comov = np.minimum(Dmax_comov, Dsurvey)
    Vmax = VolumeFromDist(Dmax_comov, solidang=solidang)
    return Vmax

def MonteCarlo_HIMF(zmax=0.1, dec1=20, dec2=80, trials=1000, zmin=0, sigma=6):
    upchan_res = width_vel2freq(del_Vrest=5)
    RMS_mJy = time2RMS(days=5*365/24, decl=np.deg2rad(20), nu=upchan_res*u.Hz).value
    #print("RMS is ", RMS_mJy)
    solidang = solid_angle(dec1=dec1, dec2=dec2, ra1=0, ra2=360)
    #print("solidang is ", solidang)
    bins = np.linspace(5, 11, 31)
    binwidth = (bins[1:])-(bins[:-1])
    bin_centers = mid_bin(bins)
    
    phi = np.zeros((trials, len(bins)-1))
    Counts = np.zeros((trials, len(bins)-1))

    #plt.figure()
    for i in np.arange(trials):
        print(i)
        alpha, M_s, phi_s = choose_SchechParams()
        HIMF = schechter_fit_lg(MHI_grid, phi_s=phi_s, M_s=M_s, alpha=alpha)
        #plt.plot(np.log10(MHI_grid), np.log10(HIMF))
        #plt.plot(np.log10(MHI_grid), np.log10(HIMF_Jones2018(MHI=MHI_grid)), label='HIMF - drawn from, Jones2018')
        catalog = Gen_Catalog(zmax=zmax, dec1=dec1, dec2=dec2, zmin=zmin, save=False, 
                              phi_s=phi_s, alpha=alpha, M_s=M_s)
        MHI = catalog[0]
        W50 = catalog[3]
        D = catalog[6]
        z = catalog[8]
        W50_broad = W50_broadened(W50)
        SNR = SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=upchan_res)
        mask = SNR > 6
        #np.save('detected_test.npy', catalog[:,mask])
        Vmax = Vmax_corr(MHI[mask], z[mask], RMS_mJy, W50_broad[mask], chan_width=upchan_res, sigma=sigma, solidang=solidang)
        counts_Vcorr, _ = np.histogram(np.log10(MHI[mask]), bins=bins, weights=1/Vmax)
        phi[i] = counts_Vcorr/binwidth
        counts, _ = np.histogram(np.log10(MHI[mask]), bins=bins)
        Counts[i] = counts
        #plt.scatter(bin_centers, np.log10(phi[i]))
        #recover_HIMF(catalog_fl='detected_test.npy', Vollim=False, RMS=RMS_mJy, sigma=6, bins=bins, figname='Plots/test_detections.png')
    #plt.show()

    np.save('catalogs_output/phi_counts_z0p1_1000.npy', np.asarray([phi, Counts]))
    #arr = np.load('catalogs_output/phi_counts.npy')
    #print(arr.shape)
    
    

if __name__ == "__main__":
    MonteCarlo_HIMF(trials=1000, zmax=0.1)
