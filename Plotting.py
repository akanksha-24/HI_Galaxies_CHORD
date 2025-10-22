import numpy as np
import matplotlib.pyplot as plt
import Galaxy_Functions as gf
import astropy.units as u
import astropy.constants as c
from matplotlib.backends.backend_pdf import PdfPages

def param_distributions(catalog, n_bins=20, flname=''):
    catalog = np.load(catalog)
    extn = ['MHI_lg', 'VHI', 'Incl', 'W50', 'RA', 'Dec', 'Distace', 'Volume', 'Redshift']
    xlabels = ['log(M$_{HI}$) [M$_{\odot}$]', 'V$_{HI}$ (km/s)', 'Inclination (deg)', 'W50 (km/s)',
               'Right Ascension (deg)', 'Declination (deg)', 'Distance (Mpc)', 'Volume (Mpc$^3$)', 'Redshift (z)']

    for i in np.arange(len(extn)):
        plt.figure(figsize=[4,3], dpi=300)
        plt.hist(catalog[extn[i]], bins=n_bins, histtype='step')
        if i==1 or i==3: # VHI or W50
            plt.yscale('log')
        plt.xlabel(xlabels[i])
        plt.tight_layout()
        plt.savefig(flname+'_'+extn[i]+'.png')

def MHI_VHI_polynomial(MHI=np.logspace(5,11,100000)):
    VHI = gf.VHI_polyFit(MHI)
    plt.figure()
    plt.plot(np.log10(MHI), VHI)
    plt.savefig('MHI_VHI.png')
    plt.show()
    
def recover_HIMF(catalog, nbins=30, VolLim=True, S21lim=None, Dmax=None, ra1=0, ra2=360, dec1=20, dec2=80, MHI=gf.MHI_grid):
    HIMF_lg = gf.schechter_fit_lg(MHI=MHI)
    solidang = gf.solid_angle(dec1, dec2, ra1, ra2)
    
    samples = np.load(catalog)
    MHI_samples = samples[:,0]
    lg_MHI = np.log10(MHI_samples)
    print("max MHI is ", np.max(lg_MHI))
    print("min MHI is ", np.min(lg_MHI))
    #Vol_drawn = samples[:,9]
    
    if VolLim:
        if Dmax==None:
            Dmax=np.max(samples[:,2])*u.Mpc
        counts, bins = np.histogram(lg_MHI, bins=nbins)
        Vmax = gf.VolumeFromDist(Dmax, solidang=solidang).value
        binwidth = (bins[1:])-(bins[:-1])
        phi = np.log10(counts/(Vmax*binwidth))
        #phi = np.log10(counts/(Vmax*(10**binwidth)))
    
    else:
        W50 = samples[:,1]
        D = samples[:,2]
        S21 = gf.MHI_toS(MHI=(10**lg_MHI)*u.solMass, delV=48*u.km/u.s, D=D*u.Mpc)
        Vmax = gf.Vmax_correct(D*u.Mpc, S21, S21lim, solidang).value
        counts_Vcorr, bins = np.histogram(lg_MHI, bins=nbins, weights=1/Vmax)
        binwidth = (bins[1:])-(bins[:-1])
        phi = np.log10(counts_Vcorr/(10**binwidth))

    bin_centers = gf.mid_bin(bins)
    
    plt.figure(figsize=[5,3],dpi=200)
    plt.plot(np.log10(MHI), np.log10(HIMF_lg), label='HIMF - drawn from, Jones2018') # HIMF
    plt.scatter(bin_centers, phi, s=3, label='HIMF - recovered', color='black') # from sample
    plt.ylabel('$\phi(M_{HI})$')
    plt.xlabel('log(M$_{HI}$/M$_{\odot}$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('HIMF_recover.png')
    plt.show()

def dndz(catalog=None, N=None, z=None, flname='', compareHans=''):
    if catalog==None:
        dz = z[1] - z[0]
        dndz = N / dz
        plt.figure()
        plt.plot(z, dndz)
        if compareHans!='':
            predictions = np.load(compareHans)
            plt.plot(predictions[0], predictions[1], linestyle='--', color='black', label='Hans Prediction')
        plt.savefig(flname)
        plt.xlabel('Redshift z')
        plt.ylabel('dn/dz')
        plt.legend()
        plt.savefig(flname)
        plt.show()

def check_Spectra(MHI_true, MHI_gen, W50, V_kms, S_Jy, freq_MHz, z, D, fname, size=None, FWHM_thermal=10):
    with PdfPages(fname) as pdf_:
        S_mJy = S_Jy * 1000
        mask = W50 < FWHM_thermal
        W50[mask] = np.sqrt(W50[mask]**2 + FWHM_thermal**2)
        f50 = W50 * gf.df_dv(W50, z)
        fobs = gf.get_fobs(z)
        if size is None: size = len(MHI_true)

        for i in range(size):
            plt.close('all')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[4,6], dpi=250)

            # Velocity axis spectra plot
            ax1.plot(V_kms[i], S_mJy[i])
            ax1.set_title(f'log(MHI)={np.log10(MHI_true[i]):.2f}, log(MHI_spec)={np.log10(MHI_gen[i]):.2f}, z={z[i]:.2f}, D={D[i]:.0f} Mpc', fontsize=9)
            ax1.text(0.01, 0.8, f"W50={W50[i]:.0f} km/s", transform=ax1.transAxes)
            ax1.set_xlabel('Velocity (km/s)')
            ax1.set_ylabel('Flux Density (mJy)')
            ax1.axvline(W50[i]/2, linestyle='--', linewidth=0.7, color='black')
            ax1.axvline(-W50[i]/2, linestyle='--', linewidth=0.7, color='black')

            # Frequency axis spectra plot
            ax2.plot(freq_MHz[i], S_mJy[i])
            ax2.text(0.01, 0.8, f"fobs={fobs[i]:.0f} MHz", transform=ax2.transAxes)
            ax2.set_xlabel('Frequency (MHz)')
            ax2.set_ylabel('Flux Density (mJy)')
            ax2.axvline(fobs[i]+f50[i]/2, linestyle='--', linewidth=0.7, color='black')
            ax2.axvline(fobs[i]-f50[i]/2, linestyle='--', linewidth=0.7, color='black')
            ax2.axvline(fobs[i], linestyle='--', linewidth=0.7, color='blue')

            plt.tight_layout()
            pdf_.savefig(fig)
            plt.close(fig)

# def Forecast_counts(MHI_lg, mask):
#     catalog = np.load(catalog)
#     extn = ['MHI_lg', 'VHI', 'Incl', 'W50', 'RA', 'Dec', 'Distace', 'Volume', 'Redshift']
#     xlabels = ['log(M$_{HI}$) [M$_{\odot}$]', 'V$_{HI}$ (km/s)', 'Inclination (deg)', 'W50 (km/s)',
#                'Right Ascension (deg)', 'Declination (deg)', 'Distance (Mpc)', 'Volume (Mpc$^3$)', 'Redshift (z)']

#     for i in np.arange(len(extn)):
#         plt.figure(figsize=[4,3], dpi=300)
#         plt.hist(catalog[extn[i]], bins=n_bins, histtype='step')
#         if i==1 or i==3: # VHI or W50
#             plt.yscale('log')
#         plt.xlabel(xlabels[i])
#         plt.tight_layout()
#         plt.savefig(flname+'_'+extn[i]+'.png')


