import numpy as np
import matplotlib.pyplot as plt
import Galaxy_Functions as gf
import astropy.units as u
import astropy.constants as c
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import cm
import Generate_Catalog as gen

def param_distributions(catalog, n_bins=20, flname=''):
    catalog = np.load(catalog)
    extn = ['MHI_lg', 'VHI', 'Incl', 'W50', 'RA', 'Dec', 'Distace', 'Volume', 'Redshift']
    xlabels = ['log(M$_{HI}$) [M$_{\odot}$]', 'V$_{HI}$ (km/s)', 'Inclination (deg)', 'W50 (km/s)',
               'Right Ascension (deg)', 'Declination (deg)', 'Distance (Mpc)', 'Volume (Mpc$^3$)', 'Redshift (z)']

    for i in np.arange(len(extn)):
        plt.figure(figsize=[4,3], dpi=300)
        plt.hist(catalog[i], bins=n_bins, histtype='step')
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

def Sky_MHIsizeDcolour(catalog_file, show=True, save=False, fname=''):
    MHI, _, _, _, ra, dec, D, _, _ = gen.load_catalogParams(catalog_file)
    ra_rad = np.deg2rad(ra)
    ra_rad = np.remainder(ra_rad + np.pi, 2 * np.pi) - np.pi  # center at 0
    dec_rad = np.deg2rad(dec)

    plt.figure(figsize=[8,6], dpi=200)
    ax = plt.axes(projection='aitoff')
    ax.grid(linewidth=0.5)
    sc = ax.scatter(ra_rad, dec_rad, #transform=ax.get_transform('world'), 
        s=(MHI/2e8).astype(int), c=D, cmap='viridis', alpha=0.5)
    plt.colorbar(sc, fraction=0.8, pad=0.1, label='Distance (Mpc)', location='bottom')
    sc.legend_elements(prop="sizes", alpha=0.6, num=3)
    handles, labels = sc.legend_elements(prop="sizes", alpha=0.6, num=3)    
    labels = ["$M_{HI}=1x10^{10}M_{\odot}$"]     
    legend = ax.legend(handles, labels, loc=[-0.1,1.1])
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")

    if save:
        plt.savefig(fname)
    if show:
        plt.show()

def Sky_MHIbins(catalog_file, fname='', MHI_bins=[6.5,8,9,10,11]):
    MHI, _, _, _, ra, dec, _, _, _ = gen.load_catalogParams(catalog_file)
    ra_rad = np.deg2rad(ra)
    ra_rad = np.remainder(ra_rad + np.pi, 2 * np.pi) - np.pi  # center at 0
    dec_rad = np.deg2rad(dec)
    color = cm.rainbow(np.linspace(0, 1, len(MHI_bins)-1))
    
    with PdfPages(fname) as pdf_:
        for i, c in enumerate(color):
            mask = (np.log10(MHI) >= MHI_bins[i]) & (np.log10(MHI) <= MHI_bins[i+1]) 
            fig = plt.figure(figsize=[8,6], dpi=200)
            ax = plt.axes(projection='aitoff')
            ax.grid(linewidth=0.5)
            sc = ax.scatter(ra_rad[mask], dec_rad[mask], alpha=0.5, color=c, s=2, label=(f'{MHI_bins[i]}-{MHI_bins[i+1]}'))
            ax.legend(loc='lower center', fontsize=8)
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
            ax.set_title(f'{MHI_bins[i]} < log(M_HI) < {MHI_bins[i+1]}', pad=15)

            pdf_.savefig(fig)
            plt.close(fig)

def Dec_Structure_MHI(catalog_file, n_bins=20, fname='', MHI_bins=[6.5,8,9,10,11]):
    MHI, _, _, _, _, dec, _, _, _ = gen.load_catalogParams(catalog_file)
    color = cm.rainbow(np.linspace(0, 1, len(MHI_bins)-1))

    plt.figure(figsize=[4,3], dpi=300)
    for i, c in enumerate(color):
        mask = (np.log10(MHI) >= MHI_bins[i]) & (np.log10(MHI) <= MHI_bins[i+1]) 
        plt.hist(dec[mask], bins=n_bins, histtype='step', label=f'{MHI_bins[i]} < log(M_HI) < {MHI_bins[i+1]}', color=c)
    plt.yscale('log')
    plt.xlabel('Declination (Deg)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.savefig(fname, bbox_inches='tight')

def Dec_Structure_MHI_pdf(catalog_file, n_bins=20, fname='', MHI_bins=[6.5,8,9,10,11]):
    MHI, _, _, _, _, dec, _, _, _ = gen.load_catalogParams(catalog_file)
    color = cm.rainbow(np.linspace(0, 1, len(MHI_bins)-1))

    with PdfPages(fname) as pdf_:
        for i, c in enumerate(color):
            fig = plt.figure(figsize=[4,3], dpi=300)
            mask = (np.log10(MHI) >= MHI_bins[i]) & (np.log10(MHI) <= MHI_bins[i+1]) 
            plt.hist(dec[mask], bins=n_bins, color=c)
            #plt.yscale('log')
            plt.xlabel('Declination (Deg)')
            plt.ylabel('Counts')
            plt.title(f'{MHI_bins[i]} < log(M_HI) < {MHI_bins[i+1]}')
            plt.tight_layout()
            pdf_.savefig(fig)
            plt.close(fig)

def Detection_counts_MHI(MHI, mask, RMS, n_bins=30, figname='', title=''):
    plt.figure()
    plt.hist(np.log10(MHI), bins=n_bins, histtype='step', label='Full Catalog')
    plt.hist(np.log10(MHI[mask]), bins=n_bins, histtype='step', label=f'Detected at RMS={RMS} mJy')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('log(MHI)')
    plt.ylabel('log(Counts)')
    plt.savefig(figname)
    plt.show()

def Detection_compareCats(MHI1, MHI2, n_bins=30, figname='', title=''):
    plt.figure(figsize=[5,4], dpi=300)
    plt.hist(np.log10(MHI1), bins=n_bins, histtype='step', label='Volume-limited Catalog')
    plt.hist(np.log10(MHI2), bins=n_bins, histtype='step', label='Flux-limited Mock-ALFALFA Catalog')
    plt.legend(loc='upper left')
    plt.yscale('log')
    plt.xlabel('log(MHI)')
    plt.ylabel('log(Counts)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()

def Detection_compare_Decs(MHI1, MHI2, Dec1, Dec2, n_bins=30, figname='', title='', MHI_bins=[6,7,8,9,10,11]):
    plt.figure(figsize=[5,4], dpi=300)
    color = cm.rainbow(np.linspace(0, 1, len(MHI_bins)-1))
    for i in range(len(MHI_bins)-1):
        mask1 = (np.log10(MHI1) >= MHI_bins[i]) & (np.log10(MHI1) <= MHI_bins[i+1]) 
        plt.hist(Dec1[mask1], bins=n_bins, histtype='step', label=f'{MHI_bins[i]} < log(M_HI) < {MHI_bins[i+1]}', color=color[i])
        mask2 = (np.log10(MHI2) >= MHI_bins[i]) & (np.log10(MHI2) <= MHI_bins[i+1]) 
        plt.hist(Dec2[mask2], bins=n_bins, histtype='step', linestyle=':', color=color[i])
    plt.legend(loc='upper right')
    plt.yscale('log')
    #plt.xlabel('W50 (km/s)')
    plt.xlabel('Dec (deg)')
    plt.ylabel('log(Counts)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()


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

#if __name__ == "__main__":
    #Sky_MHIsizeDcolour(catalog_file='catalogs_output/MockAlf_FullSky.npy', fname='Plots/MockALF_fullSky_MHIsizeDcolour.png', show=True, save=True)
    #Sky_MHIbins(catalog_file='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', fname='Plots/Volim200Mpc_fullSky_structure_MHIbins.pdf')
    #Dec_Structure_MHI_pdf(catalog_file='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', fname='Plots/Dec_MHIstructure_Volim200Mpc_CHORDSky.pdf')               


