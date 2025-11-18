import numpy as np
import matplotlib.pyplot as plt
import Galaxy_Functions as gf
import astropy.units as u
import astropy.constants as c
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import cm
import Generate_Catalog as gen
import Forecasting as forecast
import Gaussian_Estimate as gauss

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

def recover_HIMF(catalog_fl, Vollim=False, mask_fl='', RMS=0.1, sigma=6, nbins=30, marker='.', figname='',
                  MHI_grid=gf.MHI_grid, ax=None, color='', label='', ALF=False, bins=None, count_min=10, fromD=True,
                  plotWidths=False, title='', mockAlf=False):
    JonesHIMF = gf.HIMF_Jones2018(MHI=MHI_grid)
    MaHIMF = gf.HIMF_Ma2024(MHI=MHI_grid)
    W50_grid, OmanHIWF, HIWF_Schec = gf.Oman2021_HIWF()
    
    if Vollim:
        MHI, _, _, W50, _, _, _, Vol, _ = gen.load_catalogParams(catalog_file=catalog_fl)
        W50_broad = gauss.W50_broadened(W50)
        Vmax = np.max(Vol) - np.min(Vol)
        if bins is None:
            counts, bins = np.histogram(np.log10(MHI), bins=nbins)
            W50_counts, W50_bins = np.histogram(np.log10(W50_broad), bins=nbins)
        else:
            counts, bins = np.histogram(np.log10(MHI), bins=bins)
            W50_counts, W50_bins = np.histogram(np.log10(W50_broad), bins=bins)
        binwidth = (bins[1:])-(bins[:-1])
        phi = counts/(Vmax*binwidth)

        W50_binwidth = (W50_bins[1:])-(W50_bins[:-1])
        W50_phi = W50_counts/(Vmax*W50_binwidth)

    else:
        if ALF:
            Vmax, MHI, W50 = gf.Vmax_ALF(alf_fl=catalog_fl)
        elif mockAlf:
            Vmax, MHI, W50 = gf.Vmax_correct(catalog_file=catalog_fl, sigma=sigma, RMS=RMS, fromD=False, mockAlf=mockAlf)
        else:
            Vmax, MHI, W50 = gf.Vmax_correct(catalog_file=catalog_fl, sigma=sigma, RMS=RMS, fromD=fromD)
        if bins is None:
            counts_Vcorr, bins = np.histogram(np.log10(MHI), bins=nbins, weights=1/Vmax)
            counts_W50_Vcorr, W50_bins = np.histogram(np.log10(W50), bins=nbins, weights=1/Vmax)
        else:
            counts_Vcorr, bins = np.histogram(np.log10(MHI), bins=bins, weights=1/Vmax)
            counts_W50_Vcorr, W50_bins = np.histogram(np.log10(W50), bins=bins, weights=1/Vmax)
        counts, bins = np.histogram(np.log10(MHI), bins=bins)
        binwidth = (bins[1:])-(bins[:-1])
        phi = counts_Vcorr/(binwidth)

        W50_counts, W50_bins = np.histogram(np.log10(W50), bins=W50_bins)
        print("W50 is ", W50_bins)
        W50_binwidth = (W50_bins[1:])-(W50_bins[:-1])
        print("W50_binwidth is ", W50_binwidth)
        W50_phi = counts_W50_Vcorr/(W50_binwidth)
    
    phi[counts<count_min] = np.nan
    W50_phi[W50_counts<count_min] = np.nan
        
    bin_centers = gf.mid_bin(bins)
    W50_bin_centers = gf.mid_bin(W50_bins)
    if ax is None:
        plt.figure(figsize=[6,4],dpi=300)
        if plotWidths==False:
            plt.plot(np.log10(MHI_grid), np.log10(JonesHIMF), label='HIMF - drawn from, Jones2018') # HIMF
            plt.scatter(bin_centers, np.log10(phi), s=8, label='HIMF - recovered', color='black') # from sample
            plt.ylabel('$\phi(M_{HI})$')
            plt.xlabel('log(M$_{HI}$/M$_{\odot}$)')
        else:
            plt.plot(W50_grid, OmanHIWF, label='Oman+ 2021') # HIMF
            plt.plot(W50_grid, HIWF_Schec, label='Schechter Fit')
            plt.scatter(10**W50_bin_centers, W50_phi, s=8, label='HIWF - recovered', color='black') # from sample
            plt.ylabel('$\phi(w_{50})$ [$h_{70}^{3}$ Mpc$^{-3}$ dex$^{-1}$]')
            plt.xlabel('$w_{50}$ (km s$^{-1}$)')
            plt.xscale('log')
            plt.yscale('log')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linewidth=0.4)
        plt.savefig(figname)
        #plt.show()
    else:
        ax.plot(np.log10(MHI_grid), JonesHIMF, label='Jones+2018', color='gray', linewidth=1, linestyle='--') # HIMF
        ax.plot(np.log10(MHI_grid), MaHIMF, label='Ma+2024', color='purple', linewidth=1, linestyle=':') # HIMF
        ax.scatter(bin_centers, phi, s=8, label=label, color=color, marker=marker) # from sample
        ax.set_yscale('log')
        return (len(MHI))

def MHI_W50(catalog_fl1, catalog_fl2):
    cat1 = np.load(catalog_fl1)
    MHI1 = cat1[0]; W501= cat1[3]
    cat2 = np.load(catalog_fl2)
    MHI2 = cat2[0]; W502= cat2[3]
    plt.figure()
    plt.scatter(np.log10(MHI2), np.log10(W502), alpha=0.4)
    plt.scatter(np.log10(MHI1), np.log10(W501), alpha=0.4)
    plt.xlabel('log(MHI)')
    plt.ylabel('W50 (km/s)')
    plt.show()

def S21_W50(catalog_fl):
    mask, S21, W50 = forecast.detections_fromRMS(catalog_file=catalog_fl, RMS=0.1, )
    plt.figure()
    plt.scatter(np.log10(W50[mask]), np.log10(S21[mask]), alpha=0.4, s=1, label='Detected', color='blue')
    plt.scatter(np.log10(W50[~mask]), np.log10(S21[~mask]), alpha=0.4, s=1, label='Nondetected', color='lightgray')
    plt.xlabel('log(W$_{50}$/km s$^{-1}$)')
    plt.ylabel('log(S$_21$/Jy km s${-1}$')
    plt.legend()
    plt.savefig('Plots/S21_W50.png')

def MHI_Counts(catalog_fl, n_bins=30, label='', color='', hatch='', bins=None, ax=None, figname='', title=''):
    catalog = np.load(catalog_fl)
    MHI = catalog[0]
    counts = len(MHI)
    if ax is None:
        plt.figure(figsize=[5,4], dpi=300)
        if bins is None:
            plt.hist(np.log10(MHI), bins=n_bins, histtype='step', linewidth=1.5, label=f' Total: {counts}')
        else:
            plt.hist(np.log10(MHI), bins=bins, histtype='step', linewidth=1.5, label=f' Total: {counts}')
        plt.yscale('log')
        plt.ylabel('Counts')
        plt.xlabel('log($M_{HI}h^{2}_{70}/M_{\odot}$)')
        plt.legend()
        plt.grid(True, linewidth=0.4)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figname)
    else:
        counts = len(MHI)
        if bins is None:
            ax.hist(np.log10(MHI), bins=n_bins, histtype='step', label=label+f"{counts}", color=color, hatch=hatch, linewidth=1.5)
        else:
            ax.hist(np.log10(MHI), bins=bins, histtype='step', label=label+f"{counts}", color=color, hatch=hatch, linewidth=1.5)
        ax.set_yscale('log')

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
    for i in range(len(mask)):
        plt.hist(np.log10(MHI[mask[i]]), bins=n_bins, histtype='step', label=f'RMS={RMS[i]} mJy')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('log(MHI)')
    plt.ylabel('log(Counts)')
    plt.legend()
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

def W50z_Plane(RMS=0.1, sigma=6):
    z = np.linspace(0,1,1000)
    lg_W50 = np.linspace(1, 3, 1000)
    z_2d, lgW50_2d = np.meshgrid(z, lg_W50)
    MHI_2d = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS, DeltaV=10**(lgW50_2d))
    plt.figure(figsize=[5,4], dpi=300)
    plt.imshow(np.log10(MHI_2d), extent=[z.min(), z.max(), lg_W50.min(), lg_W50.max()], origin='lower', aspect='auto'     )
    plt.xlabel("Redshift z")
    plt.ylabel("log(W50/km s$^{-1}$)")
    plt.colorbar(label="log($M_{HI}$/$M_{\odot}$)")

    levels = [8,9,10,10.5,11,11.5] # in log space
    CS = plt.contour(
        z_2d, lgW50_2d, np.log10(MHI_2d),
        levels=levels, colors='black', linewidths=1, linestyles='dashed'
    )
    plt.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f"{v:.1f}")
    #plt.margins(x=0.02, y=0.02)
    plt.tight_layout()
    plt.savefig('Plots/W50z_Plane.png')
    plt.show()

def MHI_redshift(catalog_file, ax=None, figname='', title='', color='', label=''):
    catalog = np.load(catalog_file)
    MHI = catalog[0]
    z = catalog[8]
    W50 = catalog[3]
    W50_broad = gauss.W50_broadened(W50)
    if ax==None:
        plt.figure(figsize=[5,4], dpi=300)
        sc = plt.scatter(z, np.log10(MHI), c=np.log10(W50_broad), s=2, alpha=0.5)
        plt.colorbar(sc, label='log(W50/km s$^{-1}$)')
        plt.xlabel('Redshift z')
        plt.ylabel('log($M_{HI}$/$M_{\odot}$)')
        plt.title()
        plt.savefig(figname)
    else:
        ax.scatter(z, np.log10(MHI), color=color, label=label, s=5) #s=2)# alpha=0.5)

def HIWF():
    W50, HIWF = gf.Oman2021_HIWF()
    plt.figure()
    plt.plot(np.log10(W50), np.log10(HIWF))
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

if __name__ == "__main__":
    # recover_HIMF(catalog_fl='catalogs_output/MockAlf_FullSkyD200_Dec20to80_ChangeVelocity.npy', ALF=False, Vollim=False, 
    #              title='HIWF from mock-ALFALFA constrained sim',
    #              plotWidths=True, nbins=np.linspace(1,3,21), figname='Plots/HIWF_mockALF_changeVel.png', count_min=0)
    # recover_HIMF(catalog_fl='catalogs_output/ALFALFA_a100_C90.npy', ALF=True, Vollim=False, 
    #              title='HIWF from ALFALFA $\\alpha$.100',
    #              plotWidths=True, nbins=np.linspace(1,3,21), figname='Plots/HIWF_ALFALFA.png', count_min=0)
    # recover_HIMF(catalog_fl='catalogs_output/Detected_RMS0p1_VolLim_20to80deg_Dmax200.npy', ALF=False, Vollim=False, 
    #              title='HIWF from flux-limited Catalog, Detected at RMS=0.1 mJy',
    #              plotWidths=False, nbins=20, figname='Plots/HIMF_fluxLim_RMS0p1_z0p1.png', count_min=0)
    # recover_HIMF(catalog_fl='catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy', ALF=False, Vollim=True, 
    #              title='HIWF from Volume-limited Catalog',
    #              plotWidths=True, bins=np.linspace(1,3,21), figname='Plots/HIWF_VolLim_z0p1.png', count_min=0)
    #HIWF()
    #recover_HIMF(catalog_fl='catalogs_output/VolLim_20to60deg_zmin0p4_zmax1_MHI9to12.npy_rank0.npy', 
    #             Vollim=True, figname='Plots/HIMF_Volim_z0p4to1.png')
    #MHI_redshift('DetectionsVolLim_zmax0p1_5yearObs_20strips_20to80deg.npy')
    # MHI_Counts(catalog_fl='catalogs_output/Detected_VolLim_RMS0p08_20to80deg_z0p4to1_MHI9to12_new.npy',
    #           figname='Plots/z0p4to1_MHI_counts.png', bins=np.linspace(9.8,11.4,9), title='Redshift z=0.4-1')
    # recover_HIMF(catalog_fl='catalogs_output/Detected_VolLim_RMS0p08_20to80deg_z0p4to1_MHI9to12_new.npy',
    #              figname='Plots/z0p4to1_HIMF.png', sigma=6, RMS=0.08, Vollim=False, count_min=10, nbins=5)
    #W50z_Plane(RMS=)
    #S21_W50(catalog_fl='catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy')
    #recover_HIMF(catalog_fl='DetectionsVolLim_zmax0p1_fromRMS0p1_20to80deg.npy', sigma=6, RMS=0.1, Vollim=False)
    #recover_HIMF_ALF(alf_fl='catalogs_output/ALFALFA_a100_50complete.npy')
    #MHI_W50(catalog_fl1='catalogs_output/Detected_RMS0p1_peakSNR6_VolLim_20to60deg_Dmax200.npy', 
    #       catalog_fl2='catalogs_output/ALFALFA_a100.npy')
    #recover_HIMF(catalog_fl='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', Vollim=True)
    #recover_HIMF(catalog_fl='catalogs_output/Detected_RMS0p1_VolLim_20to80deg_Dmax200.npy', Vollim=False, RMS=0.1, sigma=6)
    #recover_HIMF(catalog_fl='catalogs_output/MockAlf_FullSky.npy', RMS=1, sigma=6,
    #            mask_fl='catalogs_output/maskRMS1_sigma6_MockAlf_FullSky.npy')
    #recover_HIMF(catalog_fl='catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy', RMS=0.1, sigma=6,
    #             mask_fl='catalogs_output/maskRMS0p1_sigma6_VolLim_20to60deg_zmax0p1.npy')
    #recover_HIMF(catalog='catalogs_output/MockAlf_D200_Dec20to80_ChangeVelocity.npy', VolLim=False)
    #Sky_MHIsizeDcolour(catalog_file='catalogs_output/MockAlf_FullSky.npy', fname='Plots/MockALF_fullSky_MHIsizeDcolour.png', show=True, save=True)
    #Sky_MHIbins(catalog_file='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', fname='Plots/Volim200Mpc_fullSky_structure_MHIbins.pdf')
    #Dec_Structure_MHI_pdf(catalog_file='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', fname='Plots/Dec_MHIstructure_Volim200Mpc_CHORDSky.pdf')               


