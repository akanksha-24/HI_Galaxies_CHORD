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
from matplotlib.lines import Line2D
import Galaxy_Functions as gf
from Gaussian_Estimate import *
from CHORD_Sensitivity import *
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.patches import Patch
import itertools

upchan_res = gf.width_vel2freq(del_Vrest=5) # u.Hz

#RMS_5yr = time2RMS(days=5*365/20, decl=np.deg2rad(45), nu=upchan_res*u.Hz).value
#RMS_1yr = time2RMS(days=365/20, decl=np.deg2rad(45), nu=upchan_res*u.Hz).value

def add_zmask(ax):
    x0, x1 = 0.1, 0.3   
    y0, y1 = ax.get_ylim()    
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, facecolor='red', alpha=0.3)
    ax.add_patch(rect)

    legend_patch = patches.Patch(
    facecolor='red',
    alpha=0.3,
    label='RFI'
    )
    return legend_patch

def redshift_surveysBar():
    fig, ax = plt.subplots(figsize=[6,4], dpi=300)

    surveys = ('HIPASS', 'ALFALFA', 'FASHI', 'APERTIF', 'WALLABY',
            'CHILES', 'DSA all-sky', 'MIGHTEE-HI', 'LADUMA', 'SKA1-mid',)

    redshift = [0.04, 0.06, 0.1, 0.25, 0.26, 0.45, 0.5, 0.6, 1.5, 2]

    color = ['green', 'green', 'orange', 'green', 'orange',
            'green', 'purple', 'orange', 'orange', 'purple']

    ax.barh(surveys, redshift, color=color, align='center')
    ax.yaxis.set_inverted(True)
    ax.set_xlabel('Redshift')
    rfi_patch = add_zmask(ax)

    legend_elements = [
        Patch(facecolor='green', label='Completed'),
        Patch(facecolor='orange', label='Ongoing'),
        Patch(facecolor='purple', label='Planned'),
        rfi_patch
    ]

    ax.legend(handles=legend_elements, title="Survey status")
    
    plt.tight_layout()
    plt.savefig('Plots/surveys_redshift.png')

def param_distributions(catalog, n_bins=20, flname=''):
    catalog = np.load(catalog)
    extn = ['MHI', 'VHI', 'Incl', 'W50', 'RA', 'Dec', 'Distace', 'Volume', 'Redshift']
    xlabels = ['log(M$_{HI}$) [M$_{\odot}$]', 'V$_{HI}$ (km/s)', 'Inclination (deg)', 'W50 (km/s)',
               'Right Ascension (deg)', 'Declination (deg)', 'Distance (Mpc)', 'Volume (Mpc$^3$)', 'Redshift (z)']

    for i in np.arange(len(extn)):
        plt.figure(figsize=[4,3], dpi=300)
        if i==0:
            plt.hist(np.log10(catalog[i]), bins=n_bins, histtype='step')
        else:
            plt.hist(catalog[i], bins=n_bins, histtype='step')
        #if i==1 or i==3: # VHI or W50
        #plt.yscale('log')
        plt.xlabel(xlabels[i])
        plt.tight_layout()
        plt.savefig(flname+'_'+extn[i]+'.png')

def dec_solidang(catalogs, labels, normalizeDec=True, PlotRA=False, hist2D=False, RA2D=False):
    dec_edges = np.linspace(20,80,25)
    ra_edges = np.linspace(0,360,20)
    MHI_bins = np.linspace(7.5,10.5,9)

    dec1 = dec_edges[:-1]
    dec2 = dec_edges[1:]
    solidangs = 2*np.pi * (np.sin(np.deg2rad(dec2)) - np.sin(np.deg2rad(dec1))) # Solid angle for each declination band
    plt.figure()

    for i in range(len(catalogs)):
        catalog = np.load(catalogs[i])
        lg_MHI = np.log10(catalog[0])
        dec = catalog[5]
        ra = catalog[4]
        dec_counts, _ = np.histogram(dec, bins=dec_edges)
        ra_counts, _ = np.histogram(ra, bins=ra_edges)
        total = np.sum(dec_counts)
        print("total ", total)
        density = dec_counts / (solidangs*total)
        if PlotRA:
            xlabel = "RA (deg)"
            ylabel = "Counts"
            title = "Right Ascension Distribution"
            plt.stairs(ra_counts/np.sum(ra_counts), ra_edges, fill=False, linewidth=2, label=labels[i])
        elif RA2D:
            counts, _, _ = np.histogram2d(ra, lg_MHI, bins=[ra_edges, MHI_bins])
            normalized_total = counts / (np.sum(counts, axis=0, keepdims=True))
            plt.imshow(normalized_total.T, 
                origin='lower',
                aspect='auto',
                extent=[ra_edges[0], ra_edges[-1], MHI_bins[0], MHI_bins[-1]],
                cmap='viridis')
            plt.colorbar(label='Normalized counts')
            xlabel='ra (deg)'
            ylabel='log(MHI)'
            title=''
        elif hist2D:
            counts, _, _ = np.histogram2d(dec, lg_MHI, bins=[dec_edges, MHI_bins])
            normalized_total = counts / (np.sum(counts, axis=0, keepdims=True)*solidangs[:,None])
            plt.imshow(normalized_total.T, 
                origin='lower',
                aspect='auto',
                extent=[dec_edges[0], dec_edges[-1], MHI_bins[0], MHI_bins[-1]],
                cmap='viridis')
            plt.colorbar(label='Normalized counts')
            xlabel='dec (deg)'
            ylabel='log(MHI)'
            title=''
            
        else:
            xlabel = "Declination (deg)"
            if normalizeDec:
                plt.stairs(density, dec_edges, fill=False, linewidth=2, label=labels[i])
                ylabel = "Relative Counts per Steradian"
                title = "Declination Distribution Normalized by Solid Angle and Total counts"
            else:
                ylabel = "Counts"
                title = "Declination Distribution"
                plt.stairs(dec_counts, dec_edges, fill=False, linewidth=2, label=labels[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.yscale('log')
    plt.title(title)
    #plt.legend(loc='lower right')
    plt.show()

def Dec_RA_counts(catalog, lgMHI_bin1, lgMHI_bin2, solidangs, dec_edges, ra_edges):
    catalog = np.load(catalog)
    MHI = catalog[0]
    mask = (np.log10(MHI) >= lgMHI_bin1) & (np.log10(MHI) <= lgMHI_bin2)
    dec = catalog[5, mask]
    ra = catalog[4, mask]
    dec_counts, _ = np.histogram(dec, bins=dec_edges)
    ra_counts, _ = np.histogram(ra, bins=ra_edges)
    relative_density = dec_counts / (solidangs*np.sum(dec_counts))
    relative_racounts = ra_counts/np.sum(ra_counts)
    return relative_density, relative_racounts, np.sum(dec_counts), np.sum(ra_counts)


def dec_solidang_byMHI(catalog_lss, catalog_volim, normalizeDec=True, PlotRA=False):
    dec_edges = np.linspace(20,80,20)
    ra_edges = np.linspace(0,360,20)

    dec1 = dec_edges[:-1]
    dec2 = dec_edges[1:]
    solidangs = 2*np.pi * (np.sin(np.deg2rad(dec2)) - np.sin(np.deg2rad(dec1))) # Solid angle for each declination band

    lgMHI_bins = np.linspace(7,11,5)

    mpl.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 9.5
    })

    fig, ax = plt.subplots(2, 1, figsize=(5, 8), dpi=300)
    # cmap = cm.get_cmap('Blues_r', len(lgMHI_bins))
    # colors = cmap(np.linspace(0, 1, len(lgMHI_bins)))

    #cmap = cm.get_cmap('Greens_r', len(lgMHI_bins))
    #colors_mean = cmap(np.linspace(0, 1, len(lgMHI_bins)))
    colors = cm.get_cmap('Dark2', len(lgMHI_bins)*2).colors
    linestyle = ['-', '-', '-.', ':', ':']

    for j in range(len(lgMHI_bins)-1):
        #plt.figure()
        #title = f'{lgMHI_bins[j]:.0f} < log(MHI) < {lgMHI_bins[j+1]:.0f}'
        label = f'{lgMHI_bins[j]:5.1f} < log(MHI) < {lgMHI_bins[j+1]:5.1f}'

        # Large-scale structure counts
        relative_density_lss, relative_racounts_lss, dec_counts_lss, ra_counts_lss = Dec_RA_counts(catalog_lss, lgMHI_bin1=lgMHI_bins[j],
                                                                                    lgMHI_bin2=lgMHI_bins[j+1],
                                                                                    solidangs=solidangs, dec_edges=dec_edges, 
                                                                                    ra_edges=ra_edges)
        # Mean and std based on vol-sim
        relative_density_vol, relative_racounts_vol, _, _ = Dec_RA_counts(catalog_volim, lgMHI_bin1=lgMHI_bins[j],
                                                                                    lgMHI_bin2=lgMHI_bins[j+1],
                                                                                    solidangs=solidangs, dec_edges=dec_edges, 
                                                                                    ra_edges=ra_edges)

        Dec_density_mean = np.sum(relative_density_lss)/(len(dec_edges)-1)
        Dec_density_std = np.std(relative_density_vol)
        RA_counts_mean = np.sum(relative_racounts_lss)/(len(ra_edges)-1)
        RA_counts_std = np.std(relative_racounts_vol)

        #title = "Right Ascension Distribution"
        ax[1].stairs(relative_racounts_lss, ra_edges, fill=False, linewidth=2, label=f'Total={ra_counts_lss:6.0f}', color=colors[j+2], linestyle=linestyle[j])
        #plt.axhline(y=RA_counts_mean, linestyle=':', color=colors[j])
        ax[1].fill_between(ra_edges,                        
                        RA_counts_mean - RA_counts_std,  
                        RA_counts_mean + RA_counts_std,  
                        color=colors[j+2],
                        edgecolor=colors[j+2],
                        linestyle=linestyle[j],
                        linewidth=1,
                        alpha=0.2)
        ax[1].set_xlabel("RA (deg)")
        ax[1].set_ylabel("Relative Counts")
        
        ax[0].stairs(relative_density_lss, dec_edges, fill=False, linewidth=2, label=label+f', Total={dec_counts_lss:.0f}', color=colors[j+2], linestyle=linestyle[j])
        #plt.axhline(y=Dec_density_mean, linestyle=':', color=colors[j], label='')
        ax[0].fill_between(dec_edges,                        
                    Dec_density_mean - Dec_density_std,  
                    Dec_density_mean + Dec_density_std,   
                    color=colors[j+2],
                    edgecolor=colors[j+2],
                    linestyle=linestyle[j],
                    linewidth=1,
                    alpha=0.2)
        ax[0].set_xlabel("Dec (deg)")
        ax[0].set_ylabel("Relative Counts per Steradian")
    ax[0].legend(loc='lower center', prop={'family': 'monospace'})
    #ax[1].legend(loc='upper right')
    ax[1].grid(True, linewidth=0.4)
    ax[0].grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig('Plots/LSS_RA_Dec.pdf', bbox_inches='tight')

def MHI_VHI_polynomial(MHI=np.logspace(5,11,100000)):
    VHI = gf.VHI_polyFit(MHI)
    plt.figure()
    plt.plot(np.log10(MHI), VHI)
    plt.savefig('MHI_VHI.png')
    plt.show()

def recover_HIMF(catalog_fl, Vollim=False, mask_fl='', RMS=0.1, sigma=6, nbins=30, marker='.', figname='',
                  MHI_grid=gf.MHI_grid, ax=None, color='', label='', ALF=False, bins=None, count_min=10, fromD=True,
                  plotWidths=False, title='', mockAlf=False, solidang=None):
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
            Vmax, MHI, W50 = gf.Vmax_ALF(alf_fl=catalog_fl)#, solidang=solidang)
        elif mockAlf:
            Vmax, MHI, W50 = gf.Vmax_correct(catalog_file=catalog_fl, sigma=sigma, RMS=RMS, fromD=False, mockAlf=mockAlf, solidang=solidang)
        else:
            Vmax, MHI, W50 = gf.Vmax_correct(catalog_file=catalog_fl, sigma=sigma, RMS=RMS, fromD=fromD, solidang=solidang)
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
        #print("W50 is ", W50_bins)
        W50_binwidth = (W50_bins[1:])-(W50_bins[:-1])
        #print("W50_binwidth is ", W50_binwidth)
        W50_phi = counts_W50_Vcorr/(W50_binwidth)
    
    phi[counts<count_min] = np.nan
    W50_phi[W50_counts<count_min] = np.nan
        
    bin_centers = gf.mid_bin(bins)
    W50_bin_centers = gf.mid_bin(W50_bins)
    if ax is None:
        plt.figure(figsize=[6,4],dpi=300)
        if plotWidths==False:
            plt.plot(np.log10(MHI_grid), np.log10(JonesHIMF), label='HIMF - drawn from, Jones2018') # HIMF
            #plt.fill_between(np.log10(MHI_grid))
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
            ax.hist(np.log10(MHI), bins=n_bins, histtype='step', color=color, hatch=hatch, linewidth=1.5) #label=label+f"{counts}")
        else:
            ax.hist(np.log10(MHI), bins=bins, histtype='step', color=color, hatch=hatch, linewidth=1.5) #label=label+f"{counts}",
        ax.set_yscale('log')
        return counts

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

def dndz_catalog(catalog_fl, ax=None, label='', color='', Usedz=True, bins=500):
    catalog = np.load(catalog_fl)
    z = catalog[8]
    dn, bins = np.histogram(z, bins=100)
    mid_bins = 0.5*(bins[1:] + bins[:-1])
    mask = mid_bins < 0.02
    total = np.sum(dn[mask])
    print("total is ", total)
    dz = (bins[1:])-(bins[:-1])
    dndz = dn/dz
    if ax!=None:
        if Usedz:
            ax.plot(gf.mid_bin(bins), dndz, label=label, color=color)
        else:
            ax.hist(z, bins=bins, color=color, histtype='step', label=label)
    else:
        plt.figure()
        if Usedz:
            plt.plot(gf.mid_bin(bins), dndz)
        else:
            plt.plot(gf.mid_bin(bins), dn, label=label, color=color)
        #plt.xlim(0,0.7)
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

def Sky_MHIsizeDcolour(catalog_file, show=True, save=False, fname='', sizeM=True):
    MHI, _, _, _, ra, dec, D, _, z = gen.load_catalogParams(catalog_file)
    ra_rad = np.deg2rad(ra)
    ra_rad = np.remainder(ra_rad + np.pi, 2 * np.pi) - np.pi  # center at 0
    dec_rad = np.deg2rad(dec)

    plt.figure(figsize=[8,6], dpi=200)
    ax = plt.axes(projection='aitoff')
    ax.grid(linewidth=0.5)
    if sizeM:
        sc = ax.scatter(ra_rad, dec_rad, #transform=ax.get_transform('world'), 
            s=(MHI/2e8).astype(int), c=D, cmap='viridis', alpha=0.5)
    else:
        sc = ax.scatter(ra_rad, dec_rad, #transform=ax.get_transform('world'), 
            s=1, c=z, cmap='viridis', alpha=0.5)
    #plt.colorbar(sc, fraction=0.8, pad=0.1, label='Distance (Mpc)', location='bottom')
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

def W50z_Plane(RMS=0.1, sigma=6, nearby=False):
    lg_W50 = np.linspace(1, 2.5, 1000)
    if nearby:
        z = np.linspace(0,0.01,1000)
        D = gf.Comoving_Dist(z).to_value(u.Mpc)
        print("max distance ", np.max(D))
        D_2d, lgW50_2d = np.meshgrid(D, lg_W50)
    else:
        z = np.linspace(0,1,1000)
    z_2d, lgW50_2d = np.meshgrid(z, lg_W50)
    MHI_2d = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS, DeltaV=10**(lgW50_2d))
    plt.figure(figsize=[5,4], dpi=300)
    if nearby:
        plt.imshow(np.log10(MHI_2d), extent=[D.min(), D.max(), lg_W50.min(), lg_W50.max()], origin='lower', aspect='auto')
        plt.xlabel("Distance (Mpc)")
    else:
        plt.imshow(np.log10(MHI_2d), extent=[z.min(), z.max(), lg_W50.min(), lg_W50.max()], origin='lower', aspect='auto')
        plt.xlabel("Redshift z")
    
    plt.ylabel("log(W50/km s$^{-1}$)")
    plt.colorbar(label="log($M_{HI}$/$M_{\odot}$)")

    if nearby:
        levels = [5,6,7,8,9]
        CS = plt.contour(
            D_2d, lgW50_2d, np.log10(MHI_2d),
            levels=levels, colors='black', linewidths=1, linestyles='dashed'
        )
    else:
        levels = [8,9,10,10.5,11] # in log space
        CS = plt.contour(
            z_2d, lgW50_2d, np.log10(MHI_2d),
            levels=levels, colors='black', linewidths=1, linestyles='dashed'
        )

    plt.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f}")
    #plt.margins(x=0.02, y=0.02)
    #plt.margins(x=0.1, y=0.1)
    plt.tight_layout()
    if nearby:
        plt.savefig('Plots/W50z_Plane_nearby.png')
    else:
        plt.savefig('Plots/W50z_Plane.png')
    plt.show()

def W50z_Plane_subplots(RMS=0.1, sigma=6):
    mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
    })
    fig, ax = plt.subplots(1, 2, figsize=[13,4.5], dpi=300)
    plt.subplots_adjust(wspace=0.17)
    lg_W50 = np.linspace(1, 2.7, 1000)
    z = np.linspace(0,1,10000)
    #RMS_5yr = RMS_fromDays(days=5*365/24, decl=np.deg2rad(20), z=z, nu=upchan_res*u.Hz).value
    #RMS_1yr = RMS_fromDays(days=365/12, decl=np.deg2rad(20), z=z, nu=upchan_res*u.Hz).value
    _, RMS_5yr, _ = build_survey(obs_years=5, z=z)
    print("5 year RMS is", RMS_5yr)
    _, RMS_1yr, _ = build_survey(obs_years=1, z=z, end=50)
    print("1 year RMS is", RMS_1yr)
    z_2d, lgW50_2d = np.meshgrid(z, lg_W50)
    D_2d = gf.Comoving_Dist(z_2d).to_value(u.Mpc)
    MHI_2d_5year = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_5yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
    MHI_2d_1year = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_1yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)

    levels_5yr = [[5,6,7,7.5,8],
            [9,10,10.5,11]]

    manual_locations5 = [[(0.0, 1.47), (20, 1.49), (40, 1.53), (80, 1.65), (120, 1.8)],
                    [(0.05, 1.49), (0.3, 1.53), (0.48, 1.57), (0.75, 1.62)]]

    levels_1yr = [[6,7,7.5,8],
            [10,10.5,11,11.5]]

    manual_locations1 = [[(5, 1.65), (20, 1.73), (40, 1.8), (80, 1.9)],
                    [(0.1, 1.9), (0.3, 1.95), (0.55, 2.1), (0.85, 2.3)]]
    

    im = ax[0].imshow(np.log10(MHI_2d_5year), extent=[D_2d.min(), D_2d.max(), lg_W50.min(), lg_W50.max()], origin='lower', aspect='auto')  
    ax[0].set_xlabel('Distance (Mpc)')
    ax[0].set_xlim(0,128)
    ax[0].set_ylim(1.4,2.2)
    ax[1].set_ylim(1.4,2.7)

    im = ax[1].imshow(np.log10(MHI_2d_5year), extent=[z_2d.min(), z_2d.max(), lg_W50.min(), lg_W50.max()], origin='lower', aspect='auto') 
    ax[1].set_xlim(0.03,1)
    ax[1].set_xlabel('Redshift z')

    
    CS = ax[0].contour(D_2d, lgW50_2d, np.log10(MHI_2d_5year),
        levels=levels_5yr[0], colors='black', linewidths=1, linestyles='--') 
    ax[0].clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations5[0])

    CS = ax[0].contour(D_2d, lgW50_2d, np.log10(MHI_2d_1year),
        levels=levels_1yr[0], colors='white', linewidths=1, linestyles='-.') 
    ax[0].clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations1[0])

    CS = ax[1].contour(z_2d, lgW50_2d, np.log10(MHI_2d_5year),
        levels=levels_5yr[1], colors='black', linewidths=1, linestyles='--') 
    ax[1].clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations5[1])

    CS = ax[1].contour(z_2d, lgW50_2d, np.log10(MHI_2d_1year),
        levels=levels_1yr[1], colors='white', linewidths=1, linestyles='-.') 
    ax[1].clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations1[1])

    ax[0].set_ylabel('log(W50/km s$^{-1}$)')
    ax[1].set_yticks([1.4,1.6,1.8,2,2.2,2.4,2.6])
    
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(r"log($M_{\mathrm{HI}}/M_\odot$)")
    cbar.set_ticks([4,6,8,10])

    legend_elements = [
    Line2D([0], [0], color='black', lw=1, linestyle='--', label='5 year'),#label=f'5-year survey, {RMS_5yr[0]:.1f} mJy'),
    Line2D([0], [0], color='white', lw=1, linestyle='-.', label='1 year')]#label=f'1-year survey, {RMS_1yr[0]:.1f} mJy')]

    leg = ax[1].legend(
        handles=legend_elements,
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.82, 0.90),
        loc='center',
    )

    leg.get_frame().set_facecolor('gray')
    leg.get_frame().set_alpha(0.4)


    #plt.tight_layout()
    plt.savefig("Plots/DistanceDetect_withScanTime.pdf", bbox_inches='tight')

def W50z_Plane_subplots_zD(RMS=0.1, sigma=6):
    mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
    })
    fig, ax = plt.subplots(1, 2, figsize=[14,4.5], dpi=300, sharey=True)
    plt.subplots_adjust(wspace=0.05)
    #dW = 1e-2
    #lg_W50 = np.arange(1, 2.7, dW)
    lg_W50 = np.linspace(1,2.7,1000)
    z = np.linspace(0,1,10000)
    _, RMS_5yr, _ = build_survey(obs_years=5, z=z)
    print("5 year RMS is", RMS_5yr)
    _, RMS_1yr, _ = build_survey(obs_years=1, z=z, end=50)
    print("1 year RMS is", RMS_1yr)

    z_2d, lgW50_2d = np.meshgrid(z, lg_W50)
    D_2d = gf.Comoving_Dist(z_2d).to_value(u.Mpc)
    MHI_2d_5year = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_5yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
    MHI_2d_1year = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_1yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
    print("MHI shape OG", MHI_2d_5year.shape)
    # levels_5yr = [[5,6,6.5,7,7.5,8],
    #         [9,10,10.5,11]]

    # manual_locations5 = [[(0.0, 1.2), (0.005, 1.3), (0.008, 1.4), (0.013, 1.5), (0.018, 1.6), (0.029, 1.9)],
    #                 [(0.05, 1.4), (0.3, 1.6), (0.48, 1.8), (0.75, 2.0)]]

    # levels_1yr = [[6,6.5,7,7.5,8],
    #         [10,10.5,11,11.5]]

    # manual_locations1 = [[(0.003, 1.6), (0.004, 1.65), (0.01, 1.7), (0.017, 1.75), (0.024, 1.85)],
    #                 [(0.1, 1.9), (0.45, 2.1), (0.65, 2.2), (0.85, 2.3)]]

    levels_5yr = [[5,6,7,7.5],#,8],
            [9,10,10.5,11]]

    manual_locations5 = [[(0.0, 1.3), (0.003, 1.55), (0.012, 1.7), (0.018, 1.8)],#, (0.018, 2.5)],
                    [(0.05, 1.3), (0.3, 1.6), (0.48, 1.7), (0.75, 1.8)]]

    levels_1yr = [[6,7,7.5,8],
            [10,10.5,11,11.5]]

    manual_locations1 = [[(0.003, 1.91), (0.007, 1.93), (0.01, 1.95), (0.02, 2.3)],
                    [(0.1, 1.9), (0.3, 1.95), (0.55, 2.1), (0.85, 2.3)]]
    ax[0].set_xlim(0,0.02)
    ax[1].set_xlim(0.03,1)
    ax[0].set_ylim(1.2,2.7)
    ax[1].set_ylim(1.2,2.7)
    MHI_2d_5year[np.isnan(MHI_2d_5year)==True]=10**5

    for i in range(2):
        im = ax[i].imshow(np.log10(MHI_2d_5year), extent=[z.min(), z.max(), lg_W50.min(), lg_W50.max()], origin='lower', aspect='auto')  
        ax[i].set_xlabel('Redshift')

        ax_dist = ax[i].twiny()
        ax_dist.set_xlim(gf.Comoving_Dist(ax[i].get_xlim()).to_value(u.Mpc))

        if i==0:
            z_ticks = [0,0.01,0.02]
            D_ticks = [0,20,40,60,80]
        else:
            z_ticks = [0.25,0.5,0.75,1.0]
            D_ticks = [1000,2000,3000]

        #D_ticks = gf.Comoving_Dist(z_ticks).to_value(u.Mpc)
        ax[i].set_xticks(z_ticks)
        ax_dist.set_xticks(D_ticks)
        #ax_dist.set_xticklabels([f"{d:.0f}" for d in D_ticks])
        ax_dist.set_xlabel("Distance (Mpc)")

    # end contour at W50 = 2VHI
        for j in range(len(levels_5yr[i])):
            VHI_5 = np.log10(W50_broadened(2*gf.VHI_polyFit(10**levels_5yr[i][j])))
            print(VHI_5)
            #print(np.)
            #lg_W50_5 = np.arange(1, 2*VHI_5, dW)
            #z_2d_5, lgW50_2d_5 = np.meshgrid(z, lg_W50_5)
            mask_ = lgW50_2d > VHI_5
            MHI_copy = np.log10(MHI_2d_5year).copy()
            MHI_copy2 = np.log10(MHI_2d_5year).copy()
            MHI_copy[lgW50_2d > VHI_5] = np.nan
            MHI_copy2[lgW50_2d < VHI_5] = np.nan
            #MHI_2d_5year_re = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_5yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
        
            CS = ax[i].contour(z_2d, lgW50_2d, MHI_copy,
                levels=[levels_5yr[i][j]], colors='black', linewidths=1, linestyles='--') 
            ax[i].clabel(CS, inline=True, inline_spacing=0, fontsize=10, manual=[manual_locations5[i][j]], fmt=lambda v: f" {v:.1f} ")

            CS = ax[i].contour(z_2d, lgW50_2d, MHI_copy2,
                levels=[levels_5yr[i][j]], colors='lightgray', linewidths=1, linestyles='--', alpha=0.3) 
            #ax[i].clabel(CS, inline=True, inline_spacing=0, fontsize=10, manual=[manual_locations5[i][j]], fmt=" ")

        for j in range(len(levels_1yr[i])):
            VHI_1 = np.log10(W50_broadened(2*gf.VHI_polyFit(10**levels_1yr[i][j])))
            print(VHI_1)
            #print(np.)
            #lg_W50_5 = np.arange(1, 2*VHI_5, dW)
            #z_2d_5, lgW50_2d_5 = np.meshgrid(z, lg_W50_5)
            mask_ = lgW50_2d > VHI_1
            MHI_copy = np.log10(MHI_2d_1year).copy()
            MHI_copy2 = np.log10(MHI_2d_1year).copy()
            MHI_copy[lgW50_2d > VHI_1] = np.nan
            MHI_copy2[lgW50_2d < VHI_1] = np.nan
            #MHI_2d_5year_re = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_5yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
        
            CS = ax[i].contour(z_2d, lgW50_2d, MHI_copy,
                levels=[levels_1yr[i][j]], colors='white', linewidths=1, linestyles='-.') 
            ax[i].clabel(CS, inline=True, inline_spacing=0, fontsize=10, manual=[manual_locations1[i][j]], fmt=lambda v: f" {v:.1f} ")

            CS = ax[i].contour(z_2d, lgW50_2d, MHI_copy2,
                levels=[levels_1yr[i][j]], colors='lightgray', linewidths=1, linestyles='-.', alpha=0.3) 
            #ax[i].clabel(CS, inline=True, inline_spacing=0, fontsize=10, manual=[manual_locations5[i][j]], fmt=lambda v: f" {v:.1f} ")

    ax[0].set_ylabel('log(W50/km s$^{-1}$)')
    ax[0].set_yticks([1.4,1.8,2.2,2.6])
    
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(r"log($M_{\mathrm{HI}}/M_\odot$)")
    cbar.set_ticks([4,6,8,10])

    legend_elements = [
    Line2D([0], [0], color='black', lw=1, linestyle='--',
           label='5 year'),
    Line2D([0], [0], color='white', lw=1, linestyle='-.',
           label='1 year'),
    ]

    # fig.legend(
    #     handles=legend_elements,
    #     ncol=2,
    #     frameon=True,
    #     bbox_to_anchor=(0.5, 1.02),
    #     loc='center',
    # )

    leg = ax[1].legend(
        handles=legend_elements,
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.84, 0.90),
        loc='center',
    )

    leg.get_frame().set_facecolor('gray')
    leg.get_frame().set_alpha(0.3)

    # y0, y1 = 2.2, 2.7 
    # x0, x1 = ax[0].get_xlim()    
    # rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor='lightgray', facecolor='lightgray', alpha=0.3)
    # ax[0].add_patch(rect)

    #plt.tight_layout()
    plt.savefig("Plots/DistanceDetect_zD.pdf", bbox_inches='tight')
    #plt.show()

    # if nearby:

    # else:
    #     levels = [8,9,10,10.5,11] # in log space
    #     CS = plt.contour(
    #         z_2d, lgW50_2d, np.log10(MHI_2d),
    #         levels=levels, colors='black', linewidths=1, linestyles='dashed'
    #     )

    # plt.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f}")
    # #plt.margins(x=0.02, y=0.02)
    # #plt.margins(x=0.1, y=0.1)
    # plt.tight_layout()
    # if nearby:
    #     plt.savefig('Plots/W50z_Plane_nearby.png')
    # else:
    #     plt.savefig('Plots/W50z_Plane.png')
    # plt.show()

def W50z_Plane_oneplot(RMS=0.1, sigma=6):
    mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
    })
    fig, ax = plt.subplots(1, 1, figsize=[13,4.5], dpi=300)
    #plt.subplots_adjust(wspace=0.17)
    lg_W50 = np.linspace(1, 2.7, 1000)
    z = np.linspace(0,1,10000)
    _, RMS_5yr, _ = build_survey(obs_years=5, z=z)
    print("5 year RMS is", RMS_5yr)
    _, RMS_1yr, _ = build_survey(obs_years=1, z=z, end=50)
    print("1 year RMS is", RMS_1yr)

    z_2d, lgW50_2d = np.meshgrid(z, lg_W50)
    D_2d = gf.Comoving_Dist(z_2d).to_value(u.Mpc)
    MHI_2d_5year = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_5yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
    MHI_2d_1year = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_1yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
    # levels_5yr = [[5,6,6.5,7,7.5,8],
    #         [9,10,10.5,11]]

    # manual_locations5 = [[(0.0, 1.2), (0.005, 1.3), (0.008, 1.4), (0.013, 1.5), (0.018, 1.6), (0.029, 1.9)],
    #                 [(0.05, 1.4), (0.3, 1.6), (0.48, 1.8), (0.75, 2.0)]]

    # levels_1yr = [[6,6.5,7,7.5,8],
    #         [10,10.5,11,11.5]]

    # manual_locations1 = [[(0.003, 1.6), (0.004, 1.65), (0.01, 1.7), (0.017, 1.75), (0.024, 1.85)],
    #                 [(0.1, 1.9), (0.45, 2.1), (0.65, 2.2), (0.85, 2.3)]]

    levels_5yr = [[5,6,7,7.5,8],
            [9,10,10.5,11]]

    manual_locations5 = [[(0.0, 1.5), (0.005, 1.6), (0.013, 1.7), (0.018, 1.8), (0.029, 1.9)],
                    [(0.05, 1.5), (0.3, 1.6), (0.48, 1.7), (0.75, 1.8)]]

    levels_1yr = [[6,7,7.5,8],
            [10,10.5,11,11.5]]

    manual_locations1 = [[(0.003, 1.9), (0.009, 1.95), (0.015, 2.1), (0.024, 2.3)],
                    [(0.1, 1.9), (0.3, 1.95), (0.55, 2.1), (0.85, 2.3)]]
    #ax[0].set_xlim(0,0.03)
    ax.set_ylim(1.4,2.7)
    #ax[1].set_xlim(0.03,1)

    #for i in range(2):
    im = ax.imshow(np.log10(MHI_2d_5year), extent=[z.min(), z.max(), lg_W50.min(), lg_W50.max()], origin='lower', aspect='auto')  
    ax.set_xlabel('Redshift z')

    ax_dist = ax.twiny()
    ax_dist.set_xlim(ax.get_xlim())

    z_ticks = np.round(np.linspace(*ax.get_xlim(), 5), 3)
    D_ticks = gf.Comoving_Dist(z_ticks).to_value(u.Mpc)

    ax.set_xticks(z_ticks)
    ax_dist.set_xticks(z_ticks)
    ax_dist.set_xticklabels([f"{d:.0f}" for d in D_ticks])
    ax_dist.set_xlabel("Distance (Mpc)")

    CS = ax.contour(z_2d, lgW50_2d, np.log10(MHI_2d_5year),
        levels=levels_5yr[0], colors='black', linewidths=1, linestyles='--') 
    ax.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations5[0])

    CS = ax.contour(z_2d, lgW50_2d, np.log10(MHI_2d_5year),
        levels=levels_5yr[1], colors='black', linewidths=1, linestyles='--') 
    ax.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations5[1])

    CS = ax.contour(z_2d, lgW50_2d, np.log10(MHI_2d_1year),
        levels=levels_1yr[0], colors='white', linewidths=1, linestyles='-.') 
    ax.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations1[0])

    CS = ax.contour(z_2d, lgW50_2d, np.log10(MHI_2d_1year),
        levels=levels_1yr[1], colors='white', linewidths=1, linestyles='-.') 
    ax.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations1[1])

    ax.set_ylabel('log(W50/km s$^{-1}$)')
    
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(r"log($M_{\mathrm{HI}}/M_\odot$)")
    cbar.set_ticks([4,6,8,10])

    legend_elements = [
    Line2D([0], [0], color='black', lw=1, linestyle='--',
           label='5-year survey'),
    Line2D([0], [0], color='white', lw=1, linestyle='-.',
           label='1-year survey'),
    ]

    # fig.legend(
    #     handles=legend_elements,
    #     ncol=2,
    #     frameon=True,
    #     bbox_to_anchor=(0.5, 1.02),
    #     loc='center',
    # )

    leg = ax.legend(
        handles=legend_elements,
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.72, 0.92),
        loc='center',
    )

    leg.get_frame().set_facecolor('gray')
    leg.get_frame().set_alpha(0.3)


    #plt.tight_layout()
    plt.savefig("Plots/DistanceDetect_onePlot.png")
    #plt.show()

    # if nearby:

    # else:
    #     levels = [8,9,10,10.5,11] # in log space
    #     CS = plt.contour(
    #         z_2d, lgW50_2d, np.log10(MHI_2d),
    #         levels=levels, colors='black', linewidths=1, linestyles='dashed'
    #     )

    # plt.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f}")
    # #plt.margins(x=0.02, y=0.02)
    # #plt.margins(x=0.1, y=0.1)
    # plt.tight_layout()
    # if nearby:
    #     plt.savefig('Plots/W50z_Plane_nearby.png')
    # else:
    #     plt.savefig('Plots/W50z_Plane.png')
    # plt.show()

# def W50z_Plane_oneplot(RMS=0.1, sigma=6):
#     mpl.rcParams.update({
#     'font.size': 14,
#     'axes.labelsize': 14,
#     'axes.titlesize': 14,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14
#     })
#     fig = plt.figure(figsize=[12,4], dpi=300)
#     lg_W50 = np.linspace(1, 2.7, 1000)
#     z = np.linspace(0,1,10000)
#     _, RMS_5yr, _ = build_survey(obs_years=5, z=z)
#     print("5 year RMS is", RMS_5yr)
#     _, RMS_1yr, _ = build_survey(obs_years=1, z=z, end=50)
#     print("1 year RMS is", RMS_1yr)
#     z_2d, lgW50_2d = np.meshgrid(z, lg_W50)
#     #D_2d = gf.Comoving_Dist(z_2d).to_value(u.Mpc)
#     MHI_2d_5year = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_5yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
#     MHI_2d_1year = gf.estimate_MHImax(z=z_2d, sigma=sigma, RMS_chan=RMS_1yr, DeltaV=10**(lgW50_2d), chan_width=upchan_res)
#     #levels = [5,6,7,8,9,10,11]

#     levels_5yr = [5,6,7,7.5,8,9,10,10.5,11]

#     manual_locations5 = [(0.0, 1.2), (0.005, 1.2), (0.013, 1.3), (0.018, 1.4), (0.029, 1.9), 
#                          (0.05, 1.2), (0.3, 1.3), (0.48, 1.4), (0.75, 1.5)]

#     levels_1yr = [6,7,7.5,8,10,10.5,11,11.5]

#     manual_locations1 = [(0.003, 1.6), (0.009, 1.65), (0.015, 1.75), (0.024, 1.85),
#                     (0.1, 1.9), (0.3, 1.95), (0.55, 2.1), (0.85, 2.3)]

#     im = plt.imshow(np.log10(MHI_2d_5year), extent=[z.min(), z.max(), lg_W50.min(), lg_W50.max()], origin='lower', aspect='auto')  
#     plt.xlabel('Redshift z')
#     plt.ylabel('log(W50/km s$^{-1}$)') 

#     ax = plt.axes()

#     ax_dist = ax.twiny()
#     ax_dist.set_xlim(ax.get_xlim())

#     z_ticks = np.round(np.linspace(*ax.get_xlim(), 10), 2)
#     D_ticks = gf.Comoving_Dist(z_ticks).to_value(u.Mpc)

#     ax.set_xticks(z_ticks)
#     ax_dist.set_xticks(z_ticks)
#     ax_dist.set_xticklabels([f"{d:.0f}" for d in D_ticks])
#     ax_dist.set_xlabel("Distance (Mpc)")

#     CS = plt.contour(z_2d, lgW50_2d, np.log10(MHI_2d_5year),
#         levels=levels_5yr, colors='black', linewidths=1, linestyles='--') 
#     plt.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations5)

#     CS2 = plt.contour(z_2d, lgW50_2d, np.log10(MHI_2d_1year),
#         levels=levels_1yr, colors='white', linewidths=1, linestyles='-.') 
#     plt.clabel(CS2, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ", manual=manual_locations1)

#     # CS = plt.contour(z_2d, lgW50_2d, np.log10(MHI_2d),
#     #     levels=levels, colors='black', linewidths=1, linestyles='dashed') 
#     # plt.clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f}")

#     plt.set_ylabel('log(W50/km s$^{-1}$)')
    
#     cbar = fig.colorbar(im, ax=ax, pad=0.01)
#     cbar.set_label(r"log($M_{\mathrm{HI}}/M_\odot$)")
#     cbar.set_ticks([4,6,8,10])

#     legend_elements = [
#     Line2D([0], [0], color='black', lw=1, linestyle='--',
#            label='5-year survey'),
#     Line2D([0], [0], color='white', lw=1, linestyle='-.',
#            label='1-year survey'),
#     ]
#     leg = ax[1].legend(
#         handles=legend_elements,
#         ncol=1,
#         frameon=True,
#         bbox_to_anchor=(0.72, 0.92),
#         loc='center',
#     )

#     leg.get_frame().set_facecolor('gray')
#     leg.get_frame().set_alpha(0.3)

#     plt.savefig("Plots/DistanceDetect_oneplot.png")
#     plt.show()

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
        plt.title(title)
        plt.savefig(figname)
    else:
        ax.scatter(z, np.log10(MHI), color=color, label=label, s=5) #s=2)# alpha=0.5)

def HIWF():
    W50, HIWF = gf.Oman2021_HIWF()
    plt.figure()
    plt.plot(np.log10(W50), np.log10(HIWF))
    plt.show()

def LowM_Distance(catalog):
    cat = np.load(catalog)
    lg_MHI = np.log10(cat[0])
    mask = lg_MHI < 7
    Dist = cat[6]
    plt.figure()
    plt.scatter(lg_MHI[mask], Dist[mask])
    plt.xlabel('log(M$_{HI}$/M$_{\odot}$)')
    plt.ylabel('Distance (Mpc)')
    plt.savefig('Plots/Dwarf_Distances.png')
    #plt.show()

def survey_scantime():
    fig, ax = plt.subplots(2, 1, figsize=[5,8], dpi=300)
    obs_year = [5,1]
    stop = [80,50]
    switch = 7
    start=20
    for i in range(2):
        if i==0:
            beam_sep = 3
            beam_centers = np.concatenate([np.arange(start,stop[i]+beam_sep,beam_sep)[::-1], np.arange(start+beam_sep/2,stop[i]+beam_sep/2,beam_sep)])
            print(beam_centers)
            time, RMS, beam_centers = build_survey(switch_int=switch, obs_years=obs_year[i], start=20, end=stop[i], beam_centers=np.deg2rad(beam_centers))
        else:
            time, RMS, beam_centers = build_survey(switch_int=switch, obs_years=obs_year[i], start=20, end=stop[i])
        timefull = (time + switch)/365
        time = time/365
        cumultime = np.cumsum(timefull)
        left = np.zeros(cumultime.shape[0])
        left[1:] = cumultime[:-1]
        cumulswitch = np.cumsum(time)
        if i==0:
            ax[i].barh(np.rad2deg(beam_centers), timefull, left=left, height=2.5, color='green')
            ax[i].barh(np.rad2deg(beam_centers), switch/365, left=(cumultime-switch/365), height=2.5, color='orange')
            ax[i].set_title(f'{obs_year[i]} Year Survey, Sensitivity={RMS[0]/np.sqrt(2):.2f}')
        else:
            ax[i].barh(np.rad2deg(beam_centers), 12*timefull, left=12*left, height=2.5, color='green', label='Observing Time')
            ax[i].barh(np.rad2deg(beam_centers), 12*switch/365, left=12*(cumultime-switch/365), height=2.5, color='orange', label='Re-pointing Time')
            ax[i].set_title(f'{obs_year[i]} Year Survey, Sensitivity={RMS[0]:.2f}')
        ax[i].grid(True, linewidth=0.4)
        ax[i].set_ylabel('Declination Pointing (Deg)')
        
    ax[0].set_xlabel('Time (years)')
    ax[1].set_xlabel('Time (Months)')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('Plots/scan_stratergy.pdf')

#left[1:] = cumsum
#plt.barh(np.rad2deg(beam_centers), time/365, left)


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
    redshift_surveysBar()
    #survey_scantime()
    #W50z_Plane_subplots_zD()
    #W50z_Plane_oneplot()
    #check_Spectra()
    #W50z_Plane(nearby=True)
    #LowM_Distance(catalog='catalogs_output/Detected_RMS0p08_VolLim_20to80deg_zmax0p8_full.npy')
    # dndz_catalog(catalog_fl='catalogs_output/Detected_RMS0p08_VolLim_20to80deg_zmax0p8_full.npy')
    # dec_soliandang(catalogs=['DetectionsALFALFA_Vollim_Matchsim_20to80deg_Dmax200.npy',
    #                          'DetectionsALFALFA_MockSim_changeVelocity_20to80deg_Dmax200.npy'],
    #                          labels=['Uniformly Distibuted', 'Large-scale Strcture'], PlotRA=True)
    # dec_solidang_byMHI(catalogs=['DetectionsALFALFA_MockSim_changeVelocity_20to80deg_Dmax200.npy'],
    #                         labels=['Uniformly Distibuted', 'Large-scale Strcture'], normalizeDec=True)
    #dec_solidang_byMHI(catalog_lss='catalogs_output/MockAlf_D200_Dec20to80_ChangeVelocity.npy', 
    #                  catalog_volim='DetectionsALFALFA_20to80deg_Dmax200.npy')
    # dec_solidang_byMHI(catalogs=['catalogs_output/MockAlf_D200_Dec20to80_ChangeVelocity.npy'],
    #                 labels=['Uniformly Distibuted', 'Large-scale Strcture'], normalizeDec=True)

    # dec_soliandang(catalogs=['DetectionsALFALFA_20to80deg_Dmax200.npy',
    #                         'DetectionsALFALFA_MockSim_changeVelocity_20to80deg_Dmax200.npy'],
    #                         labels=['Uniformly Distibuted', 'Large-scale Strcture'], normalizeDec=False)
    #Sky_MHIsizeDcolour(catalog_file='catalogs_output/DetectionsALFALFA_20to80deg_Dmax200_ALFboundaries.npy', 
    #                   save=True, fname='Plots/Detections_ALF_Vollim.png')
    #Sky_MHIsizeDcolour(catalog_file='catalogs_output/DetectionsALFALFA_MockSim_20to80deg_Dmax200_ALFboundaries.npy',
    #                   save=True, fname='Plots/Detections_ALF_MockSim.png')
    #Sky_MHIsizeDcolour(catalog_file='catalogs_output/ALFALFA_a100_Dmax200_ALFboundaries.npy',
    #                   save=True, fname='Plots/Detections_ALF_ALFALFA.png')
    # param_distributions(catalog='DetectionsALFALFA_20to80deg_Dmax200.npy', 
    #                     flname='Plots/distributions_DetectionsALFALFA_20to80deg_Dmax200')
    # param_distributions(catalog='DetectionsALFALFA_MockSim_Brooksvelocity_20to80deg_Dmax200.npy', 
    #                     flname='Plots/distributions_DetectionsALFALFA_MockSim_Brooksvelocity_20to80deg_Dmax200')
    # param_distributions(catalog='DetectionsALFALFA_MockSim_changeVelocity_20to80deg_Dmax200.npy', 
    #                     flname='Plots/distributions_DetectionsALFALFA_MockSim_changeVelocity_20to80deg_Dmax200')
    # recover_HIMF(catalog_fl='DetectionsALFALFA_MockSimvelocity_20to80deg_Dmax200.npy', ALF=False, Vollim=False, 
    #               title='HIMF from Mock Sim (original velocities) with ALFALFA detections', mockAlf=True,
    #               plotWidths=False, nbins=20, figname='Plots/HIMF_DectectionsALFcut_MockSimVelocity.png', count_min=0)
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
    #MHI_redshift('catalogs_output/Detected_RMS0p1_VolLim_20to80deg_Dmax200.npy', figname='Plots/SpanHower_z0p8.png')
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


