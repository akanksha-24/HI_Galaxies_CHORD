import numpy as np
import matplotlib.pyplot as plt
import Galaxy_Functions as gf
import astropy.units as u
import astropy.constants as c

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


#param_distributions('catalogs_output/VolLim_20to60deg_Dmax200.npz', flname='VolLim_20to60deg_Dmax200')
#recover_HIMF('catalogs_output/FluxLim_20to60deg_zmax0p5.npy', MHI=np.logspace(5,11,10000))
#recover_HIMF('catalogs_output/VolLim_20to60deg_Dmax500.npy', Dmax=500*u.Mpc, VolLim=True, MHI=np.logspace(5,11,10000))
#MHI_VHI_polynomial()