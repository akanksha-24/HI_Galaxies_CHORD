import numpy as np
import matplotlib.pyplot as plt
import Generate_Spectra as spec
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import Gaussian_Estimate as gauss
import Plotting as plot
from matplotlib.pyplot import cm
import cmcrameri.cm as cmc

def BusyShapes_varyParams():
    catalog = np.load('catalogs_output/Detected1yr_RMS0p18_VolLim_20to80deg_zmax0p8_full.npy')
    MHI = catalog[0]
    Vrot = catalog[1]
    incl=catalog[2]
    W50 = catalog[3]
    D = catalog[6]
    z = catalog[8]
    print("size of catalog ", MHI.shape[0])

    idx=[0,100,500,1000,5000,10000]

    default_c = np.asarray([1])
    default_b1 = np.asarray([1])
    default_b2 = np.asarray([1])
    default_xe = np.asarray([0])
    default_xp = np.asarray([0])

    cs = np.linspace(0,1,5)
    b1s = np.linspace(1,3,5)
    b2s = np.linspace(1,3,5)
    xes = np.round(np.linspace(-0.1,0.1,5), 2)
    xps = np.round(np.linspace(-0.1,0.1,5), 2)

    mpl.rcParams.update({
        'font.size': 20,         
        'axes.labelsize': 20,     
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16,
        'legend.title_fontsize': 16,
    })


    def plot_spectra(ax, label, idx, b1=default_b1, b2=default_b2, c=default_c, xe=default_xe, xp=default_xp, broaden=True):
        final_M, Vel, S_flux, freq = spec.Generate_Spectra(size=1, MHI=np.asarray([MHI[idx]]), 
                                                            W50=np.asarray([W50[idx]]), 
                                                            D_C=np.asarray([D[idx]]), 
                                                            z=np.asarray([z[idx]]), 
                                                            b1=b1,
                                                            b2=b2,
                                                            c=c,
                                                            xe=xe,
                                                            xp=xp, 
                                                            thermal_broaden=broaden)
        ax.plot(Vel, S_flux*1000, label=label)

        if broaden==False:
            w = W50[idx]
        else:
            w = np.sqrt(W50[idx]**2 + gauss.sigma2FWHM(sigma=10)**2)
            print("W quad is ", w)
        pad = w*0.8 # 80% of width
        mid = w/2
        ax.set_xlim(-(mid+pad), mid+pad)
        ax.legend()


    with PdfPages("Plots/Spectra_plot_Broadened.pdf") as pdf_:
        for i in range(len(idx)):
            plt.close('all')
            fig, ax = plt.subplots(3,2,figsize=[15,20], dpi=400, sharey=True, sharex=True)
            plt.subplots_adjust(wspace=0, hspace=0)
            for c in cs:
                plot_spectra(ax[0,0], idx=i, label=f'c={c}', c=np.asarray([c]))
            for b1 in b1s:
                plot_spectra(ax[1,0], idx=i, label=f'b1={b1}', b1=np.asarray([b1]))
            for b2 in b2s:
                plot_spectra(ax[1,1], idx=i, label=f'b2={b2}', b2=np.asarray([b2]))
            for xe in xes:
                plot_spectra(ax[2,0], idx=i, label=f'xe={xe}', xe=np.asarray([xe]))
            for xp in xps:
                plot_spectra(ax[2,1], idx=i, label=f'xp={xp}', xp=np.asarray([xp]))

            ax[0,1].axis('off')
            ax[2,0].set_xlabel('Velocity (km/s)')
            ax[2,0].set_ylabel('Flux Density (mJy)')
            fig.suptitle(f"log(MHI)={np.log10(MHI[i]):.2f} solM, final Vrot={Vrot[i]:.1f} km/s, incl={np.rad2deg(incl[i]):.1f} deg, W50={W50[i]:.1f} km/s, D={D[i]:.0f} Mpc")
            pdf_.savefig(fig)
            plt.close(fig)

def center_profile(V, S):
    profile_idx = np.argwhere(S > 0.5*np.max(S))[:,0]
    center_idx = int((np.max(profile_idx) + np.min(profile_idx))//2)
    V = V - V[center_idx]
    return V, S

def BusyShapes_PaperPlot():
    mpl.rcParams.update({
        'font.size': 15,         
        'axes.labelsize': 15,     
        'axes.titlesize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 9,
        'legend.title_fontsize': 9,
    })
    plt.figure(dpi=300, figsize=[6.5,5])
    catalog = np.load('catalogs_output/VolLim_20to80deg_Dmax200_rank0.npy')
    MHI = catalog[0]
    lg_MHI = np.log10(MHI)
    Vrot = catalog[1]
    incl=catalog[2]
    W50 = catalog[3]
    D = catalog[6]
    z = catalog[8]

    mask = (lg_MHI < 10) & (lg_MHI > 9)
    cvals=[0,0,0,1,2,4]
    b1vals=[1,5,1,2,5,4]
    b2vals=[1,5,2,4,5,1]
    avals=[1,1,1,1,1,1]
    nvals = [1,1,1,2,4,3]
    wvals = [4,3,2,1,1,4]
    xevals=[0,0,-0.1,0.1,0]
    xpvals=[0.1,0,0.1,-0.1,0]
    linestyles=['-', '--', '-.', ':', '-', '--']
    
    color = cm.get_cmap('Dark2', len(cvals)).colors
    # font = {'weight' : 'normal',
    #     'size'   : 12}

    # mpl.rc('font', **font)
    #color = cm.get_cmap('viridis', len(cvals)+1).colors
    #color = cm.Blues_r(np.linspace(0, 1, len(cvals)+3))
    #color = cmc.vanimo(np.linspace(0, 1, len(cvals)))
    SNRs = []
    for i in range(len(cvals)):
        final_M, Vel, S_flux, SNR  = spec.Generate_Spectra(size=1, MHI=np.asarray([MHI[mask][0]]), 
                                                                W50=np.asarray([W50[mask][0]]), 
                                                                D_C=np.asarray([D[mask]][0]), 
                                                                z=np.asarray([z[mask][0]]), 
                                                                b1=[b1vals[i]],
                                                                b2=[b2vals[i]],
                                                                c=[cvals[i]],
                                                                xe=[0],
                                                                xp=[0],
                                                                n=[nvals[i]], 
                                                                w=1,
                                                                thermal_broaden=True)
        Vel, S_flux = center_profile(Vel, S_flux)
        plt.plot(Vel, S_flux, label=f'c={cvals[i]}, b$_{{1}}$={b1vals[i]}, b$_{{2}}$={b2vals[i]}, n={nvals[i]}', color=color[i], linewidth=1.5, linestyle=linestyles[i])
        print("Final M is ", np.log10(final_M))
        SNRs.append(SNR[0])
    SNR_var = np.std(SNRs) / np.mean(SNRs)
    print("SNR variation is ", SNR_var*100)
    plt.legend(fontsize=10, loc='upper left')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Flux Density (mJy)')
    plt.xlim(-580,430)
    #plt.ylim(0,4)
    plt.title(f'log$(M_{{\mathrm{{HI}}}}/M_{{\odot}})$ = {lg_MHI[mask][0]:.1f}, $W_{{50}}$ = {W50[mask][0]:.0f} km s$^{{-1}}$', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plots/Busy_shapes.pdf', )
    plt.show()


BusyShapes_PaperPlot()





