from MJ_confusion_functions import *
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
import matplotlib as mpl

z_step = 1000./300000.
zstart=z_step/2
z_list = numpy.arange(zstart,1,z_step)

def fracConfusion_z():
    z_step = 500./300000.
    zstart=z_step/2
    z_list = numpy.arange(zstart,0.6,z_step)
    p_list = numpy.zeros(len(z_list))

    for i in range(len(p_list)):
        sys.stdout.write('\r')
        sys.stdout.write(str(int((i+1)*100./float(len(p_list))))+'% complete')
        sys.stdout.flush()
        
        D = co_dist(z_list[i])
        
        D_beam = phys_size((beam/60.)*numpy.pi/180.,z_list[i])
        p_list[i], _ = P_blend(D,z_list[i],D_beam,N_rands = 100000,den=False,RFI=False)

    np.save("fracz_Mdet_z0p6_zstep500.npy", p_list)

def plot_fracz_combined():
    z_step = 500./300000.
    zstart=z_step/2
    z_list = numpy.arange(zstart,0.6,z_step)
    fracz_Mdet = np.load("fracz_Mdet_z0p6_zstep500.npy")
    fracz_M0p1 = np.load("fracz_M0p1_z0p6_zstep500.npy")
    frac_Msum = np.load('../HI_stack_confusion/code/Fraction_MHI.npy')
    fracM_Mdet = np.load('fractionBlends_Detection.npy')
    fracM_M0p1 = np.load('fractionBlends_10pDetection.npy')
    #z_step_err = 500./300000.
    #z_list_err = numpy.arange(z_step_err/2,1,z_step_err)
    Sint_runs = np.load('Sint_1000runs_allz.npy')[0:360,:]
    error = np.std(Sint_runs, axis=1)
    error[error==0]=1
    print("error is ", error)

    mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10
    })
    
    fig, ax = plt.subplots(1,2, figsize=[10,4], dpi=300)
    #ax2 = ax[0].twinx()
    ax[0].fill_between(frac_Msum[0], frac_Msum[1]-error, frac_Msum[1]+error, alpha=0.3, color='gray')
    ax[0].plot(z_list, fracz_Mdet, color='Red', linewidth=3, linestyle='--', label='Rate of confusion between detections')
    ax[0].plot(z_list, fracz_M0p1, color='blue', alpha=0.5, linewidth=1.8, linestyle='-', label='Rate of confusion between a detection and $\geq 0.1\,M_{\mathrm{det}}$')
    ax[0].plot(frac_Msum[0], frac_Msum[1], color='black', alpha=0.7, linestyle='-.', linewidth=2.5, label='$M_{\mathrm{stack}} / M_{\mathrm{lim}}$')

    ax[0].legend(loc='upper right', fontsize=8)
    #ax2.legend(loc='upper right')
    ax[0].set_ylim(0, 0.82)
    #ax2.set_ylim(0, 0.82)
    ax[0].set_yticks([0,0.2,0.4,0.6,0.8])
    #ax2.set_yticks([0,0.2,0.4,0.6,0.8])
    ax[0].set_xlabel('Redshift')
    ax[0].set_ylabel('Contamination fraction')
    #ax2.set_ylabel('Fractional Mass')

    colors = cm.get_cmap('viridis')(np.linspace(0, 1, 4))
    z_step = 1000./300000.
    zstart=z_step/2
    z_list = numpy.arange(zstart,1,z_step)
    z_bins = [0,0.009,0.1,0.2,0.6]
    binwidth = 0.3
    bins = numpy.arange(5,11,binwidth)
    mid_bins = (bins[1:] + bins[:-1])*0.5
    for i in [1,2,0,3]: #range(len(z_bins)-1): 
        if i==0:
            label = 'D < 40 Mpc'
        elif i==1:
            label = 'D > 40 Mpc, z < 0.1'
        else:
            label=f'{z_bins[i]} < z < {z_bins[i+1]}'
        #ax[1].plot(mid_bins, fracM_Mdet[i], label=label, color=colors[i])
        #ax[1].plot(mid_bins, fracM_M0p1[i], label=label, color=colors[i], linestyle='--', alpha=0.4)
        ax[1].bar(mid_bins, fracM_Mdet[i], label=label, color=colors[i], width=binwidth) #edgecolor=colors[i], linewidth=2)#, edgecolor=color[i], facecolor='None')
        stair_label = r'$\geq 0.1M_{\mathrm{det}}$' if i == 0 else "_nolegend_"
        #ax[1].plot(0,0,linewidth=2, linestyle='--', color='black', label=stair_label)
        ax[1].stairs(fracM_M0p1[i], bins, color=colors[i], linewidth=2, linestyle='--', label=stair_label)
        #ax[1].step(mid_bins, fracM_M0p1[i], where='mid', color=colors[i], linewidth=2)

    handles, labels = ax[1].get_legend_handles_labels()
    order = []
    for i in range(len(z_bins) - 1):
        if i == 0:
            label = 'D < 40 Mpc'
        elif i == 1:
            label = 'D > 40 Mpc, z < 0.1'
        else:
            label = f'{z_bins[i]} < z < {z_bins[i+1]}'
        order.append(label)

    bar_handles = [handles[labels.index(o)] for o in order]
    stairs_handle = handles[labels.index(r'$\geq 0.1M_{\mathrm{det}}$')]

    # Combine everything
    final_handles = bar_handles + [stairs_handle]
    final_labels = order + [r'$\geq 0.1M_{\mathrm{det}}$']

    ax[1].legend(final_handles, final_labels, loc='upper left')

    #plt.legend()
    #plt.title('Confused with galaxy with atleast 10% of MHI')
    ax[1].set_xlabel('log($M_{\mathrm{HI}}/M_{\odot}$)')
    ax[1].set_ylabel('Fraction of Sources in blends')
    ax[0].grid(True, linewidth=0.4)
    ax[1].grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig('Plots/Confusion_paperPlot.pdf', bbox_inches='tight')
    plt.show()

def detections_And0p1(det_file, tenpercent_file):
    fraction_det = np.load(det_file)
    print(fraction_det.shape)
    tenpercent_det = np.load(tenpercent_file)
    print(tenpercent_det.shape)

    plt.figure(figsize=[5,4],dpi=300)
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, 4))
    z_bins = [0,0.009,0.1,0.2,0.6]

    binwidth = 0.5
    bins = numpy.arange(5,11,binwidth)
    mid_bins = (bins[1:] + bins[:-1])*0.5
    print("mid bins shape", mid_bins.shape)

    for i in [1,2,0,3]: #range(len(z_bins)-1): 
        if i==0:
            label = 'D < 40 Mpc'
        elif i==1:
            label = 'D > 40 Mpc, z < 0.1'
        else:
            label=f'{z_bins[i]} < z < {z_bins[i+1]}'
        plt.bar(mid_bins, fraction_det[i], label=label, edgecolor=colors[i], width=binwidth, facecolor='None') #edgecolor=colors[i], linewidth=2)#, edgecolor=color[i], facecolor='None')
        plt.bar(mid_bins, tenpercent_det[i], edgecolor=colors[i], width=binwidth, facecolor='None')

    handles, labels = plt.gca().get_legend_handles_labels()

    order = []
    for i in range(len(z_bins) - 1):
        if i==0:
            label = 'D < 40 Mpc'
        elif i==1:
            label = 'D > 40 Mpc, z < 0.1'
        else:
            label=f'{z_bins[i]} < z < {z_bins[i+1]}'
        order.append(label)

    ordered_handles = [handles[labels.index(o)] for o in order]
    plt.legend(ordered_handles, order, loc='upper left')

    #plt.legend()
    #plt.title('Confused with galaxy with atleast 10% of MHI')
    plt.xlabel('log($M_{HI}/M_{\odot}$)')
    plt.ylabel('Fraction of detections in blends')
    plt.tight_layout()
    plt.savefig('blends_MHI_z_Mdet_True.png')
    plt.show()

def fracConfusion_MHI():
    p_list = numpy.zeros(len(z_list))
    N_list = numpy.zeros(len(z_list))
    catalog_list = []
    blended_list = []

    for i in range(len(p_list)):
        sys.stdout.write('\r')
        sys.stdout.write(str(int((i+1)*100./float(len(p_list))))+'% complete')
        sys.stdout.flush()
        
        D = co_dist(z_list[i])
        
        D_beam = phys_size((beam/60.)*numpy.pi/180.,z_list[i])
        
        N_list[i] = (area/3.)*(co_dist(z_list[i]+z_step/2.)**3. - co_dist(z_list[i]-z_step/2.)**3.)*n_MHI_dist(co_dist(z_list[i]),z_list[i])
        p_list[i], _ = P_blend(D,z_list[i],D_beam,N_rands = 100000,den=False,RFI=False)

        min_mass = max(5,det_lim_gen(numpy.log10(15.),D,z_list[i]))
        max_mass = 12
        logMHI = numpy.linspace(min_mass, max_mass, 100000)
        himf = HIMF(logMHI)
        pdf = himf / numpy.sum(himf)
        catalog_list.append(numpy.random.choice(logMHI, size=int(N_list[i]), p=pdf))
        
        #logMHI = numpy.linspace(5, max_mass, 100000)
        logMHI = numpy.linspace(min_mass, max_mass, 100000)
        himf = HIMF(logMHI)
        pdf = himf / numpy.sum(himf)
        blended_list.append(numpy.random.choice(logMHI, size=int(N_list[i]*p_list[i]), p=pdf))

    with open("MJ_allsources_MHI_5year_z0to1_M01p_True.pkl", "wb") as f:
        pickle.dump(catalog_list, f)

    with open("MJ_blendedsources_MHI_5year_z0to1_M01p_True.pkl", "wb") as f:
        pickle.dump(blended_list, f)

def plot_Confusion_MHI(MHI_catfile, MHI_blendfile):
    with open(MHI_catfile, "rb") as f:
        MHI_catalog = pickle.load(f)
    with open(MHI_blendfile, "rb") as f:
        MHI_blended = pickle.load(f)

    plt.figure(figsize=[5,4],dpi=300)

    colors = cm.get_cmap('viridis')(np.linspace(0, 1, 4))
    z_bins = [0,0.009,0.1,0.2,0.6]

    binwidth = 0.3
    bins = numpy.arange(5,11,binwidth)
    mid_bins = (bins[1:] + bins[:-1])*0.5
    fraction_blends = np.zeros((len(z_bins), len(mid_bins)))

    for i in [1,2,0,3]: #range(len(z_bins)-1): 
        idx1 = numpy.argmin(numpy.abs(z_list - z_bins[i]))
        idx2 = numpy.argmin(numpy.abs(z_list - z_bins[i+1]))
        total = numpy.concatenate([numpy.array(x) for x in MHI_catalog[idx1:idx2]])
        blend = numpy.concatenate([numpy.array(x) for x in MHI_blended[idx1:idx2]])
        total_counts, bin_edges = numpy.histogram(total, bins=bins)
        print("total counts is ", total_counts)
        blended_counts, bin_edges = numpy.histogram(blend, bins=bins)
        print("blended counts is ", blended_counts)
        fraction_blended = blended_counts / total_counts
        if i==0:
            label = 'D < 40 Mpc'
        elif i==1:
            label = 'D > 40 Mpc, z < 0.1'
        else:
            label=f'{z_bins[i]} < z < {z_bins[i+1]}'
        plt.bar(mid_bins, fraction_blended, label=label, color=colors[i], width=0.5, alpha=0.2) #edgecolor=colors[i], linewidth=2)#, edgecolor=color[i], facecolor='None')
        fraction_blends[i] = fraction_blended
        #plt.set_ylim(0,0.4)

    handles, labels = plt.gca().get_legend_handles_labels()

    order = []
    for i in range(len(z_bins) - 1):
        if i==0:
            label = 'D < 40 Mpc'
        elif i==1:
            label = 'D > 40 Mpc, z < 0.1'
        else:
            label=f'{z_bins[i]} < z < {z_bins[i+1]}'
        order.append(label)

    ordered_handles = [handles[labels.index(o)] for o in order]
    plt.legend(ordered_handles, order, loc='upper left')

    #plt.legend()
    #plt.title('Confused with galaxy with atleast 10% of MHI')
    plt.xlabel('log($M_{HI}/M_{\odot}$)')
    plt.ylabel('Fraction of detections in blends')
    plt.tight_layout()
    plt.savefig('blends_MHI_z_Mdet_True.png')
    plt.show()
    np.save('fractionBlends_10pDetection.npy', fraction_blends)
    

plot_fracz_combined()
#fracConfusion_z()
#detections_And0p1(det_file='fractionBlends_Detection.npy', tenpercent_file='fractionBlends_10pDetection.npy')
#fracConfusion_MHI()
#plot_Confusion_MHI(MHI_catfile='MJ_allsources_MHI_5year_z0to1_M01p_True.pkl', 
#                    MHI_blendfile='MJ_blendedsources_MHI_5year_z0to1_M01p_True.pkl')