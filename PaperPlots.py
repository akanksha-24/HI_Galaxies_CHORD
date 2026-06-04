import numpy as np
import matplotlib.pyplot as plt
import Galaxy_Functions as gf
import astropy.units as u
import astropy.constants as c
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import cm
import Generate_Catalog as gen
import Plotting as plot
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import Galaxy_Functions as gf
import matplotlib as mpl

def Table3_Ma2024():
    # Columns: logM, HIPASS, HIPASS_err, ALFALFA, ALFALFA_err, FASHI_N, FASHI_N_err, FASHI_S, FASHI_S_err, Total, Total_err
    #columns = ['logM', 'HIPASS', 'HIPASS_err', 'ALFALFA', 'ALFALFA_err',
    #           'FASHI_N', 'FASHI_N_err', 'FASHI_S', 'FASHI_S_err', 'Total', 'Total_err']
    data = np.array([
        [7.1, 8.036e-2, 8.036e-2, 3.163e-1, 0.537e-1, 1.118e-1, 0.255e-1, 4.363e-2, 4.372e-2, 1.266e-1, 0.479e-1],
        [7.3, 8.957e-2, 4.110e-2, 2.147e-1, 0.350e-1, 1.191e-1, 0.204e-1, 2.192e-2, 1.676e-2, 1.140e-1, 0.250e-1],
        [7.5, 4.045e-2, 1.852e-2, 1.919e-1, 0.250e-1, 9.678e-2, 1.408e-2, 8.694e-2, 2.899e-2, 8.063e-2, 1.204e-2],
        [7.7, 3.278e-2, 1.108e-2, 1.022e-1, 0.151e-1, 5.811e-2, 0.758e-2, 6.954e-2, 1.831e-2, 5.191e-2, 0.718e-2],
        [7.9, 4.486e-2, 1.172e-2, 9.135e-2, 0.873e-2, 6.103e-2, 0.595e-2, 3.903e-2, 1.787e-2, 5.593e-2, 0.713e-2],
        [8.1, 2.857e-2, 0.659e-2, 6.782e-2, 0.544e-2, 4.724e-2, 0.388e-2, 4.067e-2, 0.860e-2, 3.975e-2, 0.404e-2],
        [8.3, 4.807e-2, 0.695e-2, 5.202e-2, 0.349e-2, 4.227e-2, 0.285e-2, 3.796e-2, 0.774e-2, 4.716e-2, 0.413e-2],
        [8.5, 4.373e-2, 0.513e-2, 3.989e-2, 0.226e-2, 3.746e-2, 0.227e-2, 2.687e-2, 0.380e-2, 4.098e-2, 0.304e-2],
        [8.7, 3.203e-2, 0.295e-2, 3.275e-2, 0.150e-2, 3.078e-2, 0.143e-2, 3.667e-2, 0.398e-2, 3.217e-2, 0.177e-2],
        [8.9, 2.695e-2, 0.212e-2, 2.899e-2, 0.106e-2, 3.057e-2, 0.109e-2, 2.139e-2, 0.197e-2, 2.771e-2, 0.127e-2],
        [9.1, 1.884e-2, 0.131e-2, 2.547e-2, 0.075e-2, 2.569e-2, 0.078e-2, 2.085e-2, 0.167e-2, 2.142e-2, 0.079e-2],
        [9.3, 1.650e-2, 0.097e-2, 2.019e-2, 0.050e-2, 2.129e-2, 0.054e-2, 1.641e-2, 0.102e-2, 1.806e-2, 0.058e-2],
        [9.5, 1.149e-2, 0.064e-2, 1.510e-2, 0.032e-2, 1.494e-2, 0.034e-2, 1.155e-2, 0.069e-2, 1.279e-2, 0.038e-2],
        [9.7, 7.646e-3, 0.395e-3, 1.060e-2, 0.020e-2, 1.020e-2, 0.022e-2, 8.062e-3, 0.463e-3, 8.677e-3, 0.237e-3],
        [9.9, 4.032e-3, 0.250e-3, 6.383e-3, 0.124e-3, 5.525e-3, 0.130e-3, 4.463e-3, 0.251e-3, 4.757e-3, 0.150e-3],
        [10.1, 1.629e-3, 0.117e-3, 2.773e-3, 0.064e-3, 2.358e-3, 0.067e-3, 2.085e-3, 0.144e-3, 1.995e-3, 0.070e-3],
        [10.3, 5.612e-4, 0.524e-4, 8.753e-4, 0.315e-4, 8.044e-4, 0.327e-4, 7.612e-4, 0.670e-4, 6.737e-4, 0.317e-4],
        [10.5, 1.703e-4, 0.240e-4, 1.543e-4, 0.125e-4, 1.754e-4, 0.133e-4, 1.386e-4, 0.230e-4, 1.667e-4, 0.144e-4],
        [10.7, 7.614e-6, 3.232e-6, 1.670e-5, 0.405e-5, 2.709e-5, 0.506e-5, 1.055e-5, 0.611e-5, 1.307e-5, 0.225e-5],
        [10.9, np.nan, np.nan, 1.951e-6, 1.380e-6, 4.523e-6, 2.095e-6, 3.448e-6, 3.434e-6, 1.382e-6, 0.499e-6]
        ])
    return data

def ALFALFA_data():
    data = np.array([
        [6.3,0.0276812225316,0.0276812225316,1.0],
        [6.5,0.0523692162455,0.0215726434275,6.0],
        [6.7,0.0850052199706,0.02799114127,14.0],
        [6.9,0.0414296070178,0.0104943631974,19.0],
        [7.1,0.0465782100978,0.00768057905896,39.0],
        [7.3,0.0505468482757,0.00670275759855,63.0],
        [7.5,0.0526965151773,0.00518110746144,125.0],
        [7.7,0.039598516033,0.00325602462586,161.0],
        [7.9,0.0302812477439,0.0023505200804,183.0],
        [8.1,0.0303255233158,0.00196070586552,264.0],
        [8.3,0.0261821678118,0.00151080858312,326.0],
        [8.5,0.0240819975426,0.00120627674354,443.0],
        [8.7,0.0197387368449,0.00087689519889,627.0],
        [8.9,0.0167018748042,0.000592073613198,973.0],
        [9.1,0.016787076939,0.000434203186573,1812.0],
        [9.3,0.0120179525493,0.00027072425385,2434.0],
        [9.5,0.00949498146181,0.000178300988564,3366.0],
        [9.7,0.00621279943096,0.000108839454272,3753.0],
        [9.9,0.00431128021672,7.19176803476e-05,3958.0],
        [10.1,0.00247140199652,4.64539757974e-05,2931.0],
        [10.3,0.000819443257702,2.52816960961e-05,1053.0],
        [10.5,0.000189405705098,1.20760906937e-05,246.0],
        [10.7,2.30953143642e-05,4.21660821664e-06,30.0],
    ])
    return data

# def HIPASS_data():
#     data = np.array([
#         [15411163.230713407, 0.18476769434849769],
#         [20225692.44130829, 0.1331071485590051],
#         [28236119.351101607, 0.08034371985275895],
#         [40405461.50323362, 0.08901804889000405],
#         [59266405.313007504, 0.06415497450827469],
#         [77781544.97603874, 0.045062643632990604],
#         [115507931.54233117, 0.049934623201981033],
#         [163260131.92892936, 0.04753069762123127],
#         [227919641.50541875, 0.05005909405663482],
#         [322144031.5354627, 0.05012258017161764],
#         [449729834.915456, 0.036116615980453626],
#         [643556620.0007282, 0.0287989095849403],
#         [898438537.2349018, 0.0288341278300408],
#         [1269862529.4424012, 0.020777826241888103],
#         [1794837127.8070443, 0.01880158660855637],
#         [2568384896.3243747, 0.014617546414575276],
#         [3541563157.335882, 0.01080233064766741],
#         [5067923535.6548, 0.007035166356866691],
#         [7075085030.104883, 0.004140339136558963],
#         [10000000000, 0.001709900993444972],
#         [13960520478.118752, 0.001006311669643923],
#         [19731947275.143707, 0.00040520849994348315],
#         [27208514421.709694, 0.00009350331886896894],
#         [38934974326.90512, 0.00001553021034170535],
#         [57109506573.145454, 0.000003770527659233106],
#     ])
#     return data

def HIPASS_data():
    data = np.array([
        [7.188839285714286, -0.7362637362637365],
        [7.310714285714286, -0.8846153846153846],
        [7.455357142857143, -1.1016483516483517],
        [7.6120535714285715, -1.054945054945055],
        [7.774107142857143, -1.206043956043956],
        [7.8933035714285715, -1.3489010989010992],
        [8.066071428571428, -1.3159340659340664],
        [8.214732142857143, -1.3324175824175826],
        [8.358035714285714, -1.3076923076923077],
        [8.51205357142857, -1.3076923076923077],
        [8.655357142857143, -1.4532967032967035],
        [8.806696428571428, -1.5604395604395607],
        [8.954017857142858, -1.5494505494505497],
        [9.105357142857143, -1.6923076923076925],
        [9.254017857142857, -1.7362637362637363],
        [9.408035714285713, -1.8434065934065935],
        [9.55, -1.978021978021978],
        [9.7, -2.1648351648351647],
        [9.850000000000001, -2.4065934065934065],
        [9.998660714285714, -2.7802197802197806],
        [10.143303571428572, -3.01098901098901],
        [10.295982142857143, -3.4120879120879115],
        [10.43392857142857, -4.04120879120879],
        [10.589285714285714, -4.824175824175822],
        [10.754017857142857, -5.44230769230769]
    ])

    tops = np.array([
        -0.5494505494505493,
        -0.7252747252747251,
        -0.9395604395604396,
        -0.9423076923076921,
        -1.093406593406594,
        -1.2527472527472527,
        -1.2362637362637365,
        -1.2857142857142865,
        -1.2527472527472527,
        -1.2582417582417584,
        -1.4010989010989015,
        -1.5082417582417587,
        -1.4972527472527477,
        -1.6373626373626373,
        -1.6785714285714288,
        -1.7774725274725278,
        -1.9120879120879124,
        -2.107142857142857,
        -2.3379120879120876,
        -2.7252747252747254,
        -2.956043956043956,
        -3.3571428571428577,
        -3.9862637362637363,
        -4.722527472527478,
        -5.296703296703296,
    ])

    bottoms = np.array([
        -1.1153846153846156,
        -1.1923076923076925,
        -1.3873626373626375,
        -1.236263736263736,
        -1.3928571428571428,
        -1.4725274725274722,
        -1.403846153846154,
        -1.3928571428571426,
        -1.357142857142857,
        -1.3626373626373625,
        -1.5137362637362637,
        -1.60989010989011,
        -1.6016483516483517,
        -1.7362637362637363,
        -1.7912087912087915,
        -1.89010989010989,
        -2.0219780219780223,
        -2.2115384615384612,
        -2.4478021978021975,
        -2.8351648351648366,
        -3.057692307692309,
        -3.461538461538461,
        -4.112637362637363,
        -4.96978021978022,
        -5.6923076923076925,
    ])

    H0_corr = (75/70)**3
    data[:,1] = 10**(data[:,1])*H0_corr
    error = (10**tops)*H0_corr - (10**bottoms)*H0_corr
    return data[4:-1,:], error[4:-1]

def HIMF_Counts(catalogs, labels, ALF, RMS, showSurveys=True, 
                minCount=10, mockAlf=[False,False,False], solidang=[None, None, None]):
    
    mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
    })

    fig, axs = plt.subplots(2, 1, figsize=(5.5, 5), dpi=300, sharex=True)
    bins = np.linspace(5, 11, 31)
    color = cm.get_cmap('Dark2', len(catalogs)+5).colors
    #color = cm.rainbow(np.linspace(0, 1, len(catalogs)+5))
    markers = ['v', 's', 'o']
    hatches = ['', '', '']

    HI_surveys = Table3_Ma2024()
    lg_MHI = HI_surveys[:,0]
    HIPASS = HI_surveys[:,1]
    HIPASS_err = HI_surveys[:,2]
    ALFALFA = HI_surveys[:,3]
    ALFALFA_err = HI_surveys[:,4]
    FASHI_N = HI_surveys[:,5]
    FASHI_N_err = HI_surveys[:,6] 
    FASHI_S = HI_surveys[:,7]
    FASHI_S_err = HI_surveys[:,8]
    Total = HI_surveys[:,9]
    Total_err = HI_surveys[:,10]


    for i in range(len(catalogs)):
        if i!=2:
            plot.recover_HIMF(catalog_fl=catalogs[i], ax=axs[0], label=labels[i], count_min=minCount, mockAlf=mockAlf[i], solidang=solidang[i],
                                    color=color[i], ALF=ALF[i], RMS=RMS[i], bins=bins, marker=markers[i], fromD=False)
        plot.MHI_Counts(catalog_fl=catalogs[i], ax=axs[1], label=labels[i]+f': ', 
                        color=color[i], hatch=hatches[i % len(hatches)], bins=bins)
    if showSurveys:
        axs[0].errorbar(lg_MHI, ALFALFA, yerr=ALFALFA_err,
                    fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[2], label='ALFALFA')
        axs[0].errorbar(lg_MHI, HIPASS, yerr=HIPASS_err,
                fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[3], label='HIPASS')
        #print(HIPASS_err)
        axs[0].errorbar(lg_MHI, FASHI_N, yerr=FASHI_N_err,
                fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[4], label='FASHI N')
        axs[0].errorbar(lg_MHI, FASHI_S, yerr=FASHI_S_err,
                fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[5], label='FASHI S')
    #axs[0].set_yscale('log')
    axs[0].set_ylabel('$\phi(M_{HI})h_{70}^{-3}$[Mpc$^{-3}$ dex$^{-1}$]')
    axs[1].set_ylabel('Counts')
    axs[1].set_xlabel('log($M_{HI}h^{2}_{70}/M_{\odot}$)')
    axs[0].grid(True, linewidth=0.3)
    axs[1].grid(True, linewidth=0.3)
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=8)
    axs[1].legend(loc='upper left', fontsize=8)
    #axs[0].tick_params(labelbottom=False) 

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig('Plots/WoutRFI_Highz_z0p8_HIMF_Counts.png')
    #plt.show()

def HIMF_Counts_err(err_cat, labels, ALF, RMS, showSurveys=True, 
                minCount=10, mockAlf=[False,False,False], solidang=[None, None, None]):
    
    mpl.rcParams.update({
        'font.size': 14,         
        'axes.labelsize': 14,     
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })

    fig, axs = plt.subplots(2, 1, figsize=(5.5, 7.3), dpi=300, sharex=True)#, gridspec_kw={'height_ratios': [1, 1, 0.4]})
    plt.subplots_adjust(hspace=0)

    bins = np.arange(4.9, 12.2, 0.2)
    mid_bins = 0.5*(bins[1:] + bins[:-1])
    #mid_bins = bins[:-1]
    print(mid_bins.shape)
    color = cm.get_cmap('Dark2', 7).colors
    #color = cm.rainbow(np.linspace(0, 1, len(catalogs)+5))
    markers = ['v', 's', 'o']
    hatches = ['', '', '']

    HI_surveys = Table3_Ma2024()
    lg_MHI = HI_surveys[:,0]
    # HIPASS = HI_surveys[:,1]
    # HIPASS_err = HI_surveys[:,2]
    #ALFALFA = HI_surveys[:,3]
    #ALFALFA_err = HI_surveys[:,4]
    FASHI_N = HI_surveys[:,5]
    FASHI_N_err = HI_surveys[:,6] 
    FASHI_S = HI_surveys[:,7]
    FASHI_S_err = HI_surveys[:,8]
    Total = HI_surveys[:,9]
    Total_err = HI_surveys[:,10]

    ALF_data = ALFALFA_data()
    ALF_x = ALF_data[:,0]
    ALFALFA = ALF_data[:,1]
    ALFALFA_err = ALF_data[:,2]
    ALF_count = ALF_data[:,3]
    # remove datapoints which have less than 10 sources in the mass bin
    ALF_x = ALF_x[ALF_count > minCount]
    ALFALFA = ALFALFA[ALF_count > minCount]
    ALFALFA_err = ALFALFA_err[ALF_count > minCount]

    HIPASS_dat, HIPASS_error = HIPASS_data()
    HIP_x = HIPASS_dat[:,0]
    HIPASS = HIPASS_dat[:,1]
    HIPASS_err = HIPASS_error
    print("error shape", HIPASS_err.shape)

    for i in range(len(err_cat)):
        if i < 2:
            phi_counts = np.load(err_cat[i])
            phi = phi_counts[0]
            Counts = phi_counts[1]
            Counts_mean = np.mean(Counts, axis=0)
            Counts_std = np.std(Counts, axis=0)
            total_Counts = np.sum(Counts_mean)

            phi_med = np.nanpercentile(phi, 50, axis=0)
            phi_lo  = np.nanpercentile(phi, 16, axis=0)
            phi_hi  = np.nanpercentile(phi, 84, axis=0)
            phi_med[Counts_mean<minCount] = np.nan

            if i==0:
                solidang = (gf.solid_angle(dec1=20, dec2=80, ra1=0, ra2=360)*u.sr).to_value(u.deg**2)//1000*1000
            elif i==1:
                solidang = (gf.solid_angle(dec1=20, dec2=50, ra1=0, ra2=360)*u.sr).to_value(u.deg**2)//100*100

            axs[0].scatter(mid_bins, phi_med, color=color[i], label=labels[i] + ' CHORD', s=7, marker=markers[i])
            if i==0:
                axs[0].fill_between(mid_bins[1:], phi_lo[1:], phi_hi[1:], alpha=0.3, color='gray')
            # axs[0].errorbar(mid_bins, phi_mean, yerr=phi_std,
            #                 fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[0], label='5 year survey')
            mantissa, exponent = f"{total_Counts:.0e}".split("e")
            exponent = int(exponent)
            axs[1].step(mid_bins, Counts_mean, color=color[i], label=labels[i]+f', {solidang:.0f} deg$^{{2}}$ \n\t Total: {mantissa}${{\\times}}$10$^{exponent}$', where='post')
            axs[1].fill_between(mid_bins, Counts_mean-Counts_std, Counts_mean+Counts_std, alpha=0.3, color=color[i], step='post')

            # relative_uncertainty = Counts_std / Counts_mean
            # relative_uncertainty[Counts_mean < 10] = np.nan # mask
            # poisson = np.sqrt(Counts_mean) / Counts_mean
            # #mask = (Counts_mean < 10) | (Counts_mean > 100)
            # poisson[Counts_mean < 10] = np.nan
            # axs[2].scatter(mid_bins, relative_uncertainty, color=color[i], s=5)
            # axs[2].bar(mid_bins, poisson, color=color[i], linestyle='--', alpha=0.4)
            # if i==0:
            #     axs[2].scatter(9,0,s=5, color='black', label='$\sigma/<N>$')
            #     axs[2].bar(9,0, color='black', label='Poisson', alpha=0.4)
            # axs[2].set_ylim(0,1.5)

        else:
            #plot.recover_HIMF(catalog_fl=err_cat[i], ax=axs[0], label=labels[i], count_min=minCount, mockAlf=False,
            #                            color=color[i], ALF=True, RMS=None, bins=bins, marker=markers[i], fromD=False)
            alf_counts = plot.MHI_Counts(catalog_fl=err_cat[i], ax=axs[1], label='', color=color[i], bins=bins)
            mantissa, exponent = f"{alf_counts:.0e}".split("e")
            exponent = int(exponent)
            axs[1].plot(5,0,color=color[i], label=f'ALFALFA, 7000 deg$^{{2}}$ \n\t Total: {mantissa}${{\\times}}$10$^{exponent}$')

    axs[1].set_yscale('log')
    axs[0].set_yscale('log')

    if showSurveys:
        axs[0].errorbar(ALF_x, ALFALFA, yerr=ALFALFA_err,
                    fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[2], label='ALFALFA')
        # axs[0].errorbar(lg_MHI, HIPASS, yerr=HIPASS_err,
        #         fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[3], label='HIPASS')
        axs[0].errorbar(HIP_x, HIPASS, yerr=HIPASS_err,
               fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[3], label='HIPASS')
        #print(HIPASS_err)
        axs[0].errorbar(lg_MHI, FASHI_N, yerr=FASHI_N_err,
                fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[4], label='FASHI N')
        axs[0].errorbar(lg_MHI, FASHI_S, yerr=FASHI_S_err,
                fmt='.', ecolor='gray', capsize=3, elinewidth=1, markeredgewidth=0.3, color=color[5], label='FASHI S')
    #axs[0].set_yscale('log')
    
    JonesHIMF = gf.HIMF_Jones2018(MHI=gf.MHI_grid)
    MaHIMF = gf.HIMF_Ma2024(MHI=gf.MHI_grid)

    axs[0].plot(np.log10(gf.MHI_grid), JonesHIMF, label='Jones+2018', color='gray', linewidth=1, linestyle='--') # HIMF
    axs[0].plot(np.log10(gf.MHI_grid), MaHIMF, label='Ma+2025', color='purple', linewidth=1, linestyle=':') # HIMF
    
    axs[0].set_ylabel('$\phi(M_{HI})h_{70}^{-3}$[Mpc$^{-3}$ dex$^{-1}$]')
    axs[1].set_ylabel('Counts')
    axs[1].set_xlabel('log($M_{HI}h^{2}_{70}/M_{\odot}$)')
    axs[0].set_xlim(4.8,11.6)
    axs[0].set_ylim(10**-8.5, 1)
    axs[0].set_xticks([5,6,7,8,9,10,11])
    axs[0].grid(True, linewidth=0.3)
    axs[1].grid(True, linewidth=0.3)
    # axs[2].grid(True, linewidth=0.3)
    #axs[2].set_ylabel('Relative uncertainty')

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=9)
    axs[1].legend(loc='upper left', fontsize=8.7)
    axs[1].set_ylim(0.5,10**6.8)
    axs[1].set_yticks([1,10,100,10**3,10**4,10**5,10**6])
    # axs[2].legend(fontsize=8.7, bbox_to_anchor=[0.62,1], loc='upper center')
    # axs[2].set_ylabel('Relative \n Uncertainty')
    #axs[0].tick_params(labelbottom=False) 
    #axs[0].set_title('D < 40 Mpc')
    #plt.tight_layout()
    plt.savefig('Plots/PhiCounts_error_z1_HIPASS.pdf', bbox_inches='tight')
    #plt.show()

def Counts_err(err_cat):
    phi_counts = np.load(err_cat)
    bins = np.arange(4.9, 12.2, 0.2)
    print(bins.shape)
    print(phi_counts.shape)
    mid_bins = bins[:-1]
    Counts = phi_counts[1]
    Counts_mean = np.mean(Counts, axis=0)
    Counts_std = np.std(Counts, axis=0)
    Counts_poisson = np.sqrt(Counts_mean)
    print(Counts_mean)
    print(Counts_std)

    #plt.step(mid_bins, Counts_mean, color='black', where='post')
    #plt.fill_between(mid_bins, Counts_mean-Counts_std, Counts_mean+Counts_std, alpha=0.3, color='black', step='post')
    frac = Counts_std / Counts_mean
    frac_poisson = Counts_poisson / Counts_mean
    mask = Counts_mean > 10
    #plt.plot(mid_bins[mask], frac[mask])
    #plt.plot(mid_bins[mask], frac_poisson[mask])
    plt.plot(Counts_std / Counts_poisson)
    #plt.plot()
    #plt.xlim(9.75,11)
    #plt.title('0.3 < z < 0.7')
    #plt.ylabel('Counts')
    #plt.xlabel('log($M_{\rm{HI}h^{2}_{70}/M_{\odot}$)')
    plt.show()
    #plt.savefig('Plots/Highz_counts_Std.png')



def add_zmask(ax):
    x0, x1 = 0.1, 0.3   
    y0, y1 = ax.get_ylim()    
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor='red', facecolor='red', alpha=0.25)
    ax.add_patch(rect)

    legend_patch = patches.Patch(
    facecolor='red',
    edgecolor='red',
    alpha=0.25,
    label='RFI'
    )
    return legend_patch


def MHI_redshift(catalogs, labels, figname='', mask=False):
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), dpi=280)
    color = cm.get_cmap('Dark2', len(catalogs)+5).colors

    for i in range(len(catalogs)):
        if i==3:
            plot.MHI_redshift(catalog_file=catalogs[i], color=color[0], label=labels[i], ax=axs)
        else:
            plot.MHI_redshift(catalog_file=catalogs[i], color=color[i], label=labels[i], ax=axs)

    axs.set_xlabel('Redshift z')
    axs.set_ylabel('log($M_{HI}$/$M_{\odot}$)')
    axs.grid(True, linewidth=0.3)
    axs.legend()
    if mask:
        add_zmask(ax=axs)
    plt.tight_layout()
    plt.savefig(figname)

def dndz_surveys(catalogs, labels, mask=False, figname='', Usedz=True, log=False, ALF=[False,False,True]):
    fig, axs = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
    color = cm.get_cmap('Dark2', len(catalogs)+5).colors
    for i in range(len(catalogs)):
        if ALF:
            plot.dndz_catalog(catalogs[i], ax=axs, label=labels[i], color=color[i], Usedz=Usedz, bins=50)
        else:
            plot.dndz_catalog(catalogs[i], ax=axs, label=labels[i], color=color[i], Usedz=Usedz, bins=1000)
    axs.legend()
    axs.grid(True, linewidth=0.3)
    axs.set_ylabel('dN/dz')
    axs.set_xlabel('Redshift z')
    if mask:
        add_zmask(ax=axs)
    if log:
        axs.set_yscale('log')
    if Usedz==False:
        axs.set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig(figname)

def z_Counts_err(err_cats, labels, mask=True, log=True, zmax=1):
    mpl.rcParams.update({
        'font.size': 13,         
        'axes.labelsize': 13,     
        'axes.titlesize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })
    fig, axs = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
    color = cm.get_cmap('Dark2', 7).colors
    z_step = 0.008
    z_bins = np.arange(-1*z_step/2, zmax+z_step, z_step)
    mask = np.where(z_bins <= 0.77)
    z_bins = z_bins[mask]
    #mid_bins = gf.mid_bin(z_bins)
    for i in np.arange(len(err_cats)):
        err_cat = err_cats[i]
        if i < 2:
            z_counts = np.load(err_cat)
            Counts_mean = np.mean(z_counts[:,mask], axis=0)[0]
            Counts_std = np.std(z_counts[:,mask], axis=0)[0]

            axs.step(z_bins, Counts_mean, where='post', label=labels[i], color=color[i])
            axs.fill_between(z_bins, Counts_mean-Counts_std, Counts_mean+Counts_std, alpha=0.3, color=color[i], step='post')
        else:
            plot.dndz_catalog(err_cats[i], ax=axs, label='', color=color[i], Usedz=False, bins=50)
            axs.plot(0, 0, color=color[i], label='ALFALFA')
    axs.grid(True, linewidth=0.3)
    axs.set_ylabel('Counts')
    axs.set_xlabel('Redshift')
    axs.set_ylim(5, 10**5.5)
    axs.set_xlim(-0.05, 0.65)
    if mask:
        legend_patch = add_zmask(ax=axs)
    if log:
        axs.set_yscale('log')
    handles, labels = axs.get_legend_handles_labels()
    axs.legend(handles + [legend_patch], labels + ['RFI'], fontsize=11)
    plt.tight_layout()
    plt.savefig('Plots/dndz_counts_z1.pdf', bbox_inches='tight')

def Show_Catalogs_polar(catalogs):

    fig = plt.figure(figsize=(7, 6), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], figure=fig)
    ax00 = fig.add_subplot(gs[0, 0], projection='polar') 
    ax01 = fig.add_subplot(gs[0, 1], projection='polar')  
    ax10 = fig.add_subplot(gs[1, 0])  
    ax11 = fig.add_subplot(gs[1, 1]) 

    axs = np.array([[ax00, ax01],
                    [ax10, ax11]])

    for i in range(len(catalogs)):
        MHI, _, _, _, ra, dec, D, _, z = gen.load_catalogParams(catalogs[i])

        ra_rad = np.deg2rad(ra)
        ra_rad = np.remainder(ra_rad + np.pi, 2 * np.pi) - np.pi
        dec_rad = np.deg2rad(dec)

        sc = axs[0,i].scatter(ra_rad, D, cmap='viridis', c=np.log10(MHI),
          s=3,marker='.', facecolor='white',rasterized=True)

        axs[0,i].set_rmax(240)
        axs[0,i].set_rgrids([50,100,150,200], labels=['50 Mpc','100 Mpc','150 Mpc','200 Mpc'])
        axs[0,i].set_rlabel_position(90)

        axs[1,i].hist(np.log10(MHI), histtype='step')
        axs[1,i].set_yscale('log')
        axs[1,i].set_xlabel('log($M_{HI}/M\_{odot}$)')
        axs[1,i].set_ylabel('Counts')
    
    #axs[1,0].set_colorbar(sc, fraction=0.05, pad=0.1, label='log(MHI)', location='bottom')
    plt.tight_layout()
    plt.savefig('Plots/Catalogs_show.png')
    plt.close()

def dndz_surveys(catalogs, labels, mask=False, figname='', Usedz=True, log=False, ALF=[False,False,True]):
    fig, axs = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
    color = cm.get_cmap('Dark2', len(catalogs)+5).colors
    for i in range(len(catalogs)):
        if ALF:
            plot.dndz_catalog(catalogs[i], ax=axs, label=labels[i], color=color[i], Usedz=Usedz, bins=50)
        else:
            plot.dndz_catalog(catalogs[i], ax=axs, label=labels[i], color=color[i], Usedz=Usedz, bins=1000)
    axs.legend()
    axs.grid(True, linewidth=0.3)
    axs.set_ylabel('dN/dz')
    axs.set_xlabel('Redshift z')
    if mask:
        add_zmask(ax=axs)
    if log:
        axs.set_yscale('log')
    if Usedz==False:
        axs.set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig(figname)

def Show_Catalogs(catalogs):

    fig = plt.figure(figsize=(7, 4), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2], figure=fig)
    ax00 = fig.add_subplot(gs[0, 0], projection='aitoff') 
    ax10 = fig.add_subplot(gs[1, 0], projection='aitoff')  
    ax01 = fig.add_subplot(gs[0, 1])  
    ax11 = fig.add_subplot(gs[1, 1]) 

    axs = np.array([[ax00, ax01],
                    [ax10, ax11]])

    for i in range(len(catalogs)):
        MHI, _, _, _, ra, dec, D, _, z = gen.load_catalogParams(catalogs[i])

        ra_rad = np.deg2rad(ra)
        ra_rad = np.remainder(ra_rad + np.pi, 2 * np.pi) - np.pi
        dec_rad = np.deg2rad(dec)

        sc = axs[i, 0].scatter(ra_rad, dec_rad, s=0.1, c=z,cmap='viridis', alpha=0.5)

        #axs[i, 0].invert_xaxis()  # RA increases to the left
        axs[i, 1].hist(np.log10(MHI), bins=30)

    yticks = [0, 20, 40, 60, 80]
    ax00.set_yticks(np.deg2rad(yticks))
    ax00.set_yticklabels(yticks)
    xticks = [-180, -120, -60, 0, 60, 120, 180]
    ax00.set_xticks(np.deg2rad(xticks))
    ax00.set_xticklabels(xticks)
    ax00.set_ylim(0, np.pi/2)

    ax10.set_yticks(np.deg2rad(yticks))
    ax10.set_yticklabels(yticks)
    ax10.set_xticks(np.deg2rad(xticks))
    ax10.set_xticklabels(xticks)
    ax10.set_ylim(0, np.pi/2)
    #ax00.text(0, -np.pi/2 - 0.1, "Right Ascension (degrees)", ha='center', va='top', fontsize=10, transform=ax00.transAxes)
    #ax00.text(-np.pi - 0.1, 0, "Declination (degrees)", ha='right', va='center', rotation=90, fontsize=10, transform=ax00.transAxes)
    plt.savefig('Plots/Catalogs_show.png')
    plt.close()


if __name__ == "__main__":
    # Counts_err(err_cat='phi_counts_checkpoint_5yr_z1_1000.npy')
    # z_Counts_err(err_cats=['z_counts_checkpoint_5yr_z1_1000.npy',
    #                       'z_counts_checkpoint_1yr_z1_1000.npy',
    #                       'catalogs_output/ALFALFA_a100_90complete.npy'],
    #                       labels=['5 year CHORD', '1 year CHORD', 'ALFALFA'])
    HIMF_Counts_err(err_cat=['phi_counts_checkpoint_5yr_z1_1000.npy',    
                             'phi_counts_checkpoint_1yr_z1_1000.npy',
                             'catalogs_output/ALFALFA_a100_90complete.npy'], 
                    labels=['5 year', '1 year', 'ALFALFA'], ALF=None, RMS=None, showSurveys=True, minCount=10)
    #Counts_err(err_cat='phi_counts_z0p3to0p7_100.npy')
    #HIMF_Counts_err(err_cat='phi_counts_z0p3to0p7_100.npy', labels=None, ALF=None, RMS=None, showSurveys=False, minCount=10)
    #HIMF_Counts_err(err_cat='phi_counts_z0p1_100.npy', labels=None, ALF=None, RMS=None, showSurveys=True, minCount=10)
    # Show_Catalogs_polar(catalogs=['catalogs_output/MockAlf_D200_Dec20to80_ChangeVelocity.npy',
    #                        'catalogs_output/VolLim_20to80deg_Dmax200_rank0.npy'])
    # dndz_surveys(catalogs=['catalogs_output/Detected_RMS0p08_VolLim_20to80deg_zmax0p8_full.npy',
    #                       'catalogs_output/Detected1yr_RMS0p18_VolLim_20to80deg_zmax0p8_full.npy',
    #                        'catalogs_output/ALFALFA_a100_90complete.npy'], 
    #             labels=['5 year CHORD',
    #                     '1 year CHORD',
    #                     'ALFALFA $\\alpha.100$'],
    #             figname='Plots/dndz_logCounts_sruveys_masked.png', mask=True, log=True, Usedz=False)
    # MHI_redshift(catalogs=['catalogs_output/Detected_RMS0p08_VolLim_20to80deg_zmax0p8_full.npy',
    #                       'catalogs_output/Detected1yr_RMS0p18_VolLim_20to80deg_zmax0p8_full.npy',
    #                        'catalogs_output/ALFALFA_a100_90complete.npy'], 
    #             labels=['5 year CHORD',
    #                     '1 year CHORD',
    #                     'ALFALFA $\\alpha.100$'],
    #             figname='Plots/MHI_redshift_catalogs_toz0p8_masked.png', mask=True)
    # MHI_redshift(catalogs=['DetectionsVolLim_zmax0p1_5yearObs_20strips_20to80deg.npy',
    #                       'DetectionsVolLim_zmax0p1_1yearObs_20strips_20to80deg.npy',
    #                        'catalogs_output/ALFALFA_a100_90complete.npy',
    #                        'catalogs_output/Detected_VolLim_RMS0p08_20to80deg_z0p4to1_MHI9to12_new.npy'], 
    #             labels=['5 year CHORD',
    #                     '1 year CHORD',
    #                     'ALFALFA $\\alpha.100$',
    #                     ''],
    #             figname='Plots/MHI_redshift_catalogs.png')
    # HIMF_Counts(catalogs=['catalogs_output/Detected_RMS0p08_VolLim_20to80deg_zmax0p8_full.npy',
    #                       'catalogs_output/Detected1yr_RMS0p18_VolLim_20to80deg_zmax0p8_full.npy',
    #                       'catalogs_output/ALFALFA_a100_90complete.npy'], 
    #             labels=['5 year CHORD',
    #                     '1 year CHORD',
    #                     'ALFALFA $\\alpha.100$'],
    #             ALF=[False, False, True], 
    #             RMS=[0.08, 0.18, None], minCount=10)
    # HIMF_Counts(catalogs=['catalogs_output/Detected_RMS0p1_VolLim_20to80deg_Dmax200.npy',
    #                       'DetectionsVolLim_zmax0p1_fromRMS0p1_20to80deg.npy',
    #                       'catalogs_output/ALFALFA_a100_90complete.npy',], 
    #             labels=['5 year CHORD',
    #                     '1 year CHORD',
    #                     'ALFALFA $\\alpha.100$'],
    #             ALF=[False, False, True], 
    #             RMS=[0.08, 0.18, None])
    # HIMF_Counts(catalogs=['catalogs_output/Detected_VolLim_RMS0p08_20to80deg_z0p4to1_MHI9to12_new.npy'], 
    #             labels=['z=0.4-1'],
    #             ALF=[False], minCount=0,
    #             RMS=[0.08], showSurveys=False)
    # alfsolidang = (7000*u.deg**2).to_value(u.sr)
    # HIMF_Counts(catalogs=['DetectionsALFALFA_Vollim_Matchsim_20to80deg_Dmax200.npy',
    #                       'DetectionsALFALFA_MockSim_changeVelocity_20to80deg_Dmax200.npy'], 
    #         labels=['Volume-limited',
    #                 'Constrained-sim'],
    #         ALF=[False, False], 
    #         mockAlf=[True, True],
    #         solidang=[None, None],
    #         RMS=[None, None], showSurveys=False)
    