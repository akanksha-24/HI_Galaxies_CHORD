from Generate_Catalog import *
from Galaxy_Functions import *
from Gaussian_Estimate import *
import numpy as np
import matplotlib.pyplot as plt

def get_MHIcounts_SNR(zmax=1, dec1=20, dec2=80, zmin=0, obs_year=5):
    upchan_res = width_vel2freq(del_Vrest=5)
    solidang = solid_angle(dec1=dec1, dec2=dec2, ra1=0, ra2=360)

    catalog = Gen_Catalog(zmax=zmax, dec1=dec1, dec2=dec2, zmin=zmin, save=False, Fluxlim=True, obs_year=obs_year)
    MHI = catalog[0]
    W50 = catalog[3]
    D = catalog[6]
    z = catalog[8]
    W50_broad = W50_broadened(W50)
    dec = catalog[5]

    _, RMS_mJy, _ = build_survey(switch_int=7, obs_years=obs_year, z=z, start=dec1, end=dec2)
    print(RMS_mJy)
    SNR = SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=upchan_res)
    fig, ax = plt.subplots(2,1, sharex=True, figsize=[9,8], dpi=300)
    total = len(MHI[SNR > 6])
    mask = (SNR > 6) & (SNR < 7)
    ax[0].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'6 < SNR < 7, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 7) & (SNR < 8)
    ax[0].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'7 < SNR < 8, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 8) & (SNR < 9)
    ax[0].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'8 < SNR < 9, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 9) & (SNR < 10)
    ax[0].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'9 < SNR < 10, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 10) & (SNR < 15)
    ax[0].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'10 < SNR < 15, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 15) & (SNR < 20)
    ax[0].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'15 < SNR < 20, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = SNR > 20
    ax[0].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'SNR > 20, total={len(np.log10(MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    ax[0].set_title(f'5-year Total Detections: {total}')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('log(MHI)')
    ax[0].set_ylabel('Counts')
    ax[0].legend(loc='lower center', fontsize=8)

    catalog = Gen_Catalog(zmax=zmax, dec1=dec1, dec2=50, zmin=zmin, save=False, Fluxlim=True, obs_year=1)
    MHI = catalog[0]
    W50 = catalog[3]
    D = catalog[6]
    z = catalog[8]
    W50_broad = W50_broadened(W50)
    dec = catalog[5]

    _, RMS_mJy, _ = build_survey(switch_int=7, obs_years=1, z=z, start=dec1, end=50)
    print(RMS_mJy)
    SNR = SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=upchan_res)
    total = len(MHI[SNR > 6])
    mask = (SNR > 6) & (SNR < 7)
    ax[1].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'6 < SNR < 7, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 7) & (SNR < 8)
    ax[1].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'7 < SNR < 8, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 8) & (SNR < 9)
    ax[1].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'8 < SNR < 9, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 9) & (SNR < 10)
    ax[1].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'9 < SNR < 10, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 10) & (SNR < 15)
    ax[1].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'10 < SNR < 15, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = (SNR > 15) & (SNR < 20)
    ax[1].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'15 < SNR < 20, total={len((MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    mask = SNR > 20
    ax[1].hist(np.log10(MHI)[mask], bins=np.arange(4.8,12,0.2), histtype='step', linewidth=1.5, label=f'SNR > 20, total={len(np.log10(MHI)[mask]):.0f}, {(len((MHI)[mask])/total)*100:.0f}%')
    ax[1].set_title(f'1-year Total Detections: {total}')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('log(MHI)')
    ax[1].set_ylabel('Counts')
    ax[1].legend(loc='lower center', fontsize=8)
    plt.savefig('Plots/SNR_counts.pdf')


def get_SNRCounts(zmax=1, dec1=20, dec2=80, zmin=0, obs_year=5):
    upchan_res = width_vel2freq(del_Vrest=5)
    solidang = solid_angle(dec1=dec1, dec2=dec2, ra1=0, ra2=360)

    catalog = Gen_Catalog(zmax=zmax, dec1=dec1, dec2=dec2, zmin=zmin, save=False, Fluxlim=True, obs_year=obs_year)
    MHI = catalog[0]
    W50 = catalog[3]
    D = catalog[6]
    z = catalog[8]
    W50_broad = W50_broadened(W50)
    dec = catalog[5]

    _, RMS_mJy, _ = build_survey(switch_int=7, obs_years=obs_year, z=z, start=dec1, end=dec2)
    print(RMS_mJy)
    SNR = SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=upchan_res)
    #fig, ax = plt.subplots(2,1, sharex=True, figsize=[9,8], dpi=300)
    fig = plt.figure(figsize=[9,8], dpi=300)
    total = len(MHI[SNR > 6])
    plt.hist(SNR, bins=np.arange(6,50,1), histtype='step', label=f'5-year Total Detections: {total}')
    plt.yscale('log')
    plt.xlabel('SNR')
    plt.ylabel('Counts')
    #ax[0].legend(loc='lower center', fontsize=8)

    catalog = Gen_Catalog(zmax=zmax, dec1=dec1, dec2=50, zmin=zmin, save=False, Fluxlim=True, obs_year=1)
    MHI = catalog[0]
    W50 = catalog[3]
    D = catalog[6]
    z = catalog[8]
    W50_broad = W50_broadened(W50)
    dec = catalog[5]

    _, RMS_mJy, _ = build_survey(switch_int=7, obs_years=1, z=z, start=dec1, end=50)
    print(RMS_mJy)
    SNR = SNR_int(z, MHI, W50_broad, RMS_mJy*1e-3, chan_width=upchan_res)
    total = len(MHI[SNR > 6])
    plt.hist(SNR, bins=np.arange(6,50,1), histtype='step', label=f'1-year Total Detections: {total}')
    plt.legend()
    plt.savefig('Plots/SNR_histpdf')

def ALF_SNRCounts():
    alf = np.load('catalogs_output/ALFALFA_a100_90complete.npy')
    SNR = alf[2]
    fig = plt.figure(figsize=[9,8], dpi=300)
    #total = len(MHI[SNR > 6])
    plt.hist(SNR, bins=np.arange(6,50,1), histtype='step') #label=f'ALFALFA Total Detections: {total}')
    plt.yscale('log')
    plt.xlabel('SNR')
    plt.ylabel('Counts')
    #ax[0].legend(loc='lower center', fontsize=8)
    plt.savefig('Plots/SNR_hist_Alf.pdf')

def MHI_dist_SNR():
    upchan_res = width_vel2freq(del_Vrest=5)
    lg_MHI = np.arange(5,12,1e-2)
    z = np.linspace(1e-3,1,10000)
    z_2d, lgMHI_2d = np.meshgrid(z, lg_MHI)
    _, RMS_5yr, _ = build_survey(switch_int=7, obs_years=5, z=z_2d, start=20, end=80)
    _, RMS_1yr, _ = build_survey(switch_int=7, obs_years=1, z=z_2d, start=20, end=50)
    SNR_5yr = SNR_int(z_2d, 10**lgMHI_2d, DeltaV=25, RMS_chan=RMS_5yr*1e-3, chan_width=upchan_res)
    SNR_5yr[SNR_5yr < 6] = np.nan
    SNR_1yr = SNR_int(z_2d, 10**lgMHI_2d, DeltaV=25, RMS_chan=RMS_1yr*1e-3, chan_width=upchan_res)
    SNR_1yr[SNR_1yr < 6] = np.nan

    fig, ax = plt.subplots(2,1,sharex=True, figsize=[12,8],dpi=300)
    plt.subplots_adjust(hspace=0.1)
    #CS = ax[0].contour(z_2d, lgMHI_2d, SNR_5yr, levels=[6,10,20,30], linewidths=1, linestyles='--', cmap='Blues_r') 
    #ax[0].clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ")
    ax[0].set_ylabel('log(MHI)')

    #CS = ax[1].contour(z_2d, lgMHI_2d, SNR_1yr, levels=[6,10,20,30], linewidths=1, linestyles='--', cmap='Blues_r') 
    #ax[1].clabel(CS, inline=True, inline_spacing=0, fontsize=10, fmt=lambda v: f" {v:.1f} ")
    ax[1].set_ylabel('log(MHI)')
    ax[1].set_xlabel('Redshift')
    im = ax[0].imshow(SNR_5yr, extent=[z.min(), z.max(), lg_MHI.min(), lg_MHI.max()], origin='lower', aspect='auto', vmax=50, vmin=6) 
    ax[0].set_title('5 year')
    im = ax[1].imshow(SNR_1yr, extent=[z.min(), z.max(), lg_MHI.min(), lg_MHI.max()], origin='lower', aspect='auto', vmax=50, vmin=6) 
    ax[1].set_title('1 year')

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(r"SNR")
    cbar.set_ticks(np.arange(6,50,2))

    plt.savefig('Plots/SNR_MHI_z_colour.pdf')
    
MHI_dist_SNR()

#ALF_SNRCounts()
