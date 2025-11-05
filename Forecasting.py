import Generate_Catalog as gen
import Galaxy_Functions as gf
import Gaussian_Estimate as gauss
import Plotting as plot
import numpy as np
import astropy.units as u

def SNRint_dectections(catalog_file, RMS, sigma=6, plt=True, figname='', title=''):
    MHI, _, _, W50, _, Dec, _, _, z = gen.load_catalogParams(catalog_file)
    chan_width = (1500*u.MHz/8192).to_value(u.Hz)
    RMS = (RMS*u.mJy).to_value(u.Jy)
    W50_broad = gauss.W50_broadened(W50)
    SNRint = gf.SNR_int(z, MHI, W50_broad, chan_width=chan_width, RMS_chan=RMS)
    mask = SNRint >= sigma
    if plt==True:
        plot.Detection_counts_MHI(MHI, mask, RMS, figname=figname, title=title)
    return MHI, W50_broad, Dec, mask 

def compareCatalogs(catalog1, catalog2, RMS, sigma=6, plt=True, figname='', title=''):
    MHI1, W501, Dec1, mask1 = SNRint_dectections(catalog_file=catalog1, RMS=RMS, sigma=sigma, plt=False)
    MHI2, W502, Dec2, mask2 = SNRint_dectections(catalog_file=catalog2, RMS=RMS, sigma=sigma, plt=False)
    if plt:
        #plot.Detection_compareCats(MHI1=MHI1[mask1], MHI2=MHI2[mask2], figname=figname, title=title)
        plot.Detection_compare_Decs(MHI1[mask1], MHI2[mask2], Dec1=Dec1[mask1], Dec2=Dec2[mask2], figname=figname, title=title)


if __name__ == "__main__":
    # compareCatalogs(catalog1='catalogs_output/VolLim_20to60deg_Dmax200_rank0.npy', 
    #                 catalog2='catalogs_output/MockAlf_FullSkyD200_Dec20to80_ChangeVelocity.npy',
    #                 RMS=2, figname='Plots/Compare_VolFluxCats_RMS2_D200_MHIbins.png', title='Detections at RMS=2 mJy')
    # SNRint_dectections(catalog_file='catalogs_output/MockAlf_FullSkyD200_Dec20to80_ChangeVelocity.npy', 
    #                    RMS=2, figname='Mock_ALFALFA_RMS2.png', title='Detections from Mock ALFALFA at sensitivity of 2 mJy')
    
    SNRint_dectections(catalog_file='catalogs_output/VolLim_20to60deg_zmax0p1_rank0.npy', 
                       RMS=0.1, figname='VolLim_z0p1_RMS0p1.png', title='Detections from Volume limited at sensitivity of 0.1 mJy')





