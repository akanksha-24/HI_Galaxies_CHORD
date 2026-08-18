import numpy as np
import sys
sys.path.append('source/')
from Generate_Catalog import *
from Galaxy_Functions import *
from CHORD_Sensitivity import *
from Gaussian_Estimate import *
from Generate_Spectra import *
import time

def validate_config(params):
    if params['zmax'] < params['zmin']:
        raise RuntimeError('zmax must be greater than zmin')

    if params['dec2'] < params['dec1']:
        raise RuntimeError('dec2 must be greater than dec1')

    if params['ra2'] < params['ra1']:
        raise RuntimeError('ra2 must be greater than ra1')

    if params['lgMHI_max'] < params['lgMHI_min']:
        raise RuntimeError('lgMHI_max must be greater than lgMHI_min')

    if params['lgMHI_max'] < params['lgMHI_min']:
        raise RuntimeError('lgMHI_max must be greater than lgMHI_min')

    #z_freq = (1420.40575177/params['fmin']) - 1
    #fmin_req = (params['zmax'] + 1)/1420.40575177
    #if (z_freq < params['zmax']):
    #    raise Warning(f"You frequency axis goes to {params['fmin']} but z={params['zmax']} requires a minimum frequency of {fmin_req}, some of your spectra will be cut-off")

def setup_CHORDobject(params):
    fmin = params['fmin']
    fmax = params['fmax']
    fres = params['fres'] / 1e3 # converting from kHz to MHz
    CHORD_fmax = params['CHORD_fmax']
    CHORD_fmin = params['CHORD_fmin']
    Nchans = int((CHORD_fmax - CHORD_fmin) / fres)

    bandwidth = (CHORD_fmax - CHORD_fmin) / Nchans
    channelMin = int((fmin - CHORD_fmin)/bandwidth)
    channelMax = int((fmax - CHORD_fmin)/bandwidth)

    return Nchans, channelMin, channelMax, bandwidth

def get_faxis(params):
    _, channelMin, channelMax, bandwidth = setup_CHORDobject(params)
    CHORD_fmin = params['CHORD_fmin']

    # create the same array as radvis - TelescopeFrequencySubrange
    newFreqmin = CHORD_fmin + bandwidth*channelMin
    newFreqmax = CHORD_fmin + bandwidth*channelMax
    newNfreq = int(channelMax - channelMin)
    faxis = np.linspace(newFreqmin, newFreqmax, newNfreq) # in MHz
    faxis = faxis * 1e6 # convert to Hz
    return faxis

def setup_SkyVectors(params):
    theta_SB = 4 # units arcmin
    theta_SBdeg = theta_SB/60 # convert to deg
    pixels_per_degree = params['pixel_resolution'] / theta_SBdeg 

    extent_RA = params['ra2'] - params['ra1']
    extent_Dec = params['dec2'] - params['dec1']
    base_RA = extent_RA / 2
    base_Dec = extent_Dec / 2
    npix_x =  int(extent_RA * pixels_per_degree)
    npix_y =  int(extent_Dec * pixels_per_degree)
    return extent_RA, extent_Dec, base_RA, base_Dec, npix_x, npix_y

def setup_radivs(params, brightness_threshold):
    Nchans, channelMin, channelMax, _  = setup_CHORDobject(params)
    extent_RA, extent_Dec, base_RA, base_Dec, npix_x, npix_y = setup_SkyVectors(params)

    radivs_params = [Nchans, channelMin, channelMax, extent_RA, extent_Dec, base_RA, base_Dec, npix_x, npix_y, brightness_threshold]

    np.savez(params['output_directory']+'radivs_setup.npz', Nchans=Nchans, channelMin=channelMin, channelMax=channelMax, 
             brightness_threshold=brightness_threshold, extent_RA=extent_RA, extent_Dec=extent_Dec, 
             base_RA=base_RA, base_Dec=base_Dec, npix_x=npix_x, npix_y=npix_y)

    if params['verbose']:
        print("............................ Recommended Radivs parameters ............................")
        print(f"CHORD object should be set to {Nchans} channels")
        print(f"TelescopeFrequencySubrange channel min is {channelMin}")
        print(f"TelescopeFrequencySubrange channel max is {channelMax}")
        print(f"extent RA is {extent_RA}")
        print(f"extent Dec is {extent_Dec}")
        print(f"base RA is {base_RA}")
        print(f"base Dec is {base_Dec}")
        print(f"npix_x is {npix_x}")
        print(f"npix_y is {npix_y}")
        print(f"brightness threshold is {brightness_threshold}")
        print("............................................................................")
    return radivs_params

def setup_sourceVectors(params, ra, dec):
    # import radivs
    sys.path.append(params['path_to_radivsexamples'])

    from helper import ang2vec
    theta_colat = np.deg2rad(90-dec)
    phi = np.deg2rad(ra)
    sourcesVec = ang2vec(theta_colat, phi, gridmode=False)
    return sourcesVec

def setup_catalog(params, rank, comm):
    if rank==0:
        # set the HIMF as Jones2018, if chooseHIMF=True then sample schechter values within error, else choose median values
        alpha, M_s, phi_s = choose_SchechParams(choose=params['chooseHIMF'])
        # alternatively, set HIMF to custom values
        if params['alpha']!= None: alpha = params['alpha']
        if params['M_s']!= None: alpha = params['M_s']
        if params['phi_s']!= None: alpha = params['phi_s']
    else:
        alpha = None; M_s = None; phi_s = None
    alpha, M_s, phi_s = comm.bcast((alpha, M_s, phi_s),root=0)

    catalog = Gen_Catalog(zmin=params['zmin'], zmax=params['zmax'], 
                          dec1=params['dec1'], dec2=params['dec2'],
                          ra1=params['ra1'],   ra2=params['ra2'], 
                          FluxCut=params['FluxCut'], delta_nu=params['fres'],
                          MHImin=params['lgMHI_min'], MHImax=params['lgMHI_max'], 
                          beam_sep=params['delta_dec'], switch_int=params['switching_time'],
                          obs_year=params['obs_yr'], sigma_int=params['sigma'], 
                          phi_s=phi_s, alpha=alpha, M_s=M_s, 
                          comm=comm, gather=False)
    return catalog

def make_spectra(catalog, params, rank, comm):
    if rank==0:
        print('Starting spectra generation...')
    start = time.time()
    size = catalog.shape[1]
    print("size is ", size)
    MHI = catalog[0]
    W50 = catalog[3]
    z = catalog[8]
    W50_broad = W50_broadened(W50)

    _, RMSmJy, _ = build_survey(obs_years=params['obs_yr'], z=z, start=params['dec1'], end=params['dec2'], 
                                beam_sep=params['delta_dec'], switch_int=params['switching_time'])
    RMS_Jy = RMSmJy * 1e-3
    SNR_catalog = SNR_int(z, MHI, W50_broad, RMS_Jy, chan_width=params['fres'])

    if rank==0:
        faxis = get_faxis(params)
    else:
        faxis = None
    faxis = comm.bcast((faxis),root=0)

    SNR_spectra = np.zeros(SNR_catalog.shape)
    S_Jy = np.zeros((catalog.shape[1], faxis.shape[0]))

    for j in np.arange(size):
        SNR_spectra[j], _, S_Jy[j] = Generate_Spectra(MHI=MHI[j], W50=W50[j], z=z[j], RMS_Jy=RMS_Jy[j], faxis=faxis)
    if rank==0:
        end = time.time()
        print('Completed spectra generation...')
        print()
    return S_Jy, SNR_spectra





