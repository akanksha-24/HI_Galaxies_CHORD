import numpy as np
import yaml
from mpi4py import MPI
from radivs_helper import *
    
# setup parallel run
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    # read parameters from config file
    with open("config.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
    # validate params
    validate_config(params)
    # set radivs frequency array to match
    if params['setup_radivs']:
        setup_radivs(params)
else:
    params=None
params = comm.bcast((params),root=0)

# Generate catalog (in parallel)
catalog = setup_catalog(params, rank, comm)
spectra, SNR = make_spectra(catalog, params, rank, comm)

full_catalog = comm.gather(catalog,root=0)
full_spectra = comm.gather(catalog,root=0)

if rank==0:
    full_catalog = np.concatenate(full_catalog, axis=1)
    full_spectra = np.concatenate(full_spectra, axis=0)
    sourcesVec = setup_sourceVectors(params, ra=full_catalog[4], dec=full_catalog[5])

    if params['save_catalog']:
        np.save(params['output_directory']+'catalog.npy', full_catalog)
    if params['save_spectra']:
        np.save(params['output_directory']+'spectra.npy', full_spectra)
    if params['save_sourceVectors']:
        np.save(params['output_directory']+'sourceVectors.npy', sourcesVec)




