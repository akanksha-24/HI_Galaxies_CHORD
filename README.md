# HI_GALAXIES_CHORD

This code generates a catalog of realistic galaxies drawn from the HIMF. The catalog products can then be inserted into the [Radivs](https://gitlab.com/hanshopkins/radivs/) pipeline. The catalog assumes a uniform distribution of galaxies. For more information on the catalog and spectra generation, see [Bij et al. 2026](https://arxiv.org/abs/2607.24903).

## 1. Modify config.yaml

To setup the catalog configuration, modify the `config.yaml` file to the type of catalog you are generating using the following parameters:
### Radivs Setup
- **setup_radivs**
  - Set to `True` if you are inserting the catalog to the radivs pipeline. This will save a `radivs_setup.npz` file which can then be used for the radivs CHORD simulation.
- **verbose**
  - Set to `True` if you want to print out the radivs pipeline parameters
- **path_to_radivs**
  - If using radivs, set this path to where the [Radivs](https://gitlab.com/hanshopkins/radivs/) code is located locally on your machine. Else, you may leave it blank
- **path_to_radivsexamples**
  - If using radivs, set this path to where the [Radivs_examples](https://gitlab.com/hanshopkins/radivs_examples/) code is located locally on your machine
- **pixel_resolution**
  - Set a pixel resolution for SkyVectors in radivs. This will set npix_x and npix_y based on your survey ra/dec parametes. This value is in units pixels/synthesized beam. Recommend: 3-5 pixels/beam
### Survey Parameters
- **zmax**
  - This sets the maximum redshift of the catalog 
- **zmin**
  - This sets the minimum redshift of the catalog
- **dec1**
  - Set the starting declination (in degrees) for the catalog (minimum for CHORD observations is 20 deg)
- **dec2**
  - Set the highest declination (in degrees) for the catalog (maximum for CHORD observations is 80 deg). dec2 must be greater than dec1. (dec2-dec1) will set your `extent_Dec` and (dec2-dec1)/2 is `base_Dec` in radivs.
- **ra1**
  - Set the starting RA for the catalog
- **ra2**
  - Set the ending RA for the catalog. A full drift-scan sky will have ra1=0 to ra2=360 deg (angle wrapping is accounted for). ra2 must be greater than ra1. (ra2-ra1) will set your `extent_RA` and (dec2-dec1)/2 is `base_RA` in radivs.
### HIMF + MHI Parameters
- **chooseHIMF**
  - Set to `False` if you want the median values of the [Jones et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.477....2J/abstract) Schechter HIMF fit. Set to `True` if you want to sample the Schechter fit within the quoted errors. (unless you are testing the impact of the variation in the HIMF shape, you likely want to set this to False).
- **phi_s, M_s, alpha**
  - If you want to set a custom HIMF fit (not from Jones et al. 2018), then set phi, Ms, alpha. Leave as default (~) to use Jones et al. 2018         
- **lgMHI_min**
  - minimum MHI mass to drawn from HIMF in log
- **lgMHI_max**
  - maximum MHI mass to draw from HIMF in log
- **FluxCut** 
  - This value will set whether you want a Volume-limited catalog (i.e all sources that should theoretically be present based on the HIMF) or Flux-limited catalog (i.e sources that are brighter than a given threshold). Set this number to be between 0-1. `FluxCut=0` generates a volume limited catalog. `Fluxcut > 0` applies a flux cut with MHI > MHI_det*FluxCut where MHI_det is the MHI limit expected to be detected by CHORD using Equation 10 in Bij et al. 2026. For example, `FluxCut=0.1` will produce a catalog of sources that are at least 10% as massive as the detection limit. `FluxCut=1` will make a catalog of sources expected to be above the CHORD detection limit for an integrated signal to noise > 6. 
### Spectra Parameters
- **fmin**
  - lowest frequency of the spectral axis in MHz. To not cut-off any spectra, make sure this matches zmax.    
- **fmax**
  - highest frequency of the spectral axis in MHz. fmax should be greater than fmin
- **fres**
  - frequency resolution in kHz. Note the native CHORD resolution is 195.3125 kHz. For upchannelization to 5 km/s at z=0 fres=23.689 kHz
- **flux_units**
  - Set to 1 for Jy, 1000 for mJy and 1000000 for uJy, etc. The output map from radivs will also be in the same units.
### RMS Estimation Parameters 
Set these if you are interested in comparing the RMS noise and SNR estimation from the radiometer equation to radivs. Note that these values assume full drift scans with uniform sensitivity between dec1 and dec2
- **obs_yr**
  - The number of years you will spend scanning dec1 to dec2
- **sigma**
  - The integrated signal to noise threshold above which sources are considered detected. Bij et al. 2026 uses sigma=6
- **delta_dec**
  - declination separation between pointings in degrees
- **switching_time**
  - time it takes to repoint the dishes (in days) 
### Output settings
- **output_directory**
  - path location to store all outputs
- **save_catalog**
  - True if you want to save the catalog, else False
- **save_sourceVectors**
  - True if you want to save the source_vectors to be input into radivs in shape (nsources, 3) and np.float32 - follows the same indexing as catalog
- **save_spectra**
  - True if you want to save the spectra in shape (nsources, nchans) 
- **diagnostic_Plots**
  - True if you want to make standard plots of the catalog and spectra to check the outputs
### CHORD Params
- **CHORD_fmin**
  - The minimum CHORD frequency, this should match CHORDObject in radivs
- **CHORD_fmax**
  - The maximum CHORD frequency, this should match CHORDObject in radivs

## 2. Running the catalog simulation
If you have a short run (say < 1e5 sources), then you can run it serially with:
```
python Sim_catalog.py
``` 
If you have a large run that requires a parallelization, you can use the batch script `batch_job.sh` provided and run on fir:
```
sbatch bacth_job.sh
```
This will send your job to the slurm queue. If you want to check how the job is doing run:
```
squeue -u <fir username>
```
and check the output log generated at `slurm-<jobid>.out`

## 3. Parallel slurm job parameters
For more information see the Compute Canada [guide](https://docs.alliancecan.ca/wiki/Running_jobs#MPI_job) for MPI Jobs and the CPU architecture of [FIR](https://docs.alliancecan.ca/wiki/Fir#CPU_nodes).
- **SBATCH --ntasks**
  - this is the total number of cores that will work in parallel
- **SBATCH --nodes**
  - this is the number of nodes you want to run on. For a large job, I recommend 4-16. The more nodes you ask for the longer it may queue you for.
- **SBATCH --ntasks-per-node**
  - this is the number of cores running in parallel per node. So ntasks = nodes*ntasks-per-node. For a large job, I recommend setting this to 50-100. The maximum cores is 192 for our node.
- **SBATCH --cpus-per-task=1**
  - keep this as 1 for our purposes
- **SBATCH --mem=750G**
  - Recommend using the 750G AMD EPYC 9655 (Zen 5) nodes since there are 860 of them and the memory is sufficient
- **SBATCH --time**
  - set in hh:mm:ss. Try to keep this to under 8-10 hours, otherwise you will be queued for longer

## 4. example_galaxyCatalog.py
This file shows an example of how to plug in the catalog into the radivs pipeline. You may copy this into your radivs_examples directory and run with `python example_galaxyCatalog.py` to run the radivs pipeline.

## 5. Catalog outputs
### catalog.npy 
The galaxy catalog will be saved in a file `catalog.npy` as numpy array with shape (10, nsources). The catalog contains the following information
```
catalog = np.load('catalog.npy')
```
1. `MHI = catalog[0]` is the HI mass of the source in solar masses
2. `Vrot = catalog[1]` is the rotational velocity of the source in km/s 
3.  `i = catalog[2]` is the inclination of the source in radians
4.  `W_50 = catalog[3]`is the spectral width measured at FWHM
5. `ra = catalog[4]` is the RA position in degrees
6. `dec = catalog[5]` is the declination position in degrees
7. `D = catalog[6]` is the co-moving distance to the source in Mpc
8. `Vol = catalog[7]` is the volume associated with the source distance in Mpc**3  
9. `z = catalog[8]` is the redshift of the source
10. `SNR = catalog[9]` is the signal-to-noise of the source from the integrated spectra and estimated RMS from the radiometer equation

### radivs_setup.npz
This file contains the input parameters to radivs_examples to match the catalog to the simulation. It contains:
```
setup = np.load('radivs_setup.npz')
```
1. `setup['Nchans']` is the number of channels to be specified for the CHORDObject
2. `setup['channelMin']` is the min channel to be specified for TelescopeFrequencySubrange to match fmin in the config file
3. `setup['channelMax']` is the max channel to be specified for TelescopeFrequencySubrange to match fmin in the config file
4. `setup['extent_RA']` is the RA extent to be specified when making SkyVectors for the radivs map
5. `setup['extent_Dec']` is the Dec extent to be specified when making SkyVectors for the radivs map
6. `setup['base_RA']` is the RA base to be specified when making SkyVectors for the radivs map
7. `setup['npix_x']` is the number of pixels in the x direction when making SkyVectors for the radivs map
8. `setup['npix_y']` is the number of pixels in the y direction when making SkyVectors for the radivs map
9. `setup['brightness_threshold']` is the flux cutoff (in the units of the spectra) below which the matched filter will ignore spectra channels

### spectra.npy
This is the spectra of all sources in shape (nsources, nchans). The units of the flux density of those as specified in the config file. The frequency axis is set to match CHORD_frequencySubrange.getFrequencyArray()

### sourceVectors.npy
This is the sources_vectors that will be put into radivs SourcesInfo in shape (nsources, 3). They are the x,y,z vectors pointing to the RA,Dec positions of the sources using `ang2vec` in `helper.py` in `radivs_examples`


