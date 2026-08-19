import numpy as np
import time
from CHORD_objects import CHORD_nogaps
from helper import generate_gaussian_SI, get_radec_pixelvecs, generate_single_center_SI, gaussianTemplate
from radivs import LikelihoodChangeMapFromPointSources, LikelihoodChangeMapFromNoise, TelescopeFrequencySubrange, MaximumLikelihoodMap, SourcesInfo, MatchedFilter, produceCatalogue, ScanStrategy
from plot import plot_map_radec

if __name__ == "__main__":
	path_to_catalog = '/home/akanksha/chord/HI_Galaxies_CHORD/Catalog/'
	radivs_setup = np.load(path_to_catalog+'radivs_setup.npz')
	spectra = (np.load(path_to_catalog+'spectra.npy')).astype(np.float32)
	source_vectors = np.load(path_to_catalog+'sourceVectors.npy').astype(np.float32)

	npix_x = radivs_setup['npix_x']
	npix_y = radivs_setup['npix_y']
	extent_RA = radivs_setup['extent_RA'] #deg
	extent_Dec = radivs_setup['extent_Dec'] #deg
	base_RA = radivs_setup['base_RA'] #deg
	base_Dec = radivs_setup['base_Dec'] #deg
	telescope_pointing_dec = 45.0   # make sure to write this as a float so for example 45.0 not 45
	nchannels = spectra.shape[1]
	nsources = spectra.shape[0]
	channelMin = int(radivs_setup['channelMin'])
	channelMax = int(radivs_setup['channelMax'])
	brightness_threshold = radivs_setup['brightness_threshold']
	source_width_nchannels = 1 # set for now, TBD
	

	skyVectors = get_radec_pixelvecs(npix_x, npix_y, base_Dec, base_RA, extent_Dec, extent_RA)
	skyVectors = skyVectors.astype(np.float32)
	sourcesInfo = SourcesInfo(source_vectors, spectra, brightness_threshold)
	CHORD_frequencySubrange = TelescopeFrequencySubrange(CHORD_nogaps, channelMin, channelMax)

	myScanStrategy = ScanStrategy(telescope_pointing_dec, 1)

	print("Computing LCM from point sources")
	t1 = time.time()
	LCM_sources = LikelihoodChangeMapFromPointSources(skyVectors, myScanStrategy, CHORD_frequencySubrange, sourcesInfo)
	t2 = time.time()
	print("Finished computing LCM. Total time ", t2-t1, "s")

	print("Computing LCM from noise")
	t1 = time.time()
	LCM_noise = LikelihoodChangeMapFromNoise(skyVectors, myScanStrategy, CHORD_frequencySubrange, 12345)
	t2 = time.time()
	print("Finished computing LCM. Total time ", t2-t1, "s")

	LCM_total = 100*LCM_sources + LCM_noise

	print("Computing MLM from LCM")
	t1 = time.time()
	MLM = MaximumLikelihoodMap(skyVectors, LCM_total, myScanStrategy, CHORD_frequencySubrange)
	t2 = time.time()
	print("Finished computing MLM. Total time ", t2-t1, "s")

	#we need to produce a frequency template for the matched filter
	templateSpectrum = gaussianTemplate(nchannels, source_width_nchannels, dt=np.float32)
	print("Computing Matched Filter from LCM")
	t1 = time.time()
	MF = MatchedFilter(skyVectors, templateSpectrum, LCM_total, myScanStrategy, CHORD_frequencySubrange)
	t2 = time.time()
	print("Finished computing Matched Filter. Total time ", t2-t1, "s")

	#for plotting
	#frequency_array = CHORD_frequencySubrange.getFrequencyArray()
	#plot_map_radec (LCM_total, frequency_array, npix_x, npix_y, base_Dec, base_RA, extent_Dec, extent_RA, filename="plots/LCM_animation", title="LCM animation")
	#plot_map_radec(MLM, frequency_array, npix_x, npix_y, base_Dec, base_RA, extent_Dec, extent_RA, filename="plots/MLM_animation", title="MLM animation")
	#plot_map_radec(MF, frequency_array, npix_x, npix_y, base_Dec, base_RA, extent_Dec, extent_RA, filename="plots/MF_animation", title="MF animation")
	
	#producing catalogue
	#need to create a skyinfo dict
	skyInfo = {"LikelihoodChangeMap": LCM_total, "skyVectors": skyVectors, "ScanStrategy": myScanStrategy, "telescope": CHORD_frequencySubrange}
	print("Producing catalogue")
	t1 = time.time()
	catalogue = produceCatalogue(skyInfo, templateSpectrum, 5.0, 0.1, 100.0/1.0e3, verbose=True)
	t2 = time.time()
	print("Finished computing catalogue. Total time ", t2-t1, "s")
	print("Number of detected galaxies:", catalogue["ra"].shape[0], "out of", nsources)
