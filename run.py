import Galaxy_Functions as gf
import Plotting as plot
import Generate_Catalog as cat


# cat.Gen_Catalog(zmax=0.01, npt=10000, savelarge=False, dec1=20, dec2=80, 
#             Fluxlim=False, draw=True, flname='catalogs_output/cataVolLim_20to60deg_z0p01.npy')

# cat.Gen_Catalog(zmax=1, npt=1000000, savelarge=False, dec1=20, dec2=80, 
#               Fluxlim=False, draw=True, flname='/scratch/akanksha/catalogs/VolLim_20to60deg_z1')

cat.merge_rankfiles('/scratch/akanksha/catalogs/VolLim_20to60deg_z1')

#cat.Gen_Catalog(zmax=0.5, npt=100000, dec1=20, dec2=80,
#           Fluxlim=False, flname='/scratch/akanksha/catalogs/VolLim_20to60deg_z0p5', savelarge=False)

cat.merge_rankfiles('/scratch/akanksha/catalogs/VolLim_20to60deg_z0p5')

# N, z = cat.Gen_Catalog(zmax=0.7, npt=20000, footprint=10000*u.deg**2, savelarge=False, dec1=20, dec2=80, sigma=5,
#             Fluxlim=True, noise=0.0001686*u.Jy, vel_width=58.514*u.km/u.s, draw=False, flname='catalogs_output/VolLim_20to60deg_Dmax500.npy')

#plot.dndz(catalog=None, N=N, z=z, flname='Plots/Match_Hans.png', 
#          compareHans='/Users/akankshabij/Documents/PhD/Scripts/CHORD/Hans_detectionCounts/CHORD_21cm_galaxies_counts/Hans_predictions.npy')
#plot.param_distributions('catalogs_output/VolLim_20to60deg_Dmax200.npz', flname='VolLim_20to60deg_Dmax200')
#plot.recover_HIMF('catalogs_output/FluxLim_20to60deg_zmax0p5.npy', MHI=np.logspace(5,11,10000))
#plot.recover_HIMF('catalogs_output/VolLim_20to60deg_Dmax500.npy', Dmax=500*u.Mpc, VolLim=True, MHI=np.logspace(5,11,10000))
#MHI_VHI_polynomial()
