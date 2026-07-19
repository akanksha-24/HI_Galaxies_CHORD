from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.315)
H_z = cosmo.H(z=0.1)
print(H_z)

