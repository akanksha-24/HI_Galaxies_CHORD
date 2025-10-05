from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt

def CorrelateGalaxyPos():
    # --- Parameters ---
    boxsize = 500.0     # Mpc/h
    nmesh = 256         # resolution of the grid
    Ngal = 100000       # number of galaxies to sample

    # 1. Define a linear power spectrum using cosmology
    cosmo = cosmology.Planck15
    pk = cosmology.LinearPower(cosmo, redshift=0, transfer='EisensteinHu')

    # 2. Generate a lognormal field with that power spectrum
    mesh = LogNormalCatalog(pk, BoxSize=boxsize, Nmesh=[nmesh, nmesh, nmesh],
                            seed=42, bias=1.0)

    # 3. Sample galaxies from the field
    cat = mesh.to_catalog(N=Ngal, seed=43)

    # Get RA/Dec/redshift (observational coords) if you want
    cat['RA'], cat['DEC'], cat['Redshift'] = cat['Position'].to_sky(cosmo, observer=[0,0,0])

    # --- Quick look at clustering ---
    # Compute 2PCF to verify clustering
    r_edges = np.linspace(1, 50, 20)
    result = TwoPointCorrelationFunction('rppi', edges=r_edges, data_positions1=cat['Position'])

    # Plot positions (slice)
    pos = cat['Position'][:5000].compute()
    plt.scatter(pos[:,0], pos[:,1], s=1, alpha=0.5)
    plt.xlabel("x [Mpc/h]"); plt.ylabel("y [Mpc/h]")
    plt.title("Slice of clustered galaxy positions")
    plt.show()

def predict_W50(Vrot_kms, incl_deg, Vturb_kms=8.0, Winst_kms=0.0):
    """
    Predict observed W50 (km/s) from Vrot (km/s) and inclination (degrees).
    Vturb_kms: typical turbulent sigma -> we use 2*Vturb as an FWHM-like estimate.
    Winst_kms: instrumental FWHM in km/s (or effective broadening to add in quadrature).
    """
    i = np.deg2rad(incl_deg)
    W_rot = 2.0 * np.abs(Vrot_kms) * np.sin(i)   # intrinsic rotation contribution
    W_turb = 2.0 * np.abs(Vturb_kms)             # approximate turbulent FWHM-like term
    W_obs = np.sqrt(np.maximum(0.0, W_rot**2 + W_turb**2 + Winst_kms**2))
    return W_obs
