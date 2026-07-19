import numpy as np
import Galaxy_Functions as gf
import matplotlib.pyplot as plt

MockAlfCat = np.load('catalogs_output/MockAlf_D200_Dec20to80_VelocityBrooks.npy')
VolCat = np.load('catalogs_output/VolLim_20to80deg_Dmax200_rank0.npy')
                 #VolLim_20to80deg_Dmax40.npy_rank0.npy') 


cats = [VolCat, MockAlfCat]
labels=['from HIMF', 'from Mock-ALFALFA']

plt.figure()
for i in range(2):
    cat = cats[i]
    label=labels[i]
    Dmask = (cat[6,:] < 20) & (cat[6,:] > 5)
    cat_masked = cat[:,Dmask]
    counts, bins = np.histogram(np.log10(cat_masked[0]), bins=[9,10])
    total = np.sum(counts)
    counts, bins, _ = plt.hist(np.log10(cat_masked[0]), bins=[9,10], label=label+f', Total={total}', histtype='step')
    print(counts)

plt.xlabel('log($M_{HI}/M_{\odot}$)')
plt.ylabel('Counts')
plt.legend()
plt.show()