import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl
import matplotlib.gridspec as gridspec

apf = np.loadtxt("elevation_axis_alignment_points_9Jul2025.csv", delimiter=',', skiprows=1, usecols=[1,2,3,4]) #alignment points file

x = (apf[:,0] + apf[:,2])/2
y = (apf[:,1] + apf[:,3])/2

x = x - np.min(x)
y = y - np.min(y)

#stringing together the indices for Pathfinder.
# previously 7x10 pathfinder grid
#pathfinder_indices = np.concatenate((np.arange(0,10), np.arange(23,32), np.arange(47, 56), np.arange(71,80), np.arange(95,104), np.arange(119,128), np.arange(143, 152)))
# Updtae: 8x8 pathfinder grid
pathfinder_indices = np.concatenate((np.arange(0,8), np.arange(23,31), np.arange(47, 55), np.arange(71,79), np.arange(95,103), np.arange(119,127), np.arange(143, 151), np.arange(167, 175)))

# plt.figure(figsize=[5,6], dpi=300)
# plt.scatter(np.delete(x,pathfinder_indices),np.delete(y,pathfinder_indices), label='CHORD Core')
# plt.scatter(x[pathfinder_indices],y[pathfinder_indices], label='Pathfinder')
# ax = plt.gca()
# ax.set_aspect('equal', "box")
# #plt.title("CHORD dish layout")
# plt.xlabel("E-W (m)")
# plt.ylabel("N-S (m)")
# plt.grid(True, linewidth=0.4)
# plt.legend(loc='lower center', bbox_to_anchor=(0.78, 1.01))
# plt.savefig("CHORD_dish_layout.pdf", bbox_inches="tight")

mpl.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

fig, ax = plt.subplots(1, 2, figsize=(7, 4.5), dpi=350)
colors = cm.get_cmap('Dark2', 4).colors
ax[0].scatter(np.delete(x,pathfinder_indices),np.delete(y,pathfinder_indices), color=colors[1])
ax[0].scatter(x[pathfinder_indices],y[pathfinder_indices], color=colors[0])
ax[0].set_aspect('equal', "box")
ax[0].set_xlabel("East-West Position (m)")
ax[0].set_ylabel("North-South Position (m)")
ax[0].grid(True, linewidth=0.4)
#ax[0].legend(loc='lower center', bbox_to_anchor=(0.67, 1.01), fontsize=12)
ax[0].set_xticks([0,50,100,150])
ax[0].set_yticks([0,50,100,150,200])
baselines = []
pathfinder_baselines = []
N = len(x)
print(pathfinder_indices)

for i in range(N):
    for j in range(N): 
        if i!=j: 
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            baselines.append(d)

for i in pathfinder_indices:
    for j in pathfinder_indices:
        if i!=j: 
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            pathfinder_baselines.append(d)

baselines = np.array(baselines)
print("max baseline is ", np.max(baselines))
print("min baseline is ", np.min(baselines))
print("number of baselines ", baselines.shape)
print("number of pathfinder baselines", len(pathfinder_baselines))
pathfinder_baselines = np.array(pathfinder_baselines)
counts, bins = np.histogram(baselines, bins=100)
path_counts, path_bins = np.histogram(pathfinder_baselines, bins=100)
mid_bins = (bins[1:] + bins[:-1])*0.5
mid_path = (path_bins[1:] + path_bins[:-1])*0.5
ax[1].bar(mid_bins, counts, width=2.5, color=colors[1], label='CHORD Core')  
ax[1].bar(mid_path, path_counts, width=2.5, color=colors[0], label='Pathfinder')
ax[1].set_xlabel("Baseline Length (m)")
ax[1].set_ylabel("Number of Redundant Baselines")
ax[1].set_yticks([0,2000,4000,6000,8000])
ax[1].set_yscale('log')
ax[1].set_ylim(0,10**4.5)
ax[1].set_xticks([0,100,200])
ax[1].legend(loc='upper right', fontsize=10)
ax[1].grid(True, linewidth=0.4)
plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.savefig('Paperplots/CHORD_layout_log.pdf',  bbox_inches='tight')

