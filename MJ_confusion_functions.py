import numpy, scipy, scipy.integrate, scipy.optimize,scipy.interpolate
import time, sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pickle

#Include the relevant cosmology
def E(z):
    
    #WMAP9
    H0 = 69.7 #km/s/Mpc
    O_m = 0.2814
    O_L = 1.-O_m
    c = 299792.458 #km/s
    
    return numpy.sqrt((O_m*((1.+z)**3.)) + O_L)

def co_dist(z):
    
    #WMAP9
    H0 = 69.7 #km/s/Mpc
    O_m = 0.2814
    O_L = 1.-O_m
    c = 299792.458 #km/s
    
    d_H = c/H0
    
    return d_H * scipy.integrate.quad(lambda x: 1./E(x),0.,z)[0]

def phys_size(theta,z):
    
    #Beam size changes with freq. (and thus z)
    theta_eff = theta*(1.+z)
    
    return co_dist(z)*theta_eff/(1.+z)

def line(x,a,b):
    
    return a*x + b

def skewnormal(x,alpha,omega,xi,A):
    '''Skewed Normal distribution.'''
    
    z = (x-xi)/omega
    
    return A*(1./(omega*numpy.sqrt(2.*numpy.pi)))*numpy.exp(-0.5*z**2.)*(1.+scipy.special.erf(alpha*z/numpy.sqrt(2.)))

def gauss(x,mu,sig):
    '''Normal distribution truncated at x = log10(15) and x = 3, scaled such that it always integrates to 1.'''
    
    A = 2./(scipy.special.erf((3.-mu)/(sig*numpy.sqrt(2.))) - scipy.special.erf((numpy.log10(15.)-mu)/(sig*numpy.sqrt(2.))))
    
    return A*numpy.array(numpy.greater(x,numpy.log10(15.)),dtype='float')*numpy.array(numpy.greater(3.,x),dtype='float')*numpy.exp(-((x-mu)**2.)/(2.*sig*sig))/(sig*numpy.sqrt(2.*numpy.pi))

def gumbel(x,mu,beta):
    
    zm = -(x-mu)/beta
    
    A = 1./(numpy.exp(-numpy.exp((numpy.log10(15.)-mu)/beta)) - numpy.exp(-numpy.exp((3.-mu)/beta)))
      
    return A*numpy.array(numpy.greater(x,numpy.log10(15.)),dtype='float')*numpy.array(numpy.greater(3.,x),dtype='float')*(1./beta)*numpy.exp(-(zm + numpy.exp(-zm)))

def p_w_m(w,m):
    '''Returns the probability that a given galaxy of HI mass 10^m has a velocity width of 10^w'''
    
    mu = numpy.maximum(numpy.log10(15.),0.32184878*m - 0.72761855)
    beta = numpy.minimum(-0.01580308*m + 0.31601693, -0.05778951*m + 0.72890471)
    
    
    return gumbel(w,mu,beta)

def HIMF(logMHI):

    alpha = -1.33
    phi_s = 4.8E-3
    M_s = 10.**9.96

    MHI = 10.**logMHI

    return numpy.log(10.)*phi_s*((MHI/M_s)**(alpha+1))*numpy.exp(-MHI/M_s)

def Sint(MHI, D, z):
    c = 2.356*10**5
    return MHI / ((1+z)*c*D**2)

def det_lim_gen(w,D,z,f_b=1.):
    #Returns the minimum detectable M_HI
    #W50 is in km/s
    #D is in Mpc
    #S_rms is the noise per beam, per channel in mJy
    #f_b is the fraction of the source within the beam
    
    v_ch = 299792.458*f_ch/1420405751.77 #w here is the intrinsic value
    
    W50 = 10.**w
    
    if Duffy_lim:
        f_smo = W50/(2.*v_ch)
    else:
        f_smo = numpy.minimum(W50/(2.*v_ch),(10.**2.5)/(2.*v_ch))
    
    #Use the luminosity distance - this determines how bright it will appear
    #rms noise decreases like 1+z due to effective increase in the integration time
    #Beam size increases with z so the survey area is covered more at high z
    #This does not happen with a PAF as the beams won't be made to overlap
    if PAF:
        return numpy.log10((2.356E5)*(1.+z)*SN_T*(D**2.)*W50*S_rms/(f_b*numpy.sqrt(f_smo)*1000.))
    else:
        return numpy.log10((2.356E5)*SN_T*(D**2.)*W50*S_rms/(f_b*numpy.sqrt(f_smo)*1000.))

def det_test_gen(m,w,D,z,f_b=1.):
    
    v_ch = 299792.458*f_ch/1420405751.77 #w here is the intrinsic value
    
    W50 = 10.**w
    
    if Duffy_lim:
        f_smo = W50/(2.*v_ch)
    else:
        f_smo = numpy.minimum(W50/(2.*v_ch),(10.**2.5)/(2.*v_ch))
    
    if PAF:
        m_cut = numpy.log10((2.356E5)*(1+z)*SN_T*(D**2.)*W50*S_rms/(f_b*numpy.sqrt(f_smo)*1000.))
    else:
        m_cut = numpy.log10((2.356E5)*SN_T*(D**2.)*W50*S_rms/(f_b*numpy.sqrt(f_smo)*1000.))
    
    return numpy.array(numpy.greater_equal(m,m_cut),dtype='float')

def det_test_frac(m1,m2,frac=0.1):
    #Simply tests whether the second source is above a fraction of the mass of the first
    
    return numpy.array(numpy.greater_equal(m2,numpy.maximum(5,m1+numpy.log10(frac))),dtype='float')

def w_for_det_gen(m,D,z,f_b=1.):
    
    v_ch = 299792.458*f_ch/1420405751.77 #This is for the intrinsic velocity width
    
    M_HI = 10.**m
    
    if Duffy_lim:
        if PAF:
            return numpy.log10(((1000.*f_b*M_HI/((2.356E5)*SN_T*(1+z)*(D**2.)*S_rms*numpy.sqrt(2.*v_ch))))**2.)
        else:
            return numpy.log10(((1000.*f_b*M_HI/((2.356E5)*SN_T*(D**2.)*S_rms*numpy.sqrt(2.*v_ch))))**2.)
    else:
        if PAF:
            return numpy.log10(min((1000.*f_b*M_HI*numpy.sqrt((10.**2.5)/(2.*v_ch))/((2.356E5)*SN_T*(1+z)*(D**2.)*S_rms)),((1000.*f_b*M_HI/((2.356E5)*SN_T*(1+z)*(D**2.)*S_rms*numpy.sqrt(2.*v_ch))))**2.))
        else:
            return numpy.log10(min((1000.*f_b*M_HI*numpy.sqrt((10.**2.5)/(2.*v_ch))/((2.356E5)*SN_T*(D**2.)*S_rms)),((1000.*f_b*M_HI/((2.356E5)*SN_T*(D**2.)*S_rms*numpy.sqrt(2.*v_ch))))**2.))
    
def p_m_D(m,D,z):
    #Returns the probability that a galaxy of mass m will be detectable once it's velocity width is drawn from the distribution
    
    w = w_for_det_gen(m,D,z)
    
    if w < numpy.log10(15.):
        return 0.
    if w > 3.:
        return 1.
    
    mu = numpy.maximum(numpy.log10(15.),0.32184878*m - 0.72761855)
    beta = numpy.minimum(-0.01580308*m + 0.31601693, -0.05778951*m + 0.72890471)
    
    zm = (mu-w)/beta
    zmin = (mu-3.)/beta
    zmax = (mu-numpy.log10(15.))/beta
    
    return (numpy.exp(-numpy.exp(-zmax))-numpy.exp(-numpy.exp(-zm)))/(numpy.exp(-numpy.exp(-zmax))-numpy.exp(-numpy.exp(-zmin)))

def n_MHI_dist(D,z):
    #Returns the number density of detectable sources at a given distance (and redshift)
    
    alpha = -1.33
    phi_s = 4.8E-3
    M_s = 10.**9.96
    
    MHI_min = 10.**5
    
    #return phi_s*scipy.integrate.quad(lambda x: (x**alpha)*numpy.exp(-x)*p_m_D(numpy.log10(x*M_s),D,z),MHI_min/M_s,(10.**11.)/M_s)[0]
    return phi_s*scipy.integrate.quad(lambda x: (x**alpha)*numpy.exp(-x)*p_m_D(numpy.log10(x*M_s),D,z),MHI_min/M_s,numpy.inf/M_s)[0]

def p_MHI(m,D,z):
    #Return the probability that any given galaxy is of mass m = logMHI
    
    #n_MHI = n_MHI_calc(m_intsct(D))
    n_MHI = n_MHI_dist(D,z)
    
    alpha = -1.33
    phi_s = 4.8E-3
    M_s = 10.**9.96
    
    return numpy.log(10.)*(phi_s/n_MHI)*(((10.**m)/M_s)**(alpha+1.))*numpy.exp(-((10.**m)/M_s))

def p_MHI_frac(m2,m1,n_MHI):
    #Return the probability that any given galaxy is of mass m = logMHI
    
    alpha = -1.33
    phi_s = 4.8E-3
    M_s = 10.**9.96
    
    return numpy.log(10.)*(phi_s/n_MHI)*(((10.**m2)/M_s)**(alpha+1.))*numpy.exp(-((10.**m2)/M_s))

def m_av_mod(y,x,n):
    # Akanksha - This looks like it does the correlation function
    
    A, alpha, a = 12.104434  ,   -1.13210593,   0.64058959
    b = 1./a
    
    r = A**(-1./alpha)
    
    return a*((2.*numpy.pi*n)/((alpha+2.)*(alpha+3.)*r**alpha)) * ( (y/b)*((x/a)**2.)*((r)**alpha)*(alpha+2.)*(alpha+3.) + 2.*(y/b)*((x/a)**(2.+alpha))*(alpha+3.)*scipy.special.hyp2f1(0.5,-1.-(alpha/2.),1.5,-(y*a/(x*b))**2.) - 2.*(y/b)**(alpha+3.) )

def N_exp(D,z,m1_vals,w1_vals,D_beam,N_rands=1000,m_mintest=5.0, m_maxtest=12.0):
    '''Carries out an MC integration to find the expected number of sources in the cylindrical volume.'''
    
    w2_min = numpy.log10(15.)
    w2_max = 3.
    
    rands = numpy.random.uniform(0.,1.,(2,len(w1_vals),N_rands))
    
    w2_vals = w2_min + (w2_max-w2_min)*rands[0]
    
    if M_det:
        m_min = max(5,det_lim_gen(numpy.log10(15.),D,z))
    elif M_62:
        m_min = m_mintest
    elif Ms_01:
        m_min = 8.96
    elif M_01:
        m_min = max(5,min(m1_vals)-1.)
    
    m_max = 12.
    if M_62:
        m_max = m_maxtest
    
    m_vals = m_min + (m_max-m_min)*rands[1]
    
    # sum of the spectra widths
    w_tot = 10.**numpy.transpose(numpy.tile(w1_vals,(N_rands,1))) + 10.**w2_vals
    
    if M_det:
        vals = numpy.sum(det_test_gen(m_vals,w2_vals,D,z)*HIMF(m_vals)*p_w_m(w2_vals,m_vals)*m_av_mod(w_tot/140.,D_beam,1.),axis=1)
    elif M_62:
        vals = numpy.sum(HIMF(m_vals)*p_w_m(w2_vals,m_vals)*m_av_mod(w_tot/140.,D_beam,1.),axis=1)
    elif Ms_01:
        vals = numpy.sum(HIMF(m_vals)*p_w_m(w2_vals,m_vals)*m_av_mod(w_tot/140.,D_beam,1.),axis=1)
    elif M_01:
        vals = numpy.sum(det_test_frac(numpy.transpose(numpy.tile(m1_vals,(N_rands,1))),m_vals,frac=0.1)*HIMF(m_vals)*p_w_m(w2_vals,m_vals)*m_av_mod(w_tot/140.,D_beam,1.),axis=1)

    vol = (m_max-m_min)*(w2_max-w2_min)
    
    vals = vals*vol/float(N_rands)
    
    return vals

def P_blend(D,z,D_beam,N_rands=1000,den=False,RFI=False,m_mintest=5.0, m_maxtest=12.0):
    
    #####
    #Section for using density field and RFI
    n_MHI_fac1 = 1.
    n_MHI_fac2 = 1.
    n_MHI_frac = 1.
    if den:
        norm = 0.
        cnt = 0.
        for i in range(len(overden)):
            if z - z_step/2. <= z_list[i] < z + z_step/2.:
                norm += (1.+overden[i])
                cnt += 1.
        if cnt == 0.:
            cnt = 1.
        if norm == 0.:
            norm = 1.
        n_MHI_fac1 = norm/cnt

    if RFI:
        norm = 0.
        cnt = 0.
        for i in range(len(RFImean)):
            if z - z_step/2. <= z_list[i] < z + z_step/2.:
                norm += RFImean[i]
                cnt += 1.
        if cnt == 0.:
            cnt = 1.
        if norm == 0.:
            norm = 1.
        n_MHI_fac2 = norm/cnt
        
    n_MHI_frac = n_MHI_fac1*n_MHI_fac2
    #n_MHI_frac = a40_den_corr[int((z*3.E5)/500.)]
    #####
    
    w1_min = numpy.log10(15.)
    w1_max = 3.
    
    rands = numpy.zeros((2,N_rands))
    
    for i in range(2):
        rands[i] = numpy.random.uniform(0.,1.,N_rands)
    
    w1_vals = w1_min + (w1_max-w1_min)*rands[1]
    
    m_min = max(5,det_lim_gen(numpy.log10(15.),D,z))
    m_max = 11.
    
    if m_min > m_max:
        return 0.
    
    m1_vals = m_min + (m_max-m_min)*rands[0]
    
    t0 = time.time()
    #N_vals = numpy.array(map(lambda x: N_exp_old(D,z,x,D_beam,N_rands/100),w1_vals))
    N_vals = n_MHI_frac*N_exp(D,z,m1_vals,w1_vals,D_beam,int(N_rands/10000),m_mintest=m_mintest, m_maxtest=m_maxtest)
    #print(N_vals)
    t1 = time.time()
    
    val = numpy.sum(det_test_gen(m1_vals,w1_vals,D,z)*HIMF(m1_vals)*p_w_m(w1_vals,m1_vals)*(1. - numpy.exp(-N_vals)))
        
    vol = (m_max-m_min)*(w1_max-w1_min)

    val = numpy.divide(val,n_MHI_dist(D,z))
    
    val = val*vol/float(N_rands)
    t2 = time.time()
    #print t1-t0
    #print t2-t1
    
    return val, numpy.mean(N_vals)

#Set the rms noise per beam, per channel for the relevant survey
#Also set the channel width
S_rms = 0.3#5yrCHORD #0.61#1yrCHORD  #3.4#ALFALFA  1.592#WALLABY  0.201#DINGO  13.#HIPASS  0.937#WNSHS  0.09#DINGO_UD #mJy
f_ch = 23689.8#CHORDup 195312.5#CHORDnative  24400.#ALFALFA  18270.#ASKAP  (64.E6)/1024.#HIPASS 
beam = 3.973#4 #arcmin
T_int = 1.#Integration time in multiples of ALFALFA T_int
S_rms = S_rms/numpy.sqrt(T_int)

#Set the S/N theshold for detection
SN_T = 6#Code 1 50%
Duffy_lim = False
PAF = False #This means only ASKAP PAFs, others may be set up differently

#Only one can be set to True
#Determine if blends are with other detections, all HI galaxies, galaxy above 0.1M*
M_det = False
M_62 = False
Ms_01 = False
M_01 = True

area = (4.*numpy.pi*numpy.pi/129600.)*13258.#CHORD  30940.#WALLABY  150.#DINGO  2800.#a.40  320.#PPS Strip 

# # def FracConf_MassBins():
# mass = [5,6,7,8,9,10,11]
# z_step = 500./300000.
# # #z_step = 1000./300000.
# # #z_step = 5.*70./300000.
# z_list = numpy.arange(z_step/2,1,z_step)
# p_list = numpy.zeros(len(z_list))
# # #MHI_stack = numpy.zeros(len(z_list))
# Nvals_list = numpy.zeros((len(z_list), len(mass)))
# # Nvals_list = numpy.zeros(len(z_list))

# for j in range(len(mass)):
#     for i in range(len(p_list)):
#         sys.stdout.write('\r')
#         sys.stdout.write(str(int((i+1)*100./float(len(p_list))))+'% complete')
#         sys.stdout.flush()
        
#         D = co_dist(z_list[i])
        
#         D_beam = phys_size((beam/60.)*numpy.pi/180.,z_list[i])

#         if j<(len(mass)-1):
#             p_list[i], Nvals_list[i,j] = P_blend(D,z_list[i],D_beam,N_rands = 100000,den=False,RFI=False,m_mintest=mass[j], m_maxtest=mass[j+1])
        
#         else:
#             p_list[i], Nvals_list[i,j] = P_blend(D,z_list[i],D_beam,N_rands = 100000,den=False,RFI=False,m_mintest=5.0, m_maxtest=12.0)
#         #p_list[i] = P_blend(D,z_list[i],D_beam,N_rands = 100000,den=False,RFI=False)`
    
#     sys.stdout.write('\r')
#     sys.stdout.write(str('100% complete'))
#     sys.stdout.flush()
#     print('\n')

# # font = {'size' : 12, 'family' : 'DejaVu Sans'}
# # plt.rc('font', **font)

# #plt.figsize(12,10)
# if j<(len(mass)-1):
#     label = f'$10^{int(mass[j])} > MHI > 10^{int(mass[j+1])}$'
# else:
#     label='all sources'
# plt.plot(z_list, Nvals_list, label=label)

# # # #plt.plot(numpy.array(d_sim)*70./(3.E5),p_sim,'r')
# # # #ylim(0.,0.1)
# plt.xlabel('z')
# plt.legend()
# plt.ylabel('Average number of additional non-detections in beam')
# #plt.title('All sources with MHI > 10$^{5}$')
# plt.show()

#numpy.save('N_avg_z0to1.npy', Nvals_list)

#import numpy as np
# Nvals_list = np.load('N_avg.npy')
# Nvals_sum = Nvals_list
# print(Nvals_sum.shape)
# #Nvals_sum = np.sum(Nvals_list[:,0:-1], axis=1)
# Nruns = 1000

# logMHI = np.linspace(5, 12, 100000)
# himf = HIMF(logMHI)
# pdf = himf / np.sum(himf)
# Sint_runs = np.zeros((len(z_list),Nruns))

# for j in range(Nruns):
#     for i in range(len(z_list)):
#         D = co_dist(z_list[i])
#         M_min = det_lim_gen(np.log(15.),D,z_list[i])
#         #print(M_min)
#         Sint_min = Sint(MHI=10**M_min, D=D, z=z_list[i])
#         #print(Nvals_sum[i])
#         MHIs = np.random.choice(logMHI, size=int(Nvals_sum[i]), p=pdf)
#         MHI_stack[i] = np.sum(10**MHIs)
#         #print(MHIs)
#         Sints = Sint(MHI=10**MHIs, D=D, z=z_list[i])
#         Sint_runs[i,j] = np.sum(Sints) / Sint_min

# # print(Sint_runs)
# # np.save("Sint_1000runs_allz.npy", Sint_runs)
# np.save("Mstack_1000runs_allz.npy", MHI_stack)

# plt.figure(figsize=[6,5], dpi=300)
# plt.plot(z_list, MHI_stack)
# plt.yscale('log')
# plt.show()


# import numpy as np
# Sint_runs = np.load('Sint_1000runs_allz.npy')
# Sint_mean = np.mean(Sint_runs, axis=1)
# Sint_std = np.std(Sint_runs, axis=1)
# plt.figure(figsize=[5,4],dpi=300)
# plt.errorbar(z_list, Sint_mean, yerr=Sint_std,
#              fmt='none', ecolor='gray', alpha=0.5, capsize=3)
# plt.plot(z_list, Sint_mean, '.', color='black')

# plt.xlabel('Redshift z')
# plt.ylabel('Fraction of integrated flux')
# plt.ylim(0,1)
# #plt.title('Sum of integrated flux in sources as a fraction of minimum detectable integrated flux')
# plt.savefig('Plots/Confusion_Sint_frac.png')
# plt.show()

# for j in range(len(mass)):
#     if j<(len(mass)-1):
#         label = f'{int(mass[j])} > log(MHI) > {int(mass[j+1])}'
#     else:
#         label='all sources'
#     plt.plot(z_list,Nvals_list[:,j], label=label)

# plt.xlabel('z')
# plt.legend()
# plt.ylabel('Average number of additional non-detections in beam')
# plt.yscale('log')
# plt.savefig('Plots/AvgSources_confusion_log.png')
# plt.show()


# total_sources = Nvals_list[:, (len(mass)-1)]
# sum_sources = np.sum(Nvals_list[:,0:-1], axis=1)

# plt.figure()
# plt.plot(z_list, sum_sources)
# plt.plot(z_list, total_sources)
# plt.show()

#print(Nvals_list[:, -3])


# # import numpy as np

# with open(f"MJ_allsources_MHI_5year_z1_Mdet_True.pkl", "rb") as f:
#     MHI_catalog = pickle.load(f)
# with open(f"MJ_blendedsources_MHI_5year_z1_Mdet_True.pkl", "rb") as f:
#     MHI_blended = pickle.load(f)

# print(blended_list)
# print(catalog_list)

# # MHI_bins = np.arange(5,11,0.2)
# # fraction_2d = np.zeros((len(z_list), len(MHI_bins)-1))
# # for i in np.arange(len(z_list)):
# #     total_counts, bin_edges = numpy.histogram(MHI_catalog[i], bins=MHI_bins)
# #     blended_counts, bin_edges = numpy.histogram(MHI_blended[i], bins=MHI_bins)
# #     fraction_blended = blended_counts / total_counts
# #     fraction_2d[i] = fraction_blended

# plt.figure(figsize=[5,4],dpi=300)
# # plt.imshow(fraction_2d, extent=[z_list[0], z_list[-1], MHI_bins[0], MHI_bins[-1]], origin='lower', aspect='auto')
# # plt.show()

#colors = cmap(np.linspace(0, 1, 4))
#color = cm.get_cmap('BuPu', 4).colors
# colors = cm.get_cmap('viridis')(np.linspace(0, 1, 4))
# z_bins = [0,0.009,0.1,0.2,1]
# #fig, ax = plt.subplots(4, 1, figsize=[4,6], dpi=300, sharex=True)
# #plt.subplots_adjust(hspace=0)
# for i in [1,2,0,3]: #range(len(z_bins)-1): 
#     idx1 = numpy.argmin(numpy.abs(z_list - z_bins[i]))
#     idx2 = numpy.argmin(numpy.abs(z_list - z_bins[i+1]))
#     total = numpy.concatenate([numpy.array(x) for x in MHI_catalog[idx1:idx2]])
#     blend = numpy.concatenate([numpy.array(x) for x in MHI_blended[idx1:idx2]])
#     binwidth = 0.5
#     bins = numpy.arange(5,11,0.5)
#     total_counts, bin_edges = numpy.histogram(total, bins=bins)
#     blended_counts, bin_edges = numpy.histogram(blend, bins=bins)
#     fraction_blended = blended_counts / total_counts
#     if i==0:
#         label = 'D < 40 Mpc'
#     elif i==1:
#         label = 'D > 40 Mpc, z < 0.1'
#     else:
#         label=f'{z_bins[i]} < z < {z_bins[i+1]}'
#     mid_bins = (bins[1:] + bins[:-1])*0.5
#     plt.bar(mid_bins, fraction_blended, label=label, color=colors[i], width=0.5) #edgecolor=colors[i], linewidth=2)#, edgecolor=color[i], facecolor='None')
#     #plt.set_ylim(0,0.4)

# handles, labels = plt.gca().get_legend_handles_labels()

# order = []
# for i in range(len(z_bins) - 1):
#     if i==0:
#         label = 'D < 40 Mpc'
#     elif i==1:
#         label = 'D > 40 Mpc, z < 0.1'
#     else:
#         label=f'{z_bins[i]} < z < {z_bins[i+1]}'
#     order.append(label)

# ordered_handles = [handles[labels.index(o)] for o in order]
# plt.legend(ordered_handles, order, loc='upper left')

# #plt.legend()
# #plt.title('Confused with galaxy with atleast 10% of MHI')
# plt.xlabel('log($M_{HI}/M_{\odot}$)')
# plt.ylabel('Fraction of detections in blends')
# plt.tight_layout()
# plt.savefig('blends_MHI_z_Mdet_True.png')
# plt.show()




# plt.figure(figsize=[5,4], dpi=300)
# MHI_bins = [5,7,9,11]
# MHI_fractions = numpy.zeros((len(z_list), len(MHI_bins)-1))

#for i in [1,5]:

# for j in range(len(z_list)):
#     total_counts, bin_edges = numpy.histogram(MHI_catalog[j], bins=MHI_bins)
#     blended_counts, bin_edges = numpy.histogram(MHI_blended[j], bins=MHI_bins)
#     fraction_blended = blended_counts / total_counts
#     MHI_fractions[j] = fraction_blended

# for k in range(len(MHI_bins)-1):
#     label = f"{MHI_bins[k]} < MHI < {MHI_bins[k+1]}"
#     plt.plot(z_list, MHI_fractions[:,k], label=label)

# plt.legend()
# plt.yscale('log')
# plt.show()






# MHI_catalog = numpy.concatenate([numpy.array(x) for x in catalog_list])
# MHI_blended = numpy.concatenate([numpy.array(x) for x in blended_list])

# numpy.save("MJ_allsources_MHI_1year_z0p4to1.npy", MHI_catalog)
# numpy.save("MJ_blendedsources_MHI_1year_z0p4to1.npy", MHI_blended)

# MHI_catalog5 = numpy.load("MJ_allsources_MHI_5year_z0p4to1.npy")
# MHI_blended5 = numpy.load("MJ_blendedsources_MHI_5year_z0p4to1.npy")
# MHI_catalog1 = numpy.load("MJ_allsources_MHI_1year_z0p4to1.npy")
# MHI_blended1 = numpy.load("MJ_blendedsources_MHI_1year_z0p4to1.npy")

# #print(z_list)
# color = cm.get_cmap('Dark2', 7).colors

# plt.figure(figsize=[4,3],dpi=200)
# total_counts, bin_edges = numpy.histogram(MHI_catalog5, bins=17)
# print(total_counts)
# #total_counts = total_counts.astype(float)
# #total_counts[total_counts<20]=numpy.nan
# blended_counts, bin_edges = numpy.histogram(MHI_blended5, bins=bin_edges)
# print(blended_counts)
# fraction_blended = blended_counts / total_counts
# mid_bins = (bin_edges[1:] + bin_edges[:-1])*0.5
# plt.plot(mid_bins, fraction_blended, label='5 year Survey', c=color[0])

# total_counts, bin_edges = numpy.histogram(MHI_catalog1, bins=18)
# print(total_counts)
# #total_counts = total_counts.astype(float)
# #total_counts[total_counts<20]=numpy.nan
# blended_counts, bin_edges = numpy.histogram(MHI_blended1, bins=bin_edges)
# fraction_blended = blended_counts / total_counts
# mid_bins = (bin_edges[1:] + bin_edges[:-1])*0.5
# plt.plot(mid_bins, fraction_blended, label='1 year Survey', c=color[1])

# plt.ylabel('Fraction in Blends')
# plt.xlabel('log$(M_{HI}/M_{\odot})$')
# plt.legend()
# plt.title('$0.4 < z < 1$')
# plt.tight_layout()
# plt.savefig('Plots/Confusion_vsMHI_z0p4to1.png')
# plt.show()
