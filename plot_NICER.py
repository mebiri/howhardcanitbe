#! /usr/bin/env python
"""
Created on Tue May 19 14:45:55 2026

@author: marce
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import argparse
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


parser = argparse.ArgumentParser()
parser.add_argument('--obs-file', action='append', help="REQUIRED: Filenames/paths for observations to use.")
parser.add_argument('--mode',type=str,help="REQUIRED: Supported opts: data-only, data-field")
parser.add_argument('--eos-max-m',type=float,help="Max mass to truncate mass range (pretend EOS), if wanted")
parser.add_argument('--use-lnL',action='store_true',help="If present, uses log likelihoods (recommended)")
parser.add_argument('--lnL-cut',type=float,default=1e-6,help="Value to which smaller likelihoods/lnLs are adjusted to.")
parser.add_argument('--data-on-top',action='store_true',help="Whether to render obs data above or beneath mass field")
parser.add_argument('--npts',type=int,default=1000,help="Number of points to use in field (default 1000)")
parser.add_argument('--field-M-min',type=float,default=0.4)
parser.add_argument('--field-M-max',type=float,default=2.1)
parser.add_argument('--field-R-min',type=float,default=8)
parser.add_argument('--field-R-max',type=float,default=18)
parser.add_argument('--scale-lnL',type=int,default=1,help="Scale factor to apply to likelihood calcs (applied before log)")
parser.add_argument('--plot-max',action='store_true')
parser.add_argument('--plot-causality',action='store_true')

opts = parser.parse_args()

def just_obs():
    plt.rcParams.update({'font.size': 12.0,  'mathtext.fontset': 'stix'})
    plt.rcParams['axes.unicode_minus'] = False
    #plt.rcParams['figure.figsize'] = (9.0, 7.0)
    plt.rcParams['xtick.labelsize'] = 15.0
    plt.rcParams['ytick.labelsize'] = 15.0
    plt.rcParams['axes.labelsize'] = 25.0
    plt.rcParams['lines.linewidth'] = 1.0
    
    fig1 = plt.figure(figsize=(8,6),dpi=200) 
    ax = fig1.add_subplot(111)
    
    dat_rv = [] #formerly external_ns_MR_rv
    dat_mn = []   #only used for plotting
    dat_cov = []  #only used for plotting
    dat_list = [] #only used for plotting
    stars = []    #only used for plotting
    for i in opts.obs_file:
        print("Retrieving data from file: "+i)
        starfile = i.split("/")[-1]
        if starfile[0] == 'J':
            stars.append(starfile[:5])
        else:
            stars.append(starfile.split(".")[0])
        dat = np.genfromtxt(i)[:,:2]
        print("len this file:",len(dat))
        
        mn, cov, rv = gaussian_distribution(dat)
        dat_rv.append(rv)
        dat_mn.append(mn)
        dat_cov.append(cov)
        dat_list.append(dat)  
    
    legends = []
    if opts.plot_max:
        mMax_likelihood(ax,0.3,24.0,legends)
    if opts.plot_causality:
        buchdahl_BH_limits(ax,all=False)
    
    #from plotting import plot_data_and_gaussian
    #order: ${J0740} ${J1731} ${J0030} ${J0437}  
    colors_list = ['green','purple','red','orange']
    for j in np.arange(len(dat_rv)):
        print("Plotting data for: ",stars[j])
        plot_data_and_gaussian(dat_mn[j],dat_cov[j],dat_rv[j],dat_list[j],ax,color=colors_list[j],alph=0.01,markersize=9)#,name=stars[j])
        legends.append(Line2D([0],[0],marker='o',color='w',label=stars[j],markerfacecolor=colors_list[j],alpha=0.8,markersize=9))
       
    ax.set_xlim(left=7.5,right=24.0)
    ax.set_ylim(bottom=0.3,top=2.5)
    ax.set_xlabel("Radius [km]", size="20")
    ax.set_ylabel('Mass [M/M$_\odot$]', size="20")
    ax.tick_params(axis='both', which='major', labelsize=10)  
    ax.set_yticks(ticks=[0.5,1.0,1.5,2.0,2.5])
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(handles=legends, fontsize='9', loc='lower right')
    fig1.tight_layout()
    plt.box(False)
    plt.show()
    
    plt.savefig("MR_NICER_data.png")
    print("Saved.") 


def gaussian_distribution(data):
    '''
    0th Column: Radii, 1st Column: Masses
    '''
    mn = np.mean(data,axis=0)
    cov = np.cov(data.T)
    print("mean of this dataset: (R,M) =",mn)
    print("cov:\n",cov)
    a, b = 0,1
    
    x, y = np.mgrid[min(data[:,a]):max(data[:,a]):.01, min(data[:,b]):max(data[:,b]):.01]
    #2D normal dist, axes mass vs. radius
    rv = multivariate_normal([mn[a], mn[b]], [[cov[a,a], cov[a,b]], [cov[a,b], cov[b,b]]])
    
    return mn, cov, rv


def plot_data_and_gaussian(mean, covariance, rv, data, ax, color= 'pink', name = None, color_by_obs = False, alph=0.01):    
    print(mean,covariance)
    
    lambda_, v = np.linalg.eig(covariance)
    lambda_ = np.sqrt(lambda_)
    a, b = 0,1
    
    # Check if the gradient actually looks like it should.
    if color_by_obs:
        #
        raise Exception('not implemented yet. Check color gradiant on the scatter.')
        color = None
    else: pass
    
    ax.scatter(data[:,a], data[:,b], alpha=alph, s = 2, color = color, label = name,marker="o",linewidth=0)
    #Ellipses for 1,2,3 sigma
    from matplotlib.patches import Ellipse
    for j in range(1, 4):
        ell = Ellipse(xy=(mean[a], mean[b]), width=lambda_[a]*j*2, height=lambda_[b]*j*2, angle=np.rad2deg(np.arccos(v[a, a])), lw=1.2, facecolor='None', alpha=0.6, edgecolor= 'black')
        # or try np.degrees(np.arctan2(*vecs[:,0][::-1])) for angle
        ax.add_artist(ell)
    
    return


def mMax_likelihood(ax,alph,xlim,leg):    
    m = [2.14, 2.01, 1.908] #3 high-mass pulsars (Dietrich et al. 2020)
    sig = [0.1, 0.04, 0.016]
    names = ["J0740","J0348","J1614"]
    colors= ["green","yellow","blue"]
    
    x = np.linspace(7.5,xlim, 500)

    for i in np.arange(len(m)):
        ax.fill_between(x,(m[i]-sig[i])*np.ones(500),(m[i]+sig[i])*np.ones(500),color=colors[i],alpha=alph)#,label=names[i])
        leg.append(Patch(facecolor=colors[i], edgecolor=colors[i],label=names[i]))


def buchdahl_BH_limits(ax, all = True):
    import lal
    m = np.linspace(1.5,4.5, 300)
    r = 2.824*m*lal.MRSUN_SI/1e3 #10.1146/annurev-nucl-102711-095018
    ax.fill_between(r,m, 3*m, color='gainsboro')
    if all:
        r = 9*m/4*lal.MRSUN_SI/1e3 # 10.1103/PhysRevLett.121.161101
        ax.fill_between(r,m, 3*m, color='lightgrey')
        r = 2*m*lal.MRSUN_SI/1e3
        ax.fill_between(r,m, 2*m, color='silver')
    if all:
        ax.text(9.0, 2.4, "Causality limit", rotation=35)
        ax.text(8.0, 2.55, "Buchdahl limit", rotation=41)
        ax.text(7.3, 2.7, "BH limit", rotation=45)
    else:
        ax.text(8.5, 2.1, "Causality limit", rotation=50)
    return


def obs_and_MR():
    import matplotlib
    
    #mass grid
    #dM = 0.001
    Mmax = opts.field_M_max
    Mmin = opts.field_M_min
    #M = np.arange(Mmin,Mmax,dM)  # range of masses
    M = np.random.uniform(Mmin,Mmax,opts.npts)
    if opts.eos_max_m is not None:
        M = M[np.where(M <= opts.eos_max_m)]
    #radius grid
    #R = reprimand_object.circ_radius_from_grav_mass(M)*reprimand_object.units_to_SI.length/1e3
    Rmax = opts.field_R_max
    Rmin = opts.field_R_min
    #dR = (Rmax-Rmin)/len(M)
    #R = np.arange(8,25,dR)
    R = np.random.uniform(Rmin,Rmax,len(M))
    print("len M:",len(M),"; len R:",len(R))
    scale_factor = opts.scale_lnL*(Mmax-Mmin)/(max(M)-min(M))
    print("scale factor:",scale_factor)
    
    like_grid = np.zeros((len(M),len(opts.obs_file)))
    print("likelihood grid size:",like_grid.shape)
        
    #Create 2D Gaussians for each provided NICER data set:
    dat_rv = [] #formerly external_ns_MR_rv
    dat_mn = []   #only used for plotting
    dat_cov = []  #only used for plotting
    dat_list = [] #only used for plotting
    stars = []    #only used for plotting
    obs_num = 0
    for obs in opts.obs_file:
        print("Retrieving data from file: "+obs)
        starfile = obs.split("/")[-1]
        if starfile[0] == 'J':
            stars.append(starfile[:5])
        else:
            stars.append(starfile.split(".")[0])
        
        dat_here = np.genfromtxt(obs)[:,:2] #only get 1st 2 cols (R,M)
        mn, cov, rv = gaussian_distribution(dat_here)
        dat_rv.append(rv)
        dat_mn.append(mn)
        dat_cov.append(cov)
        dat_list.append(dat_here)    
    

        likelihood_local_list = rv.pdf(np.c_[R,M])*scale_factor
        print("total likelihood:",np.sum(likelihood_local_list))
        if opts.use_lnL:
            likelihood_local_list = np.log(likelihood_local_list)
        if opts.lnL_cut is not None:
            likelihood_truncate_list = np.where(likelihood_local_list > opts.lnL_cut, likelihood_local_list, opts.lnL_cut)
        else:
            likelihood_truncate_list = likelihood_local_list
        like_grid[:,obs_num] = likelihood_truncate_list
        obs_num += 1
    
    print("Test, first line eval:")
    if opts.use_lnL:
        #likelihood_array = np.zeros(len(M))  
        res1 = 0.
        for res in like_grid[0]: res1 += res
        likelihood_array = np.sum(like_grid,axis=1)
    else:
        #likelihood_array = np.ones(len(M))
        res1 = 1.
        for res in like_grid[0]: res1 *= res
        likelihood_array = np.prod(like_grid,axis=1)
    print(like_grid[0]," = ",res1) 
    print("Sample of likelihood_array:\n",likelihood_array[:10])
    
    #for j in np.arange(len(like_grid[0])):
    #    like_arr2[:] += like_grid[:,j]   
    #for i in np.arange(len(M)):
    #    for res in like_grid[i]: likelihood_array[i] *= res     
    
    
    matplotlib.rcParams.update({'font.size': 12.0,  'mathtext.fontset': 'stix'})
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)
    matplotlib.rcParams['xtick.labelsize'] = 15.0
    matplotlib.rcParams['ytick.labelsize'] = 15.0
    matplotlib.rcParams['axes.labelsize'] = 25.0
    matplotlib.rcParams['lines.linewidth'] = 2.0
    #plt.style.use('seaborn-v0_8-whitegrid')
    #fig,(ax1,ax2) = plt.subplots(2,1)
    
    fig = plt.figure(); ax = fig.add_subplot(111)
    figname = "MR_NICER"
    if opts.use_lnL: figname += "_lnL"
    else: figname += "_likelihood"
    for s in stars: figname += "_"+s
    figname += "_field"
    
    if opts.data_on_top:
        ax.scatter(R,M,c=likelihood_array,marker=".")
        figname += "_bottom"

    from plotting import plot_data_and_gaussian
    for j in np.arange(len(dat_rv)):
        print("Plotting data for: ",stars[j])
        plot_data_and_gaussian(dat_mn[j],dat_cov[j],dat_rv[j],dat_list[j],ax)
    
    if not opts.data_on_top:
        ax.scatter(R,M,c=likelihood_array,marker=".")
        figname += "_top"
    
    lmax = max(likelihood_array)
    lmax_loc = np.where(likelihood_array == lmax)[0][0] #index(lmax)
    #lmin = min(likelihood_array)
    print("Max likelihood:",lmax,"at (R, M) = (",R[lmax_loc],M[lmax_loc],")")
    print("Mean likelihood:",np.mean(likelihood_array))
    print("First element in list:",likelihood_array[0])
    
    #for i in likelihood_dict:
        #MRcolor = plt.cm.gist_rainbow((likelihood_dict[i]-lmin)/(lmax-lmin))
    #ratio = (likelihood_array[:]-lmin)/(lmax-lmin)
    #ax.scatter(R,M,c=likelihood_array,marker=".")
    #ax.scatter(R,M, alpha = ratio, color = 'b')
    
    #ax.set_xlim(7,20)
    ax.set_xlabel('Radius [km]')
    ax.set_ylabel('Mass [M/M$_\odot$]')
    plt.savefig(figname+'.png',format = 'png')
    
    plt.show()


if opts.mode == "data-only":
    just_obs()
elif opts.mode == "data-field":
    obs_and_MR()
else:
    print("ERROR: unsupported mode option provided. Exiting.")


