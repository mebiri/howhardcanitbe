#! /usr/bin/env python
"""
Created on Tue May 19 14:45:55 2026

@author: marce
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--obs-file', action='append', help="REQUIRED: Filenames (NOT PATHS) for observations used for likelihood calculation and plots generated here.")#Supported: j0740 j1731 j0030 j0437")
parser.add_argument('--mode',type=str,help="Supported opts: data-only, data-field")
parser.add_argument('--eos-max-m',type=float,help="Max mass to truncate mass range (pretend EOS), if wanted")

opts = parser.parse_args()

def just_obs():
    fig1 = plt.figure(figsize=(5,5),dpi=250) 
    ax = fig1.add_subplot(111)
    
    for i in opts.obs_file:
        if i[0] == 'J':
            starname = i[:5]
        else:
            starname = i.split(".")[0]
        dat = np.genfromtxt(i)
        print("len this file:",len(dat))
        R = dat[:1000,0]
        M = dat[:1000,1]
    
        ax.scatter(R,M,marker=".",label=starname)
        
    #ax.set_xlim(left=1.0,right=2.0)
    ax.set_ylim(bottom=1.0,top=2.0)
    ax.set_xlabel("radius", size="11")
    ax.set_ylabel("mass", size="11")
    ax.tick_params(axis='both', which='major', labelsize=10) 
    ax.grid(True)
    ax.legend(fontsize='9', loc='lower right')
    fig1.tight_layout()
    plt.show()
    
    plt.savefig("nicer_data.png")
    print("Saved.") 


def gaussian_distribution(data):
    '''
    0th Column: Radii, 1st Column: Masses
    '''
    mn = np.mean(data,axis=0)
    cov = np.cov(data.T)
    a, b = 0,1
    
    x, y = np.mgrid[min(data[:,a]):max(data[:,a]):.01, min(data[:,b]):max(data[:,b]):.01]
    #pos = np.dstack((x, y))
    #2D normal dist, axes mass vs. radius
    rv = multivariate_normal([mn[a], mn[b]], [[cov[a,a], cov[a,b]], [cov[a,b], cov[b,b]]])
    
    return mn, cov, rv


def obs_and_MR():
    import matplotlib
    import matplotlib.pyplot as plt
    
    #mass grid
    dM = 0.001
    Mmax = 2.1
    Mmin = 0.4
    M = np.arange(Mmin,Mmax,dM)  # range of masses
    if opts.eos_max_m is not None:
        M = M[np.where(M <= opts.eos_max_m)]
    #radius grid
    #R = reprimand_object.circ_radius_from_grav_mass(M)*reprimand_object.units_to_SI.length/1e3
    Rmax = 25
    Rmin = 8
    dR = (Rmax-Rmin)/len(M)
    R = np.arange(8,25,dR)
    print("len M:",len(M),"; len R:",len(R))
    assert len(M) == len(R)
    scale_factor = (Mmax-Mmin)/(max(M)-min(M))
    
    like_grid = np.zeros((len(M),len(opts.obs_file)))
    print("like grid size:",like_grid.shape)
        
    #Create 2D Gaussians for each provided NICER data set:
    dat_rv = [] #formerly external_ns_MR_rv
    dat_mn = []   #only used for plotting
    dat_cov = []  #only used for plotting
    dat_list = [] #only used for plotting
    stars = []    #only used for plotting
    obs_num = 0
    for obs in opts.obs_file:
        print("Retrieving data from file: "+obs)
        if obs[0] == 'J':
            stars.append(obs[:5])
        else:
            stars.append(obs.split(".")[0])
        
        dat_here = np.genfromtxt(obs)
        mn, cov, rv = gaussian_distribution(dat_here)
        dat_rv.append(rv)
        dat_mn.append(mn)
        dat_cov.append(cov)
        dat_list.append(dat_here)    
    

        likelihood_local_list = obs.pdf(np.c_[R,M])*scale_factor
        print("likelihood res:",np.sum(likelihood_local_list))
        #likelihood_dict.append(np.sum(likelihood_local_list))
        like_grid[:,obs_num] = likelihood_local_list
      
    likelihood_array = np.ones(len(M))
    for i in np.arange(len(M)):
        for res in like_grid[i]: likelihood_array[i] *= res     
    
    
    matplotlib.rcParams.update({'font.size': 12.0,  'mathtext.fontset': 'stix'})
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)
    matplotlib.rcParams['xtick.labelsize'] = 15.0
    matplotlib.rcParams['ytick.labelsize'] = 15.0
    matplotlib.rcParams['axes.labelsize'] = 25.0
    matplotlib.rcParams['lines.linewidth'] = 2.0
    plt.style.use('seaborn-v0_8-whitegrid')
    #fig,(ax1,ax2) = plt.subplots(2,1)
    
    fig = plt.figure(); ax = fig.add_subplot(111)
    from plotting import plot_data_and_gaussian
    for j in np.arange(len(dat_rv)):
        plot_data_and_gaussian(dat_mn[j],dat_cov[j],dat_rv[j],dat_list[j],ax)
    
    #likelihood_array = np.empty((0))
    #for i in likelihood_dict: likelihood_array = np.append(likelihood_array, likelihood_dict[i])
    
    lmax = max(likelihood_array)
    lmin = min(likelihood_array)
    
    #for i in likelihood_dict:
        #MRcolor = plt.cm.gist_rainbow((likelihood_dict[i]-lmin)/(lmax-lmin))
    ratio = (likelihood_array[:]-lmin)/(lmax-lmin)
    ax.scatter(R,M,c=likelihood_array,marker=".")
    #ax.scatter(R,M, alpha = ratio, color = 'b')
    
    #ax.set_xlim(7,20)
    ax.set_xlabel('Radius [km]')
    ax.set_ylabel('Mass [M/M$_\odot$]')
    
    #plt.savefig('MR_likelihood_sample_mMax.pdf',format = 'pdf')
    plt.savefig('MR_likelihood_sample_mMax_field.png',format = 'png')
    
    plt.show()


if opts.mode == "data-only":
    just_obs()
elif opts.mode == "data-field":
    obs_and_MR()
else:
    print("ERROR: unsupported mode option provided. Exiting.")


