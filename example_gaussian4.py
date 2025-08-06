# -*- coding: utf-8 -*-
"""
Not example_gaussian.py, but its mutant cousin.
Goal is minimal edits to example_gaussian.py: should pull points from file
instead of drawing multivariate directly, and calculate product Likelihood 
instead of summed (mixed model) likelihood.

@author: marce
"""


#! /usr/bin/env python
#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import argparse
import RIFT.lalsimutils as lalsimutils
#import os

'''
import RIFT.lalsimutils as lalsimutils
from RIFT.physics import EOSManager as EOSManager

from scipy.integrate import nquad
#import EOS_param as ep
import os
from EOSPlotUtilities import render_eos
import lal
'''


####################### Gaussian #######################

obs = None
pop_params = None
def initialize_me(**kwargs):
    #Initial grid (mass only, assume no uncertainty for now):
    obs_name = kwargs["input_file_name"]
    obs = np.genfromtxt(obs_name,dtype='str')
    
    pop_params = np.genfromtxt(kwargs["eos_file_name"],dtype='str')
    

#x_offset=4
def likelihood_evaluation():
    
    #call syntax: m1m2(mchirp, eta), returns m1, m2
    m1, m2 = lalsimutils.m1m2(obs[0],obs[1])
    #print("First line of obs:")
    #print(obs[0])
    
    
    from scipy.stats import multivariate_normal
    #rv = multivariate_normal([-x_offset,0,0], [[2,0,0], [0,2,0], [0,0,2]])
    
    
    #likelihood_dict = {}
    
    for i in np.arange(opts.eos_start_index, opts.eos_end_index):
        #likelihood_dict[i] = rv.pdf([eoss[i,2:]])*rv2.pdf([eoss[i,2:]])
        #print("Case",i+1)
        #Check that the passed uncertainty is positive (scipy will crash otherwise): 
        sig_test = np.float64(pop_params[i][4])                                                                                                                                   
        if sig_test < 0.0:
            print("WARNING: sigma < 0 encountered; updating:",sig_test,"->",abs(sig_test))
            sig_test = abs(sig_test)
            pop_params[i][4] = str(sig_test)
        #Check that passed sigma is above min (diff from above, which allows abs(sigma) > 0.1)
        if sig_test < 0.1:
            print("WARNING: sigma < 0.1 [min] encountered; updating:",sig_test,"-> 0.1")
            pop_params[i][4] = 0.1

        rv = multivariate_normal(pop_params[i][2:4], pop_params[i][4])
        #print(obs[i,:2])
        #print("All o:",[rv.pdf(o[:2]) for o in obs])
        #print("Product:",np.prod([rv.pdf(o[:2]) for o in obs]))
        
        #print("Obs i:",rv.pdf([obs[i,:2]]))
        
        #likelihood_dict[i] = np.prod([rv.pdf(o[:2]) for o in obs])#ignore uncertainties, if any
        
        pop_params[i,0] = np.sum([np.log(rv.pdf(o[:2])) for o in obs])
        #pop_params[i,0] = np.log(likelihood_dict[i])
        pop_params[i,1] = 0.001  # nominal integration error
    
    postfix = ''
    if opts.conforming_output_name:
        postfix = '+annotation.dat'
    
    # opts.fname is not None only when using RIFT as is in RIT-matters/20230623
    if opts.fname is None: np.savetxt(opts.outdir+"/"+opts.fname_output_integral+postfix, pop_params[opts.eos_start_index: opts.eos_end_index], fmt = '%10s', header="lnL     sigma_lnL   " + ' '.join(dat_orig_names))
    else: np.savetxt(opts.fname_output_integral+postfix, pop_params[opts.eos_start_index: opts.eos_end_index], fmt = '%10s', header="lnL     sigma_lnL   " + ' '.join(dat_orig_names))
    print("Done; saved.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname",type=str,help="Dummy argument required by API")
    parser.add_argument('--using-eos', type=str, help="Send eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
    parser.add_argument('--using-eos-index',type=int, help="Line number for single calculation.  Single-line calculation only")
    parser.add_argument('--eos_start_index',type=int, help="Line number from where to start the likelihood calculation.")
    parser.add_argument('--eos_end_index',type=int, help="Line number which needs likelihood for which needs to be evaluated.")
    parser.add_argument('--plot', action='store_true', help="Enable to plot resultant M-R and Gaussians.")
    parser.add_argument('--outdir', type=str, help="Output eos file directory.")
    parser.add_argument('--outdir-clean', type=str, help="Delete CleaOutile direc before starting the runtory.")
    parser.add_argument('--fname-output-integral', type=str, help="Output eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
    parser.add_argument('--fname-output-samples', type=str, help="NEVER USED, but is enabled. Output eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
    parser.add_argument('--n-events-to-analyze', type=int,default=None, help="Number of events to analyze")
    parser.add_argument("--conforming-output-name",action='store_true')
    
    #TODO: This can't be here anymore (as is) - CIP won't handle, need to get from 
    parser.add_argument("--init_grid",type=str,help="Initial grid of data to test, format [mu1 mu2 sig1 sig2]")
    
    opts = parser.parse_args()
    
    native_flag = False
    #i.e., using Spyder on local machine rather the IGWN environment
    if opts.using_eos == None:
        print("No eos options spotted.")
        #Set some local defaults. Need test_params.txt and initgrid.txt in pwd
        native_flag = True
        opts.fname = "Test"
        opts.using_eos = "test_params.txt"
        opts.using_eos_index = None
        opts.eos_start_index = 0
        opts.eos_end_index = 20
        opts.fname_output_integral = "results.txt"
        opts.n_events_to_analyze = 1
        opts.init_grid = "initgrid.txt"
        
        #print(opts)
    
    
    if not(opts.using_eos_index is None):
        opts.eos_start_index = opts.using_eos_index
        opts.eos_end_index = opts.using_eos_index + opts.n_events_to_analyze
    
    dat_orig_names = None
    fname_eos = opts.using_eos
    fname_eos = fname_eos.replace("file:",'')
    with open(fname_eos,'r') as f:
        header_str = f.readline()
        header_str = header_str.rstrip()
        dat_orig_names = header_str.replace('#','').split()[2:]
    print("Original field names ", dat_orig_names)
    
    
    if opts.outdir_clean:
        import shutil
        try: shutil.rmtree(opts.outdir)
        except: pass
        del shutil
    elif opts.outdir is None:
        opts.outdir = "."
        
    if not native_flag:
        from pathlib import Path
        Path(opts.outdir).mkdir(parents=True, exist_ok=True)
        del Path
    
    likelihood_evaluation()



