# -*- coding: utf-8 -*-
"""
External prior code for hyperpipe. 
Possesses an initialize_me() function and a likelihood evaluation function. 
Calculates likelihood of initialized population parameters from a norm.
"""

#! /usr/bin/env python
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#import os
import numpy as np
import argparse
from scipy.stats import multivariate_normal
try:
    import RIFT.lalsimutils as lalsimutils
except:
    print("RIFT not available.")

'''
import RIFT.lalsimutils as lalsimutils
from RIFT.physics import EOSManager as EOSManager

from scipy.integrate import nquad
#import EOS_param as ep
import os
from EOSPlotUtilities import render_eos
import lal
'''

#Local variant, for when no RIFT access (e.g., Spyder)
def m1m2_local(Mc, eta):
    """Compute component masses from Mc, eta. Returns m1 >= m2"""
    
    #WARNING: ADDED LINES VS ORIGINAL:----------
    eta = np.float64(eta)
    Mc = np.float64(Mc)
    #-------------------------------------------
    
    etaV = np.array(1-4*eta,dtype=float) 
    if isinstance(eta, float):
        if etaV < 0:
            etaV = 0
            etaV_sqrt =0
        else:
            etaV_sqrt = np.sqrt(etaV)
    else:
        indx_ok = etaV>=0
        etaV_sqrt = np.zeros(len(etaV),dtype=float)
        etaV_sqrt[indx_ok] = np.sqrt(etaV[indx_ok])
        etaV_sqrt[np.logical_not(indx_ok)] = 0 # set negative cases to 0, so no sqrt problems
    m1 = 0.5*Mc*eta**(-3./5.)*(1. + etaV_sqrt)
    m2 = 0.5*Mc*eta**(-3./5.)*(1. - etaV_sqrt)
    return m1, m2


################## Initialization #####################
sigma1d = 0.1
rv = None
n_dim = None
pop_params = None

def initialize_me(**kwargs):
    #kwargs will take this form:
    #{'input_line' : dat_as_array, 'param_names':param_names}
    #where dat_as_array = dat.view((float, len(param_names))) (whatever this means)
    #& param_names = dat.dtype.names
    print("--- Initializing External Prior ---")
    global rv
    global n_dim
    global pop_params
    if 'input_file_name' in kwargs:
        input_file_name = kwargs['input_file_name']  # filename with x0 lines
        input_file_index = kwargs['input_file_index'] # line in the input filename to use
        print("Loading file '"+input_file_name+"' line "+input_file_index)
        #Load input file, pulling out just the indicated index (1 line):
        pop_params = np.loadtxt(input_file_name)[input_file_index] #this should be obs, I think
    elif 'input_line' in kwargs:
        pop_params = kwargs['input_line']
    
    #cf. rv = multivariate_normal(mean=x0, cov = sigma1d*sigma1d*np.diag(np.ones(n_dim)))
    #presumably the above len(x0)=2, so generate unit covariance
    rv = multivariate_normal(pop_params[2:4], pop_params[4]) #assumes only 2D - not good
    n_dim = len(pop_params[2:])-1
    
    print("pop_params:",pop_params)
    print("n_dim=",n_dim)
 
#from external_prior_example.py:
# =============================================================================
#     global rv
#     print(" ==== INITIALIZING EXTERNAL PRIOR === ")
#     if 'input_file_name' in kwargs:
#         input_file_name = kwargs['input_file_name']  # filename with x0 lines
#         input_file_index = kwargs['input_file_index'] # line in the input filename to use
#         # Parse input file
#         x0 = np.loadtxt(input_file_name)[input_file_index]
#     elif 'input_line' in kwargs:
#         x0 = kwargs['input_line']
#     print(" INITIAL PRIOR, MEAN IN MASSES ", x0)
#     n_dim = len(x0)
#     rv = multivariate_normal(mean=x0, cov = sigma1d*sigma1d*np.diag(np.ones(n_dim)))
# =============================================================================


def retrieve_eos(**kwargs):
    #kwargs will take this form:
    #{'input_line' : dat_as_array, 'param_names':param_names}
    #where dat_as_array = dat.view((float, len(param_names))) (whatever this means)
    #& param_names = dat.dtype.names
    print("Hello! I've been trying to reach you about your car's extended warranty.")
    
    #Open EOS file here


####################### LIKELIHOOD EVAL #######################

def likelihood_evaluation(m1, m2):    
    #Wow, so complex:
    return rv.logpdf([m1,m2]) 


#RIFT
# =============================================================================
# def ln_external_prior(*X):
#     print(" ==== CALLING EXTERNAL PRIOR === ")
#     # Populate function on the grid
#     x_here = np.array(X).T
#     # note first two coordinates are mc, delta_mc! must convert!
#     dat_out = lalsimutils.convert_waveform_coordinates(x_here[:, :2], low_level_coord_names=['mc','delta_mc'],coord_names=['m1','m2'])
#     x_here[:,0] = dat_out[:,0]
#     x_here[:,1] = dat_out[:,1]
#     #    Ly = np.zeros( len(x_here[:,0] ))
#     #    print(x_here.shape, Ly.shape)
#     #    is_set = False
#     Ly = rv.logpdf(x_here)
#     return Ly
# =============================================================================


if __name__ == '__main__':    
    #I'm a pretend CIP! I hold all the strings... mwahahahaha!
    #using_eos = "test_params.txt" #contains population parameters
    obsname = "grid-random-0-0.txt" #the "observations" CIP is testing
    
    
    #-----Adapted from util_ConstructIntrinsicPosterior.py-----
    parser = argparse.ArgumentParser()
    
    # Supplemental likelihood factors: convenient way to effectively change the mass/spin prior in arbitrary ways for example
    # Note this supplemental factor is passed the *fitting* arguments, directly.  Use with extreme caution, since we often change the dimension in a DAG 
    parser.add_argument("--supplementary-likelihood-factor-code", default="ext_prior1",type=str,help="Import a module (in your pythonpath!) containing a supplementary factor for the likelihood.  Used to impose supplementary external priors of arbitrary complexity and external dependence (e.g., external astro priors). EXPERTS-ONLY")
    parser.add_argument("--supplementary-likelihood-factor-function", default="likelihood_evaluation",type=str,help="With above option, specifies the specific function used as an external likelihood. EXPERTS ONLY")
    parser.add_argument("--supplementary-likelihood-factor-ini", default=None,type=str,help="With above option, specifies an ini file that is parsed (here) and passed to the preparation code, called when the module is first loaded, to configure the module. EXPERTS ONLY")
    #parser.add_argument("--supplementary-prior-code",default=None,type=str,help="Import external priors, assumed in scope as extra_prior.prior_dict_pdf, extra_prior.prior_range.  Currentlyonly supports seperable external priors")
    parser.add_argument("--using-eos", type=str, default="test_params.txt", help="Name of EOS.  Fit parameter list should physically use lambda1, lambda2 information (but need not). If starts with 'file:', uses a filename with EOS parameters ")
    parser.add_argument("--using-eos-for-prior", action='store_true', default=True, help="Alternate (hacky) implementation, which overrides using-eos and using-eos-index, to handle loading in a hyperprior")
    parser.add_argument("--using-eos-index", type=int, default=0, help="Index of EOS parameters in file.")    
    
    opts = parser.parse_args()
    print(opts)
    
    has_retrieve_eos = False #not dealing with this yet
    
    supplemental_ln_likelihood= None
    supplemental_ln_likelihood_prep =None
    supplemental_ln_likelihood_parsed_ini=None
    # Supplemental likelihood factor. Must have identical call sequence to 'likelihood_function'. Called with identical raw inputs (including cosines/etc)
    if opts.supplementary_likelihood_factor_code and opts.supplementary_likelihood_factor_function:
        print(" EXTERNAL SUPPLEMENTARY LIKELIHOOD FACTOR : {}.{} ".format(opts.supplementary_likelihood_factor_code,opts.supplementary_likelihood_factor_function))
        #__import__(opts.supplementary_likelihood_factor_code) #example_gaussian
        #external_likelihood_module = sys.modules[opts.supplementary_likelihood_factor_code] #example_gaussian
        supplemental_ln_likelihood = likelihood_evaluation #getattr(external_likelihood_module,opts.supplementary_likelihood_factor_function) #look for likelihood_evaluation()
        name_prep = "prepare_"+opts.supplementary_likelihood_factor_function #"prepare_likelihood_evaluation"
        if opts.using_eos_for_prior:
            #This gets one line of data; it will also get the names for each column, after header:
            dat = np.genfromtxt(opts.using_eos,names=True)[opts.using_eos_index]   # Parse file for them, to reduce need for burden parsing, and avoid burden/confusion.
            param_names = dat.dtype.names #separate out the names from the data
            dat_as_array = dat.view((float, len(param_names)))
            print(dat_as_array)
            args_init = {'input_line' : dat_as_array, 'param_names':param_names}  # pass the recordarray broken into parts, for convenience
            
            dat_orig_names = param_names[2:] #Adapted from ye old example_gaussian
            print("Original field names ", dat_orig_names)
            
            supplemental_init = initialize_me #getattr(external_likelihood_module, 'initialize_me') #find initialize_me()
            supplemental_init(**args_init) #run initialize_me('input_line'=dat_as_array, 'param_names'=param_names)
            # CHECK IF WE RETRIEVE AN EOS from these hyperparameters too, so we can do both. 
            if has_retrieve_eos:
                fake_eos = False  # using EOS hyperparameter conversion! 
                supplemental_eos = retrieve_eos #getattr(external_likelihood_module, 'retrieve_eos')
                supplemental_eos(**args_init) #run retrieve_eos('input_line'=dat_as_array, 'param_names'=param_names)
                my_eos = supplemental_eos(**args_init) #why is it called twice...?            
          
        #Fake CIP integral:
        obs = np.genfromtxt(obsname)
        print("obs line 1:",obs[0])
        
        part_sum = 0.0
        for i in range(len(obs)):
            #lol what an integral....
            part_sum += supplemental_ln_likelihood(obs[i][1],obs[i][2])
        print("Final 'integral' value =",part_sum)


    #from external_prior_example.py:
# =============================================================================
#     x0 = [20,5] # remember these are MASSES
#     rv = multivariate_normal(mean=x0, cov = sigma1d*sigma1d*np.diag(np.ones(len(x0))))
#     dat = rv.rvs(100).T
#     dat_alt = dat.T
#     # force so m1 > m2
#     m1 = np.maximum(dat_alt[:,0], dat_alt[:,1])
#     m2 = np.minimum(dat_alt[:,0], dat_alt[:,1])
#     #    print(m1,m2)
#     # convert input colums to mc, delta_mc, as assumed by inputs
#     mcV = lalsimutils.mchirp(m1,m2)
#     deltaV =  (m1 - m2)/(m1+m2)
#     print(dat.shape, mcV.shape)
#     dat_alt[:,0] = mcV # view into dat still !
#     dat_alt[:,1] = deltaV
#     
#     out = likelihood_evaluation(*dat)
#     print(out)
# =============================================================================



