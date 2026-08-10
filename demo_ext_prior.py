# -*- coding: utf-8 -*-
"""
External prior code for hyperpipe. 
Possesses an initialize_me() function and a likelihood evaluation function. 
Calculates likelihood of initialized population parameters from a norm.

--!! N.B. POPULATION/EOS FILE EXPECTED TO CONTAIN THESE COLUMNS !!--
    # lnL sigma_lnL x0 x1 w
"""

#! /usr/bin/env python
import numpy as np
from scipy.stats import norm, multivariate_normal
import sys

'''
Imports used later in code:
 import RIFT.lalsimutils as lalsimutils
'''

################## Initialization #####################
rv = None
nm = 1
eos = None
rift = False
scale = 1.0

#Try to import lalsimutils (will fail on local machines)
try:
    import RIFT.lalsimutils as lalsimutils
    rift = True
except:
    print("WARNING: Unable to import RIFT or RIFT.lalsimutils.")
print(" Initializing external prior: RIFT status is:",rift)


def initialize_me(**kwargs):
    '''
    **kwargs MUST take this form:
    {'input_line':dat_as_array, 'param_names':param_names, 'cip_param_names':coord_names}
    where: 
        dat_as_array = dat.view((float, len(param_names))) - a 1D array of float values
        param_names = dat.dtype.names - IN THIS ORDER: lnL lnL_err {EOS PARAMS} m1 m2 sigma (sigma same error for both m1 & m2)
        cip_param_names = [str] - given coordinates that CIP is working in, order doesn't matter (hopefully)
    '''
    print("----- INITIALIZING EXTERNAL PRIOR -----")
    if 'input_file_name' in kwargs:
        input_file_name = kwargs['input_file_name']  # filename with x0 lines
        input_file_index = kwargs['input_file_index'] # line in the input filename to use
        print("Loading file '"+input_file_name+"' line "+input_file_index)
        #Load input file, pulling out just the indicated index (1 line):
        all_params = np.loadtxt(input_file_name)[input_file_index] 
    elif 'input_line' in kwargs: #typical usage
        all_params = kwargs['input_line']
    
    #This code can do any number of things, to set up any values that will be 
    #needed in the likelihood_evaluation() function, or to create an EOS object
    #for CIP (not done here)
    
    #----- Initialize parameter data -----
    global rv
    global scale
    #weight parameter also gets used for covariance here, just for fun
    rv = multivariate_normal(mean=all_params[:2], cov=(all_params[2]**2)*np.diag(np.ones(2)))
    #random tuning nonsense to set the weight factor:
    scale = 100.0*all_params[2] 
    if all_params[0] < 0.9 or all_params[0] > 1.1 or all_params[1] < 0.9 or all_params[1] > 1.1:
        scale += -200.0 #downweight outside of box
    else:
        scale += 10*norm.pdf(all_params[0],loc=1.0,scale=0.1)*norm.pdf(all_params[1],loc=1.0,scale=0.1) #bias towards center
    
    #----- Initialize EOS object -----
    global eos
    if (rift):
        eos = None #create EOS object here if using with CIP
    
    #----- Initialize normalization constant -----
    global nm
    #xbounds = [1,3] 
    #ybounds = [1,2]
    nm = 1 #should compute normalization factor, usually
    if nm == -1:
        return -2.5e6 #specific failcode for debugging purposes
    print("Normalization constant set to",nm)
    
    print("----- END EXTERNAL PRIOR INITIALIZATION -----")


####################### EOS RETRIEVAL #######################

#Optional function; used with CIP: Get the previously-initialized EOS object
def retrieve_eos(**kwargs): 
    '''
    **kwargs MUST take this form:
    {'input_line':dat_as_array, 'param_names':param_names, 'cip_param_names':coord_names}
    where: 
        dat_as_array = dat.view((float, len(param_names))) - an array of float values
        param_names = dat.dtype.names - IN THIS ORDER: lnL lnL_err m1 m2 sigma (sigma same error for both m1 & m2)
        cip_param_names = [str] - given coordinates that CIP is working in, order doesn't matter
    '''
    
    print("Retrieving EOS object from external initialization.")
    if eos is not None:
        return eos
    else:
        print("Unfortunately, we have no EOS for you today; sorry.")
        return None #CIP will probably crash, CIP_faster will definitely crash on lal conversion


####################### LIKELIHOOD EVAL #######################

def likelihood_evaluation(*X):
    #*X contains data list in same order as params given to initialize_me()
    x_in = np.asarray([X[0],X[1]],dtype=np.float64) 
    #x_in = np.asarray([X[cv_params[0][1]],X[cv_params[1][1]]],dtype=np.float64).T

    #Likelihood (w/ normalization constant):
    if nm == 0:
        return -np.inf
    else:
        return rv.logpdf(x_in) - np.log(nm) + scale


