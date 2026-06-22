#! /usr/bin/env python

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
#import RIFT.lalsimutils as lalsimutils
#import lalsimulation as lalsim
#import lal
#import functools
#import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--inj-file", help="Name of dat file")
parser.add_argument("--inj-file-out", default="output-puffball.dat", help="Name of dat file")
parser.add_argument("--puff-factor", default=1,type=float)
parser.add_argument("--force-away", default=0,type=float,help="If >0, uses the icov to compute a metric, and discards points which are close to existing points")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--no-correlation", type=str,action='append', help="Pairs of parameters, in format [mc,eta]  The corresponding term in the covariance matrix is eliminated")
#parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
parser.add_argument("--random-parameter", action='append',help="These parameters are specified at random over the entire range, uncorrelated with the grid used for other parameters.  Use for variables which correlate weakly with others; helps with random exploration")
parser.add_argument("--random-parameter-range", action='append', type=str,help="Add a range (pass as a string evaluating to a python 2-element list): --parameter-range '[0.,1000.]'   MUST specify ALL parameter ranges (min and max) in order if used.  ")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--reflect-parameter",action='append',type=str)
parser.add_argument("--regularize",action='store_true',help="Add some ad-hoc terms based on priors, to help with nearly-singular matricies")
parser.add_argument("--use-rotated-spectral-coords",action='store_true',help="Apply rotated coord sys to spectral EOS params; from Wysocki 2020 https://arxiv.org/pdf/2001.01747")
parser.add_argument("--downselect-mass-range",action='store_true',help="Reject points where m2 > m1 for population models (slightly hacky; prefer diff coord sys)")
parser.add_argument("--rotated-coord-buffer",default=0.0,type=float,help="Fractional buffer (e.g., 0.1; default 0) to extend rotated hypercube space (APPLIES AS % OF BOUND VALUE)")
parser.add_argument("--reflect-parameters",action='store_true',help="Toggle parameter reflection, even if no --reflect-parameter provided (for rotated coord reflection)")
parser.add_argument("--use-alternate-buffer",default=0.05,type=float,help="Buffer expands hypercube by x% of its full width (2x% total expansion), instead of by x% of bound value.")

#to unlink downselect & reflection:
parser.add_argument("--parameter-range",action='append',type=str)
#for generalization:
parser.add_argument("--supplementary-coordinate-code", default=None,type=str,help="Coordinate conversion/prior code. Accepts: the literal 'rift_default' (use RIFT.lalsimutils.convert_waveform_coordinates plus RIFT-standard priors); a filesystem path ending in .py (loaded as a plugin); or any importable dotted module name.")
parser.add_argument("--supplementary-coordinate-function", default=None, type=str, help="Name of the entry-point callable inside the module named by --supplementary-coordinate-code. Defaults to 'convert_coordinates'.")

opts=  parser.parse_args()


if opts.random_parameter is None:
    opts.random_parameter = []

# Extract parameter names
coord_names = opts.parameter # Used  in fit
#if opts.parameter_nofit:
#    coord_names = coord_names + opts.parameter_nofit
if coord_names is None:
    sys.exit(0)

# match up pairs in --no-correlation
corr_list = None
if not(opts.no_correlation is None):
    corr_list = []
    corr_name_list = list(map(eval,opts.no_correlation))
#    print opts.no_correlation, corr_name_list
    for my_pair in corr_name_list:
        
        i1 = coord_names.index(my_pair[0])
        i2 = coord_names.index(my_pair[1])

        if i1>-1 and i2 > -1:
            corr_list.append([i1,i2])
#        else:
#            print i1, i2
#    print opts.no_correlation, coord_names, corr_list

downselect_dict = {}
reflect_dict={}


if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = list(map(eval,opts.downselect_parameter_range)) #change to parameter_range
else: #change to elif and put reflect_parameter code from below here
    dlist = []
    dlist_ranges = []
    opts.downselect_parameter =[]
if len(dlist) != len(dlist_ranges): #check rlist as well
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)): #probably generalize to a param_dict & plist
    downselect_dict[dlist[indx]] = dlist_ranges[indx]

indx_reflect=[]
rlist=[]
if opts.reflect_parameter:
    rlist  = opts.reflect_parameter
    indx_reflect = [coord_names.index(param) for param in opts.reflect_parameter]
if len(rlist) > len(coord_names):
    print(" reflection parameters inconsistent", rlist, coord_names)
    raise Exception(" Reflection only allowed for coordinates ")
for indx in np.arange(len(rlist)):
    if not(rlist[indx] in coord_names):
        raise Exception(" Reflection only allowed for coordinates (--parameter) ")
    if not(rlist[indx] in downselect_dict):
        raise Exception(" Reflection requires parameter range specified as a downselection ")
    reflect_dict[rlist[indx]] = downselect_dict[rlist[indx]]



# Load data, keep parameter names
dat_raw = np.genfromtxt(opts.inj_file,names=True)
X= np.zeros((len(dat_raw), len(coord_names)))
# Copy over the parameters we use.  Note we have no way to create linear combinations or alternate coordinates here
for p in coord_names:
#    indx_p = list(dat_raw.dtype.names).index(p)
    indx_in = coord_names.index(p)
    X[:,indx_in] = dat_raw[p]


# Measure covariance matrix and generate random errors
if len(coord_names) >1:
    cov_in = np.cov(X.T)
    cov = cov_in*opts.puff_factor*opts.puff_factor

    # Check for singularities
    if np.min(np.linalg.eig(cov)[0])<1e-10:
        print(" ===> WARNING: SINGULAR MATRIX: are you sure you varied this parameters? <=== ")
        icov_pseudo = np.linalg.pinv(cov)
        # Prior range for each parameter is 1000, so icov diag terms are 10^(-6)
        # This is somewhat made up, but covers most things
        diag_terms = 1e-6*np.ones(len(cov))
        # 
        icov_proposed = icov_pseudo+np.diag(diag_terms)
        cov= np.linalg.inv(icov_proposed)

    cov_orig = np.array(cov)  # force copy
    # Remove targeted covariances
    if not(corr_list is None):
      for my_pair in corr_list:
        if my_pair[0] != my_pair[1]:
            cov[my_pair[0],my_pair[1]]=0
            cov[my_pair[1],my_pair[0]]=0
            

    # Compute errors
    rv = scipy.stats.multivariate_normal(mean=np.zeros(len(coord_names)), cov=cov,allow_singular=True)  # they are just complaining about dynamic range of parameters, usually
    delta_X = rv.rvs(size=len(X))
    X_out = X#+delta_X
    
    # Reflection
    for indx in indx_reflect:
        param = coord_names[indx]
        print("   Reflecting into range : {} [{}, {}]".format(param,reflect_dict[param][0],reflect_dict[param][1]))
        # put in range [0,2 L]
        tmp = reflect_dict[param][0] + np.mod(X_out[:,indx] - reflect_dict[param][0], 2*(reflect_dict[param][1] - reflect_dict[param][0]) )
        # final reflection
        tmp = np.where( tmp > reflect_dict[param][1], 2*reflect_dict[param][1] - tmp, tmp)
        X_out[:,indx] = tmp
        # DELETE parameter from downselet_dict : no longer needed
        del downselect_dict[param]
        
else:
    sigma = np.std(X)
    cov = sigma*sigma
    delta_X =np.random.normal(size=len(coord_names), scale=sigma)
    X_out = X+delta_X


# Downselect
names_downselect = list(downselect_dict.keys())
indx_ok = np.ones(len(X_out),dtype=bool)
print(" Initial data length:",len(X_out))

#Apply rotation to spectral EOS params from Wysocki et. al 2020 https://arxiv.org/pdf/2001.01747
if opts.use_rotated_spectral_coords:
    print(" Applying rotated coordinate transformation to spectral parameters")
    dan_rot = [[0.43801, -0.53573, 0.52661, -0.49379],
               [-0.76705, 0.17169, 0.31255, -0.53336],
               [0.45143, 0.67967, -0.19454, -0.54443],
               [0.12646, 0.47070, 0.76626, 0.41868]]
    scaled_mean = [0.89421, 0.33878, -0.07894, 0.00393]
    scaled_sig = [0.35700, 0.25769, 0.05452, 0.00312]
    dan_inv = np.linalg.inv(dan_rot)
    
    #get gammas' indices in X_out(= X) from coord names
    r_tilde = np.zeros((len(X_out),4))
    rot_cols = []
    for i in np.arange(4):
        #do one coord at a time
        indx = coord_names.index("gamma"+str(i))
        rot_cols.append(indx)

        #convert gammas to r_tilde using equation: r_tilde = (gamma - u)/sig
        r_tilde[:,i] = (X_out[:,indx] - scaled_mean[i])/scaled_sig[i]

    #apply transform: r_prime = S*r_tilde ( [4 x 4].([N x 4].T) )
    r_prime = np.matmul(dan_rot,r_tilde.T).T
    
    rot_coords = {}
    rot_coords["r0"] = [-4.37722, 4.91227]
    rot_coords["r1"] = [-1.82240, 2.06387]
    rot_coords["r2"] = [-0.32445, 0.36469]
    rot_coords["r3"] = [-0.09529, 0.11426]
    
    for indx, param in enumerate(rot_coords.keys()):
        # apply hypercube buffer
        if opts.use_alternate_buffer: #new_bound = bound +/- buffer*(width of cube in param)
            ubound = rot_coords[param][1] + opts.rotated_coord_buffer*abs(rot_coords[param][1]-rot_coords[param][0])
            lbound = rot_coords[param][0] - opts.rotated_coord_buffer*abs(rot_coords[param][1]-rot_coords[param][0])
        else:
            ubound = rot_coords[param][1] + opts.rotated_coord_buffer*abs(rot_coords[param][1])
            lbound = rot_coords[param][0] - opts.rotated_coord_buffer*abs(rot_coords[param][0])
        if opts.reflect_parameters:
            # Reflection
            print("   Reflecting into range : {} [{}, {}]".format(param,lbound,ubound))
            # put in range [0,2 L]
            #tmp = rot_coords[param][0] + np.mod(r_prime[:,indx] - rot_coords[param][0], 2*(rot_coords[param][1] - rot_coords[param][0]) )
            tmp = lbound + np.mod(r_prime[:,indx] - lbound, 2*(ubound - lbound) )
            # final reflection
            #tmp = np.where( tmp > rot_coords[param][1], 2*rot_coords[param][1] - tmp, tmp)
            tmp = np.where( tmp > ubound, 2*ubound - tmp, tmp)
            r_prime[:,indx] = tmp 
        else:
            # Downselection
            #print(" Downselecting:",name,"; indx:",indx,"[",lbound,ubound,"]; buffer =",opts.rotated_coord_buffer)
            indx_ok = np.logical_and(indx_ok,  r_prime[:,indx]<= ubound )
            indx_ok = np.logical_and(indx_ok,  r_prime[:,indx]>= lbound )
            print('   Increment downselect : {} {} '.format(param, np.sum(indx_ok) ))
    
    if opts.reflect_parameters:
        #apply inverse: S-1*r_prime = S-1*S*r_tilde = r_tilde ( [4 x 4].([N x 4].T) )
        r_tilde_post = np.matmul(dan_inv,r_prime.T).T
        
        for i, col in enumerate(rot_cols):    
            #r_tilde = (gamma - u)/sig  ->  gamma = r_tilde*sig + u
            X_out[:,col] = (r_tilde_post[:,i]*scaled_sig[i]) + scaled_mean[i]
            

#print("post rotational downselect:")
#print(X_out[:,:4])

# no conversion needed
for name in names_downselect:
    indx = coord_names.index(name)
    #print(" Downselecting:",name,"; indx:",indx,"[",downselect_dict[name][0],downselect_dict[name][1],"]")
    indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isnan(X_out[:,indx])))
    indx_ok = np.logical_and(indx_ok,  X_out[:,indx]<= downselect_dict[name][1] )
    indx_ok = np.logical_and(indx_ok,  X_out[:,indx]>= downselect_dict[name][0] )
    print('   Increment downselect : {} {} '.format(name, np.sum(indx_ok) ))

#enforce m1 >= m2 for populations
if opts.downselect_mass_range and ("m1" in coord_names) and ("m2" in coord_names):
    m1_indx = coord_names.index("m1")
    m2_indx = coord_names.index("m2")
    indx_ok = np.logical_and(indx_ok,  X_out[:,m1_indx]>= X_out[:,m2_indx] )
    print('   Increment downselect : {} {} '.format("m1>=m2", np.sum(indx_ok) ))

X_out = X_out[indx_ok]
dat_raw = dat_raw[indx_ok] # must downselect here as well!
    
# Write data back into correct format and save
for p in coord_names:
#    indx_p = dat_raw.dtype.names.index(p)
    indx_in = coord_names.index(p)
    dat_raw[p] = X_out[:,indx_in]

np.savetxt(opts.inj_file_out, dat_raw,header=" ".join(dat_raw.dtype.names))