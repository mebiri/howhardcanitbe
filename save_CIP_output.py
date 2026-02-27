# -*- coding: utf-8 -*-
"""
Extension of hyperpipe: save CIP-like output files, without using CIP
For alternate CIP scripts, crash avoidances, etc.
"""
#! /usr/bin/env python

import numpy as np
import argparse
import sys

internal_dtype = np.float32

parser = argparse.ArgumentParser()

#Arguments used by this code (REQUIRED):
parser.add_argument("--fname-output-samples",default="output-ILE-samples",help="output posterior samples (default output-ILE-samples -> output-ILE)")
parser.add_argument("--fname-output-integral",default="integral_result",help="output filename for integral result. Postfixes appended")
parser.add_argument("--n-eff",default=3e3,type=float)
parser.add_argument("--failstate",type=int,default=0,help="EXCLUSIVE here: tells code how to handle incoming data; 0 leaves input unchanged.")
#(OPTIONAL):
parser.add_argument("--save-all",default=True,help="EXCLUSIVE here: how many of the CIP output files to make; True=6, False=1 (+annotation.dat)")
#parser.add_argument("--outdat",default=None,help="EXCLUSIVE here: direct line(s) of output to save in CIP file formats")
parser.add_argument("--using-eos", type=str, default=None, help="Name of EOS.  Fit parameter list should physically use lambda1, lambda2 information (but need not). If starts with 'file:', uses a filename with EOS parameters ")
parser.add_argument("--using-eos-index", type=int, default=None, help="Index of EOS parameters in file.")  
parser.add_argument("--n-events-to-analyze",default=1,type=int,help="Number of EOS realizations to analyze. Currently only supports 1")

#Other typical CIP arguments, handled but not used here:
#filenames
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")

#eos management
parser.add_argument("--using-eos-for-prior", action='store_true', default=None, help="Alternate (hacky) implementation, which overrides using-eos and using-eos-index, to handle loading in a hyperprior")
parser.add_argument("--input-tides",action='store_true',help="Use input format with tidal fields included.")
parser.add_argument("--input-eos-index",action='store_true',help="Use input format with eos index fields included")
parser.add_argument("--eos-param", type=str, default=None, help="parameterization of equation of state")
parser.add_argument("--eos-param-values", default=None, help="Specific parameter list for EOS")

#output management
parser.add_argument("--no-save-samples",action='store_true')
parser.add_argument("--n-output-samples",default=3000,type=int,help="output posterior samples (default 3000)")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--no-plots",action='store_true')

#used by CIP but ignored here
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
parser.add_argument("--parameter-nofit", action='append', help="Parameter used to initialize the implied parameters, and varied at a low level, but NOT the fitting parameters")
parser.add_argument("--no-adapt-parameter",action='append',help="Disable adaptive sampling in a parameter. Useful in cases where a parameter is not well-constrainxed, and the a prior sampler is well-chosen.")
parser.add_argument("--mc-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--eta-range",default=None,help="Eta range. Important if we have a BNS or other item that has a strong constraint.")
parser.add_argument("--input-distance",action='store_true',help="Use input format with distance fields (but not tidal fields?) enabled.")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--no-downselect",action='store_true',help='Prevent using downselection on output points' )
parser.add_argument("--no-downselect-grid",action='store_true',help='Prevent using downselection on input points. Applied only to mc range' )
parser.add_argument("--aligned-prior", default="uniform",help="Options are 'uniform', 'volumetric', and 'alignedspin-zprior'. Only influences s1z, s2z")
parser.add_argument("--transverse-prior", default="uniform",help="Options are  (default), 'uniform-mag', 'taper-down',  'sqrt-prior',  'alignedspin-zprior', and 'Rbar-singular'. Only influences s1x,s1y,s2x,s2y,Rbar.  Usually NOT intended for final work, except for Rbar-singular.")
parser.add_argument("--chi-max", default=1,type=float,help="Maximum range of 'a' allowed.  Use when comparing to models that aren't calibrated to go to the Kerr limit.")
parser.add_argument("--lnL-offset",type=float,default=np.inf,help="lnL offset. ONLY POINTS within lnLmax - lnLoffset are used in the calculation!  VERY IMPORTANT - default value chosen to include all points, not viable for production with some fit techniques like gp")
parser.add_argument("--lnL-cut",type=float,default=None,help="lnL cut [MANUAL]. Remove points below this likelihood value from consideration.  Generally should not use")
parser.add_argument("--n-max",default=3e8,type=float)
parser.add_argument("--fit-method",default="rf",help="rf (default) : rf|gp|quadratic|polynomial|gp_hyper|gp_lazy|cov|kde.  Note 'polynomial' with --fit-order 0  will fit a constant")
parser.add_argument("--protect-coordinate-conversions", action='store_true', help="Adds an extra layer to coordinate conversions with range tests. Slows code down, but adds layer of safety for out-of-range EOS parameters for example")
parser.add_argument("--source-redshift",default=0,type=float,help="Source redshift (used to convert from source-frame mass [integration limits] to arguments of fitting function.  Note that if nonzero, integration done in SOURCE FRAME MASSES, but the fit is calculated using DETECTOR FRAME")
parser.add_argument("--sampler-method",default="adaptive_cartesian",help="adaptive_cartesian|GMM|adaptive_cartesian_gpu|portfolio")
parser.add_argument("--internal-use-lnL",action='store_true',help="integrator internally manipulates lnL. ONLY VIABLE FOR GMM AT PRESENT. ---TREATED AS TRUE IN THIS CODE BY DEFAULT---")
parser.add_argument("--tripwire-fraction",default=0.05,type=float,help="Fraction of nmax of iterations after which n_eff needs to be greater than 1+epsilon for a small number epsilon")

# Supplemental likelihood factors
parser.add_argument("--supplementary-likelihood-factor-code", default=None,type=str,help="Import a module (in your pythonpath!) containing a supplementary factor for the likelihood.  Used to impose supplementary external priors of arbitrary complexity and external dependence. EXPERTS-ONLY")
parser.add_argument("--supplementary-likelihood-factor-function", default=None,type=str,help="With above option, specifies the specific function used as an external likelihood. EXPERTS ONLY")

opts = parser.parse_args()

#can be called externally, maybe (Alt_marg.py pref.) - need to setup missing opts
def save_results(out_grid, header, save_all=True):
    #NEED TO MATCH ALL CIP OUTPUT FILES FOR HYPERPIPE
    #Note: CIP filenames not formatted to support multiple lines, at present
    for line in out_grid:
        res = line[0]
        if res == -np.inf:
            print("Note: lnL = -inf detected; skipping filesaves for this line.")
            continue
        var_out = line[1]**2 
        if line[1] == 0.0:
            var_out = 0.00001 
        elif not save_all:
            var_out = line[1]
        else:
            var_out = np.sqrt(line[1]**2)/res
        ln_integrand_value = res
        neff = opts.n_eff
        
        # Save result -- needed for odds ratios, etc.
        #   Warning: integral_result.dat uses *original* prior, before any reweighting
        if save_all:
            #File (1/7): MARG-0-0.dat
            np.savetxt(opts.fname_output_integral+".dat", [ln_integrand_value])#+lnL_shift])
        
        eos_extra = []
        params_here = line[2:]#np.loadtxt(fname)[opts.using_eos_index][2:]
        annotation_header = header # this will/must be lnL sigma_lnL and then parameter names, which we want to preserve
        with open(opts.fname_output_integral+"+annotation.dat", 'w') as file_out:
            file_out.write("# " + annotation_header + "\n")
            file_out.write(" {} {} ".format(ln_integrand_value, var_out) + ' '.join(map(str,params_here)))
            #File (2/7): MARG-0-0+annotation.dat
       
        print(" Max lnL ", ln_integrand_value) #just to preserve output consistency
        
        if save_all:
            n_ESS = -1
            np.savetxt(opts.fname_output_integral+"+annotation_ESS.dat",[[ln_integrand_value, var_out, neff, n_ESS]],header=" lnL sigmaL neff n_ESS ")
            #File (3/7): MARG-0-0+annotation_ESS.dat
            
            lnLmax = ln_integrand_value
            weights = 1#np.exp(lnL-lnLmax)#*p/ps #will be e^0 = 1
            
            log_res_reweighted = lnLmax + np.log(np.mean(weights))
            sigma_reweighted= np.std(weights,dtype=np.float64)/np.mean(weights)
            np.savetxt(opts.fname_output_integral+"_withpriorchange.dat", [log_res_reweighted])  # should agree with the usual result, if no prior changes. Erm... about that...
            #File (4/7): MARG-0-0_withpriorchange.dat
            with open(opts.fname_output_integral+"_withpriorchange+annotation.dat", 'w') as file_out:
                str_out = list(map(str,[log_res_reweighted, sigma_reweighted, neff]))
                file_out.write("# " + annotation_header + "\n")
                file_out.write(' '.join( str_out + eos_extra + ["\n"]))
                #File (5/7): MARG-0-0_withpriorchange+annotation.dat
            
            #lalsimutils.ChooseWaveformParams_array_to_xml(P_list[:n_output_size],fname=opts.fname_output_samples,fref=P.fref)
            #File (6/7): MARG-0-0.xml.gz - hopefully not needed...
            lnL_list = [line[0]] #still just for the 1 line; sampler results in CIP
            lnL_list = np.array(lnL_list,dtype=internal_dtype)#lnL_list created during downselecting in CIP
            np.savetxt(opts.fname_output_samples+"_lnL.dat", lnL_list)
            #File (7/7): MARG-0-0_lnL.dat - usually contains lnL of all samples    
        
        print("All files saved for this line.")

#more original code from CIP:
# =============================================================================
#         eos_extra = []
#         annotation_header = "lnL sigmaL neff "
#         fname = opts.using_eos.replace('file:','')
#         params_here = np.loadtxt(fname)[opts.using_eos_index:opts.using_eos_index+opts.n_events_to_analyze][:,2:]
#         linefirst =''
#         with open(fname) as f:
#             linefirst = f.readline()
#         linefirst = linefirst[2:]
#         annotation_header = linefirst # this will/must be lnL sigma_lnL and then parameter names, which we want to preserve
#         with open(opts.fname_output_integral+"+annotation.dat", 'w') as file_out:
#             file_out.write("# " + annotation_header + "\n")
#             file_out.write(" {} {} ".format(ln_integrand_value, np.sqrt(var)/res) + ' '.join(map(str,params_here)))
#             #File (2/7): MARG-0-0+annotation.dat
#         
#         #Don't have access to sampled points b/c used scipy
#         #samples = sampler._rvs
#         #samples_type_names = list(samples.keys())
#         #print(samples_type_names)
#         #n_params = len(opts.parameter) + len(opts.parameter_implied) #coord_names
#         #dat_mass = np.zeros((len(samples[low_level_coord_names[0]]),n_params+3))
#         dat_logL = np.zeros(len(grid))#samples[low_level_coord_names[0]]))
#         if not(opts.internal_use_lnL):
#             if 'log_integrand' in samples_type_names:
#                 dat_logL = np.log(samples["log_integrand"])
#             elif 'integrand' in samples_type_names:
#                 dat_logL = np.log(samples["integrand"])
#             else:
#                 raise Exception("Failure : cannot identify lnL field")
#         else:
#             if 'log_integrand' in samples_type_names:
#                 dat_logL = samples['log_integrand']
#             else:
#                 dat_logL = samples["integrand"]
#         dat_logL = grid[:,0] #not really the same thing but whatever
#         lnLmax = np.max(dat_logL[np.isfinite(dat_logL)])
#         print(" Max lnL ", np.max(dat_logL))
#     
#         n_ESS = -1
#         # Compute n_ESS.  Should be done by integrator!
#         if 'log_joint_s_prior' in  samples:
#             weights_scaled = np.exp(dat_logL - lnLmax + samples["log_joint_prior"] - samples["log_joint_s_prior"])
#             # dictionary, write this to enable later use of it
#             samples["joint_s_prior"] = np.exp(samples["log_joint_s_prior"])
#             samples["joint_prior"] = np.exp(samples["log_joint_prior"])
#         else:
#             weights_scaled = np.exp(dat_logL - lnLmax)*sampler._rvs["joint_prior"]/sampler._rvs["joint_s_prior"]
#         weights_scaled = weights_scaled/np.max(weights_scaled)  # try to reduce dynamic range
#         n_ESS = np.sum(weights_scaled)**2/np.sum(weights_scaled**2)
#         print(" n_eff n_ESS ", neff, n_ESS)
#         np.savetxt(opts.fname_output_integral+"+annotation_ESS.dat",[[ln_integrand_value, np.sqrt(var)/res, neff, n_ESS]],header=" lnL sigmaL neff n_ESS ")
#         #File (3/7): MARG-0-0+annotation_ESS.dat
#         
#         #p = samples["joint_prior"]
#         #ps =samples["joint_s_prior"]
#         lnL = dat_logL
#         lnLmax = np.max(lnL)
#         weights = np.exp(lnL-lnLmax)#*p/ps #will probably be e^0 = 1
#         
#         log_res_reweighted = lnLmax + np.log(np.mean(weights))
#         sigma_reweighted= np.std(weights,dtype=np.float64)/np.mean(weights)
#         #neff_reweighted = np.sum(weights)/np.max(weights)
#         np.savetxt(opts.fname_output_integral+"_withpriorchange.dat", [log_res_reweighted])  # should agree with the usual result, if no prior changes. Erm... about that...
#         #File (4/7): MARG-0-0_withpriorchange.dat
#         with open(opts.fname_output_integral+"_withpriorchange+annotation.dat", 'w') as file_out:
#             str_out = list(map(str,[log_res_reweighted, sigma_reweighted, neff]))
#             file_out.write("# " + annotation_header + "\n")
#             file_out.write(' '.join( str_out + eos_extra + ["\n"]))
#             #File (5/7): MARG-0-0_withpriorchange+annotation.dat
#         
#         #n_output_size = np.min([len(P_list),opts.n_output_samples])
#         #lalsimutils.ChooseWaveformParams_array_to_xml(P_list[:n_output_size],fname=opts.fname_output_samples,fref=P.fref)
#         #File (6/7): MARG-0-0.xml.gz - hopefully not needed...
#         lnL_list = []
#         for l in range(len(grid)):#so pointless...
#             lnL_list.append(grid[l][0])
#         lnL_list = np.array(lnL_list,dtype=internal_dtype)#lnL_list created during downselecting in CIP
#         np.savetxt(opts.fname_output_samples+"_lnL.dat", lnL_list)
#         #File (7/7): MARG-0-0_lnL.dat - usually contains lnL of all samples
# =============================================================================


#------------------------------------------------------------------------------
if opts.using_eos is None:
    print("--Warning: Test Mode: using preset files--")
    opts.using_eos="file:test_pop_eos_Parametrized-EoS_maxmass_EoS_samples.txt"
    opts.using_eos_index = 0

#Access EOS file data:
fname = opts.using_eos.replace('file:', '')
pop_dat = None
try:
    check_dat = np.genfromtxt(fname,names=True)[opts.using_eos_index] #test for index being out of range
    pop_dat = np.genfromtxt(fname,names=True)[opts.using_eos_index:opts.using_eos_index+opts.n_events_to_analyze] #should be 1 line if n_events=1
except Exception as e:
    print(" Fail: EOS index out of range:\n   ",e)
    sys.exit(0)
param_names = list(pop_dat.dtype.names)
pop_as_array = pop_dat.view((float, len(param_names)))#[:,2:] #skip first 2 cols
print(pop_as_array)
print(" retrieved dat size: (",len(pop_as_array),len(pop_as_array[0]),")")

dat_out = np.zeros((len(pop_as_array),len(pop_as_array[0])))
savetype=opts.save_all
if opts.failstate == 3:
    #EOS creation failed in CIP
    dat_out[:,2:] = pop_as_array[:,2:]
    dat_out[:,0] = -1000000.0 #fiducial value
    dat_out[:,1] = 0.001 #fiducial value
    savetype=False
else:
    dat_out[:,:] = pop_as_array[:,:]

lineheader = ' '.join(map(str,param_names))+"\n" #to match CIP extracted header
save_results(dat_out,lineheader,save_all=savetype)


