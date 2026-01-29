# -*- coding: utf-8 -*-
"""
hyperpipe alternate marginalizer for NS mass distribution

Compute the term L_k = \prod_k w_k = \sum_k ln(w_k), where w_k is the integral
   w_k = \int p(m) g((m-n)_obs, sig_obs) dm 
   over m_obs-3sig_obs < m < m_obs+3sig_obs for k real NS obs, 
   a gaussian population distribution p(m), and a gaussian g(\mu, sig)
Essentially \int g(mu, sig) is a cumulative PDF over the integration region
Full region is not necessary as sig is small. 
cf. Eqn. (2) in Kedia et al. 2025
   
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
import argparse
import sys


internal_dtype = np.float32

parser = argparse.ArgumentParser()
#filenames
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--fname-output-samples",default="output-ILE-samples",help="output posterior samples (default output-ILE-samples -> output-ILE)")
parser.add_argument("--fname-output-integral",default="integral_result",help="output filename for integral result. Postfixes appended")

#eos management
parser.add_argument("--using-eos", type=str, default=None, help="Name of EOS.  Fit parameter list should physically use lambda1, lambda2 information (but need not). If starts with 'file:', uses a filename with EOS parameters ")
parser.add_argument("--using-eos-for-prior", action='store_true', default=None, help="Alternate (hacky) implementation, which overrides using-eos and using-eos-index, to handle loading in a hyperprior")
parser.add_argument("--using-eos-index", type=int, default=None, help="Index of EOS parameters in file.")
parser.add_argument("--input-tides",action='store_true',help="Use input format with tidal fields included.")
parser.add_argument("--input-eos-index",action='store_true',help="Use input format with eos index fields included")
parser.add_argument("--n-events-to-analyze",default=1,type=int,help="Number of EOS realizations to analyze. Currently only supports 1")
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
parser.add_argument("--n-eff",default=3e3,type=float)
parser.add_argument("--fit-method",default="rf",help="rf (default) : rf|gp|quadratic|polynomial|gp_hyper|gp_lazy|cov|kde.  Note 'polynomial' with --fit-order 0  will fit a constant")
parser.add_argument("--protect-coordinate-conversions", action='store_true', help="Adds an extra layer to coordinate conversions with range tests. Slows code down, but adds layer of safety for out-of-range EOS parameters for example")
parser.add_argument("--source-redshift",default=0,type=float,help="Source redshift (used to convert from source-frame mass [integration limits] to arguments of fitting function.  Note that if nonzero, integration done in SOURCE FRAME MASSES, but the fit is calculated using DETECTOR FRAME")
parser.add_argument("--sampler-method",default="adaptive_cartesian",help="adaptive_cartesian|GMM|adaptive_cartesian_gpu|portfolio")
parser.add_argument("--internal-use-lnL",action='store_true',help="integrator internally manipulates lnL. ONLY VIABLE FOR GMM AT PRESENT. ---TREATED AS TRUE IN THIS CODE BY DEFAULT---")
parser.add_argument("--tripwire-fraction",default=0.05,type=float,help="Fraction of nmax of iterations after which n_eff needs to be greater than 1+epsilon for a small number epsilon")

# Supplemental likelihood factors (not presently used here)
parser.add_argument("--supplementary-likelihood-factor-code", default=None,type=str,help="Import a module (in your pythonpath!) containing a supplementary factor for the likelihood.  Used to impose supplementary external priors of arbitrary complexity and external dependence. EXPERTS-ONLY")
parser.add_argument("--supplementary-likelihood-factor-function", default=None,type=str,help="With above option, specifies the specific function used as an external likelihood. EXPERTS ONLY")

opts=  parser.parse_args()


def loop_manager(m_obs,sig_obs,pop_dat,pop_idx,eos_len,out_pts=100,match=True):
    npts = len(pop_dat) #should be 1 in hyperpipe
    print("npts:",npts)
    dat_out = None
    
    if not match:
        npts = out_pts #FOR TESTING PURPOSES ONLY!   
    dat_out = np.zeros((npts,len(pop_dat[0]))) #effectively forcing a deep copy of pop_dat
    
    for n in np.arange(npts):
        line = pop_dat[n,pop_idx:pop_idx+3]
        print("line",n,":",line)
        
        #check 0 < sig < 0.5 (protection against puffing):
        if abs(line[2]) >= 0.5:
            line[2] = 0.49
        else:
            line[2] = abs(line[2])
        
        #2D Gaussian of population 
        rv = multivariate_normal(mean=line[:2], cov=(line[2]**2)*np.diag(np.ones(2)))
        
        dat_out[n][0] = compute_product(m_obs,sig_obs,rv)
        dat_out[n][1] = 0.001
        dat_out[n][2:2+eos_len] = pop_dat[n,2:2+eos_len]
        dat_out[n][pop_idx:pop_idx+3] = line
    print(dat_out[0])
    
    return dat_out
    

def compute_product(m_obs,sig_obs,pop_norm):
    partial_sum = 0.0
    for i in range(len(m_obs)):
        #distribution around real mass:
        g_k = multivariate_normal(mean=m_obs[i], cov=np.diag([sig_obs[i][0]**2,sig_obs[i][1]**2]))
                #norm(loc=0,scale=sig_obs[i])
        
        #integrand is product of gaussians: p(m)*g_k(m)
        int_rv = lambda y, x: pop_norm.pdf([x,y])*g_k.pdf([x,y])
        #prod = lambda x: pop_norm.pdf(x)*g_k.pdf(x-m_obs[i])
        
        #initial integration range (rectangle)
        lxbd = m_obs[i][0] - 3*sig_obs[i][0] #left x bound
        rxbd = m_obs[i][0] + 3*sig_obs[i][0] #right x bound
        lybd = m_obs[i][1] - 3*sig_obs[i][1] #lower y bound
        tybd = m_obs[i][1] + 3*sig_obs[i][1] #upper y bound
        
        #truncate bounds to be within 1 < m1 < 3, 1 < m2 < 3 (rectangle)
        if lxbd < 1.0: 
            lxbd = 1.0
        if rxbd > 3.0:
            rxbd = 3.0
        if lybd < 1.0:
            lybd = 1.0
        if tybd > 3.0:
            tybd = 3.0
        
        w_k = 0.0
        if lxbd >= tybd: 
            #smallest m1 >= largest m2 -> rectangle fully within triangle
            #integrate over rectangle:
            w_k, err = dblquad(int_rv, lxbd, rxbd, lybd, tybd)
        else: 
            #some amount of rectangle outside m2 < m1 triangle region
            if rxbd <= tybd: 
                #largest m1 <= largest m2 -> top edge of rect outside of triangle
                #integrate over trapezoid:
                w_k, err = dblquad(int_rv, lxbd, rxbd, lybd, lambda x: x)
            else: 
                #largest m1 > largest m2 -> top left corner of rectangle outside of triangle
                #split region into trapezoid + rectangle at m1 = max(m2):
                w_k1, err1 = dblquad(int_rv, lxbd, tybd, lybd, lambda x: x)
                w_k2, err2 = dblquad(int_rv, tybd, rxbd, lybd, tybd)
                w_k = w_k1 + w_k2
        
        partial_sum += np.log(w_k) #equivalent to opts.internal_use_lnl = True
    
    print(partial_sum)
    return partial_sum


def save_results(grid, eos_names):
    #NEED TO MATCH ALL CIP OUTPUT FILES FOR HYPERPIPE
    if len(grid) == 1:
        res = grid[0][0]
        var = grid[0][1]**2
        ln_integrand_value = res
        neff = opts.n_eff
        # Save result -- needed for odds ratios, etc.
        #   Warning: integral_result.dat uses *original* prior, before any reweighting
        #File (1/7): MARG-0-0.dat
        np.savetxt(opts.fname_output_integral+".dat", [ln_integrand_value])#+lnL_shift])
        
        eos_extra = []
        annotation_header = "lnL sigmaL neff "
        fname = opts.using_eos.replace('file:','')
        params_here = np.loadtxt(fname)[opts.using_eos_index][2:]
        linefirst =''
        with open(fname) as f:
            linefirst = f.readline()
        linefirst = linefirst[2:]
        annotation_header = linefirst # this will/must be lnL sigma_lnL and then parameter names, which we want to preserve
        with open(opts.fname_output_integral+"+annotation.dat", 'w') as file_out:
            file_out.write("# " + annotation_header + "\n")
            file_out.write(" {} {} ".format(ln_integrand_value, np.sqrt(var)/res) + ' '.join(map(str,params_here)))
            #File (2/7): MARG-0-0+annotation.dat
       
        print(" Max lnL ", ln_integrand_value) #just to preserve output consistency
    
        n_ESS = -1
        # Compute n_ESS.  Should be done by integrator!
    # =============================================================================
    #     if 'log_joint_s_prior' in  samples:
    #         weights_scaled = np.exp(dat_logL - lnLmax + samples["log_joint_prior"] - samples["log_joint_s_prior"])
    #         # dictionary, write this to enable later use of it
    #         samples["joint_s_prior"] = np.exp(samples["log_joint_s_prior"])
    #         samples["joint_prior"] = np.exp(samples["log_joint_prior"])
    #     else:
    #         weights_scaled = np.exp(dat_logL - lnLmax)*sampler._rvs["joint_prior"]/sampler._rvs["joint_s_prior"]
    #     weights_scaled = weights_scaled/np.max(weights_scaled)  # try to reduce dynamic range
    #     n_ESS = np.sum(weights_scaled)**2/np.sum(weights_scaled**2)
    #     print(" n_eff n_ESS ", neff, n_ESS)
    # =============================================================================
        np.savetxt(opts.fname_output_integral+"+annotation_ESS.dat",[[ln_integrand_value, np.sqrt(var)/res, neff, n_ESS]],header=" lnL sigmaL neff n_ESS ")
        #File (3/7): MARG-0-0+annotation_ESS.dat
        
        lnLmax = ln_integrand_value
        weights = 1#np.exp(lnL-lnLmax)#*p/ps #will be e^0 = 1
        
        log_res_reweighted = lnLmax + np.log(np.mean(weights))
        sigma_reweighted= np.std(weights,dtype=np.float64)/np.mean(weights)
        #neff_reweighted = np.sum(weights)/np.max(weights)
        np.savetxt(opts.fname_output_integral+"_withpriorchange.dat", [log_res_reweighted])  # should agree with the usual result, if no prior changes. Erm... about that...
        #File (4/7): MARG-0-0_withpriorchange.dat
        with open(opts.fname_output_integral+"_withpriorchange+annotation.dat", 'w') as file_out:
            str_out = list(map(str,[log_res_reweighted, sigma_reweighted, neff]))
            file_out.write("# " + annotation_header + "\n")
            file_out.write(' '.join( str_out + eos_extra + ["\n"]))
            #File (5/7): MARG-0-0_withpriorchange+annotation.dat
        
        #n_output_size = np.min([len(P_list),opts.n_output_samples])
        #lalsimutils.ChooseWaveformParams_array_to_xml(P_list[:n_output_size],fname=opts.fname_output_samples,fref=P.fref)
        #File (6/7): MARG-0-0.xml.gz - hopefully not needed...
        lnL_list = [grid[0][0]]
        lnL_list = np.array(lnL_list,dtype=internal_dtype)#lnL_list created during downselecting in CIP
        np.savetxt(opts.fname_output_samples+"_lnL.dat", lnL_list)
        #File (7/7): MARG-0-0_lnL.dat - usually contains lnL of all samples    
    else:
        #Needs work to make the output look good (values trapped in lists, etc.)
        res = grid[:,0]
        var = grid[:,1]**2
        ln_integrand_value = res
        neff = opts.n_eff
        # Save result -- needed for odds ratios, etc.
        #   Warning: integral_result.dat uses *original* prior, before any reweighting
        #File (1/7): MARG-0-0.dat
        np.savetxt(opts.fname_output_integral+".dat", [ln_integrand_value])#+lnL_shift])
        
        eos_extra = []
        annotation_header = "lnL sigmaL neff "
        fname = opts.using_eos.replace('file:','')
        params_here = np.loadtxt(fname)[opts.using_eos_index:opts.using_eos_index+opts.n_events_to_analyze][:,2:]
        print(params_here)
        linefirst =''
        with open(fname) as f:
            linefirst = f.readline()
        linefirst = linefirst[2:]
        annotation_header = linefirst # this will/must be lnL sigma_lnL and then parameter names, which we want to preserve
        print(' '.join(map(str,params_here)))
        print(" {} {} ".format(ln_integrand_value, np.sqrt(var)/res) + ' '.join(map(str,params_here)))
        with open(opts.fname_output_integral+"+annotation.dat", 'w') as file_out:
            file_out.write("# " + annotation_header + "\n")
            file_out.write(" {} {} ".format(ln_integrand_value, np.sqrt(var)/res) + ' '.join(map(str,params_here)))
            #File (2/7): MARG-0-0+annotation.dat
        
        #Don't have access to sampled points b/c used scipy
        #samples = sampler._rvs
        #samples_type_names = list(samples.keys())
        #print(samples_type_names)
        #n_params = len(opts.parameter) + len(opts.parameter_implied) #coord_names
        #dat_mass = np.zeros((len(samples[low_level_coord_names[0]]),n_params+3))
        dat_logL = np.zeros(len(grid))#samples[low_level_coord_names[0]]))
    # =============================================================================
    #     if not(opts.internal_use_lnL):
    #         if 'log_integrand' in samples_type_names:
    #             dat_logL = np.log(samples["log_integrand"])
    #         elif 'integrand' in samples_type_names:
    #             dat_logL = np.log(samples["integrand"])
    #         else:
    #             raise Exception("Failure : cannot identify lnL field")
    #     else:
    #         if 'log_integrand' in samples_type_names:
    #             dat_logL = samples['log_integrand']
    #         else:
    #             dat_logL = samples["integrand"]
    # =============================================================================
        dat_logL = grid[:,0] #not really the same thing but whatever
        lnLmax = np.max(dat_logL[np.isfinite(dat_logL)])
        print(" Max lnL ", np.max(dat_logL))
    
        n_ESS = -1
        # Compute n_ESS.  Should be done by integrator!
    # =============================================================================
    #     if 'log_joint_s_prior' in  samples:
    #         weights_scaled = np.exp(dat_logL - lnLmax + samples["log_joint_prior"] - samples["log_joint_s_prior"])
    #         # dictionary, write this to enable later use of it
    #         samples["joint_s_prior"] = np.exp(samples["log_joint_s_prior"])
    #         samples["joint_prior"] = np.exp(samples["log_joint_prior"])
    #     else:
    #         weights_scaled = np.exp(dat_logL - lnLmax)*sampler._rvs["joint_prior"]/sampler._rvs["joint_s_prior"]
    #     weights_scaled = weights_scaled/np.max(weights_scaled)  # try to reduce dynamic range
    #     n_ESS = np.sum(weights_scaled)**2/np.sum(weights_scaled**2)
    #     print(" n_eff n_ESS ", neff, n_ESS)
    # =============================================================================
        np.savetxt(opts.fname_output_integral+"+annotation_ESS.dat",[[ln_integrand_value, np.sqrt(var)/res, neff, n_ESS]],header=" lnL sigmaL neff n_ESS ")
        #File (3/7): MARG-0-0+annotation_ESS.dat
        
        #p = samples["joint_prior"]
        #ps =samples["joint_s_prior"]
        lnL = dat_logL
        lnLmax = np.max(lnL)
        weights = np.exp(lnL-lnLmax)#*p/ps #will probably be e^0 = 1
        
        log_res_reweighted = lnLmax + np.log(np.mean(weights))
        sigma_reweighted= np.std(weights,dtype=np.float64)/np.mean(weights)
        #neff_reweighted = np.sum(weights)/np.max(weights)
        np.savetxt(opts.fname_output_integral+"_withpriorchange.dat", [log_res_reweighted])  # should agree with the usual result, if no prior changes. Erm... about that...
        #File (4/7): MARG-0-0_withpriorchange.dat
        with open(opts.fname_output_integral+"_withpriorchange+annotation.dat", 'w') as file_out:
            str_out = list(map(str,[log_res_reweighted, sigma_reweighted, neff]))
            file_out.write("# " + annotation_header + "\n")
            file_out.write(' '.join( str_out + eos_extra + ["\n"]))
            #File (5/7): MARG-0-0_withpriorchange+annotation.dat
        
        #n_output_size = np.min([len(P_list),opts.n_output_samples])
        #lalsimutils.ChooseWaveformParams_array_to_xml(P_list[:n_output_size],fname=opts.fname_output_samples,fref=P.fref)
        #File (6/7): MARG-0-0.xml.gz - hopefully not needed...
        lnL_list = []
        for l in range(len(grid)):#so pointless...
            lnL_list.append(grid[l][0])
        lnL_list = np.array(lnL_list,dtype=internal_dtype)#lnL_list created during downselecting in CIP
        np.savetxt(opts.fname_output_samples+"_lnL.dat", lnL_list)
        #File (7/7): MARG-0-0_lnL.dat - usually contains lnL of all samples
        
    print("All files saved.")
        

  
#Access NS pulsar mass data:
opts.fname="NSmasses.txt" #TEST CODE
mass_dat = np.genfromtxt(opts.fname,names=True) #will be NSmasses.txt (renamed from all.marg_net to event-0.net)
param_names = list(mass_dat.dtype.names)
dat_as_array = mass_dat.view((float, len(param_names)))[:,1:] #skip first col
print(dat_as_array)

#Split data for no real reason:
mass_list = dat_as_array[:,:2]#np.concatenate([dat_as_array[:,0], dat_as_array[:,1]])
#print(mass_list)
sig_list = dat_as_array[:,2:]#np.concatenate([dat_as_array[:,2], dat_as_array[:,3]])
#print(sig_list)

#Access pop data via EOS file:
#TEST CODE---
opts.using_eos="file:test_pop_eos_Parametrized-EoS_maxmass_EoS_samples.txt"
opts.using_eos_index = 0

fname = opts.using_eos.replace('file:', '')
pop_dat = np.genfromtxt(fname,names=True)[opts.using_eos_index:opts.using_eos_index+opts.n_events_to_analyze] #should be 1 line if n_events=1
param_names = list(pop_dat.dtype.names)
pop_as_array = pop_dat.view((float, len(param_names)))#[:,2:] #skip first 2 cols
print(pop_as_array)
print(len(pop_as_array),len(pop_as_array[0]))

#adapted from ext_prior1.py
eos_names = []
pop_params_names = [] 
pop_params_lib = ['m1','m2','sig'] #can be added to for other populations
pop_params_indx = 0
for i in param_names[2:]: #should be anything past lnL, sig_lnL
    if i == "m1":
        pop_params_indx = param_names.index(i)
    if i in pop_params_lib:
        pop_params_names.append(i)
    else: #anything that isn't m1, m2, sig
        eos_names.append(i)
print("Population parameters found:",pop_params_names,"starting at index",pop_params_indx,
      "\nEOS parameters found:",eos_names)
if len(pop_params_names) < 3:
    print("ERROR: Population data could not be initialized: 3 or more columns required.")
    sys.exit(0)

dat_out = loop_manager(mass_list,sig_list,pop_as_array,pop_params_indx,len(eos_names))

#print(dat_out[:,0])
#print(len(dat_out))
save_results(dat_out,eos_names)
#print("Result:",res)

# =============================================================================
# x = np.linspace(1, 2,10)
# straight_line = [y for y in x]
# 
# import matplotlib.pyplot as plt
# fig1 = plt.figure(figsize=(5,5),dpi=250) 
# ax = fig1.add_subplot(111)
# ax.errorbar(mass_list[:,0],mass_list[:,1],yerr=3*sig_list[:,1],xerr=3*sig_list[:,0],fmt='none',linestyle='')
# ax.plot(x,straight_line)
# ax.set_xlim(left=1.0,right=2.0)
# ax.set_ylim(bottom=1.0,top=2.0)
# ax.set_xlabel("$\mu_1$", size="11")
# ax.set_ylabel("$\mu_2$", size="11")
# ax.tick_params(axis='both', which='major', labelsize=10) 
# fig1.tight_layout()
# plt.show(block=False)
# =============================================================================


