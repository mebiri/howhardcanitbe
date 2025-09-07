# -*- coding: utf-8 -*-
"""
External prior code for hyperpipe. 
Possesses an initialize_me() function and a likelihood evaluation function. 
Calculates likelihood of initialized population parameters from a norm.

--!!N.B. POPULATION/EOS FILE EXPECTED TO CONTAIN THESE COLUMNS!!--
    # lnL sigma_lnL {EOS columns} m1 m2 sig
"sig" is an uncertainty for both m1 and m2 (may allow separate sigs in future)  
"""

#! /usr/bin/env python
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#import os
import numpy as np
import argparse
from scipy.stats import norm, multivariate_normal

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
#Also most likely conversion, so faster than importing lalsimutils
def m1m2_local(Mc, eta):
    """Compute component masses from Mc, eta. Returns m1 >= m2"""
    
    #WARNING: ADDED LINES VS ORIGINAL:----------
    #eta = np.float64(eta)
    #Mc = np.float64(Mc)
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


def conversion_check(params_list):
    print("Checking given CIP coordinates:",params_list)
    
    #Find what coordinates are in the CIP list & where
    mass_params = ['mc','eta','m1','m2','delta_mc','mtot','q','mc_ecc']
    print("Valid coordinates:",mass_params)
    
    #This should determine which coords are at which index in the CIP list,
    #and how many valid coords have been passed:
    cip_idx = []
    for m in mass_params:
        if m in params_list:
            m_idx = params_list.index(m)
            cip_idx.append([m,m_idx])
            
    print("Search results:",cip_idx)
    
    if len(cip_idx) != 2:
        if len(cip_idx) > 2:
            print("WARNING: Too many viable parameters passed! Can't decide!")
        elif len(cip_idx) < 2:
            print("WARNING: Not enough valid parameters passed! Exiting.")
        return 0, cip_idx
    
    func = 0
    #this is fine b/c order will always be fixed in cip_idx:
    if cip_idx[0][0] == 'm1' and cip_idx[1][0] == 'm2':
        func = 1 #no conversion
    elif cip_idx[0][0] == 'mc' and cip_idx[1][0] == 'eta':
        func = 2 #common conversion, so use local version as shortcut
    else:
        func = 3 #lalsimutils conversion will be requested
    
    #list of lalsimutils conversions available:
    #Mceta(m1, m2)
    #m1m2(Mc,eta)
    #...that's it. Send everything else direct to convert_waveform_coordinates()
    return func, cip_idx


#For computing the integral of rv over the domain
def int_rv(m2,m1):
    rv.pdf(m1,m2)#TODO: might be some problems with rv not being initialized by b_i_c
    

#Check location of pop masses to avoid integrating (and importing scipy) if possible (faster)
def boundary_integration_checks(pop,mass_bounds):
    #Integration bounds; currently for BH masses:
    m_min = mass_bounds[0]
    m_max = mass_bounds[1]
    
    #--NOTE: THIS ALL ASSUMES 1 UNCERTAINTY FOR 2 MASSES (2D PROBLEM)--
    global pop_params
    near_dist = 3*pop_params[2] #3*sigma distance cutoff from means of rv (somewhat lazy but saves time)
    d3 = (pop_params[0]-pop_params[1])/np.sqrt(2) #d3 = (m1-m2)/sqrt(2) = distance from m1=m2 line
    d1 = m_max - pop_params[0] #d1 = distance from right border
    d2 = pop_params[1] - m_min #d2 = distance above bottom border
    
    #TODO: check that d1, d2, d3 are positive -> negative means outside pop range, should be error
    d1c = False
    d2c = False
    d3c = False
    checks = 0
    if d1 < near_dist: d1c = True; checks += 1 #m1 close to m_max (right)
    if d2 < near_dist: d2c = True; checks += 1 #m2 close to m_min (bottom)
    if d3 < near_dist: d3c = True; checks += 1 #(m1,m2) close to m1=m2 (hypotenuse)
    print("Boundary checks: m1:",d1c,"m2:",d2c,"m1=m2:",d3c,"Total:",checks)
    
    #d1 < 3sig --> m1 near right boundary
    #d2 < 3sig --> m2 near bottom boundary
    #d3 < 3sig --> (m1,m2) near m1=m2 boundary
    
    #TODO: The 7 Deadly Tests:
        #1. d1 < 3sig, d2, d3 > 3sig -> CDF(d1) (close to right)
        #2. d2 < 3sig, d1, d3 > 3sig -> CDF(d2) (close to bottom)
        #3. d3 < 3sig, d1, d2 > 3sig -> CDF ??? (close to m1=m2 diag)
        #4. d1, d2 < 3sig, d3 > 3sig -> int -3sig->m_max (bottom right corner)
        #5. d1, d3 < 3sig, d2 > 3sig -> int -3sig->m_max (upper right corner)
        #6. d2, d3 < 3sig, d1 > 3sig -> int m_min->+3sig (bottom left corner)
        #7. d1, d2, d3 < 3sig -> int over whole region (center, large sig)
    
    nm_val = 0
    #if m_max - pop_params[0] < near_dist or pop_params[1] - m_min < near_dist or d3 < near_dist:
    if checks == 0: 
        nm_val = 1 #nothing near edges, just set normalization to 1
    elif checks == 1:
        if d1c: #test 1
            nm_val = norm.cdf(d1,loc=0,scale=pop_params[2])
        elif d2c: #test 2
            nm_val = norm.cdf(d2,loc=0,scale=pop_params[2])
        elif d3c: #test 3
            nm_val = norm.cdf(d3,loc=0,scale=pop_params[2]) #Note this still uses the 1D sigma
        else:
            print("Error: total checks is 1 but no distance checks are true.")
            nm_val = 1 #failsafe
    elif checks > 1:
        #No choice now but to integrate...
        from scipy.integrate import dblquad #does a double integral

        #if close to 2 bounds -> corner -> integrate
        #if narrow, reduce integration bounds to be closer to coord, so no failure
        if pop_params[2]/(m_max-m_min) < 0.1: #sig < 10% width of mass range
            rd_bound = 2*near_dist
        else:
            rd_bound = 0#TODO: THIS WON'T WORK
        
        if d1c and d2c and d3c: #test 7
            #Integrate rv over domain: #TODO: This is a rectangle - need y bounds to vary to get triangle
            nm_val, nm_err = dblquad(int_rv, m_min, m_max, m_min, m_max)
        elif not d3c: #test 4
            #reduce integration bounds to [m1-6sig,m_max] and [m_min,m2+6sig]
            nm_val, nm_err = dblquad(int_rv, pop_params[0]-rd_bound, m_max, m_min, pop_params[1]+rd_bound)
        elif not d2c: #test 5
            nm_val, nm_err = dblquad(int_rv, pop_params[0]-rd_bound, m_max, m_min, pop_params[1]-rd_bound)
        elif not d1c: #test 6
            nm_val, nm_err = dblquad(int_rv, m_min, pop_params[0]+rd_bound, m_min, pop_params[1]+rd_bound)
        else: 
            print("Error: total checks is > 1 but no distance check combinations are true.")
            nm_val = 1 #failsafe
        
    return nm_val


################## Initialization #####################
sigma1d = 0.1
rv = None
n_dim = None
pop_params = None
nm = 1
cfunc = 0
cv_params = None
eos = None

#Try to import lalsimutils (will fail on local machines)
try:
    import RIFT.lalsimutils as lalsimutils
    cfunc = 4#lalsimutils.convert_waveform_coordinates 
except:
    print("WARNING: Unable to import RIFT or RIFT.lalsimutils.")
    cfunc = 1


def initialize_me(**kwargs):
    '''
    **kwargs MUST take this form:
    {'input_line':dat_as_array, 'param_names':param_names, 'cip_param_names':coord_names}
    where: 
        dat_as_array = dat.view((float, len(param_names))) - an array of float values
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
    elif 'input_line' in kwargs:
        all_params = kwargs['input_line']#used by CIP - single line of data from eos file
        print("Given params:",kwargs)
    
    #----- Initialize unit conversion function -----
    global cfunc, cv_params
    cvtest, cv_params = conversion_check(kwargs['cip_param_names'])
    
    if cvtest == 0:
        print("ERROR: could not find valid mass conversion. Something bad will happen now....")
        return #this will break CIP, most likely #TODO: improve this reaction, if possible
    else:
        if cfunc != 1 or cvtest != 3: #lalsimultils imported or not needed
            cfunc = cvtest #cvtest can be 1, 2, or 3
            print("Data will be converted via method",cvtest)
        else: #no lalsimutils (cfunc == 1) and it is needed (cvtest == 3)
            print("RIFT unavailable; data will not be converted.") #i.e., cfunc = 1 still

    
    #----- Initialize population -----
    global pop_params
    global rv
    global n_dim
    global eos
    eos_names = []
    
    #Expected header names: # lnL sigma_lnL g0 g1 g2 g3 m1 m2 sig
    #Split columns into pop and EOS:
    if cvtest != 0:
        #assume population is a go
        pop_params = []
        pop_params_lib = ['m1','m2','sig'] #can be added to for other populations
        for i in kwargs['param_names'][2:]: #should be anything past lnL, sig_lnL
            if i in pop_params_lib:
                pop_params.append(all_params[kwargs['param_names'].index(i)])
            else: #anything that isn't m1, m2, sig
                eos_names.append(i)
                eos.append(all_params[kwargs['param_names'].index(i)])
        
        print("Population parameters found:",pop_params_lib[:len(pop_params)],
              "\nEOS parameters found:",eos_names)
        
        #for i in pop_params_lib:
        #    mdx = kwargs['param_names'].index(i) 
        #    pop_params.append(all_params[mdx]) 
            #TODO: allow flex for 2 sigmas? - easy to add to pop_params_lib
    
        #cf. rv = multivariate_normal(mean=x0, cov = sigma1d*sigma1d*np.diag(np.ones(n_dim)))
        rv = multivariate_normal(pop_params[:2], pop_params[2]) #assumes only 2D - not great
        n_dim = len(pop_params)-1 #TODO: assumes only 1 sigma column (see above)
    else:
        print("ERROR: Population data could not be initialized: data headers not found.")
    
    print("pop_params:",pop_params)
    print("n_dim=",n_dim)
    
    #----- Initialize EOS object -----
    if len(eos_names) > 0:
        try:
            from RIFT.physics import EOSManager as EOSManager
            print("Able to make EOS")
            
            #TODO: make it here, idk
            
        except:
            print("ERROR: Unable to create EOS object.")#should only happen to local runs
    else:
        print("Warning: no EOS parameters found; no EOS object will be created.")
        eos = None
    
    #----- Initialize normalization constant -----
    if pop_params is not None:#TODO: This isn't rigorous enough; checks require len(pop) == 3
        #check population width: if narrow width or far from edges -> nm = 1 (normal)
        global nm
        nm = boundary_integration_checks(pop_params,[3,30])
    #else: nm is set to 1 by default (i.e., entire normal curve is within domain)
    print("Normalization constant set to",nm)
    
    print("----- END EXTERNAL PRIOR INITIALIZATION -----")


#Get the previously-initialized EOS object,
def retrieve_eos(**kwargs): #not sure the kwargs are needed anymore
    '''
    **kwargs MUST take this form:
    {'input_line':dat_as_array, 'param_names':param_names, 'cip_param_names':coord_names}
    where: 
        dat_as_array = dat.view((float, len(param_names))) - an array of float values
        param_names = dat.dtype.names - IN THIS ORDER: lnL lnL_err m1 m2 sigma (sigma same error for both m1 & m2)
        cip_param_names = [str] - given coordinates that CIP is working in, order doesn't matter
    '''
    
    if eos is not None:
        return eos
    else:
        print("Hello! I've been trying to reach you about your car's extended warranty.")
        print("Did you know that, because you own a 2004 Honda Prius, you are entitled to up to $16,000 of insurance for the next 5 years at your local Kia dealership?")
        print("All you have to do is fill in your name, address, and registration (andgivemeallofyourmoney), and we can get you set up in just 10 minutes!")
        print("Oh, yeah, and there's no EOS here; sorry. :)")
        return None #CIP will ignore the EOS (hopefully)


####################### LIKELIHOOD EVAL #######################

def likelihood_evaluation(*X):
    #This looks arduous and slow for a function that will be called 1000 times in a for loop...
    #*X contains data list in same order as cip_params given to initialize_me()
    
    #NOTE: the names in cv_params are ALWAYS in standard order, but the indicies may not be
    #Convert to m1,m2 coords from whatever CIP is passing:
    if cfunc == 1:    
        m1m2 = [X[cv_params[0][1]],X[cv_params[1][1]]] #no conversion
    elif cfunc == 2:
        m1m2 = m1m2_local(X[cv_params[0][1]],X[cv_params[1][1]]) #local mc,eta conversion
    else:
        m1m2 = lalsimutils.convert_waveform_coordinates([X[cv_params[0][1]],X[cv_params[1][1]]], low_level_coord_names=[cv_params[0][0],cv_params[1][0]],coord_names=['m1','m2'])
        
    #Likelihood (w/ normalization constant):
    return rv.logpdf(m1m2) - np.log(nm)


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
    
    #-----Adapted from util_ConstructIntrinsicPosterior.py-----
    parser = argparse.ArgumentParser()
    
    # Supplemental likelihood factors: convenient way to effectively change the mass/spin prior in arbitrary ways for example
    # Note this supplemental factor is passed the *fitting* arguments, directly.  Use with extreme caution, since we often change the dimension in a DAG 
    parser.add_argument("--supplementary-likelihood-factor-code", default="ext_prior1",type=str,help="Import a module (in your pythonpath!) containing a supplementary factor for the likelihood.  Used to impose supplementary external priors of arbitrary complexity and external dependence (e.g., external astro priors). EXPERTS-ONLY")
    parser.add_argument("--supplementary-likelihood-factor-function", default="likelihood_evaluation",type=str,help="With above option, specifies the specific function used as an external likelihood. EXPERTS ONLY")
    parser.add_argument("--supplementary-likelihood-factor-ini", default=None,type=str,help="With above option, specifies an ini file that is parsed (here) and passed to the preparation code, called when the module is first loaded, to configure the module. EXPERTS ONLY")
    #parser.add_argument("--supplementary-prior-code",default=None,type=str,help="Import external priors, assumed in scope as extra_prior.prior_dict_pdf, extra_prior.prior_range.  Currentlyonly supports seperable external priors")
    parser.add_argument("--using-eos", type=str, default="test_pop_m1_m2.txt", help="Name of EOS.  Fit parameter list should physically use lambda1, lambda2 information (but need not). If starts with 'file:', uses a filename with EOS parameters ")
    parser.add_argument("--using-eos-for-prior", action='store_true', default=True, help="Alternate (hacky) implementation, which overrides using-eos and using-eos-index, to handle loading in a hyperprior")
    parser.add_argument("--using-eos-index", type=int, default=0, help="Index of EOS parameters in file.")    
    parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
    parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")

    opts = parser.parse_args()
    print(opts)
    
    #coordinates for CIP to use:
    coord_names = ['mc','eta']#opts.parameter # Used  in fit
    #if coord_names is None:
    #    coord_names = []
    #if opts.parameter_implied:
    #    coord_names = coord_names+opts.parameter_implied
    
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
            args_init = {'input_line' : dat_as_array, 'param_names':param_names, 'cip_param_names':coord_names}  # pass the recordarray broken into parts, for convenience
            
            dat_orig_names = param_names[2:] #Adapted from ye old example_gaussian.py
            print("Original field names:", dat_orig_names)
            
            supplemental_init = initialize_me #getattr(external_likelihood_module, 'initialize_me') #find initialize_me()
            supplemental_init(**args_init) #run initialize_me('input_line'=dat_as_array, 'param_names'=param_names)
            # CHECK IF WE RETRIEVE AN EOS from these hyperparameters too, so we can do both. 
            if has_retrieve_eos:
                fake_eos = False  # using EOS hyperparameter conversion! 
                supplemental_eos = retrieve_eos #getattr(external_likelihood_module, 'retrieve_eos')
                supplemental_eos(**args_init) #run retrieve_eos('input_line'=dat_as_array, 'param_names'=param_names)
                my_eos = supplemental_eos(**args_init) #why is it called twice...?            
    
            
    #Fake CIP integral:
    # Result shifted by lnL_shift
    #fn_passed = likelihood_function
    if supplemental_ln_likelihood:
        #fn_passed =  lambda *x: np.exp(supplemental_ln_likelihood(*x))
    #if opts.internal_use_lnL:
    #    fn_passed = log_likelihood_function   # helps regularize large values
    #    if supplemental_ln_likelihood:
    #        fn_passed =  lambda *x: log_likelihood_function(*x) + supplemental_ln_likelihood(*x)
        #extra_args.update({"use_lnL":True,"return_lnI":True})
        
        #This is cute and all, but not helpful for testing.
        #print("Accessing data file.")
        #obsname = "grid-random-0-0.txt" #the "observations" CIP is testing
        #obs = np.genfromtxt(obsname)
        #print("obs line 1:",obs[0])
        
        #Create npts pairs of random points in a grid space 
        x0=[20,10]
        rvdat = multivariate_normal(mean=x0, cov=0.01*np.diag(np.ones(len(x0))))
        dat = rvdat.rvs(10)
        #dat_alt = dat.T #copy so dat is unaffected by the below
        #Force m1 > m2:
        m1 = np.maximum(dat[:,0], dat[:,1])
        m2 = np.minimum(dat[:,0], dat[:,1])
        #print(m1,m2)
        
        if 'mc' in coord_names: 
            #Convert to mc:
            mcV = (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)
            
            #Convert to other:
            otherV = []
            if 'delta_mc' in coord_names:
                #Convert to delta_mc
                otherV =  (m1 - m2)/(m1+m2)
            elif 'eta' in coord_names:
                #Convert to eta:
                otherV = m1*m2/(m1+m2)/(m1+m2)
                                    
            print("Shape check:",dat.shape, mcV.shape)
            dat[:,0] = mcV
            dat[:,1] = otherV
        elif 'm1' in coord_names:
            print("Shape check:",dat.shape, m1.shape)
            dat[:,0] = m1
            dat[:,1] = m2
        
        obs = np.zeros((len(dat),3))
        #obs[:,0] = -1 - technically, but I don't care
        obs[:,1] = dat[:,0]
        obs[:,2] = dat[:,1]
        #print(obs)
        
        print("'Integrating' over obs.")
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


