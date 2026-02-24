# -*- coding: utf-8 -*-
"""
External prior code for hyperpipe. 
Possesses an initialize_me() function and a likelihood evaluation function. 
Calculates likelihood of initialized population parameters from a norm.
Also able to initial parametric EOS model, default type spectral.

--!! N.B. POPULATION/EOS FILE EXPECTED TO CONTAIN THESE COLUMNS !!--
    # lnL sigma_lnL {EOS columns} m1 m2 sig
"sig" is an uncertainty for both m1 and m2 (may allow separate sigs in future)

--!! N.B.: NS-BH systems NOT handled by this code; only BNS/BBH systems !!--
"""

#! /usr/bin/env python
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import argparse
from scipy.stats import norm, multivariate_normal
import sys

'''
Imports used later in code:
 import RIFT.physics.EOSManager as EOSManager
 import RIFT.lalsimutils as lalsimutils
 from scipy.stats import norm
 from scipy.integrate import dblquad
'''

#Local variant, for when no RIFT access (e.g., Spyder)
#Also most likely conversion, so faster than going thru lalsimutils (probably)
def m1m2_local(Mc, eta):
    """Compute component masses from Mc, eta. Returns m1 >= m2"""
    #print("Received:",Mc,eta)
    etaV = np.array(1-4*eta,dtype=float) 
    #print("etaV:",etaV)
    if isinstance(eta, float):
        if etaV < 0:
            etaV = 0
            etaV_sqrt =0
        else:
            etaV_sqrt = np.sqrt(etaV)
    else:
        indx_ok = etaV>=0
        #print("indx_ok:",indx_ok)
        etaV_sqrt = np.zeros(len(etaV),dtype=float)
        #print("etaV_sqrt:",etaV_sqrt)
        etaV_sqrt[indx_ok] = np.sqrt(etaV[indx_ok])
        etaV_sqrt[np.logical_not(indx_ok)] = 0 # set negative cases to 0, so no sqrt problems
        #print("etaV_sqrt, post:",etaV_sqrt)
    m1 = 0.5*Mc*eta**(-3./5.)*(1. + etaV_sqrt)
    m2 = 0.5*Mc*eta**(-3./5.)*(1. - etaV_sqrt)
    return m1, m2


#Exists to reduce time spent running thru lalsimutils
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
    #hardcoded indices are fine b/c order will always be fixed in cip_idx:
    if cip_idx[0][0] == 'm1' and cip_idx[1][0] == 'm2':
        func = 1 #no conversion
    elif cip_idx[0][0] == 'mc' and cip_idx[1][0] == 'eta':
        func = 2 #common conversion, so use local version as shortcut
    else:
        func = 3 #lalsimutils conversion will be requested
    
    return func, cip_idx
    

#Check location of pop masses to avoid integrating (and importing scipy) if possible (faster)
def boundary_integration_checks(pop,mass_bounds):
    #Integration bounds; currently for BH masses:
    m_min = mass_bounds[0]
    m_max = mass_bounds[1]
    
    #--NOTE: THIS ASSUMES 1 UNCERTAINTY FOR 2 MASSES WITH M1 > M2 (2D PROBLEM)--
    global pop_params
    near_dist = 3*pop_params[2] #3*sigma distance cutoff from means of rv (somewhat lazy but saves time)
    d3 = (pop_params[0]-pop_params[1])/np.sqrt(2) #d3 = (m1-m2)/sqrt(2) = distance from m1=m2 line
    d1 = m_max - pop_params[0] #d1 = distance from right border
    d2 = pop_params[1] - m_min #d2 = distance above bottom border
    
    #Check that d1, d2, d3 are positive -> negative means outside pop range
    if d1 < 0 or d2 < 0 or d3 < 0 or pop_params[0] < m_min or pop_params[1] > m_max:
        print("=====\n FAILSTATE 4: 1 OR MORE MASSES OUTSIDE VALID RANGE. EXITING.\n=====")
        sys.exit(0)
        #return 0 #set normalization constant to 0, so lnL = -infinity
    
    d1c = False
    d2c = False
    d3c = False
    checks = 0
    if d1 < near_dist: d1c = True; checks += 1 #m1 near m_max (right boundary)
    if d2 < near_dist: d2c = True; checks += 1 #m2 near m_min (bottom boundary)
    if d3 < near_dist: d3c = True; checks += 1 #(m1,m2) near m1=m2 (hypotenuse boundary)
    print("Boundary checks: m1:",d1c,"m2:",d2c,"m1=m2:",d3c,"Total:",checks)
    
    #The 7 Deadly Tests:
        #1. d1 < 3sig, d2, d3 > 3sig -> CDF(d1) (close to right)
        #2. d2 < 3sig, d1, d3 > 3sig -> CDF(d2) (close to bottom)
        #3. d3 < 3sig, d1, d2 > 3sig -> CDF ??? (close to m1=m2 diag)
        #4. d1, d2 < 3sig, d3 > 3sig -> int -3sig->m_max (bottom right corner)
        #5. d1, d3 < 3sig, d2 > 3sig -> int -3sig->m_max (upper right corner)
        #6. d2, d3 < 3sig, d1 > 3sig -> int m_min->+3sig (bottom left corner)
        #7. d1, d2, d3 < 3sig -> int over whole region (center, large sig)
    
    nm_val = 0
    if checks == 0: 
        nm_val = 1 #nothing near edges, just set normalization to 1
    elif checks == 1:
        #from scipy.stats import norm
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
        #if close to 2 bounds -> corner -> No choice now but to integrate...
        from scipy.integrate import dblquad #does a double integral

        #if narrow, reduce integration bounds to be closer to coord, so no failure
        if pop_params[2]/(m_max-m_min) < 0.05: #sig < 5% width of mass range
            #reduce integration bounds to m +/- 6sig:
            rxbd = pop_params[0]-(2*near_dist) #lower x bound for right side
            lxbd = pop_params[0]+(2*near_dist) #upper x bound for left corner
            tybd = pop_params[1]-(2*near_dist) #lower y bound for top corner
            bybd = pop_params[1]+(2*near_dist) #upper y bound for bottom side
        else:
            rxbd = m_min #lower x bound for right side
            lxbd = m_max #upper x bound for left corner
            tybd = m_min #lower y bound for top corner
            bybd = m_max #upper y bound for bottom side
        
        global rv
        int_rv = lambda y, x: rv.pdf([x,y])
        if d1c and d2c and d3c: #test 7
            #Integrate rv over (triangular) domain:
            nm_val, nm_err = dblquad(int_rv, m_min, m_max, m_min, lambda x: x)
        elif not d3c: #test 4
            nm_val, nm_err = dblquad(int_rv, rxbd, m_max, m_min, bybd)
        elif not d2c: #test 5 
            nm_val, nm_err = dblquad(int_rv, rxbd, m_max, tybd, lambda x: x)
        elif not d1c: #test 6 
            nm_val, nm_err = dblquad(int_rv, m_min, lxbd, m_min, lambda x: x)
        else: 
            print("Error: total checks is > 1 but no distance check combinations are true.")
            nm_val = 1 #failsafe
        
    return nm_val


#NOTE: ONLY THESE EOS_PARAMS HANDLED CURRENTLY: spectral, cs_spectral, PP
def generate_eos(eos_line, eos_headers, eos_param="spectral"):
    print("Creating EOS object of type",eos_param,"using given data line.")
    
    eos_names = eos_headers
    if ((eos_param == "spectral" or eos_param == "cs_spectral") and eos_names[0] != "gamma1") or (eos_param=="PP" and eos_names[1] != "gamma1"):
        print("WARNING: Unsupported gamma labels in EOS names found:",eos_names,"will relabel.")
        counter = 0
        indx= 0
        while counter < 4 and indx < len(eos_headers):#max 4 gamma cols, or stop at end of list
            if eos_headers[indx][0] == 'g' and eos_headers[indx][-1] == str(counter):#ensure gamma col
                counter += 1
                eos_names[indx] = "gamma"+str(counter)
            indx+=1
        print("Relabeled EOS headers:",eos_names)  
        #TODO: may need to handle re-sorting for spectral types if g1 not first, as a precaution
    
    #Better than CIP, for sure...
    spec_param_array = eos_line 
    spec_params ={}

    for i in range(len(eos_names)):
        spec_params[eos_names[i]]=spec_param_array[i]
    print("EOS data:\n",spec_params)
    
    try: #test code
        import RIFT.physics.EOSManager as EOSManager
    except:
        print("-- ERROR: could not import EOSManager. --") #test code, only on local machine
        #return None
    
    eos_name="default_eos_name"
    eos_base = None
    try:
        if eos_param == 'spectral':
            #expect cols: gamma1, gamma2, gamma3, gamma4 (or fewer; must be at least 2 cols)
            eos_base = EOSManager.EOSLindblomSpectral(name=eos_name,spec_params=spec_params,use_lal_spec_eos=True)
        elif eos_param == 'cs_spectral' and len(spec_param_array) >=4:
            #expect cols: gamma1, gamma2, gamma3, gamma4
            eos_base = EOSManager.EOSLindblomSpectralSoundSpeedVersusPressure(name=eos_name,spec_params=spec_params,use_lal_spec_eos=True)
        elif eos_param == 'PP' and len(spec_param_array) >=4:
            #expect cols: logP1, gamma1, gamma2, gamma3
            eos_base = EOSManager.EOSPiecewisePolytrope(name=eos_name,params_dict=spec_params)
        else:
            raise Exception("Unknown method for parametric EOS data file {} : {} ".format(eos_name,eos_param))
    except Exception as e:
        print("=====\n FAILSTATE 3: EOS CREATION FAILED. Exception:\n     ",type(e),":",e,"\n EXITING.\n=====")
        sys.exit(64) #special exit code for shell_wrapper_cip.sh to detect (hopefully)!
        #print(" WARNING: RETURNED EOS OBJECT WILL BE",type(eos_base),"!\n=====")
    
    return eos_base


################## Initialization #####################
sigma1d = 0.1
rv = None
n_dim = 2
pop_params = None
nm = 1
cfunc = 0
cv_params = None
eos = None
constraint_mmax_factor = 0.0

#Try to import lalsimutils (will fail on local machines)
try:
    import RIFT.lalsimutils as lalsimutils
    cfunc = 4#can call lalsimutils.convert_waveform_coordinates 
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
        #print("Given params:",kwargs)
    
    #----- Initialize unit conversion function -----
    global cfunc, cv_params
    rift = False
    if cfunc == 4: rift = True
    cvtest, cv_params = conversion_check(kwargs['cip_param_names'])
    
    if cvtest == 0:
        print("  FAILSTATE 1: could not find valid mass coordinate conversion. Exiting.")
        #return #this will break CIP, most likely
        sys.exit(0)
    else:
        if (rift) or cvtest != 3: #lalsimultils imported or not needed
            cfunc = cvtest #cvtest can be 1, 2, or 3
            print("Data will be converted via method",cvtest)
        else: #no lalsimutils (cfunc == 1) and it is needed (cvtest == 3)
            print("RIFT unavailable; data will not be converted.") #i.e., cfunc = 1 still

    #----- Initialize population & EOS data -----
    global pop_params
    global rv
    global n_dim
    eos_dat = []
    eos_names = []
    
    #Expected header names: # lnL sigma_lnL g0 g1 g2 g3 m1 m2 sig
    #Split columns into pop and EOS:
    #if cvtest != 0: #always true
    #assume population is a go
    pop_params = []
    pop_params_names = [] #yes this is literally just for the one print statement
    pop_params_lib = ['m1','m2','sig'] #can be added to for other populations
    for i in kwargs['param_names'][2:]: #should be anything past lnL, sig_lnL
        if i in pop_params_lib:
            pop_params_names.append(i)
            pop_params.append(all_params[kwargs['param_names'].index(i)])
        else: #anything that isn't m1, m2, sig
            eos_names.append(i)
            eos_dat.append(all_params[kwargs['param_names'].index(i)])
    
    print("Population parameters found:",pop_params_names,
          "\nEOS parameters found:",eos_names)
    
    if len(pop_params) < 3:
        print("  FAILSTATE 2: could not initialize population data: 3 or more columns required. Exiting.")
        #pop_params = None
        #n_dim = 0
        sys.exit(0)
    else:
        #NOTE: supports 2+1 or 2+2-type mass/sig columns. Not 3+1, etc.
        n_dim = (len(pop_params)%2)+int(len(pop_params)/2) #expect 1 sigma per mass or pair of masses
        
        #check 0 < sig < 0.5 (protection against puffing):
        if abs(pop_params[n_dim]) >= 0.5:
            pop_params[n_dim] = 0.49
        else:
            pop_params[n_dim] = abs(pop_params[n_dim])
        
        #cf. rv = multivariate_normal(mean=x0, cov = sigma1d*sigma1d*np.diag(np.ones(n_dim)))
        rv = multivariate_normal(mean=pop_params[:n_dim], cov=(pop_params[n_dim]**2)*np.diag(np.ones(n_dim))) #assumes only 2D - not great
    #else:
    #    print("ERROR: Population data could not be initialized: data headers not found.")
    
    print("pop_params:",pop_params)
    print("n_dim=",n_dim)
    
    #----- Initialize EOS object -----
    global eos
    global constraint_mmax_factor
    if len(eos_names) > 0 and (rift):
        eos = generate_eos(eos_dat, eos_names)
        constraint_mmax_factor = mmax_constraint(eos.mMaxMsun) 
    else:
        print("ERROR: Unable to create EOS object.") #Likely a no-CIP-test route only
        eos = None
    
    #----- Initialize normalization constant -----
    if pop_params is not None:
        #check population width: if narrow width or far from edges -> nm = 1 (normal)
        global nm
        if len(pop_params) == 3:
            if len(eos_names) != 0: #if EOS, must be BNS, else BBH; NSBH not handled
                mbounds = [1,3] #Neutron star mass range
            else:
                mbounds = [3,30] #Black hole mass range
            nm = boundary_integration_checks(pop_params,mbounds)#only supports [m1 m2 sig]
        else:
            print("WARNING: unable to check population boundaries. Assuming normalized population.")
    #else: nm is set to 1 by default (i.e., entire normal curve is within domain)
    print("Normalization constant set to",nm)
    
    print("----- END EXTERNAL PRIOR INITIALIZATION -----")


####################### EOS RETRIEVAL #######################

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
    
    print("Retrieving EOS Object from external initialization.")
    if eos is not None:
        return eos
    else:
        print("Hello! I've been trying to reach you about your car's extended warranty.")
        #print("Did you know that, because you own a 2004 Honda Prius, you are entitled to up to $16,000 of insurance for the next 5 years at your local Kia dealership?")
        #print("All you have to do is fill in your name, address, and registration (andgivemeallofyourmoney), and we can get you set up in just 10 minutes!")
        print("Unfortunately, we have no EOS for you today; sorry.")
        return None #CIP will ignore the EOS (hopefully)


####################### LIKELIHOOD EVAL #######################

def mmax_constraint(mmax_EOS):
    #Hardcoded to avoid file access (unavailable via CIP, at present)
    mass_NS = [2.14, 2.01, 1.908] #3 high-mass pulsars (Dietrich et al. 2020)
    mass_NS_sig = [0.1, 0.04, 0.016]
    
    partial_prod = 1.
    for i in np.arange(len(mass_NS)):
        partial_prod *= norm.cdf(mmax_EOS,loc=mass_NS[i],scale=mass_NS_sig[i])
    return partial_prod


def likelihood_evaluation(*X):
    #This looks arduous and slow for a function that will be called 1000 times in a for loop...
    #*X contains data list in same order as cip_params given to initialize_me()
    
    #NOTE: the names in cv_params are ALWAYS in standard order, but the indices may not be
    #Convert to m1,m2 coords from whatever CIP is passing:
    x_in = np.asarray([X[cv_params[0][1]],X[cv_params[1][1]]],dtype=np.float64).T
    if cfunc == 1:    
        m1m2 = x_in #no conversion
    elif cfunc == 2:
        m1m2=np.asarray(m1m2_local(x_in[:,cv_params[0][1]],x_in[:,cv_params[1][1]]),dtype=np.float64).T #local mc,eta conversion
    else:
        m1m2 = lalsimutils.convert_waveform_coordinates(x_in, low_level_coord_names=[cv_params[0][0],cv_params[1][0]],coord_names=['m1','m2'])
        #lalsimutils.convert_waveform_coordinates  lalcutout - for local testing
    #print(m1m2[:5])
    
    #Likelihood (w/ normalization constant):
    if nm == 0:
        return -np.inf
    else:
        return rv.logpdf(m1m2) - np.log(nm) + np.log(constraint_mmax_factor)


#USED FOR TESTING ONLY---
def lalcutout(x_in,coord_names=['mc', 'eta'],low_level_coord_names=['m1','m2'],enforce_kerr=False,source_redshift=0):
    print("lal received:",x_in,", length:",len(x_in))
    #print("cn:",coord_names," llcn:",low_level_coord_names)
    
    x_out = np.zeros( (len(x_in), len(coord_names) ) )
    
    coord_names_reduced = coord_names.copy() 
    for p in low_level_coord_names:
        if p in coord_names:
            indx_p_out = coord_names.index(p)
            indx_p_in = low_level_coord_names.index(p)
            coord_names_reduced.remove(p)
            x_out[:,indx_p_out] = x_in[:,indx_p_in]
    #print("coord_names_reduced:",coord_names_reduced)
            
    if 'mc' in low_level_coord_names and ('eta' in low_level_coord_names or 'delta_mc' in low_level_coord_names):
        indx_mc = low_level_coord_names.index('mc')
        eta_vals = np.zeros(len(x_in))
        
        #print("indx_mc:",indx_mc)
        if ('delta_mc' in low_level_coord_names):
                indx_delta = low_level_coord_names.index('delta_mc')
                #print("indx_delta:",indx_delta)
                #print("Shape of x_in:",x_in.shape)
                eta_vals = 0.25*(1- x_in[:,indx_delta]**2)
        
        if 'm1' in coord_names_reduced:
            m1_vals =np.zeros(len(x_in))  
            m2_vals =np.zeros(len(x_in)) 
            print("Passing to m1m2 via lal:")
            print(x_in[:,indx_mc],eta_vals)
            m1_vals,m2_vals = m1m2_local(x_in[:,indx_mc],eta_vals)
            indx_p_out = coord_names.index('m1')
            x_out[:,indx_p_out] = m1_vals
            coord_names_reduced.remove('m1')
            if 'm2' in coord_names_reduced:
                indx_p_out = coord_names.index('m2')
                x_out[:,indx_p_out] = m2_vals
                coord_names_reduced.remove('m2')
    
    if len(coord_names_reduced)<1:
        return x_out
    return x_out


if __name__ == '__main__':    
    #-----Adapted from util_ConstructIntrinsicPosterior_GenericCoordinates.py-----
    parser = argparse.ArgumentParser()
    
    # Supplemental likelihood factors: convenient way to effectively change the mass/spin prior in arbitrary ways for example
    # Note this supplemental factor is passed the *fitting* arguments, directly.  Use with extreme caution, since we often change the dimension in a DAG 
    parser.add_argument("--supplementary-likelihood-factor-code", default="ext_prior1",type=str,help="Import a module (in your pythonpath!) containing a supplementary factor for the likelihood.")
    parser.add_argument("--supplementary-likelihood-factor-function", default="likelihood_evaluation",type=str,help="With above option, specifies the specific function used as an external likelihood.")
    parser.add_argument("--using-eos", type=str, default="test_pop_m1_m2_eos.txt", help="Name of EOS.  Fit parameter list should physically use lambda1, lambda2 information (but need not).")
    parser.add_argument("--using-eos-for-prior", action='store_true', default=True, help="Alternate (hacky) implementation, which overrides using-eos and using-eos-index, to handle loading in a hyperprior")
    parser.add_argument("--using-eos-index", type=int, default=0, help="Index of EOS parameters in file.")    
    parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
    parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")

    opts = parser.parse_args()
    print(opts)
    
    opts.using_eos = "test_pop_eos_Parametrized-EoS_maxmass_EoS_samples.txt"
    
    #coordinates for CIP to use:
    coord_names = ['m1','m2']#opts.parameter # Used  in fit
    
    has_retrieve_eos = True 
    
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
                my_eos = supplemental_eos(**args_init) #run retrieve_eos('input_line'=dat_as_array, 'param_names'=param_names)           
      
    #Fake CIP integral:
    # Result shifted by lnL_shift
    #fn_passed = likelihood_function
    if supplemental_ln_likelihood:
        #fn_passed =  lambda *x: np.exp(supplemental_ln_likelihood(*x))
    #if opts.internal_use_lnL:
    #    fn_passed = log_likelihood_function   # helps regularize large values
    #    if supplemental_ln_likelihood:
    #        fn_passed =  lambda *x: log_likelihood_function(*x) + supplemental_ln_likelihood(*x)
        
        #Create npts pairs of random points in a grid space 
        x0=[1.9,1.1] #N.B. Drawing m1, m2 pairs!
        rvdat = multivariate_normal(mean=x0, cov=0.01*np.diag(np.ones(len(x0))))
        dat = rvdat.rvs(5)
        #dat_alt = dat.T #copy so dat is unaffected by the below
        #Force m1 > m2:
        m1 = np.maximum(dat[:,0], dat[:,1])
        m2 = np.minimum(dat[:,0], dat[:,1])
        #print(m1,m2)
        
        #Convert drawn masses to CIP coordinates as needed:
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
        print(obs)
        
        print("'Integrating' over obs.")
        likes = supplemental_ln_likelihood(obs[:,1],obs[:,2])
        print("length of likes:",len(likes))        


