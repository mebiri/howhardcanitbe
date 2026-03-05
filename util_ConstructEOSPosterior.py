#!/usr/bin/env python
#
#  util_ConstructEOSPosterior.py
#     - takes in *generic-format* hyperparameter likelihood data
#     - uses *uniform* prior on hyperparameters.  [non-uniform priors can  be applied by the user with a supplementary function]
#     - generates posterior distribution by weighted Monte Carlo
#
# EXAMPLE:
#   python `which util_ConstructEOSPosterior.py` --fname fake_int_grid.dat  --parameter gamma1 --parameter gamma2 --lnL-offset 50

import RIFT.interpolators.BayesianLeastSquares as BayesianLeastSquares

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import scipy.stats
import functools
import itertools

import joblib  # http://scikit-learn.org/stable/modules/model_persistence.html

# GPU acceleration: NOT YET, just do usual
xpy_default=numpy  # just in case, to make replacement clear and to enable override
identity_convert = lambda x: x  # trivial return itself
cupy_success=False

no_plots = True
internal_dtype = np.float32  # only use 32 bit storage! Factor of 2 memory savings for GP code in high dimensions

 
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.lines as mlines
    import corner

    no_plots=False
except ImportError:
    print(" - no matplotlib - ")


from sklearn.preprocessing import PolynomialFeatures
if True:
#try:
    import RIFT.misc.ModifiedScikitFit as msf  # altenative polynomialFeatures
else:
#except:
    print(" - Faiiled ModifiedScikitFit : No polynomial fits - ")
from sklearn import linear_model

from igwn_ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)

import RIFT.integrators.mcsampler as mcsampler
try:
    import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
    mcsampler_gmm_ok = True
except:
    print(" No mcsamplerEnsemble ")
    mcsampler_gmm_ok = False
try:
    import RIFT.integrators.mcsamplerGPU as mcsamplerGPU
    mcsampler_gpu_ok = True
    mcsamplerGPU.xpy_default =xpy_default  # force consistent, in case GPU present
    mcsamplerGPU.identity_convert = identity_convert
except:
    print( " No mcsamplerGPU ")
    mcsampler_gpu_ok = False
try:
    import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAdaptiveVolume
    mcsampler_AV_ok = True
except:
    print(" No mcsamplerAV ")
    mcsampler_AV_ok = False
try:
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPortfolio
    mcsampler_Portfolio_ok = True
except:
    print(" No mcsamplerPortolfio ")





def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = numpy.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b


parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="filename of *.dat file (EOS-format: lnL sigma_lnL p1 p2 ... .  ASSUME any stacking over events already performed.")
parser.add_argument("--fname-output-samples",default="output-EOS-samples",help="output grid")
parser.add_argument("--fname-output-integral",default="output-EOS-samples",help="for evidencees and pipeline compatibility")
parser.add_argument("--n-output-samples",default=2000,type=int,help="output posterior samples (default 3000)")
parser.add_argument("--eos-param", type=str, default=None, help="parameterization of equation of state [spectral only, for now]")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior. Currently can only specify gamma1,gamma2, ..., and these MUST be columns in --fname. IF NOT PROVIDED, DEFAULTS TO LIST IN FILE.  ")
parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo. For EOS objects, only possible for physical quantities like R1.4, etc. NOT YET PROVIDED")
#parser.add_argument("--no-adapt-parameter",action='append',help="Disable adaptive sampling in a parameter. Useful in cases where a parameter is not well-constrained, and the a prior sampler is well-chosen.")
parser.add_argument("--parameter-nofit", action='append', help="Parameter used to initialize the implied parameters, and varied at a low level, but NOT the fitting parameters.")
parser.add_argument("--integration-parameter-range",action='append', help="Integration parameter ranges. Syntax is name:[a,b]")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--no-downselect",action='store_true')
parser.add_argument("--aligned-prior", default="uniform",help="Options are 'uniform', 'volumetric', and 'alignedspin-zprior'")
parser.add_argument("--cap-points",default=-1,type=int,help="Maximum number of points in the sample, if positive. Useful to cap the number of points ued for GP. See also lnLoffset. Note points are selected AT RANDOM")
parser.add_argument("--lambda-max", default=4000,type=float,help="Maximum range of 'Lambda' allowed.  Minimum value is ZERO, not negative.")
parser.add_argument("--lnL-shift-prevent-overflow",default=None,type=float,help="Define this quantity to be a large positive number to avoid overflows. Note that we do *not* define this dynamically based on sample values, to insure reproducibility and comparable integral results. BEWARE: If you shift the result to be below zero, because the GP relaxes to 0, you will get crazy answers.")
parser.add_argument("--lnL-offset",type=float,default=np.inf,help="lnL offset")
parser.add_argument("--lnL-cut",type=float,default=None,help="lnL cut [MANUAL]")
parser.add_argument("--sigma-cut",type=float,default=0.6,help="Eliminate points with large error from the fit.")
parser.add_argument("--ignore-errors-in-data",action='store_true',help='Ignore reported error in lnL. Helpful for testing purposes (i.e., if the error is zero)')
parser.add_argument("--lnL-peak-insane-cut",type=float,default=np.inf,help="Throw away lnL greater than this value. Should not be necessary")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
parser.add_argument("--n-max",default=3e5,type=float)
parser.add_argument("--n-step",default=1e5,type=int)
parser.add_argument("--n-eff",default=3e3,type=int)
parser.add_argument("--pool-size",default=3,type=int,help="Integer. Number of GPs to use (result is averaged)")
parser.add_argument("--fit-method",default="rf",help="rf (default) : rf|gp|quadratic|polynomial|gp_hyper|gp_lazy|cov|kde.  Note 'polynomial' with --fit-order 0  will fit a constant")
parser.add_argument("--fit-load-gp",default=None,type=str,help="Filename of GP fit to load. Overrides fitting process, but user MUST correctly specify coordinate system to interpret the fit with.  Does not override loading and converting the data.")
parser.add_argument("--fit-save-gp",default=None,type=str,help="Filename of GP fit to save. ")
parser.add_argument("--fit-order",type=int,default=2,help="Fit order (polynomial case: degree)")
parser.add_argument("--no-plots",action='store_true')
parser.add_argument("--using-eos-type", type=str, default=None, help="Name of EOS parameterization (must match what is used for inputs). Will use EOS parameterization to identify appropriate field headers")
parser.add_argument("--sampler-method",default="adaptive_cartesian",help="adaptive_cartesian|GMM|adaptive_cartesian_gpu")
parser.add_argument("--sampler-portfolio",default=None,action='append',type=str,help="comma-separated strings, matching sampler methods other than portfolio")
parser.add_argument("--sampler-portfolio-args",default=None, action='append', type=str, help='eval-able dictionary to be passed to that sampler_')
parser.add_argument("--internal-use-lnL",action='store_true',help="integrator internally manipulates lnL..   ")
parser.add_argument("--internal-correlate-parameters",default=None,type=str,help="comman-separated string indicating parameters that should be sampled allowing for correlations. Must be sampling parameters. Only implemented for gmm.  If string is 'all', correlate *all* parameters")
parser.add_argument("--internal-n-comp",default=1,type=int,help="number of components to use for GMM sampling. Default is 1, because we expect a unimodal posterior in well-adapted coordinates.  If you have crappy coordinates, use more")
parser.add_argument("--force-no-adapt",action='store_true',help="Disable adaptation, both of the tempering exponent *and* the individual sampling prior(s)")
parser.add_argument("--tripwire-fraction",default=0.05,type=float,help="Fraction of nmax of iterations after which n_eff needs to be greater than 1+epsilon for a small number epsilon")

# Supplemental likelihood factors: convenient way to effectively change the mass/spin prior in arbitrary ways for example
# Note this supplemental factor is passed the *fitting* arguments, directly.  Use with extreme caution, since we often change the dimension in a DAG 
parser.add_argument("--supplementary-likelihood-factor-code", default=None,type=str,help="Import a module (in your pythonpath!) containing a supplementary factor for the likelihood.  Used to impose supplementary external priors of arbitrary complexity and external dependence (e.g., imposing alternate EOS priors)")
parser.add_argument("--supplementary-likelihood-factor-function", default=None,type=str,help="With above option, specifies the specific function used as an external likelihood. EXPERTS ONLY")
parser.add_argument("--supplementary-likelihood-factor-ini", default=None,type=str,help="With above option, specifies an ini file that is parsed (here) and passed to the preparation code, called when the module is first loaded, to configure the module. EXPERTS ONLY")
opts=  parser.parse_args()

#print(" WARNING: Always use internal_use_lnL for now ")
#opts.internal_use_lnL=True

no_plots = no_plots |  opts.no_plots
lnL_shift = 0
lnL_default_large_negative = -500
if opts.lnL_shift_prevent_overflow:
    lnL_shift  = opts.lnL_shift_prevent_overflow



### Comparison data (from LI)
###

downselect_dict = {}
dlist = []
dlist_ranges=[]
if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = map(eval,opts.downselect_parameter_range)
else:
    dlist = []
    dlist_ranges = []
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]

if opts.no_downselect:
    downselect_dict={}


test_converged={}

###
### Retrieve data
###
#  int_sig sigma/L gamma1 gamma2 ...
col_lnL = 0
dat_orig = dat = np.loadtxt(opts.fname)
dat_orig = dat[dat[:,col_lnL].argsort()] # sort  http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
print(" Original data size = ", len(dat), dat.shape)
dat_orig_names = None
with open(opts.fname,'r') as f:
    header_str = f.readline()
    header_str = header_str.rstrip()
dat_orig_names = header_str.replace('#','').split()[2:]

###
### Parameters in use
###

coord_names = opts.parameter # Used  in fit
if coord_names is None:
    coord_names = dat_orig_names
low_level_coord_names = coord_names # Used for Monte Carlo
if opts.parameter_implied:
    coord_names = coord_names+opts.parameter_implied
if opts.parameter_nofit:
    if opts.parameter is None:
        low_level_coord_names = opts.parameter_nofit # Used for Monte Carlo
    else:
        low_level_coord_names = opts.parameter+opts.parameter_nofit # Used for Monte Carlo
error_factor = len(coord_names)
name_index_dict ={}
for name in dat_orig_names:
    try:
        name_index_dict[name] = 2+dat_orig_names.index(name)
    except:
        raise Exception(" Currently fitting parameter names must match columns in data file ")
# TeX dictionary
print(" Coordinate names for fit :, ", coord_names, " from ", dat_orig_names, " indexed as ", name_index_dict)
print(" Coordinate names for Monte Carlo :, ", low_level_coord_names)


###
### Integration ranges
###

param_ranges = {}
for range_code  in opts.integration_parameter_range:
    name, range_str  = range_code.split(':')
    range_expr =     eval(range_str)  # define. Better to split on , for example
    param_ranges[name]  = np.array(range_expr)

# Add in integration range for everything else, if nothing specified
for name in dat_orig_names:
    if not name in param_ranges:
        vals = dat_orig[:,name_index_dict[name]]
        param_ranges[name] = [np.min(vals), np.max(vals)]

###
### Prior functions : default is UNIFORM, since it is unmodeled and generic
###

def uniform_prior(x):
    return np.ones(x.shape)

prior_map = {}
for name in low_level_coord_names:
    prior_map[name] = uniform_prior
    if not(name in param_ranges):
        raise Exception(" {} not provided a parameter range ".format(name))  # change later, should fall back to using prior range from above


prior_range_map = param_ranges

# prior_map  = { 'gamma1':eos_param_uniform_prior, 'gamma2':eos_param_uniform_prior,
# }
# # Les: somewhat more aggressive: 
# #    gamma1: 0.2,2
# #    gamma2: -1.67, 1.7
# prior_range_map = { 'gamma1':  [0.707899,1.31], 'gamma2':[-1.6,1.7], 'gamma3':[-0.6,0.6], 'gamma4':[-0.02,0.02]
# }

#supplemental code deleted - not used currently


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

def adderr(y):#unused, seemingly
    val,err = y
    return val+error_factor*err


def fit_rf(x,y,y_errors=None,fname_export='nn_fit'):
#    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    # Instantiate model. Usually not that many structures to find, don't overcomplicate
    #   - should scale like number of samples
    rf = ExtraTreesRegressor(n_estimators=100, verbose=True,n_jobs=-1) # no more than 5% of samples in a leaf
    if y_errors is None:
        rf.fit(x,y)
    else:
        rf.fit(x,y,sample_weight=1./y_errors**2)

    ### reject points with infinities : problems for inputs
    def fn_return(x_in,rf=rf):
        f_out = -lnL_default_large_negative*np.ones(len(x_in))
        # remove infinity or Nan
        indx_ok = np.all(np.isfinite(np.array(x_in,dtype=float)),axis=-1)
        # rf internally uses float32, so we need to remove points > 10^37 or so ! 
        #    ... this *should* never happen due to bounds constraints, but ...
        indx_ok_size = np.all( np.logical_not(np.greater(np.abs(x_in),1e37)), axis=-1)
        indx_ok = np.logical_and(indx_ok, indx_ok_size)
        f_out[indx_ok] = rf.predict(x_in[indx_ok])
        return f_out
#    fn_return = lambda x_in: rf.predict(x_in) 

    print( " Demonstrating RF")   # debugging
    residuals = rf.predict(x)-y
    print( "    std ", np.std(residuals), np.max(y), np.max(fn_return(x)))
    return fn_return





# initialize
dat_mass  = [] 
weights = []
n_params = -1


 ###
 ### Convert data.   RIGHT NOW JUST DOWNSELECTING, no intermediate fitting parameters defined
 ###

dat_out = []
for line in dat:
  dat_here= np.zeros(len(coord_names)+2)
  if line[col_lnL+1] > opts.sigma_cut:
      print("skipping", line)
      continue
  dat_here[:-2] = line[2:len(coord_names)+2]  # modify to use names!
  dat_here[-2] = line[0]
  dat_here[-1] = line[1]
  dat_out.append(dat_here)
dat_out= np.array(dat_out)
# Repack data
X =dat_out[:,0:len(coord_names)]
Y = dat_out[:,-2]
if np.max(Y)<0 and lnL_shift ==0: 
    lnL_shift  = -100 - np.max(Y)   # force it to be offset/positive -- may help some configurations. Remember our adaptivity is silly.
Y_err = dat_out[:,-1]
# Save copies for later (plots)
X_orig = X.copy()
Y_orig = Y.copy()



# Eliminate values with Y too small
max_lnL = np.max(Y)
if np.isinf(opts.lnL_offset):
    indx_ok= np.ones(len(Y),dtype=bool)  # default case, we preserve all the data
else:
    indx_ok = np.array(Y>np.max(Y)-opts.lnL_offset,dtype=bool)  # force cast : sometimes indx_ok is a mappable object?
n_ok = np.sum(indx_ok)
# Provide some random points, to insure reasonable tapering behavior away from the sample
print(" Points used in fit : ", n_ok, " out of ", len(indx_ok), " given max lnL ", max_lnL)
if max_lnL < 10 and np.mean(Y) > -10: # second condition to allow synthetic tests not to fail, as these often have maxlnL not large
    print(" Resetting to use ALL input data -- beware ! ")
    # nothing matters, we will reject it anyways
    indx_ok = np.ones(len(Y),dtype=bool)
elif n_ok < 10: # and max_lnL > 30:
    # mark the top 10 elements and use them for fits
    # this may be VERY VERY DANGEROUS if the peak is high and poorly sampled
    idx_sorted_index = np.lexsort((np.arange(len(Y)), Y))  # Sort the array of Y, recovering index values
    indx_list = np.array( [[k, Y[k]] for k in idx_sorted_index])     # pair up with the weights again
    indx_list = indx_list[::-1]  # reverse, so most significant are first
    indx_ok = list(map(int,indx_list[:10,0]))
    print(" Revised number of points for fit: ", np.sum(indx_ok), len(indx_ok), indx_list[:10])
X_raw = X.copy()

my_fit= None
if opts.fit_method == 'rf':
    print( " FIT METHOD ", opts.fit_method, " IS RF ")
    # NO data truncation for NN needed?  To be *consistent*, have the code function the same way as the others
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
    if opts.ignore_errors_in_data:
        Y_err=None
    my_fit = fit_rf(X,Y,y_errors=Y_err)


# Sort for later convenience (scatterplots, etc)
indx = Y.argsort()#[::-1]
X=X[indx]
Y=Y[indx]



###
### Integrate posterior
###


sampler = mcsampler.MCSampler()

#if opts.sampler_method == "GMM":
    #sampler = mcsamplerEnsemble.MCSampler()
if opts.sampler_method == "AV":
    print("Sampler method is AV")
    sampler = mcsamplerAdaptiveVolume.MCSampler()
    opts.internal_use_lnL= True  # required!



##
## Loop over param names
##
for p in coord_names:
    prior_here = prior_map[p]
    range_here = prior_range_map[p]
    
    print("Paramter",p,"added to sampler.")
    sampler.add_parameter(p, pdf=np.vectorize(lambda x:1), prior_pdf=prior_here,left_limit=range_here[0],right_limit=range_here[1],adaptive_sampling=True)

likelihood_function = None
log_likelihood_function = None
def convert_coords(x):
    return x
def log_likelihood_function(*args):
    return my_fit(convert_coords(np.array([*args]).T ))

if len(coord_names) ==1:
    def likelihood_function(x):  
        if isinstance(x,float):
            return np.exp(my_fit([x]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x])))
if len(coord_names) ==2:
    def likelihood_function(x,y):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y])))
if len(coord_names) ==3:
    def likelihood_function(x,y,z):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z])))
if len(coord_names) ==4:
    def likelihood_function(x,y,z,a):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z,a],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a])))
if len(coord_names) ==5:
    def likelihood_function(x,y,z,a,b):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b])))
if len(coord_names) ==6:
    def likelihood_function(x,y,z,a,b,c):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b,c],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c])))
if len(coord_names) ==7:
    def likelihood_function(x,y,z,a,b,c,d):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d])))
if len(coord_names) ==8:
    def likelihood_function(x,y,z,a,b,c,d,e):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e])))
if len(coord_names) ==9:
    def likelihood_function(x,y,z,a,b,c,d,e,f):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e,f]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e,f])))
if len(coord_names) ==10:
    def likelihood_function(x,y,z,a,b,c,d,e,f,g):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e,f,g]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e,f,g])))




n_step = opts.n_step
my_exp = np.min([1,0.8*np.log(n_step)/np.max(Y)])   # target value : scale to slightly sublinear to (n_step)^(0.8) for Ymax = 200. This means we have ~ n_step points, with peak value wt~ n_step^(0.8)/n_step ~ 1/n_step^(0.2), limiting contrast
if np.max(Y_orig) < 0:   # for now, don't use a weight exponent if we are negative: can't use guess based from GW experience
    my_exp = 1
#my_exp = np.max([my_exp,  1/np.log(n_step)]) # do not allow extreme contrast in adaptivity, to the point that one iteration will dominate
print(" Weight exponent ", my_exp, " and peak contrast (exp)*lnL = ", my_exp*np.max(Y), "; exp(ditto) =  ", np.exp(my_exp*np.max(Y)), " which should ideally be no larger than of order the number of trials in each epoch, to insure reweighting doesn't select a single preferred bin too strongly.  Note also the floor exponent also constrains the peak, de-facto")


extra_args={}

extra_args.update({
    "n_adapt": 100, # Number of chunks to allow adaption over
    "history_mult": 10, # Multiplier on 'n' - number of samples to estimate marginalized 1D histograms with, 
    "force_no_adapt":opts.force_no_adapt,
    "tripwire_fraction":opts.tripwire_fraction
})

fn_passed = likelihood_function
#if supplemental_ln_likelihood:
#    fn_passed =  lambda *x: likelihood_function(*x)*np.exp(supplemental_ln_likelihood(*x))
if opts.internal_use_lnL:
    fn_passed = log_likelihood_function   # helps regularize large values
    #if supplemental_ln_likelihood:
    #    fn_passed =  lambda *x: log_likelihood_function(*x) + supplemental_ln_likelihood(*x)
    extra_args.update({"use_lnL":True,"return_lnI":True})


print("Integral here.")
res, var, neff, dict_return = sampler.integrate(fn_passed, *coord_names,  verbose=True,nmax=int(opts.n_max),n=n_step,neff=opts.n_eff, save_intg=True,tempering_adapt=True, floor_level=1e-3,igrand_threshold_p=1e-3,convergence_tests=test_converged,adapt_weight_exponent=my_exp,no_protect_names=True,**extra_args)  # weight ecponent needs better choice. We are using arbitrary-name functions


# Save result -- needed for odds ratios, etc.
np.savetxt("integral_result.dat", [np.log(res)])

if neff < len(coord_names):
    print(" PLOTS WILL FAIL ")
    print(" Not enough independent Monte Carlo points to generate useful contours")


samples = sampler._rvs
print(samples.keys())
n_params = len(coord_names)
dat_mass = np.zeros((len(samples[coord_names[0]]),n_params+3))
if not(opts.internal_use_lnL):
    dat_logL = np.log(samples["integrand"])
else:
    if 'log_integrand' in samples:
        dat_logL = samples['log_integrand']
    else:
        dat_logL = samples["integrand"]
lnLmax = np.max(dat_logL[np.isfinite(dat_logL)])
print(" Max lnL ", np.max(dat_logL))

n_ESS = -1
if True:
    # Compute n_ESS.  Should be done by integrator!
    if 'log_joint_s_prior' in  samples:
        weights_scaled = np.exp(dat_logL - lnLmax + samples["log_joint_prior"] - samples["log_joint_s_prior"])
        # dictionary, write this to enable later use of it
        samples["joint_s_prior"] = np.exp(samples["log_joint_s_prior"])
        samples["joint_prior"] = np.exp(samples["log_joint_prior"])
    else:
        weights_scaled = np.exp(dat_logL - lnLmax)*sampler._rvs["joint_prior"]/sampler._rvs["joint_s_prior"]
    weights_scaled = weights_scaled/np.max(weights_scaled)  # try to reduce dynamic range
    n_ESS = np.sum(weights_scaled)**2/np.sum(weights_scaled**2)
    print(" n_eff n_ESS ", neff, n_ESS)


# Throw away stupid points that don't impact the posterior
indx_ok = np.ones(len(dat_logL),dtype=bool)
if not('log_joint_s_prior' in samples):
    indx_ok=samples["joint_s_prior"]>0
indx_ok = np.logical_and(dat_logL > np.max(dat_logL)-opts.lnL_offset ,indx_ok)
for p in coord_names:
    samples[p] = samples[p][indx_ok]
dat_logL  = dat_logL[indx_ok]
print(samples.keys())
samples["joint_prior"] =samples["joint_prior"][indx_ok]
samples["joint_s_prior"] =samples["joint_s_prior"][indx_ok]



###
### 1d posteriors of the coordinates used for sampling  [EQUALLY WEIGHTED, BIASED because physics cuts aren't applied]
###

p = samples["joint_prior"]
ps =samples["joint_s_prior"]
lnL = dat_logL
lnLmax = np.max(lnL)
weights = np.exp(lnL-lnLmax)*p/ps



print(" ---- Subset for posterior samples (and further corner work) --- ")


p_norm = (weights/np.sum(weights))
indx_list = np.random.choice(np.arange(len(weights)), p=p_norm.astype(np.float64),size=opts.n_output_samples)


dat_out = np.zeros( (opts.n_output_samples,2+len(dat_orig_names)) )

if len(coord_names) < len(dat_orig_names): # not needed if all params are in fit

    if len(dat) < opts.n_output_samples:
        print(" NOTE: original data shorter than  requested output; adding",opts.n_output_samples-len(dat),"duplicate fill lines from original data.")
        newlines = None
        if opts.n_output_samples > 2*len(dat):
            newlines = dat[:]
            newlen = len(newlines)
            while newlen < opts.n_output_samples:
                newerlines = dat[:opts.n_output_samples-newlen] #will only get up to len(dat) lines
                newlines = np.concatenate((newlines,newerlines), axis=0)
                newlen = len(newlines)
        else:
            newlines = dat[:opts.n_output_samples-len(dat)] #duplicate lines to fill
        dat = np.concatenate((dat,newlines), axis=0) #should be fine since dat isn't used after this

    for c in np.arange(len(dat_orig_names)):
        if dat_orig_names[c] not in coord_names:
            print("  Not in coord_names:",dat_orig_names[c],"; adding to output as constant.")
            outidx = name_index_dict[dat_orig_names[c]]   # write in correct place
            if len(dat) > opts.n_output_samples:
                dat_out[:,outidx] = dat[:opts.n_output_samples,outidx] #truncate original data to fit (not ideal)
            else: #len(dat) <= n_output_samples (if dat was <, should now be =)
                dat_out[:,outidx] = dat[:,outidx]
                
for indx in np.arange(len(coord_names)):
    vals = samples[coord_names[indx]][indx_list]   # load in data for this column
    outindx = name_index_dict[ coord_names[indx]]   # write in correct place
    dat_out[:,outindx] = vals

#If one of the masses carried as const, re-sort to enforce m1 > m2
if ("m1" not in coord_names) or ("m2" not in coord_names):
    print(" NOTE: re-sorting masses so m1 > m2 (precaution)")
    m1dx = name_index_dict["m1"]
    print("Minimum m1 (pre-sort):",min(dat_out[:,m1dx]))
    print("Minimum m2 (pre-sort):",min(dat_out[:,m1dx+1]))
    m1 = np.maximum(dat_out[:,m1dx], dat_out[:,m1dx+1]) #N.B.: assumes m2 col index after m1 col
    m2 = np.minimum(dat_out[:,m1dx], dat_out[:,m1dx+1])
    dat_out[:,m1dx] = m1
    dat_out[:,m1dx+1] = m2
    print("Minimum m1 (post-sort):",min(dat_out[:,m1dx]))
    print("Minimum m2 (post-sort):",min(dat_out[:,m1dx+1]))

print(" Saving to ", opts.fname_output_samples+".dat")
np.savetxt(opts.fname_output_samples+".dat",dat_out,header=" lnL sigma_lnL " + ' '.join(dat_orig_names))
