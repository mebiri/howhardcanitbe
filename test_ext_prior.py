#! /usr/bin/env python
import argparse
import os

# note we are using mc, q coordinates for CIP, since that is more sane and lets us specify mc ranges
my_dim_list = ['mc','delta_mc', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
my_dim_ranges = [[8,18], [0.1,1], [-1,1], [-1,1], [-1,1], [-1,1], [-1,1], [-1,1]]
indx_offset = 1 # 1 for m1,m2 ; 3 if I start with s1x

parser = argparse.ArgumentParser()
parser.add_argument("--test-name",default='ciptest',type=str)
parser.add_argument("--n-dim",default=1,type=int)
parser.add_argument("--n-max",default=3e5,type=int)
parser.add_argument("--n-output-samples",default=15000,type=int)
#parser.add_argument("--lnL-offset",default=None,type=float) # unused in this code
parser.add_argument("--fit-method",default='rf',type=str)
#parser.add_argument("--pool-size",default=10,type=int)
parser.add_argument("--composite-file",default="all.net",type=str)
parser.add_argument("--external-prior-code",default="ext_prior1",type=str)
parser.add_argument("--external-likelihood-func",default="likelihood_evaluation",type=str)
parser.add_argument("--eos-pop-file",default="",type=str)
opts = parser.parse_args()

n_dim = opts.n_dim

os.chdir(opts.test_name)
os.environ['DIR_PARAMS'] =os.getcwd()
mc_str = str(my_dim_ranges[0]).replace(' ','')
cmd="util_ConstructIntrinsicPosterior_GenericCoordinates.py  --sampler-method AV --mc-range " + mc_str + " --eta-range [0.2,0.24999]  --no-downselect  --verbose  --use-precessing  --ignore-errors-in-data --no-plots --fname " + opts.composite_file + "  --use-precessing --parameter " +  (' --parameter '.join(my_dim_list[:n_dim])) + "  --fit-method rf  --supplementary-likelihood-factor-code '" + opts.external_prior_code + "' --supplementary-likelihood-factor-function '" + opts.external_likelihood_func + "'  --using-eos '" + opts.eos_pop_file + "'  --using-eos-index  --using-eos-for-prior  --n-max " + str(opts.n_max)  + " --n-output-samples " + str(opts.n_output_samples) # + ' > cip.log '
#  --no-downselect  # danger using --no-downselect in general, unless fit stabilized by random points
print(cmd)
os.system(cmd)