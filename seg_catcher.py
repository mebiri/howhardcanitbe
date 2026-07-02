#! /usr/bin/env python
"""
make replacement file for EOS lines that segfault
"""
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--using-eos-index",type=int)
parser.add_argument("--using-eos",type=str)
parser.add_argument("--fname-output",type=str)
#parser.add_argument("--fname-output-samples",type=str)
parser.add_argument("--save-all-files",action='store_true')
parser.add_argument("--mode",type=str)

opts = parser.parse_args()

fname = opts.using_eos.replace('file:', '')
dat = np.genfromtxt(fname,names=True)[opts.using_eos_index] #get single line of grid
param_names = list(dat.dtype.names)
dat = dat.view((float, len(param_names)))
print(" size of dat:",len(dat),dat.shape)

if len(dat) != 11:
    print("--- ERROR: Incorrect data length:",len(dat)," ---")
    import sys
    sys.exit(0)
    
if opts.mode == "CIP":
    dat[0] = -1.5e6 #CIP EOS failure error code
elif opts.mode == "NICER":
    dat[0] = -4e6 #NICER EOS fail code
else:
    print("--- ERROR: INVALID MODE ---")
    import sys
    sys.exit(0)

dat[1] = 1000 #seg fault identifier

lineheader = ' '.join(map(str,param_names))+"\n" #to match CIP extracted header
#import save_CIP_output
#save_CIP_output.save_results(dat, lineheader, opts.fname_output, opts.fname_output, save_all=opts.save_all_files) 

params_here = dat[2:]
annotation_header = lineheader # this will/must be lnL sigma_lnL and then parameter names, which we want to preserve
with open(opts.fname_output+"+annotation.dat", 'w') as file_out:
    file_out.write("# " + annotation_header + "\n")
    file_out.write(" {} {} ".format(dat[0], dat[1]) + ' '.join(map(str,params_here)))
    #File (2/7): MARG-0-0+annotation.dat


