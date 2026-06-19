#! /usr/bin/env python
"""
cry me a bucket
"""

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--posterior",type=str)
parser.add_argument("--param",action='append',type=str)

opts = parser.parse_args()

try:
    dat = np.genfromtxt(opts.posterior,names=True)
except:
    print("Could not open provided file. Exiting")
    import sys
    sys.exit(0)
    
param_names = dat.dtype.names #separate out the names from the data
dat_as_array = dat.view((float, len(param_names)))

for p in opts.param:
    try:
        indx = param_names.index(p)
    except:
        print("Could not find parameter "+p+"in file!")
        continue
    
    dat_here = dat[:,indx]
    
    mn = np.mean(dat_here)
    sig = np.std(dat_here)
    print("Results for parameter: ",p)
    print("mean =",mn)
    print("sig =",sig)


