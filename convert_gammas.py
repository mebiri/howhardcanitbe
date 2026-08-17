#! /usr/bin/env python
"""
convert gamma points to shifted, rotated coords
Sum_k g_k x^k -> Sum_g c_g (x - x0)^g
where
c0 = g0 + g1(x0) + g2(x0)^2 + g3(x0)^3
c1 = g1 + 2g2(x0) + 3g3(x0)^2
c2 = 2g2 + 6g3(x0)
c3 = 6g3
"""
### -*- coding: utf-8 -*-

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--x0",type=float,default=1,help="Pressure value to shift by (x = p/p0 so x0 scales p/p0)")
parser.add_argument("--posterior-file",type=str,help="posterior grid file to convert")

opts = parser.parse_args()

#opts.posterior_file = "pop_eos_Parametrized-EoS_maxmass_EoS_samples_grid.txt"
#opts.x0 = 1#2.7e14

eos = None
param_names = None
try:
    eos_dat = np.genfromtxt(opts.posterior_file,names=True)
    coord_names = list(eos_dat.dtype.names)
    eos = eos_dat.view((float, len(coord_names)))
except:
    print("ERROR: could not open provided file. Exiting.")
    import sys
    sys.exit(0)
    
#find g cols
gindx = []
gammas = ["gamma0","gamma1","gamma2","gamma3"]
for g in gammas:
    gindx.append(coord_names.index(g))
    
if len(gindx) < 4:
    print("ERROR: not all gamma cols found! Exiting.")
    import sys
    sys.exit(0)

#convert, save to different array for output    
c_dat = np.zeros((len(eos),4))
c_dat[:,0] = eos[:,gindx[0]] + eos[:,gindx[1]]*opts.x0 + eos[:,gindx[2]]*(opts.x0**2) + eos[:,gindx[3]]*(opts.x0**3)
c_dat[:,1] = eos[:,gindx[1]] + 2.0*eos[:,gindx[2]]*opts.x0 + 3.0*eos[:,gindx[3]]*(opts.x0**2)
c_dat[:,2] = 2.0*eos[:,gindx[2]] + 6.0*eos[:,gindx[3]]*opts.x0
c_dat[:,3] = 6.0*eos[:,gindx[3]]

#fill new values
for i,g in enumerate(gindx):
    eos[:,g] = c_dat[:,i]

#write out
outx = str(opts.x0).replace(".","p")
outname = "grid_con-"+outx+".dat" #+ opts.posterior_file.split("/")[-1].split["."][0]
headers = "lnL " + " ".join(coord_names[1:])
np.savetxt(outname, eos,header=headers)
print("Done.")


#plot for testing
# =============================================================================
# import matplotlib.pyplot as plt
# fig1 = plt.figure(figsize=(8,5),dpi=250) 
# ax = fig1.add_subplot(111)
# ax.scatter(eos[:,gindx[0]],eos[:,gindx[1]],marker=".")
# ax.scatter(c_dat[:,0],c_dat[:,1],marker=".")
# ax.set_xlabel("$\mu_1$", size="11")
# ax.set_ylabel("$\mu_2$", size="11")
# ax.tick_params(axis='both', which='major', labelsize=10) 
# fig1.tight_layout()
# plt.show(block=False)
# =============================================================================


