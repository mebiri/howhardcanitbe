# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 13:49:43 2025

@author: marce
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
    
parser.add_argument("--post",type=str,default=None)
parser.add_argument("--param",type=str,default=None)

opts = parser.parse_args()
print(opts)

post_dat = np.genfromtxt(opts.post,names=True)
param_names = post_dat.dtype.names #separate out the names from the data
pdat_as_array = post_dat.view((float, len(param_names)))

pdx = param_names.index(opts.param)

pvals = pdat_as_array[:,pdx]
lnL = pdat_as_array[:,0]

#Scatterplot:
fig1 = plt.figure(figsize=(8,5),dpi=250) 
ax = fig1.add_subplot(111)
ax.scatter(pvals,lnL,marker=".")
ax.set_xlabel("$\mu_1$", size="11")
ax.set_ylabel("$lnL$", size="11")
ax.tick_params(axis='both', which='major', labelsize=10) 
fig1.tight_layout()
plt.show()
#plt.savefig(plotname+".png")
#print("Plot saved as "+plotname+".png")



