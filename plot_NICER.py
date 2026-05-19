#! /usr/bin/env python
"""
Created on Tue May 19 14:45:55 2026

@author: marce
"""

import matplotlib.pyplot as plt
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--obs-file', action='append', help="REQUIRED: Filenames (NOT PATHS) for observations used for likelihood calculation and plots generated here.")#Supported: j0740 j1731 j0030 j0437")

opts = parser.parse_args()


fig1 = plt.figure(figsize=(5,5),dpi=250) 
ax = fig1.add_subplot(111)

for i in opts.obs_file:
    dat = np.genfromtxt(i)
    print("len this file:",len(dat))
    R = dat[:1000,0]
    M = dat[:1000,1]

    ax.plot(R,M)
    
ax.set_xlim(left=1.0,right=2.0)
ax.set_ylim(bottom=1.0,top=2.0)
ax.set_xlabel("radius", size="11")
ax.set_ylabel("mass", size="11")
ax.tick_params(axis='both', which='major', labelsize=10) 
ax.grid(True)
fig1.tight_layout()
plt.show()

plt.savefig("nicer_data.png")
print("Saved.") 
