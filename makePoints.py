# -*- coding: utf-8 -*-
"""
Hyperpipe modified. Part 1: point generator!

@author: marce
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--numpts',type=int, help="Number of test points to produce.")
parser.add_argument('--outdir', type=str, help="Output eos file directory.")
parser.add_argument('--fname-output-samples', type=str, help="Output eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
parser.add_argument("--conforming-output-name",action='store_true')

opts = parser.parse_args()

npts = opts.numpts

if opts.outdir is None:
    opts.outdir = "."

from pathlib import Path
Path(opts.outdir).mkdir(parents=True, exist_ok=True)
del Path

def make_Lambda(npts,mu=0,sig=1):
    #Create npts pairs of random points in a grid space
    with open('test_params.txt', 'w') as f:
        f.write("# lnL sigma_lnL x y z")#Update to correct RIFT format
        for n in range(npts):
            #next_pair = np.asarray(np.random.normal(loc=1.5, scale=0.5, size=2),dtype=np.float64)
            next_pair = np.random.normal(loc=mu, scale=sig, size=3)#want 3D Gaussian
            #popdata.append([next_pair[0],next_pair[1]])
            f.write("\n0 0 "+str(next_pair[0])+" "+str(next_pair[1])+" "+str(abs(next_pair[2])))

    print("Fake population (" + str(npts) + " points) data created.")


def make_better_Lambda(npts,mu1,mu2,sig):
    #Create npts pairs of random points in a grid space
    with open('test_params.txt', 'w') as f:
        f.write("# lnL sigma_lnL m1 m2 sig")#Update to correct RIFT format
        for n in range(npts):
            #next_pair = np.asarray(np.random.normal(loc=1.5, scale=0.5, size=2),dtype=np.float64)
            n1 = np.random.normal(loc=mu1, scale=sig, size=1)#want 3D Gaussian
            n2 = np.random.normal(loc=mu2, scale=sig, size=1)
            ns = abs(np.random.normal(loc=sig, scale=1, size=1))#must have sig > 0
            
            if n2 > n1: #ensure m1 > m2
                n3 = n1
                n1 = n2
                n2 = n3
            
            #popdata.append([next_pair[0],next_pair[1]])
            f.write("\n0 0 "+str(n1[0])+" "+str(n2[0])+" "+str(ns[0]))

    print("Fake population (" + str(npts) + " points) data created.")
    

def get_Lambda():
    #Get text for describing everything:
    rv = np.genfromtxt('test_params.txt',dtype='str')
    #print(rv)
    #print("1st element:",rv[0])
    
    #dat = []
    #for i in range(1,len(rv)):
        #newtext = rv[i].split(" ")
        #newdat = np.float64(newtext)#, dtype=np.float64)
       # newdat = []
        #for j in range(len(newtext)):
         #   newdat.append(np.float64(newtext[j]))
        #dat.append(newdat)
    
    print("Sample of txt file:")
    print(rv[0])
    print(rv[0][0])
    #print("Sample math:",2*rv[0][2])
    
    from scipy.stats import multivariate_normal
    rv2 = multivariate_normal([5,0,0], [[2,0,0], [0,2,0], [0,0,2]])
    
    dat= {}
    for i in range(len(rv)):
        dat[i] = rv2.pdf([rv[i,2:]])
        rv[i,0] = np.log(dat[i])
        rv[i,1] = 0.001
    
    #print("Final data:\n",rv)
    print("Length of dat:",len(dat))
    #print("Value of dat:\n",dat)
    #print("1st value of dat:",dat[0])
    dat2 = [dat[i] for i in range(len(dat))]
    #print("dat post-conversion:",dat2)
    #x = np.linspace(0,5,20)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.scatter(rv[:,2],rv[:,3],c=dat2)
    #ax.plot(rv[:,2], dat2)
    plt.show()


if __name__ == "__main__":
    #Fix the random generator's seed to produce consistent results (for testing):
    np.random.seed(42179)
    
    #make_Lambda(20)
    
    make_better_Lambda(20,20,10,1)
    
    #get_Lambda()
    

