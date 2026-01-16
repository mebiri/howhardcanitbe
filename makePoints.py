# -*- coding: utf-8 -*-
"""
Hyperpipe modified. Part 1: point generator!

@author: marce
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#parser.add_argument('--mode',type=int,default=0,help="REQUIRED. 0 is defaults; 1 is custom.")
parser.add_argument('--npts',type=int,default=3000,help="Number of test points to produce.")
parser.add_argument('--static-eos-line',type=int,default=0,help="Line of EOS file to use for static model.")
parser.add_argument('--mass-mean',type=float,default=1.4,help="mean to draw pop masses from.")
parser.add_argument('--mass-sig',type=float,default=0.1,help="width of pop for drawing masses")
parser.add_argument('--eos-file',type=str,default="Parametrized-EoS_maxmass_EoS_samples.txt")

opts = parser.parse_args()
# =============================================================================
# parser = argparse.ArgumentParser()
# parser.add_argument('--numpts',type=int, help="Number of test points to produce.")
# parser.add_argument('--outdir', type=str, help="Output eos file directory.")
# parser.add_argument('--fname-output-samples', type=str, help="Output eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
# parser.add_argument("--conforming-output-name",action='store_true')
# 
# opts = parser.parse_args()
# 
# npts = opts.numpts
# 
# if opts.outdir is None:
#     opts.outdir = "."
# 
# from pathlib import Path
# Path(opts.outdir).mkdir(parents=True, exist_ok=True)
# del Path
# =============================================================================

def mchirp(m1, m2):
    """Compute chirp mass from component masses"""
    return (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)

def symRatio(m1, m2):#this is eta
    """Compute symmetric mass ratio from component masses"""
    return m1*m2/(m1+m2)/(m1+m2)


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
    

def make_different_Lambda(npts,means,sig,units):
    #Create npts pairs of random points in a grid space    
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mean=means, cov=0.01*np.diag(np.ones(len(means))))
    dat = rv.rvs(npts).T
    dat_alt = dat.T #copy so dat is unaffected by the below
    #Force m1 > m2:
    m1 = np.maximum(dat_alt[:,0], dat_alt[:,1])
    m2 = np.minimum(dat_alt[:,0], dat_alt[:,1])
    #print(m1,m2)
    
    if 'mc' in units: 
        #Convert to mc:
        mcV = mchirp(m1,m2)
        
        #Convert to other:
        otherV = []
        if 'delta_mc' in units:
            #Convert to delta_mc
            otherV =  (m1 - m2)/(m1+m2)
        elif 'eta' in units:
            #Convert to eta:
            otherV = symRatio(m1,m2)
    
        print("Shape check:",dat.shape, mcV.shape)
        dat_alt[:,0] = mcV
        dat_alt[:,1] = otherV
    else:
        print("Shape check:",dat.shape, m1.shape)
        dat_alt[:,0] = m1
        dat_alt[:,1] = m2
    
    ns = abs(np.random.normal(loc=sig, scale=sig, size=npts)) #must have sig>0
    ns_alt = ns.T
    
    grid = np.zeros((npts,len(means)+3))
    print("size of grid:",grid.shape)
    
    grid[:,2] = dat_alt[:,0]
    grid[:,3] = dat_alt[:,1]
    grid[:,4] = ns_alt[:]
    
    filename = 'test_pop_' + units[0] + "_" + units[1] + ".txt"
    headers = "lnL sigma_lnL {} {} sig".format(units[0],units[1])
    np.savetxt(filename,grid,header=headers,fmt='%.18s')

    print("Fake population (" + str(npts) + " points) data created.")


def make_Lambda_with_eos(npts,means,sig,units,eos_cols=None,eos_file=None,match_eos=True):
    
    
    eos_dat = None
    eos_names = eos_cols
    dat_len = npts
    eos_title = ""
    if eos_file is not None:
        eos_dat = np.genfromtxt(eos_file,names=True)
        print("size of eos data:",eos_dat.shape)
        print("Sample of eos data:",eos_dat[0])
        eos_names = eos_dat.dtype.names
        eos_title = "_"+eos_file[:len(eos_file)-4]
        if match_eos:#len(eos_dat) < npts:
            dat_len = len(eos_dat)
            print("Note: data will be truncated to",dat_len,"lines.")
    
    num_eos_cols = len(eos_names)
    print(num_eos_cols,"EOS columns:",eos_names)
    
    #Create npts pairs of random points in a grid space    
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mean=means, cov=0.01*np.diag(np.ones(len(means))))
    dat = rv.rvs(dat_len).T
    dat_alt = dat.T #copy so dat is unaffected by the below
    #Force m1 > m2:
    m1 = np.maximum(dat_alt[:,0], dat_alt[:,1])
    m2 = np.minimum(dat_alt[:,0], dat_alt[:,1])
    #print(m1,m2)
    
    if 'mc' in units: 
        #Convert to mc:
        mcV = mchirp(m1,m2)
        
        #Convert to other:
        otherV = []
        if 'delta_mc' in units:
            #Convert to delta_mc
            otherV =  (m1 - m2)/(m1+m2)
        elif 'eta' in units:
            #Convert to eta:
            otherV = symRatio(m1,m2)
    
        print("Shape check:",dat.shape, mcV.shape)
        dat_alt[:,0] = mcV
        dat_alt[:,1] = otherV
    else:
        print("Shape check:",dat.shape, m1.shape)
        dat_alt[:,0] = m1
        dat_alt[:,1] = m2
    
    ns = abs(np.random.normal(loc=sig, scale=sig, size=dat_len)) #must have sig>0
    ns_alt = ns.T
    
    grid = np.zeros((dat_len,len(means)+3+num_eos_cols))
    print("size of grid:",grid.shape)
    
    if eos_dat is not None:
        for i in range(num_eos_cols):
            for l in range(dat_len):
                grid[l,2+i] = eos_dat[l][i]
    grid[:,num_eos_cols+2] = dat_alt[:dat_len,0]
    grid[:,num_eos_cols+3] = dat_alt[:dat_len,1]
    grid[:,num_eos_cols+4] = ns_alt[:dat_len]
    
    
    filename = 'test_pop_' + units[0] + "_" + units[1] + "_eos"+eos_title+".txt"
    headers = "lnL sigma_lnL "+" ".join(i for i in eos_names)+" {} {} sig".format(units[0],units[1])
    np.savetxt(filename,grid,header=headers,fmt='%.18s')
    
    print("Fake population (" + str(npts) + " points) data created.")


def make_pop_with_eos(npts,mu,sig=0.2,eos_file=None,match_eos=True):
    eos_dat = None
    eos_names = []
    dat_len = npts
    eos_title = ""
    if eos_file is not None:
        eos_dat = np.genfromtxt(eos_file,dtype='float64',names=True)
        print("size of eos data:",eos_dat.shape)
        print("Sample of eos data:",eos_dat[0])
        eos_names = eos_dat.dtype.names
        eos_title = "_"+eos_file[:len(eos_file)-4]
        if match_eos:
            dat_len = len(eos_dat)
        else:
            print("Note: data will be truncated to",dat_len,"lines.")
    
    offset = 0
    print("Original eos file columns:",eos_names)
    if eos_names[0] == "lnL":
        eos_names = eos_names[2:]
        offset = 2
    num_eos_cols = len(eos_names)
    print(num_eos_cols,"EOS columns:",eos_names)
    
    #Create pairs of random points, centered on mu & truncated to [1,2]   
    from scipy.stats import norm
    rv = norm(loc=mu, scale=sig)
    dat = rv.rvs(size=(2,dat_len))
    #print("np dat:\n",dat2)
    dat_alt = dat.T
    #print("np dat_alt:\n",dat_alt2)
    m1 = np.maximum(dat_alt[:,0], dat_alt[:,1])
    m2 = np.minimum(dat_alt[:,0], dat_alt[:,1])
    #print("m1:\n",m1)
    #print("m2:\n",m2)
    #truncate to between 1 and 2:
    np.ceil(m1,out=m1,where=(m1 < 1.0))
    np.floor(m1,out=m1,where=(m1 > 2.0))
    np.ceil(m2,out=m2,where=(m2 < 1.0))
    np.floor(m2,out=m2,where=(m2 > 2.0))
    #print("m1:\n",m1)
    #print("m2:\n",m2) 
    print("Shape check:",dat.shape, m1.shape)
    dat_alt[:,0] = m1
    dat_alt[:,1] = m2
    
    #biases uncertainties to around .2:
    ns = abs(np.random.normal(loc=sig, scale=sig/4, size=dat_len)) #must have sig>0
    ns_alt = ns.T
    
    grid = np.zeros((dat_len,5+num_eos_cols))
    print("size of grid:",grid.shape)
    
    if eos_dat is not None:
        for i in range(num_eos_cols):
            for l in range(dat_len):
                grid[l,2+i] = eos_dat[l][offset+i]
    grid[:,num_eos_cols+2] = dat_alt[:,0]
    grid[:,num_eos_cols+3] = dat_alt[:,1]
    grid[:,num_eos_cols+4] = ns_alt[:]
    
    
    filename = 'test_pop_eos'+eos_title+".txt"
    headers = "lnL sigma_lnL "+" ".join(i for i in eos_names)+" m1 m2 sig"
    np.savetxt(filename,grid,header=headers,fmt='%.18e')
    
    print("Fake population (" + str(dat_len) + " points) data created.")


def make_pop_with_static_eos(npts,mu,sig=0.1,eos_file=None,line=0,hold=0):
    eos_dat = None
    eos_names = []
    dat_len = npts
    eos_title = ""
    if eos_file is not None:
        eos_dat = np.genfromtxt(eos_file,dtype='float64',names=True)[line]
        print("size of eos data:",eos_dat.shape)
        print("eos data:",eos_dat)
        eos_names = eos_dat.dtype.names
        eos_title = "_"+eos_file[:len(eos_file)-4].split("_")[0]+"_"#.split("-")[0]
        
    
    offset = 0
    print("Original eos file columns:",eos_names)
    if eos_names[0] == "lnL":
        eos_names = eos_names[2:]
        offset = 2
    num_eos_cols = len(eos_names)
    print(num_eos_cols,"EOS columns:",eos_names)
    
    #Create pairs of random points, centered on mu & truncated to [1,2]   
    from scipy.stats import norm
    
    single_pop = True
    single_sig = True
    try:
        mu2 = mu[1]
        single_pop = False
    except:
        print("Single pop mean detected.")
    try:
        sig2 = sig[1]
        single_sig = False
    except:
        print("Single sig detected.")
    
    mu1 = mu2 = 1.4
    if single_pop:
        mu2 = mu1 = mu
    else:
        mu1 = mu[0]
        mu2 = mu[1]
    name_info = str(int(mu1*10))+"_"+str(int(mu2*10))+"_h"+str(hold)
    
    sig1 = sig2 = 0.1
    if single_sig:
        sig2 = sig1 = sig
    else:
        sig1 = sig[0]
        sig2 = sig[1]
    
    if hold == 1:
        sig1 = 0.001
    elif hold == 2:
        sig2 = 0.001
    
    #mu1 = 1.4
    #sig1 = .1
    #mu2 = 1.1
    #sig2 = 0.0001
    #draw m1:
    rv1 = norm(loc=mu1, scale=sig1)
    dat1 = rv1.rvs(size=dat_len)
    #print("np dat:\n",dat2)
    dat1_alt = dat1.T
    #print("np dat_alt:\n",dat_alt2)
    #draw m2:
    rv2 = norm(loc=mu2, scale=sig2)
    dat2 = rv2.rvs(size=dat_len)
    #print("np dat:\n",dat2)
    dat2_alt = dat2.T
    
    #if hold > 0:
    #    np.ceil(m1,out=m1,where=(m1 < 1.0))
    #    np.floor(m1,out=m1,where=(m1 > 2.0))
    
    m1 = np.maximum(dat1_alt[:], dat2_alt[:])
    m2 = np.minimum(dat1_alt[:], dat2_alt[:])
    
    print("min m1:\n",min(m1))
    print("min m2:\n",min(m2))
    #truncate to between 1 and 2:
    np.floor(m1,out=m1,where=(m1 > 2.0))
    if hold == 1:
        print("Holding 1 not implemented yet. Exiting.")
        return #temp
    elif hold == 2:
        m2cut = 0.01
        m2 = np.where(m2 < mu2-m2cut, mu2-m2cut, m2) #raise m2 vals to m2 min cutoff
        m2 = np.where(m2 > mu2+m2cut, mu2+m2cut, m2) #lower m2 vals to m2 max cutoff
        #m1 = np.where(m1 < mu2-(.9*m2cut), m1+0.01, m1) #raise m1 vals above m2 min cutoff
        m1 = np.where(m1 < m2, m2+0.01, m1) #raise m1 vals above m2 min cutoff
    else:
        np.ceil(m1,out=m1,where=(m1 < 1.0))
        np.ceil(m2,out=m2,where=(m2 < 1.0))
        np.floor(m2,out=m2,where=(m2 > 2.0))
    
    print("min m1:\n",min(m1))
    print("min m2:\n",min(m2))
    
    #print("m1:\n",m1)
    #print("m2:\n",m2) 
    print("Shape check:",dat1.shape, m1.shape,dat2.shape, m2.shape)
    #dat_alt = np.zeros((2,dat_len)).T
    #dat_alt[:,0] = m1
    #dat_alt[:,1] = m2
    
    #fix uncertainties to constant:
    ns = np.zeros(dat_len)#or this: np.random.uniform(0.1,0.2,npts)
    if hold == 1:
        ns.fill(sig2)
    elif hold == 2:
        ns.fill(sig1)
    else:
        sig_avg = np.average([sig1,sig2])
        ns = abs(np.random.normal(loc=sig_avg, scale=sig_avg/4, size=dat_len)) #must have sig>0
    ns_alt = ns.T
    
    grid = np.zeros((dat_len,5+num_eos_cols))
    print("size of grid:",grid.shape)
    
    if eos_dat is not None:
        for i in range(num_eos_cols):
            grid[:,2+i].fill(eos_dat[offset+i])
    grid[:,num_eos_cols+2] = m1#dat_alt[:,0]
    grid[:,num_eos_cols+3] = m2#dat_alt[:,1]
    grid[:,num_eos_cols+4] = ns_alt[:]
    
    #print(grid)
    
    filename = 'static_pop_'+name_info+eos_title+str(line)+'.txt'
    headers = "lnL sigma_lnL "+" ".join(i for i in eos_names)+" m1 m2 sig"
    np.savetxt(filename,grid,header=headers,fmt='%.18e')
    
    #TEMP########
    #fig1 = plt.figure(figsize=(8,5),dpi=250) 
    #ax = fig1.add_subplot(111)
    #ax.scatter(m1,m2,marker=".")
    #ax.set_xlabel("$m_1$", size="11")
    #ax.set_ylabel("$m_2$", size="11")
    #ax.tick_params(axis='both', which='major', labelsize=10) 
    #fig1.tight_layout()
    #plt.show()
    
    print("Fake population (" + str(dat_len) + " points) data created.")


def generate_mass_Lambda_grid(mu, sig, npts, filepath, eos_index):
    #Create pairs of random points, centered on mu & with variance sig^2 
    from scipy.stats import norm
    rv = norm(loc=mu, scale=sig)
    dat = rv.rvs(size=(2,npts))
    #print("np dat:\n",dat2)
    dat_alt = dat.T
    #print("np dat_alt:\n",dat_alt2)
    m1 = np.maximum(dat_alt[:,0], dat_alt[:,1])
    m2 = np.minimum(dat_alt[:,0], dat_alt[:,1])
    #print("m1:\n",m1)
    #print("m2:\n",m2)
    #truncate to between 1 and 2:
    #np.ceil(m1,out=m1,where=(m1 < 1.0))
    #np.floor(m1,out=m1,where=(m1 > 2.0))
    #np.ceil(m2,out=m2,where=(m2 < 1.0))
    #np.floor(m2,out=m2,where=(m2 > 2.0))
    #print("m1:\n",m1)
    #print("m2:\n",m2) 
    print("Shape check:",dat.shape, m1.shape)
    dat_alt[:,0] = m1
    dat_alt[:,1] = m2
    
    #This gets one line of data; it will also get the names for each column, after header:
    eos_dat = np.genfromtxt(filepath,names=True)[eos_index]   
    param_names = eos_dat.dtype.names #separate out the names from the data
    dat_as_array = eos_dat.view((float, len(param_names)))
    print(dat_as_array)
    
    print("Original field names:", param_names)
    
    #Create EOS object:
    try:
        import ext_prior1
        my_eos = ext_prior1.generate_eos(dat_as_array, param_names) 
    
        #Use EOS object to compute lambda values for each m:
        l1 = [my_eos.lambda_from_m(my_eos, m) for m in m1]
        l2 = [my_eos.lambda_from_m(my_eos, n) for n in m2]
    except:
        print("Error: could not create EOS object.")
        l1 = np.zeros(npts)
        l2 = np.zeros(npts)
    
    grid = np.zeros((npts,4+len(param_names)))
    print("size of grid:",grid.shape)
    
    if eos_dat is not None:
        for i in range(len(param_names)):
            grid[:,4+i].fill(eos_dat[i])
    grid[:,0] = m1#dat_alt[:,0]
    grid[:,1] = m2#dat_alt[:,1]
    grid[:,2] = l1
    grid[:,3] = l2
    
    #print(grid)
    eos_title = "_"+filepath.split("_")[0]+"_"#.split("-")[0]

    filename = 'mass_lambda_grid'+eos_title+str(eos_index)+'.txt'
    headers = "m1 m2 Lambda1 Lambda2 "+" ".join(i for i in param_names)
    np.savetxt(filename,grid,header=headers,fmt='%.18e')
    
  
    
    

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
    np.random.seed(75108)#42179)
    
    #make_Lambda(20)
    
    #if opts.mode == 0:
    #    num_pop = 3000
    init_means = [20,10]
    sc = 0.5
    out_units = ['m1','m2']
        #ln = 0
        #eos = "Parametrized-EoS_maxmass_EoS_samples.txt"
    #else:
    #    num_pop = opts.npts
    #    mu = opts.mass_mean
    #    sigma = opts.mass_sig
    #    ln = opts.static_eos_line
    #    eos = opts.eos_file
    
    
    #make_better_Lambda(num_pop,init_means[0],init_means[1],sc)
    
    #make_different_Lambda(num_pop, init_means, sc, out_units)
    
    #make_Lambda_with_eos(num_pop, init_means, sc, out_units,eos_cols=["gamma1","gamma2","gamma3","gamma4"])
    
    #make_Lambda_with_eos(num_pop, init_means, sc, out_units,eos_file="Parametrized-EoS_maxmass_EoS_samples.txt")
    
    #make_pop_with_eos(num_pop,1.4,sig=.1,eos_file="Parametrized-EoS_maxmass_EoS_samples.txt")#"Parametrized-EoS_maxmass_EoS_samples.txt")
    
    #eos_file = "Parametrized-EoS_maxmass_EoS_samples.txt"
    #eos_title = "_"+eos_file[:len(eos_file)-4].split("_")[0].split("-")[0]
    #print(eos_title)
    
    opts.mass_mean = [1.4,1.1]
    opts.mass_sig = [0.1,0.001]
    
    #make_pop_with_static_eos(opts.npts,opts.mass_mean,sig=opts.mass_sig,line=opts.static_eos_line,eos_file=opts.eos_file,hold=2)

    generate_mass_Lambda_grid(1.39, 0.14, 100, "Parametrized-EoS_maxmass_EoS_samples.txt", 0)

    #get_Lambda()
    

