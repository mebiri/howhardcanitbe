# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 11:51:56 2025

@author: marce
"""

import numpy as np
import argparse
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.integrate import dblquad #does a double integral
from scipy.integrate import quad


#
# Mass parameter conversion functions - note they assume m1 >= m2
#
def m1m2q(Mc, q):
    """Compute component masses from Mc, q. Returns m1 >= m2"""
    #uses Mc = (m1*m2)^3/5 / (m1+m2)^1/5 ; q = m1/m2
    m1 = Mc*(q**(2./5.))*((q+1)**(1./5.))
    m2 = Mc*(q**(-3./5.))*((q+1)**(1./5.))
    return m1, m2

def Mcq(m1,m2):
    """Compute chirp mass & mass ratio from component masses"""
    mc = (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)
    q = m2/m1 #np.sqrt(1 - 4*eta), eta = mass1 * mass2 / mt**2 = (m1-m2)/(m1+m2) = (1-m2/m1)/(1+m2/m1) = (1-q)/(1+q)
    return mc, q

def m1q(Mc,q):
    return Mc*(q**(2./5.))*((q+1)**(1./5.))

def m2q(Mc,q):
    return Mc*(q**(-3./5.))*((q+1)**(1./5.))



#Check location of pop masses to avoid integrating (and importing scipy) if possible (faster)
def boundary_integration_checks(pop,mass_bounds):
    #Integration bounds; currently for BH masses:
    m_min = mass_bounds[0]
    m_max = mass_bounds[1]
    
    #--NOTE: THIS ASSUMES 1 UNCERTAINTY FOR 2 MASSES WITH M1 > M2 (2D PROBLEM)--
    pop_params = pop
    near_dist = 3*pop_params[2] #3*sigma distance cutoff from means of rv (somewhat lazy but saves time)
    d3 = (pop_params[0]-pop_params[1])/np.sqrt(2) #d3 = (m1-m2)/sqrt(2) = distance from m1=m2 line
    d1 = m_max - pop_params[0] #d1 = distance from right border
    d2 = pop_params[1] - m_min #d2 = distance above bottom border
    
    #Check that d1, d2, d3 are positive -> negative means outside pop range
    if d1 < 0 or d2 < 0 or d3 < 0 or pop_params[0] < m_min or pop_params[1] > m_max:
        print("ERROR: 1 or more masses outside valid range ["+str(m_min)+","+str(m_max)+"]; cannot normalize population.")
        return 0 #set normalization constant to 0, so lnL = -infinity
    
    d1c = False
    d2c = False
    d3c = False
    checks = 0
    if d1 < near_dist: d1c = True; checks += 1 #m1 near m_max (right boundary)
    if d2 < near_dist: d2c = True; checks += 1 #m2 near m_min (bottom boundary)
    if d3 < near_dist: d3c = True; checks += 1 #(m1,m2) near m1=m2 (hypotenuse boundary)
    #print("Boundary checks: m1:",d1c,"m2:",d2c,"m1=m2:",d3c,"Total:",checks)
    
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

rv = None
nm = 1

def initialize_me(**kwargs):
    '''
    **kwargs MUST take this form:
    {'input_line':dat_as_array, 'param_names':param_names, 'cip_param_names':coord_names}
    where: 
        dat_as_array = dat.view((float, len(param_names))) - an array of float values
        param_names = dat.dtype.names - IN THIS ORDER: lnL lnL_err {EOS PARAMS} m1 m2 sigma (sigma same error for both m1 & m2)
        cip_param_names = [str] - given coordinates that CIP is working in, order doesn't matter (hopefully)
    '''
    #----- INITIALIZING EXTERNAL PRIOR -----
    all_params = kwargs['input_line']#used by CIP - single line of data from eos file
    
    #----- Initialize population data -----
    global rv
    global nm
    
    #Expected header names: # lnL sigma_lnL g0 g1 g2 g3 m1 m2 sig
    #Split columns into pop and EOS:
    pop_params = []
    pop_params_lib = ['m1','m2','sig'] #can be added to for other populations
    for i in kwargs['param_names'][2:]: #should be anything past lnL, sig_lnL
        if i in pop_params_lib:
            pop_params.append(all_params[kwargs['param_names'].index(i)])
    
    #NOTE: supports 2+1 or 2+2-type mass/sig columns. Not 3+1, etc.
    n_dim = (len(pop_params)%2)+int(len(pop_params)/2) #expect 1 sigma per mass or pair of masses
    
    #check 0 < sig < 1 (protection against puffing):
    if abs(pop_params[n_dim]) >= 0.5:
        pop_params[n_dim] = 0.49
    else:
        pop_params[n_dim] = abs(pop_params[n_dim])
    
    rv = multivariate_normal(mean=pop_params[:n_dim], cov=(pop_params[n_dim]**2)*np.diag(np.ones(n_dim))) 
    
    #----- Initialize normalization constant -----
    #This is extremely slow...
    if False: #pop_params is not None:
        #check population width: if narrow width or far from edges -> nm = 1 (normal)
        global nm
        if len(pop_params) == 3:
            if pop_params[0] < 3.0: #if m1 is NS, so is m2 b/c m1>m2
                mbounds = [1,3] #Neutron star mass range
            else:
                mbounds = [3,30] #Black hole mass range
            nm = boundary_integration_checks(pop_params,mbounds)#only supports [m1 m2 sig]
        else:
            print("WARNING: unable to check population boundaries. Assuming normalized population.")
    #else: nm is set to 1 by default (i.e., entire normal curve is within domain)
    #print("Normalization constant set to",nm)
    
    #----- END EXTERNAL PRIOR INITIALIZATION -----
    
    
#compute \int dq p(m1(mc,q),m2(mc,q))
#integrate (sum) p(m1(mc,q), m2(mc,q)) from q=q1 to q=q2, w/ mc=const.

def likelihood_evaluation(*X):
    #*X contains data list in same order as cip_params given to initialize_me()
    
    #Convert to m1,m2 coords from whatever CIP is passing:
    m1m2 = np.asarray([X[0],X[1]],dtype=np.float64).T
    
    #Likelihood (w/ normalization constant):
    if nm == 0:
        return -np.inf
    else:
        return rv.logpdf(m1m2) - np.log(nm)


def do_rv(q,mc):
    m1, m2 = m1m2q(mc,q)
    return likelihood_evaluation([m1], [m2])


def integrate(mc,q_min,q_max):
    int_rv = lambda q: do_rv(q,mc) 
    likelihood = quad(int_rv, q_min, q_max)
    return likelihood[0]


def post_plot(m1,m2,m1_q,m2_q, log_L, dat, plotname):
    print("Plotting.")
    
    #Scatterplot:
    fig1 = plt.figure(figsize=(8,5),dpi=250) 
    ax = fig1.add_subplot(111)
    ax.scatter(m1,m2,c=log_L,marker=".")
    if dat is not None:
        ax.scatter(dat[:,-3],dat[:-2],c=dat[:,0],marker=".")
    ax.plot(m1_q,m2_q)
    ax.set_xlabel("$\mu_1$", size="11")
    ax.set_ylabel("$\mu_2$", size="11")
    ax.tick_params(axis='both', which='major', labelsize=10) 
    fig1.tight_layout()
    plt.savefig(plotname+".png")
    print("Plot saved as "+plotname+".png")
    #plt.show(block=False)
    

if __name__=="__main__":
    
    #-----Adapted from util_ConstructIntrinsicPosterior_GenericCoordinates.py-----
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--using-eos", type=str, default="test_pop_m1_m2_eos.txt", help="Name of EOS")
    parser.add_argument("--posterior",type=str,default=None)
    parser.add_argument("--mass-grid",type=str,default=None)
    parser.add_argument("--use-eos-file",action='store_true',default=False)
    parser.add_argument("--create",action='store_true',default=False)

    opts = parser.parse_args()
    print(opts)
    
    if (opts.posterior is None) and (opts.mass_grid is None):
        opts.use_eos_file = False
        opts.create = True
        print("Will create test field.")
        
    #coordinates for CIP to use:
    coord_names = ['m1','m2']
        
    dat_as_array = None
    likelist = None
    m1_indx = 0
    m2_indx = 0
    if opts.mass_grid is not None:
        try:
            dat = np.genfromtxt(opts.mass_grid,names=True)
            param_names = dat.dtype.names #separate out the names from the data
            dat_as_array = dat.view((float, len(param_names))) 
            
            dat_orig_names = param_names[2:] #Adapted from ye old example_gaussian.py
            print("Original field names:", dat_orig_names)
            m1_indx = param_names.index("m1")
            m2_indx = param_names.index("m2")
            likelist = dat_as_array[:,0]
        except:
            print("Could not open provided file; will create new points.")
    
    if opts.mass_grid is None or dat_as_array is None:
        param_names = []
        if opts.use_eos_file:
            dat = np.genfromtxt(opts.using_eos,names=True)
            param_names = dat.dtype.names #separate out the names from the data
            dat_as_array = dat.view((float, len(param_names)))
            print(dat_as_array[0])
            for i in range(len(dat_as_array)):
                if dat_as_array[i,-1] < 0.1:
                    dat_as_array[i,-1] = 0.1
        else:
            npts = 5000
            param_names = ["lnL","sigma_lnL","m1","m2","sig"]
            dat_as_array = np.zeros((npts,len(param_names)))
            datm = np.random.uniform(1.0,2.0,(npts,2))
            m1 = np.maximum(datm[:,0], datm[:,1])
            m2 = np.minimum(datm[:,0], datm[:,1])
            ns = np.random.uniform(0.1,0.2,npts)
            ns_alt = ns.T
            
            dat_as_array[:,2] = m1
            dat_as_array[:,3] = m2
            dat_as_array[:,4] = ns_alt[:]
            #print(dat_as_array[:5])
    
        dat_orig_names = param_names[2:] #Adapted from ye old example_gaussian.py
        print("Original field names:", dat_orig_names)
        m1_indx = param_names.index("m1")
        m2_indx = param_names.index("m2")
        
        likelist = []
        for l in range(len(dat_as_array)):
            args_init = {'input_line' : dat_as_array[l], 'param_names':param_names, 'cip_param_names':coord_names}
    
            initialize_me(**args_init) 
        
            res = integrate(1.188,1.0,2.0)
            likelist.append(res+100)
        
        mindx = likelist.index(min(likelist))
        print("minimum likelihood:",min(likelist),"at index ",mindx)
        print("min sigma:",min(ns_alt))
        print("at min: dat=",dat_as_array[mindx])
        print("maximum likelihood:",max(likelist))
        
        if opts.create:
            new_params = ["lnL","sigma_lnL","m1","m2","sig"]
            new_grid = np.zeros((len(dat_as_array),len(new_params)))
            new_grid[:,0] = likelist
            new_grid[:,2] = dat_as_array[:,m1_indx]
            new_grid[:,3] = dat_as_array[:,m2_indx]
            new_grid[:,4] = dat_as_array[:,-1]
            print(new_grid[:4])
            
            filename = "mass_field_test.txt"
            headers = "lnL sigma_lnL m1 m2 sig"
            np.savetxt(filename,new_grid,header=headers,fmt='%.18e')
            print("Grid saved as "+filename)
            
    
    q_range = np.linspace(1.0, 2.0, 100)
    m1_q = m1q(1.188, q_range)
    m2_q = m2q(1.188, q_range)    
    
    post_dat = None
    if opts.posterior is not None:
        post_dat = np.genfromtxt(opts.posterior,names=True)
        param_names = post_dat.dtype.names #separate out the names from the data
        pdat_as_array = post_dat.view((float, len(param_names)))
        
        post_plot(dat_as_array[:,m1_indx], dat_as_array[:,m2_indx], m1_q,m2_q,likelist,pdat_as_array,"mass_field_with_posterior")

    else:
        post_plot(dat_as_array[:,m1_indx], dat_as_array[:,m2_indx], m1_q,m2_q,likelist,None,"mass_field_test")
    
    
    