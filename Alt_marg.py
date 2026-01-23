# -*- coding: utf-8 -*-
"""
hyperpipe alternate marginalizer

Compute the term L_k = \prod_k w_k = \sum_k ln(w_k), where w_k is the integral
   w_k = \int p(m) g((m-n)_obs, sig_obs) dm 
   over m_obs-3sig_obs < m < m_obs+3sig_obs for k real NS obs, 
   a gaussian population distribution p(m), and a gaussian g(\mu, sig)
Essentially \int g(mu, sig) is a cumulative PDF over the integration region
Full region is not necessary as sig is small. 
cf. Eqn. (2) in Kedia et al. 2025
   
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm, multivariate_normal


def loop_manager(m_obs,sig_obs,pop_list,out_pts=100,match=True):
    npts = len(pop_list)
    if not match:
        npts = out_pts
    dat_out = np.zeros((npts,2+len(pop_list[0])))
    
    for i in range(npts):
        line = pop_list[i]
        
        rv = multivariate_normal(mean=line[:2], cov=(line[2]**2)*np.diag(np.ones(2)))
        
        dat_out[i][0] = compute_product(m_obs,sig_obs,rv)
        dat_out[i][1] = 0.001
        dat_out[i][2:] = line
    
    filename = 'output_pop_dat.txt'
    headers = "lnL sigma_lnL m1 m2 sig"
    np.savetxt(filename,dat_out,header=headers,fmt='%.18e')
    print("Saved as "+filename+".")
    
    

def compute_product(m_obs,sig_obs,pop_norm):
    
    partial_sum = 0.0
    for i in range(len(m_obs)):
        #distribution around real mass:
        g_k = multivariate_normal(mean=m_obs[i], cov=np.diag([sig_obs[i][0]**2,sig_obs[i][1]**2]))
                                  #norm(loc=0,scale=sig_obs[i])
        
        #integrand is product of gaussians: p(m)*g_k(m)
        int_rv = lambda y, x: pop_norm.pdf([x,y])*g_k.pdf([x,y])
        #prod = lambda x: pop_norm.pdf(x)*g_k.pdf(x-m_obs[i])
        
        #integrate:
        w_k, err = dblquad(int_rv, m_obs[i][0]-3*sig_obs[i][0], m_obs[i][0]+3*sig_obs[i][0], m_obs[i][1]-3*sig_obs[i][1], lambda x: x)
        
        partial_sum += np.log(w_k)
    
    print(partial_sum)
    return partial_sum


if __name__ == "__main__":    
    #Access mass data; it will also get the names for each column, after header:
    mass_dat = np.genfromtxt("NSmasses.txt",names=True)  
    param_names = list(mass_dat.dtype.names) #separate out the names from the data
    dat_as_array = mass_dat.view((float, len(param_names)))[:,1:] #skip first col
    #print(dat_as_array)
    
    #flatten data for integral: -> split data for no real reason
    mass_list = dat_as_array[:,:2]#np.concatenate([dat_as_array[:,0], dat_as_array[:,1]])
    print(mass_list)
    sig_list = dat_as_array[:,2:]#np.concatenate([dat_as_array[:,2], dat_as_array[:,3]])
    print(sig_list)
    
    #Access pop data; it will also get the names for each column, after header:
    pop_dat = np.genfromtxt("test_pop_eos.txt",names=True)  
    param_names = list(pop_dat.dtype.names) #separate out the names from the data
    pop_as_array = pop_dat.view((float, len(param_names)))[:,2:] #skip first 2 cols
    #print(dat_as_array)
    
    
    loop_manager(mass_list,sig_list,pop_as_array,out_pts=10,match=False)
    #print("Result:",res)


