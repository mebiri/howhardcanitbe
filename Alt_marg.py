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
from scipy.integrate import quad
from scipy.stats import norm


def compute_product(m_obs,sig_obs,mu_pop,sig_pop):
    #population distribution:
    pm = norm(loc=mu_pop, scale=sig_pop)
    
    partial_sum = 0.0
    for i in range(len(m_obs)):
        #distribution around real mass:
        g_k = norm(loc=0,scale=sig_obs[i])
        
        #integrand is product of gaussians: p(m)*g_k(m)
        prod = lambda x: pm.pdf(x)*g_k.pdf(x-m_obs[i])
        
        #integrate:
        w_k, err = quad(prod, m_obs[i]-3*sig_obs[i], m_obs[i]+3*sig_obs[i])
        
        partial_sum += np.log(w_k)
        print(partial_sum)
    
    return partial_sum


if __name__ == "__main__":    
    #Access mass data; it will also get the names for each column, after header:
    eos_dat = np.genfromtxt("NSmasses.txt",names=True)  
    param_names = list(eos_dat.dtype.names) #separate out the names from the data
    dat_as_array = eos_dat.view((float, len(param_names)))[:,1:] #skip first col
    #print(dat_as_array)
    
    #flatten data for integral:
    mass_list = np.concatenate([dat_as_array[:,0], dat_as_array[:,1]])
    print(mass_list)
    sig_list = np.concatenate([dat_as_array[:,2], dat_as_array[:,3]])
    print(sig_list)
    
    res = compute_product(mass_list,sig_list,1.39,0.14)
    print("Result:",res)


