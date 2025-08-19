# -*- coding: utf-8 -*-
"""
Bakey a cakey as fast as you can!

* How to build a fake composite file
** make synthetic 'true' obs : one line per 'event', each has some m1,m2 say (initgrid.txt)
** pick a error scale in mass: sigma_mass. (hardcoded for now)
** Draw two random numbers error1, error2 from a normal with this width: rv.rvs(scale=sigma_mass), 
    and make mu1 = mass1 + error1 , mu2= mass2+error2
** for each obs, draw 1000 random points from the whole prior range, (i.e., limits in mc, eta). 
** Convert  to m1,m2, these Evaluate
    lnL = 100 +  rv.logpdf( dat_masses)   # rv is multivariate normal, with center mu1, mu2
** put that lnL in the .composite file output, and save, one file per 'obs'/event

@author: marce
"""

#! /usr/bin/env python
import numpy as np
import scipy.stats as stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test-name",default='multitest',type=str) #unused
parser.add_argument("--n-grid",default=200,type=int)
parser.add_argument("--n-grid-post",default=10,type=int,help='Number of elements to adjoin from the true posterior')
parser.add_argument("--iteration-number",default=0,type=int)
#parser.add_argument("--skip-lnL",action='store_true') #unused
parser.add_argument("--lnL-fixed-offset",default=100,type=float,help="A large constant offset, comparable to what we expect for a modest detection, so that the tendency of the GP to return to being near 0 does not contaminate the posterior for our gaussian tests! Equivalent to setting a minimum SNR")
#parser.add_argument("--verbose",action='store_true') #unused 
parser.add_argument("--base-data-file",default='initgrid3.txt',help="filename/path of synthetic data file in #m1 m2 format.")
parser.add_argument("--single",action='store_true',help="For testing; will only make composite file for first line in synthetic data file.")
parser.add_argument("--plot-composite",action='store_true',help="Scatterplot of generated points, using lnL as color scale.") 
opts = parser.parse_args()

print(opts)
n_grid = opts.n_grid
n_grid_post = opts.n_grid_post
offset = opts.lnL_fixed_offset
run_single = opts.single
save_plot = opts.plot_composite
gen_plot = False

## Retrieve test parameters, from current working directory
#dat =np.loadtxt("x0.dat")[0]   # these are lines with different gaussian parameters in each line, we just need ONE of them

n_dim = 2#len(dat) - hardcoding 2 dimensions for now (overwritten later)
my_dim_list = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
my_dim_ranges = [[1.5,2], [1,1.49], [-1,1], [-1,1], [-1,1], [-1,1], [-1,1], [-1,1]] #old m1,m2 ranges: [15,25], [8,12]
indx_offset = 1 #1 if including m1,m2; 3 if starting with s1x

sigma_mass = 0.1
m1_err, m2_err = stats.norm.rvs(scale=sigma_mass, size=2)
print("mass errors:",m1_err,m2_err)


#unused at present; seems unnecessary
def get_dat():
    #dat =np.loadtxt("x0.dat")[0]   # these are lines with different gaussian parameters in each line, we just need ONE of them
    filename = opts.base_data_file
    xdat = np.genfromtxt(filename,dtype='str')
    #m1 = xdat[0][0]
    #m2 = xdat[0][1]
    #print(m1,m2)
    
    return xdat


def generate_obs_grid(m1, m2, rv, idx):
    print("Generating points for masses",m1,m2)
    
    #Make grid to fill in:
    #12 columns: -1 m1 m2 s1x s1y s1z s2x s2y s2z lnL lnL_err -1
    data_grid = np.zeros((n_grid+n_grid_post,12))
    
    #First and last columns all -1, for reasons:
    data_grid[:,0] = -1
    data_grid[:,11] = -1
    
    #For each mass (dim), draw an in-range random # for each line of the grid:
    for i in np.arange(n_dim):
        #This draws column by column in the grid (many lines, few columns):
        data_grid[:n_grid, i+indx_offset] = np.array(np.random.uniform(low=my_dim_ranges[i][0], high=my_dim_ranges[i][1], size=n_grid))
    
    #(Bonus lines) Part with random draws from the posterior - first make random draws from all 
    for z in np.arange(n_grid_post):
        #This draws line by line in the grid (few lines, few columns):
        data_grid[n_grid+z, indx_offset:indx_offset+n_dim] = rv.rvs() #double-check this
    
    #Fill in likelihood:
    for l in range(n_grid+n_grid_post):
        lnL = offset + rv.logpdf(data_grid[l][indx_offset:indx_offset+n_dim])
        data_grid[l][indx_offset+n_dim+6] = lnL #not generalized for diff offset
    
    #Nominal likelihood error:
    data_grid[:,10] = 0.1
    
    #Must save in a structured format:
    # -1 m1 m2 s1x s1y s1z s2x s2y s2z lnL lnL_err -1 #ntot neff
    prefix = "indx "
    if indx_offset >2: #i.e., started with s1x
        prefix += "m1 m2"
    #Default filename format: 'grid-random-0-#.txt'
    np.savetxt("grid-random-"+str(opts.iteration_number)+"-"+str(idx)+".txt",data_grid,header=prefix + (' '.join(my_dim_list)) + ' lnL lnL_err -1')
    
    if save_plot:# or gen_plot:
        plot_composite(data_grid, idx)
        
# =============================================================================
# rv = stats.multivariate_normal(mean=dat, cov=sigma_mass**2 * np.diag(np.ones(n_dim)) ) # naive unit covariance
# 
# 
# def random_draw():
#     rv.rvs()
# 
# # Grid.  
# my_grid = np.zeros((n_grid+n_grid_post, 11))
# # Uniform part. Independent draws in all dimensions
# for indx in np.arange(n_dim):
#     my_grid[:n_grid,indx+indx_offset] = np.array(np.random.uniform(low=my_dim_ranges[indx][0], high=my_dim_ranges[indx][1],size=n_grid))
# # Part with random draws from the posterior
# #   - first make random draws from all 
# for z in np.arange(n_grid_post):
#     my_grid[n_grid+z, indx_offset:indx_offset+n_dim] = random_draw()
# 
# 
# # Must save in a structured format
# #   -1 m1 m2 s1x s1y s1z s2x s2y s2z lnL err ntot neff
# prefix = "indx "
# if indx_offset >2:
#     prefix += "m1 m2"
# np.savetxt("grid-random-"+str(opts.iteration_number)+".txt",data_grid,header=prefix + (' '.join(my_dim_list)) + ' lnL lnL_err -1')
# =============================================================================
    

def plot_composite(grid, idx):
    import matplotlib.pyplot as plt
    
    #Scatterplot:
    fig1 = plt.figure(figsize=(8,5),dpi=250) 
    ax = fig1.add_subplot(111)
    ax.scatter(grid[:,indx_offset],grid[:,indx_offset+1],c=grid[:,indx_offset+n_dim+6])
    ax.set_xlabel("x mass", size="11")
    ax.set_ylabel("y mass", size="11")
    ax.tick_params(axis='both', which='major', labelsize=10) 
    fig1.tight_layout()
# =============================================================================
#     if gen_plot:
#         plt.show(block=False)
#     if save_plot:
#         plotname = "grid-random-"+str(opts.iteration_number)+"-"+str(idx)+".png"
#         plt.savefig(plotname,format = 'png')
#         print("Scatterplot saved as "+plotname)
# =============================================================================
    

if __name__ == '__main__':
    #Tasty defaults:
    #TODO remove these
    if n_grid == 200: #assume running locally 
        n_grid = 1000
        n_grid_post = 0
        run_single = True
        save_plot = True
        gen_plot = False #TODO set to False
        print("Local settings: n_grid=",n_grid,"n_grid_post=",n_grid_post,"single=",run_single,"plot: save",save_plot,"gen",gen_plot)
        
    #exact ("true") obs file (should contain masses or mc/eta?):
    #x = get_dat() #seems pretty unnecessary
    x = np.genfromtxt(opts.base_data_file,dtype='str')
    n_dim = len(x[0])
    
    for idx in range(len(x)):
        #Offset "exact" masses with error:
        mu1 = np.float64(x[idx][0]) + m1_err
        mu2 = np.float64(x[idx][1]) + m2_err
        print("mu for index "+str(idx)+":",mu1,mu2)
        #rv = stats.multivariate_normal(mean=dat, cov=sigma_mass**2 * np.diag(np.ones(n_dim))) #naive unit covariance
        rv = stats.multivariate_normal(mean=[mu1,mu2], cov=(sigma_mass**2)*np.diag(np.ones(n_dim)))
        generate_obs_grid(mu1, mu2, rv, idx)
        if run_single:
            break
    print("Done.")
    

