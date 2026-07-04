#! /usr/bin/env python
"""
custom corner plot, with hypercube shown in back & data superimposed in front.
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--using-eos', type=str,help="REQUIRED: Send eos file with [lnL, sigma_lnL, gamma0, gamma1, gamma2, gamma3, m1, m2, sig] as the parameters.")
parser.add_argument('--buffer',type=float,default=0.1,help="buffer size (default 10%)")
parser.add_argument('--npts-cube',type=int,default=2000,help="number of points to draw to fill hypercube")
parser.add_argument('--lnL-cut',type=float,default=None,help="maybe curoff lnLs below certain val, to reduce points plotted")
parser.add_argument('--use-alt-buffer',action='store_true',help="use symmetric buffer implementation (total buffer = 2x opts.buffer)")

opts = parser.parse_args()

#opts.using_eos = "consolidated_0_0.txt"
#opts.lnL_cut= 15

def get_puff_bounds(use_alt_buffer=False, buffer=0.1,ret=False):
    rot_coords = {}
    rot_coords["r0"] = [-4.37722, 4.91227]
    rot_coords["r1"] = [-1.82240, 2.06387]
    rot_coords["r2"] = [-0.32445, 0.36469]
    rot_coords["r3"] = [-0.09529, 0.11426]
    
    new_bounds = []
    for indx, param in enumerate(rot_coords.keys()):
        # apply hypercube buffer
        if use_alt_buffer: #new_bound = bound +/- buffer*(width of param range) SYMMETRIC buffer
            ubound = rot_coords[param][1] + buffer*abs(rot_coords[param][1]-rot_coords[param][0])
            lbound = rot_coords[param][0] - buffer*abs(rot_coords[param][1]-rot_coords[param][0])
        else: #new_bound = bound +/- buffer*|bound| -> asymmetric buffer
            ubound = rot_coords[param][1] + buffer*abs(rot_coords[param][1])
            lbound = rot_coords[param][0] - buffer*abs(rot_coords[param][0])
        
        print("bounds for param: {} [{}, {}]".format(param,lbound,ubound))
        new_bounds.append([lbound,ubound])
    if ret:
        return new_bounds


def build_plot(gammas,g_dat,lnL_list):    
    fig1 = plt.figure(figsize=(8,7.5),dpi=250) 
    
    ax1 = fig1.add_subplot(331)
    ax1.scatter(gammas[:,0],gammas[:,1],marker=".")
    ax1.scatter(g_dat[:,0],g_dat[:,1],c=lnL_list,marker=".")
    #ax.set_xlim(left=1.0,right=2.0)
    #ax.set_ylim(bottom=1.0,top=2.0)
    #ax1.set_xlabel("$\gamma_0$", size="11")
    ax1.set_ylabel("$\gamma_1$", size="11")
    ax1.tick_params(axis='both', which='major', labelsize=6) 
    #ax1.grid(True)
    
    ax2 = fig1.add_subplot(335)
    ax2.scatter(gammas[:,1],gammas[:,2],marker=".")
    ax2.scatter(g_dat[:,1],g_dat[:,2],c=lnL_list,marker=".")
    #ax2.set_xlabel("$\gamma_1$", size="11")
    #ax2.set_ylabel("$\gamma_2$", size="11")
    ax2.tick_params(axis='both', which='major', labelsize=6) 
    
    ax3 = fig1.add_subplot(339)
    ax3.scatter(gammas[:,2],gammas[:,3],marker=".")
    ax3.scatter(g_dat[:,2],g_dat[:,3],c=lnL_list,marker=".")
    ax3.set_xlabel("$\gamma_2$", size="11")
    #ax3.set_ylabel("$\gamma_3$", size="11")
    ax3.tick_params(axis='both', which='major', labelsize=6) 
    
    ax4 = fig1.add_subplot(334)
    ax4.scatter(gammas[:,0],gammas[:,2],marker=".")
    ax4.scatter(g_dat[:,0],g_dat[:,2],c=lnL_list,marker=".")
    #ax4.set_xlabel("$\gamma_0$", size="11")
    ax4.set_ylabel("$\gamma_2$", size="11")
    ax4.tick_params(axis='both', which='major', labelsize=6) 
    
    ax5 = fig1.add_subplot(337)
    ax5.scatter(gammas[:,0],gammas[:,3],marker=".")
    ax5.scatter(g_dat[:,0],g_dat[:,3],c=lnL_list,marker=".")
    ax5.set_xlabel("$\gamma_0$", size="11")
    ax5.set_ylabel("$\gamma_3$", size="11")
    ax5.tick_params(axis='both', which='major', labelsize=6) 
    
    ax6 = fig1.add_subplot(338)
    ax6.scatter(gammas[:,1],gammas[:,3],marker=".")
    ax6.scatter(g_dat[:,1],g_dat[:,3],c=lnL_list,marker=".")
    ax6.set_xlabel("$\gamma_0$", size="11")
    #ax6.set_ylabel("$\gamma_2$", size="11")
    ax6.tick_params(axis='both', which='major', labelsize=6) 
    
    fig1.tight_layout()
    plt.show(block=False)


npts = opts.npts_cube
do_alt_buff = False
if opts.use_alt_buffer:
    do_alt_buff = True
r_bounds = np.array(get_puff_bounds(use_alt_buffer=do_alt_buff, buffer=opts.buffer,ret=True))

rs = np.zeros((npts,4))
rs[:,0] = np.random.uniform(r_bounds[0,0], r_bounds[0,1],npts)
rs[:,1] = np.random.uniform(r_bounds[1,0], r_bounds[1,1],npts)
rs[:,2] = np.random.uniform(r_bounds[2,0], r_bounds[2,1],npts)
rs[:,3] = np.random.uniform(r_bounds[3,0], r_bounds[3,1],npts)

coord_names = ["gamma0","gamma1","gamma2","gamma3"]
low_level_coord_names = coord_names
import dan_rotation_conversion as dan
r_gammas = dan.inverse_dan_rotation(rs, coord_names, low_level_coord_names)

fname = opts.using_eos
dat = np.genfromtxt(fname,names=True)
param_names = list(dat.dtype.names)
all_dat = dat.view((float, len(param_names)))
print("size of imported data:",len(all_dat),all_dat.shape)

maxlnL = max(all_dat[:,0])
print("max lnL:",maxlnL)
if opts.lnL_cut:
    indx_ok = np.ones(len(all_dat),dtype=bool)
    indx_ok = all_dat[:,0] > maxlnL - opts.lnL_cut
    print(" Length of cut data:",np.sum(indx_ok))
    all_dat = all_dat[indx_ok]

lnLs = all_dat[:,0]
g_indx = [param_names.index(k) for k in coord_names]
all_dat = all_dat[:,g_indx]

print("size of selected data:",len(all_dat),all_dat.shape)
print("length of likelihood data:",len(lnLs))

build_plot(r_gammas, all_dat, lnLs)


