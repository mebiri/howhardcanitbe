#! /usr/bin/env python
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-
custom corner plot, with hypercube shown in back & data superimposed in front.
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
#from matplotlib import colormaps

parser = argparse.ArgumentParser()

parser.add_argument('--composite', type=str,help="REQUIRED: Send eos file with [lnL, sigma_lnL, gamma0, gamma1, gamma2, gamma3, ...] as the parameters.")
parser.add_argument('--buffer',type=float,default=0.1,help="buffer size (default 10%)")
parser.add_argument('--npts-cube',type=int,default=2000,help="number of points to draw to fill hypercube")
parser.add_argument('--lnL-cut',type=float,default=9,help="maybe curoff lnLs below certain val, to reduce points plotted")
parser.add_argument('--use-alt-buffer',action='store_true',help="use symmetric buffer implementation (total buffer = 2x opts.buffer)")
parser.add_argument('--use-all-composite-but-grayscale',action='store_true',help="plot all points in greyscale, color points on top")
parser.add_argument('--match-hypercube',action='store_true',help="Fix axis bounds around hypercube")
parser.add_argument('--cube-color',type=str,default="tab:blue",help="Color for hypercube points")
parser.add_argument('--custom-bound',type=str,action='append',help="Custom bounds for plots; supply empty str for unmodded param (will match hypercube)")
parser.add_argument('--use-cart-bounds',action='store_true',help="Use old Cartesian bounds from Carney et al (2018) for hypercube")
parser.add_argument('--posterior', type=str,default=None,action='append',help="grid.dat file(s) with [lnL, sigma_lnL, gamma0, gamma1, gamma2, gamma3, ...]. Can be used instead of or in addition to --composite.")
parser.add_argument('--post-color',type=str,action='append',help="color for posterior points (eventually hopefully contour), one per posterior file")
parser.add_argument('--post-label',type=str,action='append',help="Legend labels for posterior(s). If not provided, defaults to filenames.")

opts = parser.parse_args()

if not opts.composite and not opts.posterior:
    print(" ERROR: must supply either composite or posterior file. Exiting.")
    import sys
    sys.exit(0)


def get_puff_bounds(use_alt_buffer=False, buffer=0.1,ret=False,use_Cart=False):
    rot_coords = {}
    rot_coords["r0"] = [-4.37722, 4.91227] if not use_Cart else [0.2,2.0]
    rot_coords["r1"] = [-1.82240, 2.06387] if not use_Cart else [-1.6,1.7]
    rot_coords["r2"] = [-0.32445, 0.36469] if not use_Cart else [-0.6,0.6]
    rot_coords["r3"] = [-0.09529, 0.11426] if not use_Cart else [-0.02,0.02]
    
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


def build_plot(gammas,g_dat,lnL_list,filename,colormap=None,grey_dat=None):    
    fig1 = plt.figure(figsize=(8,7.5),dpi=250) 
    
    #Figure things out
    grey = False
    if grey_dat is not None: #opts.plot_all_composite_but_grayscale
        grey = True
    sc = opts.cube_color
    cm = 'rainbow_r' #colormap for composite
    post = False #posterior flag
    comp = False #composite flag
    if lnL_list is None: #no composite, expect posterior
        post = True
    else:
        comp = True
        if len(g_dat) > 1:
            post = True
            
    #posterior colors and labels
    if post:
        pc_list=['black', 'red', 'green', 'blue','yellow','C0','C1','C2','C3']
        if opts.post_color:
            pc_list = opts.post_color + pc_list
        leg_list = [f.split("/")[-1].split(".")[0] for f in filename]
        if comp: leg_list = leg_list[1:] #ditch composite filename
        if opts.post_label: 
            if len(opts.post_label) < len(leg_list):
                leg_list = opts.post_label + leg_list[len(opts.post_label):]
            else:
                leg_list = opts.post_label
    
    gmin = []
    gmax = []
    for i in np.arange(4):
        gmin.append(min(gammas[:,i])-(0.1*np.abs(min(gammas[:,i]))))#/(i+1)))
        gmax.append(max(gammas[:,i])+(0.1*np.abs(max(gammas[:,i]))))#/(i+1)))
    gmin[3] = gmin[3]-0.01
    gmax[3] = gmax[3]+0.01
    if opts.custom_bound:
        for i, r in enumerate(opts.custom_bound):
            if r != "":
                gmin[i] = np.float64(r.replace("[","").replace("]","").split(",")[0])
                gmax[i] = np.float64(r.replace("[","").replace("]","").split(",")[1])
    
    ax1 = fig1.add_subplot(331)
    ax1.scatter(gammas[:,0],gammas[:,1],marker=".",color=sc)
    if grey: ax1.scatter(grey_dat[:,0],grey_dat[:,1],marker=".",s=1,color='0.5')
    #ax1.scatter(g_dat[:,0],g_dat[:,1],c=c_list,marker=".",s=1,cmap=cm) #replace w/ below once contours work
    if comp: ax1.scatter(g_dat[0][:,0],g_dat[0][:,1],c=lnL_list,marker=".",s=1,cmap=cm)
    if post: 
        for i,p in enumerate(g_dat[1:]):
            ax1.scatter(p[:,0],p[:,1],c=pc_list[i],marker=".",s=1,label=leg_list[i])
        #ax1.contour(g_dat[:,0],g_dat[:,1],np.ones((len(g_dat),len(g_dat))),levels=[1]) #TODO: doesn't work!
    if opts.match_hypercube or opts.custom_bound: 
        ax1.set_xlim(left=gmin[0],right=gmax[0])
        ax1.set_ylim(bottom=gmin[1],top=gmax[1])
    #ax1.set_xlabel("$\gamma_0$", size="11")
    ax1.set_xticklabels([])
    ax1.set_ylabel("$\gamma_1$", size="11")
    ax1.tick_params(axis='both', which='major', labelsize=10) 
    
    ax2 = fig1.add_subplot(335)
    ax2.scatter(gammas[:,1],gammas[:,2],marker=".",color=sc)
    if grey: ax2.scatter(grey_dat[:,1],grey_dat[:,2],marker=".",s=1,color='0.5')
    ax2.scatter(g_dat[:,1],g_dat[:,2],c=lnL_list,marker=".",s=1,cmap=cm)
    if opts.match_hypercube or opts.custom_bound: 
        ax2.set_xlim(left=gmin[1],right=gmax[1])
        ax2.set_ylim(bottom=gmin[2],top=gmax[2])
    #ax2.set_xlabel("$\gamma_1$", size="11")
    #ax2.set_ylabel("$\gamma_2$", size="11")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.tick_params(axis='both', which='major', labelsize=10) 
    
    ax3 = fig1.add_subplot(339)
    ax3.scatter(gammas[:,2],gammas[:,3],marker=".",color=sc)
    if grey: ax3.scatter(grey_dat[:,2],grey_dat[:,3],marker=".",s=1,color='0.5')
    ax3.scatter(g_dat[:,2],g_dat[:,3],c=lnL_list,marker=".",s=1,cmap=cm)
    if opts.match_hypercube or opts.custom_bound: 
        ax3.set_xlim(left=gmin[2],right=gmax[2])
        ax3.set_ylim(bottom=gmin[3],top=gmax[3])
    ax3.set_xlabel("$\gamma_2$", size="11")
    #ax3.set_ylabel("$\gamma_3$", size="11")
    ax3.set_yticklabels([])
    ax3.tick_params(axis='both', which='major', labelsize=10) 
    
    ax4 = fig1.add_subplot(334)
    ax4.scatter(gammas[:,0],gammas[:,2],marker=".",color=sc)
    if grey: ax4.scatter(grey_dat[:,0],grey_dat[:,2],marker=".",s=1,color='0.5')
    ax4.scatter(g_dat[:,0],g_dat[:,2],c=lnL_list,marker=".",s=1,cmap=cm)
    if opts.match_hypercube or opts.custom_bound: 
        ax4.set_xlim(left=gmin[0],right=gmax[0])
        ax4.set_ylim(bottom=gmin[2],top=gmax[2])
    #ax4.set_xlabel("$\gamma_0$", size="11")
    ax4.set_xticklabels([])
    ax4.set_ylabel("$\gamma_2$", size="11")
    ax4.tick_params(axis='both', which='major', labelsize=10) 
    
    ax5 = fig1.add_subplot(337)
    ax5.scatter(gammas[:,0],gammas[:,3],marker=".",color=sc)
    if grey: ax5.scatter(grey_dat[:,0],grey_dat[:,3],marker=".",s=1,color='0.5')
    ax5.scatter(g_dat[:,0],g_dat[:,3],c=lnL_list,marker=".",s=1,cmap=cm)
    if opts.match_hypercube or opts.custom_bound: 
        ax5.set_xlim(left=gmin[0],right=gmax[0])
        ax5.set_ylim(bottom=gmin[3],top=gmax[3])
    ax5.set_xlabel("$\gamma_0$", size="11")
    ax5.set_ylabel("$\gamma_3$", size="11")
    ax5.tick_params(axis='both', which='major', labelsize=10) 
    
    ax6 = fig1.add_subplot(338)
    ax6.scatter(gammas[:,1],gammas[:,3],marker=".",color=sc)
    if grey: ax6.scatter(grey_dat[:,1],grey_dat[:,3],marker=".",s=1,color='0.5')
    ax6.scatter(g_dat[:,1],g_dat[:,3],c=lnL_list,marker=".",s=1,cmap=cm)
    if opts.match_hypercube or opts.custom_bound: 
        ax6.set_xlim(left=gmin[1],right=gmax[1])
        ax6.set_ylim(bottom=gmin[3],top=gmax[3])
    ax6.set_xlabel("$\gamma_1$", size="11")
    #ax6.set_ylabel("$\gamma_2$", size="11")
    ax6.set_yticklabels([])
    ax6.tick_params(axis='both', which='major', labelsize=10) 
    
    #ax7 = fig1.add_subplot(336)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig1.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig1.legend(lines, labels, loc='upper right')
    
    fig1.tight_layout()
    fig1.subplots_adjust(hspace=0.05,wspace=0.05)
    save_name = "custom_corner_"+"_".join([f.split("/")[-1].split(".")[0] for f in filename])
    save_name+="_b"+str(opts.buffer).replace(".","p")
    if comp and opts.lnL_cut:
        save_name+="_Lcut"+str(opts.lnL_cut).split(".")[0]
    if comp and opts.use_all_composite_but_grayscale:
        save_name+="_allcomp"
    if opts.match_hypercube:
        save_name+="_matchcube"
    if opts.use_cart_bounds:
        save_name+="_Cartesian"
    if opts.custom_bound:
        save_name+="_bounded"
    fig1.savefig(save_name+".png",dpi=250)
    plt.show()
    print("EOS mass-radius figure saved as "+save_name+".png")


npts = opts.npts_cube
do_alt_buff = False
if opts.use_alt_buffer:
    do_alt_buff = True
r_bounds = np.array(get_puff_bounds(use_alt_buffer=do_alt_buff, buffer=opts.buffer,ret=True,use_Cart=opts.use_cart_bounds))

rs = np.zeros((npts,4))
rs[:,0] = np.random.uniform(r_bounds[0,0], r_bounds[0,1],npts)
rs[:,1] = np.random.uniform(r_bounds[1,0], r_bounds[1,1],npts)
rs[:,2] = np.random.uniform(r_bounds[2,0], r_bounds[2,1],npts)
rs[:,3] = np.random.uniform(r_bounds[3,0], r_bounds[3,1],npts)

coord_names = ["gamma0","gamma1","gamma2","gamma3"]
low_level_coord_names = coord_names
if not opts.use_cart_bounds:
    import dan_rotation_conversion as dan
    r_gammas = dan.inverse_dan_rotation(rs, coord_names, low_level_coord_names)
else:
    r_gammas = rs

g_dat_list = [] #this will be very large
filenames = [] #for all filenames
lnL = None
cname = None
if opts.composite: #process composite to always be first
    cname = opts.composite
    filenames.append(cname)
    print("Retrieving composite data from file:",cname)
    dat = np.genfromtxt(cname,names=True)
    param_names = list(dat.dtype.names)
    all_dat = dat.view((float, len(param_names)))
    print("size of imported data:",len(all_dat),all_dat.shape)

    g_indx = [param_names.index(k) for k in coord_names]
    g_dat_orig = all_dat[:,g_indx]
    maxlnL = max(all_dat[:,0])
    print("max lnL:",maxlnL)
    if opts.lnL_cut:
        indx_ok = np.ones(len(all_dat),dtype=bool)
        indx_ok = all_dat[:,0] > maxlnL - opts.lnL_cut
        print(" Length of truncated data:",np.sum(indx_ok))
        all_dat = all_dat[indx_ok]
    
    lnL = all_dat[:,0] 
    g_dat = all_dat[:,g_indx]

    #stolen from plot_posterior_corner.py:
    #cm = colormaps['rainbow_r']
    indx_sorted = lnL.argsort()
    y_span = lnL.max() - lnL.min()
    print(" Composite file : lnL span ", y_span)
    #y_min = lnL.min()
    #cm2 = lambda x: cm( (x - y_min)/y_span)
    #my_cmap_values = cm((lnL-y_min)/y_span)
     
    # reverse order ... make sure largest plotted last
    g_dat = g_dat[indx_sorted]   # Sort by lnL
    #my_cmap_values = my_cmap_values[indx_sorted]

    print("size of selected data:",len(all_dat),all_dat.shape)
    print("length of likelihood data:",len(lnL))
    for i in np.arange(len(g_dat[0])):
        print(" coord range {} : [{}, {}]".format(coord_names[i],min(g_dat[:,i]),max(g_dat[:,i])))
    g_dat_list.append(g_dat)
    
post = False #whether data file is posterior (with lnL = 0)    
pname = None
if opts.posterior:
    pname = opts.posterior
    filenames += pname
    post = True
    for file in pname:
        print("Retrieving data from posterior file:",file)
        dat = np.genfromtxt(file,names=True)
        param_names = list(dat.dtype.names)
        all_dat = dat.view((float, len(param_names)))
        print("size of imported data:",len(all_dat),all_dat.shape)
    
        g_indx = [param_names.index(k) for k in coord_names]
        g_dat = all_dat[:,g_indx]
        
        for i in np.arange(len(g_dat[0])):
            print(" coord range {} : [{}, {}]".format(coord_names[i],min(g_dat[:,i]),max(g_dat[:,i])))
        g_dat_list.append(g_dat)

if opts.match_hypercube or opts.custom_bound:
    print(" Will override bounds")

if opts.composite and opts.use_all_composite_but_grayscale:
    build_plot(r_gammas, g_dat_list, lnL, filenames, grey_dat=g_dat_orig)
else:
    build_plot(r_gammas, g_dat_list, lnL, filenames)


