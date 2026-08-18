#! /usr/bin/env python
"""
Pick out specific EOS points (specifically OOB points) & plot them P(rho)
"""

import numpy as np
import argparse

#import RIFT.physics.EOSManager as EOSManager
import RIFT.plot_utilities.EOSPlotUtilities as eosplot

parser = argparse.ArgumentParser()
#eos arguments
parser.add_argument('--eos-file',type=str,help='REQUIRED, even if loading pyr dat! ')
parser.add_argument('--points-in', default=1,type=int,help="Number of completely in-bounds EOS lines of posterior to pick/plot")
parser.add_argument('--points-oob',default=1,type=int,help="Number of completely out-of-bounds EOS lines in posterior to pick/plot")
parser.add_argument('--no-plot',action='store_true',help="Do not plot EOS curves, just report EOS lines selected")
parser.add_argument('--param-bounds',action='append',default=None,type=str,help="format: param:[min,max] for each param (will match to EOS). If none provided, defaults to Carney 2018 gamma bounds")

parser.add_argument('--load-pyr-obj-dir',action='append',type=str,default=None,help="Dir(s)/basename(s) containing pyr objects to load for MR plot (create using NICER code with --save-pyr in Hyperpipe)")
#flags
parser.add_argument('--plot-pd',action='store_true',help='Plot pressure vs. density')
parser.add_argument('--plot-mr',action='store_true',help='Plot mass vs. radius; uses pyreprimand')
#plot opts
parser.add_argument('--eos-label', action='append',help='Label(s) for the EOS file(s) - order must be the same as eos-file option (use underscores for spaces)')
parser.add_argument('--eos-color', action='append',help='Line colors for the plot. If not provided, colors will be chosen automatically. Use white to have no line (must specify fill-color)')
parser.add_argument('--fill-color',action='append',help="Fill colors for region between percentiles; leave blank for no fill")
parser.add_argument('--plot-pd-name',type=str,default=None,help='Filename for the pressure vs. density plot')
parser.add_argument('--plot-mr-name',type=str,default=None,help='Filename for the mass vs. radius plot')
parser.add_argument('--show-grid',action='store_true',help="Show gridlines on plot; for MR plot only, right now")
parser.add_argument('--fill-alpha',type=float,default=0.1,help="Alpha value for shaded regions. Default is 0.1; set to 0 where no --fill-color provided.")
parser.add_argument('--xvar-single',type=str,default='rest_mass_density',help="X coord for --render-eos-objects plot")
parser.add_argument('--yvar-single',type=str,default='pressure',help="Y coord for --render-eos-objects plot")

opts = parser.parse_args()

if opts.eos_file is None:
    print("no eos provided, dumbass")
    import sys
    sys.exit(0)

dat = np.genfromtxt(opts.eos_file,names=True)
param_names = list(dat.dtype.names)
all_dat = dat.view((float, len(param_names)))
print("size of imported data:",len(all_dat),all_dat.shape)

my_bounds = {}
if opts.param_bounds is None:
    my_bounds["gamma0"] = [0.2,2.0]
    my_bounds["gamma1"] = [-1.6,1.7]
    my_bounds["gamma2"] = [-0.6,0.6]
    my_bounds["gamma3"] = [-0.02,0.02]
else:
    for p in opts.param_bounds:
        plist = p.split(":")
        strbounds = plist[1].replace("[","").replace("]","").split(",")
        my_bounds[plist[0]] = [float(strbounds[0]),float(strbounds[1])]

oob_lines_list = []
in_lines_list = []
oob_indx = []
in_indx = []
files_list_pd = []
files_list_mr = []
plot_opts_list = []
fill_opts_list = []

if opts.plot_mr and (opts.load_pyr_obj_dir is None):
    print("ERROR: no supplied paths to MR data for requested MR plot. Will not generate!")
    opts.plot_mr = False
elif opts.plot_mr:
    print("Not implemented yet.")
# =============================================================================
#     #get pyr h5 files that exist first, use those indices to draw from EOS file
#     for i in np.arange(len(opts.load_pyr_obj_dir)):
#         h5_files = glob.glob(opts.load_pyr_obj_dir[i]+"*.h5")
#         #reconstruct indices present:
#         valid_indx = np.zeros(len(h5_files))
#         for f in np.arange(len(h5_files)):
#             name_bits = h5_files[f].split("/")[-1].split("_") #/~/~/MARG-1-0_reprimand.tov.seq_0.h5 -> [MARG-1-0 , reprimand.tov.seq , 0.h5]
#             valid_indx[f] = int(name_bits[0].split("-")[-1]) + int(name_bits[2][0])
#         
#         print("Length of valid .h5 files collected:",len(valid_indx))
#         valid_indx = valid_indx.astype(int)
#     
#         if (int(opts.draw_eos) != 0) and (len(valid_indx) > int(opts.draw_eos)):
#             lines_to_use = np.random.choice(len(valid_indx),size=int(opts.draw_eos),replace=False)
#             print("Drawing",len(lines_to_use),"random lines from this file.")
#         else:
#             print("Using all collected lines from this file; total:",len(h5_files))
#             lines_to_use = np.arange(len(valid_indx)) 
#         files_list_mr.append(np.array(h5_files)[lines_to_use]) #list of FILEPATHS
#         lines_to_use_list.append(valid_indx[lines_to_use]) #save line numbers for eos_files
#     
# =============================================================================
        #TODO: need to assert len(eos_files) == len(pyr_obj_dirs)
    
    #if False: #opts.eos_file and (len(lines_to_use_list) > 0):
    #    #TODO: make sure lines_to_use doesn't exceed length of eos_file - shouldn't happen
    #    for i in np.arange(len(opts.eos_file)):
    #        dat = np.genfromtxt(opts.eos_file[i])[:,0]
    #        if len(dat) < lines_to_use_list[i][-1]:
    #            print("ERROR: Cannot use this set of lines. Deal with it.")
                #delete lines_to_use_list[i]?
else: #basically: opts.plot_pd or opts.eos_file
    #get EOS lines from grid file
    indx = 0
    while (len(oob_lines_list) < opts.points_oob) or (len(in_lines_list) < opts.points_in):
        line = dat[indx]
        oob_checks = 0
        in_checks = 0
        for p in my_bounds.keys():
            if p in param_names:
                col = param_names.index(p)
                if line[col] < my_bounds[p][0] or line[col] > my_bounds[p][1]:
                    oob_checks += 1
                else:
                    in_checks += 1
        if oob_checks == len(my_bounds.keys()):
            oob_lines_list.append(line)
            oob_indx.append(indx)
        elif in_checks == len(my_bounds.keys()):
            in_lines_list.append(line)
            in_indx.append(indx)
        indx += 1

#oob_indx = oob_indx[:opts.points_oob] 
#in_indx = in_indx[:opts.points_in]   
print("OOB indices (length",len(oob_indx),"total):\n",oob_indx[:opts.points_oob])
print("in indices (length",len(in_indx),"total):\n",in_indx[:opts.points_in])


#directly render all eos in provided range using their own axes
if not opts.no_plots: 
    try:
        print("Importing matplotlib...")
        import matplotlib #super slow import
        print(" Matplotlib backend ", matplotlib.get_backend())
        if matplotlib.get_backend() == 'agg':
            pass
        else:
            matplotlib.use('agg')
        fig_extension = '.png'
        bNoInteractivePlots =True
        bNoPlots=False
    except:
        print(" Error setting backend")
        
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 12.0,  'mathtext.fontset': 'stix'})
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)
    matplotlib.rcParams['xtick.labelsize'] = 15.0
    matplotlib.rcParams['ytick.labelsize'] = 15.0
    matplotlib.rcParams['axes.labelsize'] = 25.0
    matplotlib.rcParams['lines.linewidth'] = 2.0
    matplotlib.rcParams['legend.loc'] = 'lower right'
    
    import plot_eos_inference as pei
    xvar = opts.xvar_single
    yvar = opts.yvar_single
    oob_eos_list = pei.build_eos_sequence(opts.eos_file,oob_lines_list)
    if oob_eos_list is None:
        print("All provided EOS parameters failed; exiting.")
        sys.exit(0)
    print("EOS list initialized; total:",len(oob_eos_list))
    
    in_eos_list = pei.build_eos_sequence(opts.eos_file,in_lines_list)
    if in_eos_list is None:
        print("All provided EOS parameters failed; exiting.")
        sys.exit(0)
    print("EOS list initialized; total:",len(in_eos_list))
    
    oob_opts = {}
    in_opts = {}
    if opts.eos_color:
        oob_opts['color'] = opts.eos_color[0]
        if len(opts.eos_color) > 1:
            in_opts['color'] = opts.eos_color[1]
    
    for e in in_eos_list[:opts.points_in]:
        eosplot.render_eos(e,xvar, yvar,**in_opts) #'rest_mass_density', 'pressure'
    for e in oob_eos_list[:opts.points_oob]:
        eosplot.render_eos(e,xvar, yvar,**oob_opts) #'rest_mass_density', 'pressure'
    print("All EOS rendered.")
    
    if xvar == 'rest_mass_density':
        xlab = r"log$_{10}\, \rho$ [g cm$^{-3}$]"
    elif xvar == 'pressure':
        xlab = r"log$_{10}\,\, p$"
    else:
        xlab = xvar
    if yvar == 'pressure':
        ylab = r"log$_{10}\, P$ [dyn cm$^{-2}$]"
    elif yvar == 'energy_density':
        ylab = r"log$_{10}\,\, \epsilon$"
    else:
        ylab = yvar
    plt.xlabel(xlab) #"\rho$ [g cm$^{-3}$]")
    plt.ylabel(ylab) #"P$ [dyn cm$^{-2}$]")
    dpi_base=200
    res_base = 4*dpi_base
    plt.savefig("EOS_OOB_plot_"+yvar+"_vs_"+xvar+fig_extension,dpi=res_base)
    print("EOS figure saved.")


