#! /usr/bin/env python
'''
Code for making P vs. rho and M vs. R EOS inference plots for hyperpipe
Utilizes RIFT.plot_utilities.EOSPlotUtilities, although uses locally-improved
versions of some of those functions.

Minimal input is the path to the posterior files the user wants plotted,
flags corresponding to which plots to make, and the number of EOS lines from
each file to use (num-eos: either total lines on plot, or lines use to interp).
User can also provide labels, line colors, and fill colors for each posterior.

To make P vs. rho plots, must supply eos_file(s)
To make M vs. R plots, must have pyreprimand installed and supply pyr objects
for each EOS in the eos_file(s) (pyr data files not yet supported)
'''
import numpy as np
import argparse
import sys
import copy

import RIFT.physics.EOSManager as EOSManager
import RIFT.plot_utilities.EOSPlotUtilities as eosplot

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

parser = argparse.ArgumentParser()
#eos arguments
parser.add_argument('--eos-file',action='append',help='REQUIRED, even if loading pyr dat! The last shall be first (top layer), and the first shall be last (bottom)',required=True)
parser.add_argument('--num-eos', default=1,type=int,help="Number of lines of each posterior file to plot EOS curves for (starts at 0). Enter 0 to do all lines (use draw_eos with this)")
parser.add_argument('--draw-eos',default=500,help="Number of random EOS to interpolate from posterior file, if file length > provided value")
parser.add_argument('--load-pyr-obj-dir',action='append',type=str,default=None,help="Dir(s)/basename(s) containing pyr objects to load for MR plot (create using NICER code with --save-pyr in Hyperpipe)")
parser.add_argument('--load-pyr-dat-dir',action='append',type=str,default=None,help="Dir(s)/basenames(s) containing mass-radius pyr dat files for MR plot (create using NICER code with --save-all-files in Hyperpipe)")
parser.add_argument("--percentile-bounds",type=str,nargs='*',default='[0.05,0.95]',help="percentile bounds for interpolation; e.g., [0.05,0.95] (NO SPACES)")
#flags
parser.add_argument('--render-eos-objects',action='store_true',help="Plot each eos line separately, instead of interpolating to a grid")
parser.add_argument('--plot-pd',action='store_true',help='Plot pressure vs. density')
parser.add_argument('--plot-mr',action='store_true',help='Plot mass vs. radius; uses pyreprimand')
#plot opts
parser.add_argument('--eos-label', action='append',help='Label(s) for the EOS file(s) - order must be the same as eos-file option (use underscores for spaces)')
parser.add_argument('--eos-color', action='append',help='Line colors for the plot. If not provided, colors will be chosen automatically. Use white to have no line (must specify fill-color)')
parser.add_argument('--fill-color',action='append',help="Fill colors for region between percentiles; leave blank for no fill")
parser.add_argument('--plot-pd-name',type=str,default=None,help='Filename for the pressure vs. density plot')
parser.add_argument('--plot-mr-name',type=str,default=None,help='Filename for the mass vs. radius plot')
parser.add_argument('--show-grid',action='store_true',help="Show gridlines on plot; for MR plot only, right now")

parser.add_argument('--verbose', action = 'store_true', help = 'Print information on the progress of the code')

opts = parser.parse_args()

#get percentile range into proper form - only works with one set of bounds, right now
quant_bounds = None#np.zeros((len(opts.percentile_bounds),2))
#for i in range(len(opts.percentile_bounds)): 
strbounds = opts.percentile_bounds.replace("[","").replace("]","").split(",")
if len(strbounds) > 2:
    print("ERROR: provided percentile range invalid; will use default (90%).")
    quant_bounds = [0.05,0.95]
else:
    quant_bounds = [float(x) for x in strbounds]
    #quant_bounds[i,:] = eval(opts.percentile_bounds[i]) #eval is ultra-sketchy but oh well


def plot_credible():
    # Writing on the plot what Credible Intervals are being shown. - copied from Atul
    quantile_text = ''
    #for i in range(len(quant_bounds)):
        #if i ==0:
    quantile_text += str(round((quant_bounds[1] - quant_bounds[0])*100))+'% CI'
        #else: quantile_text += '\n'+ str(round((quant_bounds[i][1] - quant_bounds[i][0])*100))+'% CI'
    plt.text(18.2, 2.3, quantile_text, bbox={'facecolor':'white','alpha':1,'edgecolor':'black'})


posterior_header = None
#NOTE: ONLY THESE EOS_PARAMS HANDLED CURRENTLY: spectral, cs_spectral, PP
def generate_eos(eos_line, eos_headers, eos_param="spectral",save_header=True,verbose=False):
    if verbose: print("Creating EOS object of type",eos_param,"using given data line.")
    
    global posterior_header
    if eos_headers == posterior_header:
        eos_names = posterior_header
        if verbose: print("Relabeling EOS using existing headers:",eos_names) 
    else:
        eos_names = eos_headers
        if ((eos_param == "spectral" or eos_param == "cs_spectral") and eos_names[0] != "gamma1") or (eos_param=="PP" and eos_names[1] != "gamma1"):
            print("WARNING: Unsupported gamma labels in EOS names found:",eos_names,"will relabel.")
            counter = 0
            indx= 0
            while counter < 4 and indx < len(eos_headers):#max 4 gamma cols, or stop at end of list
                if eos_headers[indx][0] == 'g' and eos_headers[indx][-1] == str(counter):#ensure gamma col
                    counter += 1
                    eos_names[indx] = "gamma"+str(counter)
                indx+=1
            print("Relabeled EOS headers:",eos_names)  
        if save_header:
            posterior_header = eos_names
            
    spec_param_array = eos_line 
    spec_params ={}

    for i in range(len(eos_names)):
        spec_params[eos_names[i]]=spec_param_array[i]
    if verbose: print("EOS data:\n",spec_params)
    
    eos_name="default_eos_name"
    eos_base = None
    try:
        if eos_param == 'spectral':
            #expect cols: gamma1, gamma2, gamma3, gamma4 (or fewer; must be at least 2 cols)
            eos_base = EOSManager.EOSLindblomSpectral(name=eos_name,spec_params=spec_params,use_lal_spec_eos=True)
        elif eos_param == 'cs_spectral' and len(spec_param_array) >=4:
            #expect cols: gamma1, gamma2, gamma3, gamma4
            eos_base = EOSManager.EOSLindblomSpectralSoundSpeedVersusPressure(name=eos_name,spec_params=spec_params,use_lal_spec_eos=True)
        elif eos_param == 'PP' and len(spec_param_array) >=4:
            #expect cols: logP1, gamma1, gamma2, gamma3
            eos_base = EOSManager.EOSPiecewisePolytrope(name=eos_name,params_dict=spec_params)
        else:
            raise Exception("Unknown method for parametric EOS data file {} : {} ".format(eos_name,eos_param))
    except Exception as e:
        if verbose:
            print("=====\n FAILSTATE 3: EOS CREATION FAILED. Exception:\n     ",type(e),":",e,"\n EXITING.\n=====")
        else:
            print("=== EOS Creation Failed:",e,"===")
        eos_base = None
    
    return eos_base


def build_eos_sequence(filename, lines):
    #This gets 1+ lines of data; it will also get the names for each column, after header:
    dat = np.genfromtxt(filename,names=True)[lines]
    param_names = dat.dtype.names #separate out the names from the data
    all_params = dat.view((float, len(param_names)))
    
    #load eos data directly from file, make EOSs via EOSManager
    eos_names = []
    eos_dat = np.zeros((len(all_params),len(param_names[2:])))
    pop_params_lib = ['m1','m2','sig'] #can be added to for other populations
    j= 0
    for i in param_names[2:]: #should be anything past lnL, sig_lnL
        if i in pop_params_lib:
            continue
        else: #anything that isn't m1, m2, sig
            eos_names.append(i)
            eos_dat[:,j] = all_params[:,param_names.index(i)]
            j+=1
    
    if len(eos_names) > 0:
        eos_list = []
        for i in np.arange(len(eos_dat)):
            new_eos = generate_eos(eos_dat[i], eos_names)
            if new_eos is None:
                print("  eos line",i,"failed to generate.")
            else:
                eos_list.append(new_eos.eos)
        return eos_list
    else:
        print("ERROR: No EOS columns found. Unable to create EOS object.")
        return None


#ported from EOSPlotUtilities for modifications
def render_eos_list_quantiles_vs(eos_list, quantile_bounds=None, xvar='energy_density', xgrid=None,yvar='pressure', units='cgs',use_monotonic=True,use_log=True,return_outvals=False,input_outvals=None,show_traces=False,plot_kwargs={},fill_kwargs={}):
    outvals_here=None
    if input_outvals is None:
        outvals_here  = eosplot.eval_eos_list_vs(eos_list, xvar=xvar , xgrid=xgrid, yvar=yvar, units=units, use_monotonic=use_monotonic)
    else:
        outvals_here = input_outvals

    if outvals_here is None:
        raise Exception(" failure generating eval list, should never happen this way")

    xgrid_here = np.array(xgrid)
    upper_vals = np.percentile(outvals_here,quantile_bounds[0]*100,1)
    lower_vals = np.percentile(outvals_here,quantile_bounds[1]*100,1)
    if use_log:
        xgrid_here = np.log10(xgrid_here)
        upper_vals = np.log10(upper_vals)
        lower_vals = np.log10(lower_vals)

    if show_traces:
        #print(outvals_here.shape, xgrid_here.shape)
        for indx in np.arange(len(outvals_here)):
            if use_log:
                plt.plot(xgrid_here,np.log10(outvals_here[:,indx]),color='k')
        
    plt.plot(xgrid_here, upper_vals, **plot_kwargs) 
    plot_kwargs['label'] = '' #this is the modification
    plt.plot(xgrid_here, lower_vals, **plot_kwargs)
    plt.fill_between(xgrid_here, lower_vals,upper_vals,**fill_kwargs)
    #plt.ylim(min(lower_vals)-0.1,max(upper_vals)+0.1) #also new here
    if return_outvals:
        return outvals_here #bug in original code: undefined variable "outvals"
    return None


lines_to_use_list = []
plot_opts_list = []
fill_opts_list = []
for i in np.arange(len(opts.eos_file)):
    #NEED to do this regardless, for consistency of random lines
    #See how many lines of data there are:
    if opts.num_eos == 0:
        dat = np.genfromtxt(opts.eos_file[i])[:,0]
    else:
        dat = np.genfromtxt(opts.eos_file[i])[:opts.num_eos,0]
    
    print("Initial data length for file:",len(dat))
    
    if (int(opts.draw_eos) != 0) and (len(dat) > int(opts.draw_eos)):
        lines_to_use = np.random.choice(len(dat),size=int(opts.draw_eos),replace=False)
        print("Drawing",len(lines_to_use),"random lines from this file.")
        dat = dat[lines_to_use]
        if opts.verbose: print("Length of dat is now:",len(dat))
    else:
        print("Using all lines from this file; total:",len(dat))
        lines_to_use = np.arange(len(dat)) #needed to get pyr files
    lines_to_use_list.append(lines_to_use)
    
    if opts.verbose: print("First line of data:\n",dat[0])
    
    #gather plot stuff together:
    label_here = None
    plot_opts_here = {}
    fill_opts_here = {}
    if opts.eos_label and len(opts.eos_label) > i:
        label_here = opts.eos_label[i].replace("_"," ")
    if opts.fill_color and len(opts.fill_color) > i:
        fill_opts_here['color'] = opts.fill_color[i] 
        fill_opts_here['alpha'] = 0.1
    else:
        fill_opts_here['alpha'] = 0.0 #transparent
    if opts.eos_color and len(opts.eos_color) > i:
        plot_opts_here['color'] = opts.eos_color[i]
        if opts.eos_color[i] == "white" and label_here:
            fill_opts_here['label'] = label_here
            label_here = None
    
    plot_opts_here['label'] = label_here
    plot_opts_list.append(plot_opts_here)
    fill_opts_list.append(fill_opts_here)

print("Plot options collected:\n",plot_opts_list,"\n",fill_opts_list)

if opts.render_eos_objects: #directly render all eos in provided range using their own axes
    for i in np.arange(len(opts.eos_file)):
        my_eos_list = build_eos_sequence(opts.eos_file[i],lines_to_use_list[i])
        if my_eos_list is None:
            print("All provided EOS parameters failed; exiting.")
            sys.exit(0)
        print("EOS list initialized; total:",len(my_eos_list))
        
        for e in my_eos_list:
            eosplot.render_eos(e.eos,'rest_mass_density', 'pressure')
    print("All EOS rendered.")
    dpi_base=200
    res_base = 4*dpi_base
    plt.savefig("test_eos_pd_plot"+fig_extension,dpi=res_base)
    print("EOS figure saved.")
    sys.exit(0)
    
if opts.plot_pd: 
    print("Creating pressure-density figure...")
    for i in np.arange(len(opts.eos_file)): 
        my_eos_list = build_eos_sequence(opts.eos_file[i],lines_to_use_list[i])
            
        if my_eos_list is None: 
            print("ERROR: no valid EOSs were created for this file.")
            continue 
        else:
            print("EOS list total:",len(my_eos_list))
            density_grid = 10**np.linspace(14,16,200) #need to choose good range of densities
            
            plot_opts = copy.deepcopy(plot_opts_list[i])
            fill_opts = copy.deepcopy(fill_opts_list[i])
            render_eos_list_quantiles_vs(my_eos_list, quantile_bounds=[0.05,0.95], xvar='rest_mass_density', xgrid=density_grid,yvar='pressure',use_log=True,plot_kwargs=plot_opts,fill_kwargs=fill_opts)
        
    print("All pressure-density EOS rendered.")
    plt.xlabel(r"log$_{10}\, \rho$ [g cm$^{-3}$]")
    plt.ylabel(r"log$_{10}\, P$ [dyn cm$^{-2}$]")
    plt.xlim(14,16)
    #plt.ylim() - set in render_eos_list_quantiles_vs()
    if opts.eos_label:
        plt.legend(frameon=False)
    dpi_base=200
    res_base = 4*dpi_base
    if opts.plot_pd_name:
        save_name = opts.plot_pd_name
    else:
        save_name = "EOS_PDplot"
        for e in opts.eos_file:
            save_name += "_"+e.split("/")[-1].split(".")[0]
    plt.savefig(save_name+fig_extension,dpi=res_base)
    plt.show()
    print("EOS pressure-density figure saved as "+save_name+fig_extension)

if opts.plot_mr:
    plt.clf()
    print("Creating mass-radius figure...")
    import pyreprimand as pyr
    import glob
    import lal
    mass_range = None
    for i in np.arange(len(opts.eos_file)): 
        if opts.load_pyr_obj_dir:
            #fetch pyr object files matching lines_to_use indices
            #opts = "path/MARG-0-"
            chunk_files = glob.glob(opts.load_pyr_obj_dir[i]+"0_reprimand.tov.seq_*.h5")
            nchunk = len(chunk_files)
            eos_sequence = []
            for j in lines_to_use_list[i]:
                if nchunk == 1:
                    loadname = opts.load_pyr_obj_dir[i]+str(j)+"_reprimand.tov.seq_0.h5"
                else:
                    loadname = opts.load_pyr_obj_dir[i]+str(j-(j%nchunk))+"_reprimand.tov.seq_"+str(j%nchunk)+".h5"
                try:
                    tov_seq_reprimand = pyr.load_star_branch(loadname, pyr.units.geom_solar(msun_si=lal.MSUN_SI))
                    eos_sequence.append(tov_seq_reprimand)
                except:
                    print(" WARNING: could not find file: "+loadname)
                    continue
            
            plot_opts = plot_opts_list[i]
            fill_opts = fill_opts_list[i]
            mass_range = [0.7,2.5]
            eosplot.render_reprimand_tovsequence_list_quantiles_vs(eos_sequence, quantile_bounds=[0.05,0.95], 
                                                                   xvar='radius', yvar='mass', range_mass='[0.7,2.5]', 
                                                                   percentile_method = 'use nan percentile', plot_kwargs=plot_opts, fill_kwargs=fill_opts)
        elif opts.load_pyr_dat_dir:
            print(" WARNING: using pyr dat not fully implemented or tested. Use with caution!")
            from scipy.interpolate import UnivariateSpline, PchipInterpolator
            #load .txt files of pyr dat
            #opts = "path/MARG-0-"
            eos_dat = []
            for j in lines_to_use:
                #if opts.plot_pd: - can't use this
                #    dat_here = np.loadtxt(opts.load_pyr_dat_dir+str(j)+"_pressure-density_"+str(j)+".txt")
                #    #do something with this
                #if opts.plot_mr:
                dat_here = np.loadtxt(opts.load_pyr_dat_dir[i]+str(j)+"_mass-radius_"+str(j)+".txt")[:,:2]
                #mass = 0, radius = 1
                eos_dat.append(dat_here) #will become very large, possibly ragged 3D array (list of 2D Nx2 arrays)
            
            r_grid = np.linspace(8,20,1200) #0.01 resolution, same as pd plot
            npts = len(r_grid)
            #    print(npts,n_eos)
            # LARGE ALLOCATION potentially, so watch out -- usually I just need quantiles
            outvals  = np.zeros((npts,len(eos_dat)))

            for indx in np.arange(len(eos_dat)):
                if True:
                    intp_func = PchipInterpolator(np.log(eos_dat[indx][:,1]),np.log(eos_dat[indx][:,0]))
                else:
                    intp_func = UnivariateSpline(np.log(eos_dat[indx][:,1]),np.log(eos_dat[indx][:,0]))
                ygrid = np.exp(intp_func(np.log(r_grid)))
                outvals[:,indx] = ygrid
            
            xgrid_here = np.array(r_grid)
            upper_vals = np.percentile(outvals,quant_bounds[0]*100,1)
            lower_vals = np.percentile(outvals,quant_bounds[1]*100,1)
            mass_range = [min(upper_vals),max(upper_vals)]
            
            plot_opts = plot_opts_list[i]
            plt.plot(xgrid_here, upper_vals, plot_opts)
            plot_opts['label'] = ''
            plt.plot(xgrid_here, lower_vals, plot_opts)
            plt.fill_between(xgrid_here, lower_vals,upper_vals,fill_opts_list[i])    
    
    print("All mass-radius EOS rendered.")
    plt.xlabel("$R$ [km]")
    plt.ylabel(r"$M$ [M$_{\odot}$]")
    plt.xlim(8,18)
    plt.ylim(mass_range[0],mass_range[1])
    if opts.show_grid:
        plt.grid(True)
    if opts.eos_label:
        plt.legend(frameon=False)
    dpi_base=200
    res_base = 4*dpi_base
    plt.tight_layout()
    if opts.plot_mr_name:
        save_name = opts.plot_mr_name
    else:
        save_name = "EOS_MRplot"
        for e in opts.eos_file:
            save_name += "_"+e.split("/")[-1].split(".")[0]
    plt.savefig(save_name+fig_extension,dpi=res_base)
    plt.show()
    print("EOS mass-radius figure saved as "+save_name+fig_extension)


