#! /usr/bin/env python

'''
Adapted from Askold Vilkha's code for plotting tabular EOS inference results.
(https://github.com/askoldvilkha, av7683@rit.edu, askoldvilkha@gmail.com)
This version plots for regular spectral EOS, not tabular.
It also supersedes the RIFT.plot_utilities.EOSPlotUtilities code, reusing and
improving it locally. 

# This script is used to plot the results of the tabular EOS inference. 
# Minimal input is the path to the tabular EOS file and posterior samples files the user wants to have on the plot.
# The user will have to specify which plots to make by setting the corresponding flags to True.
# The script has a functionality to choose custom labels and colors for the posterior samples.
# If multiple tabular EOS files are provided, the user will have to specify which ones to plot and to use for the posterior samples.
'''
import numpy as np
import argparse
import warnings
import sys
import ast
import textwrap

import RIFT.physics.EOSManager as EOSManager
import RIFT.plot_utilities.TabularEOSPlotUtilities as tabplot
import RIFT.plot_utilities.EOSPlotUtilities as eosplot
import lalsimulation as lalsim

try:
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

parser = argparse.ArgumentParser()#description = argparse_help_dict['help'])

# basic arguments
parser.add_argument('--eos-file', action = 'append',help='The last shall be first (top layer), and the first shall be last (bottom)')#, help = argparse_help_dict['tabular-eos-file'], required = True)
parser.add_argument('--num-eos',default=1,type=int,help="Number of lines of posterior file to plot EOS curves for (starts at 0). Enter 0 to do all lines (use draw_eos with this)")
#parser.add_argument('--posterior-file', action = 'append', help = 'Path to the posterior samples file(s). If none are provdided, only priors will be plotted.')
parser.add_argument('--plot-p-vs-rho', action = 'store_true', help = 'Plot pressure vs. density')
parser.add_argument('--plot-m-vs-r', action = 'store_true', help = 'Plot mass vs. radius')
parser.add_argument('--eos-label', action = 'append', help = 'Label(s) for the EOS file(s) - order must be the same as eos-file option')
#parser.add_argument('--posterior-label', action = 'append', help = 'Label for the posterior samples file')
parser.add_argument('--color', action = 'append', help = 'Line colors for the plot. If not provided, colors will be chosen automatically')
parser.add_argument('--use-bgcgb-colormap', action = 'store_true', help = 'Use the BlackGreyCyanGreenBlue colormap for the plots')
parser.add_argument('--verbose', action = 'store_true', help = 'Print information on the progress of the code')
parser.add_argument('--plot-p-vs-rho-title', action = 'store', help = 'Title for the pressure vs. density plot')
parser.add_argument('--plot-m-vs-r-title', action = 'store', help = 'Title for the mass vs. radius plot')
parser.add_argument('--render-eos-objects',action='store_true',help='I do not know what this was for')
parser.add_argument('--render-eos-files',action='store_true',help='!! Always use this with posterior files !!')
parser.add_argument('--fill-color',action='append',help="Fill colors for region between percentiles; leave blank for no fill")
parser.add_argument('--draw-eos',default=500,help="Number of random EOS to use from file, if file length > provided value")


opts = parser.parse_args()

posterior_header = ""
print("posterior_header is:",posterior_header)

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
            #global posterior_header
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
        #sys.exit(64) #special exit code for shell_wrapper_cip.sh to detect (hopefully)!
        eos_base = None
    
    return eos_base


#taken straight from EOSPlotUtilities
def render_eos(eos, xvar='energy_density', yvar='pressure',units='cgs',npts=100,label=None,logscale=True,verbose=False,**kwargs):

    min_pseudo_enthalpy = 0.005
    max_pseudo_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos)
    hvals = max_pseudo_enthalpy* 10**np.linspace( np.log10(min_pseudo_enthalpy/max_pseudo_enthalpy),  -1e-4,num=npts)
    if verbose:
        print(hvals,min_pseudo_enthalpy, max_pseudo_enthalpy)

    qry = EOSManager.QueryLS_EOS(eos)

    xvals = qry.extract_param(xvar,hvals)
    yvals = qry.extract_param(yvar,hvals)
    if verbose:
        print(np.c_[xvals,yvals])
        
    
    if logscale:
        plt.loglog(xvals, yvals,label=label,**kwargs)
    else:
        plt.plot(xvals, yvals,label=label,**kwargs)
    return None


#ported from EOSPlotUtilities for modifications
def render_eos_list_quantiles_vs(eos_list, quantile_bounds=None, xvar='energy_density', xgrid=None,yvar='pressure', units='cgs',use_monotonic=True,use_log=True,return_outvals=False,input_outvals=None,show_traces=False,plot_kwargs={},fill_kwargs={},plot_label=""):
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
        
    plt.plot(xgrid_here, upper_vals, label=plot_label, **plot_kwargs) #this is the change
    plt.plot(xgrid_here, lower_vals, **plot_kwargs)
    plt.fill_between(xgrid_here, lower_vals,upper_vals,**fill_kwargs)
    if return_outvals:
        return outvals_here #bug in original code: undefined variable "outvals"
    return None


my_eos_list = None
def initialize_eos(eos_file):
    #This gets 1+ lines of data; it will also get the names for each column, after header:
    if opts.num_eos == 0:
        dat = np.genfromtxt(eos_file,names=True)
    else:
        dat = np.genfromtxt(eos_file,names=True)[:opts.num_eos]
    
    param_names = dat.dtype.names #separate out the names from the data
    all_params = dat.view((float, len(param_names)))
    
    if len(all_params) > int(opts.draw_eos):
        lines_to_use = np.choice(len(all_params),size=int(opts.draw_eos),replace=False)
        print("Drawing",len(lines_to_use),"random lines from this file.")
        all_params = all_params[lines_to_use]
        print("Length of dat is now:",len(all_params))
    
    print("First line:",all_params[0])

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
    
    global my_eos_list
    my_eos_list = []
    if len(eos_names) > 0:
        for i in np.arange(len(eos_dat)):
            my_eos = generate_eos(eos_dat[i], eos_names)
            if my_eos is None:
                print("  eos line",i,"failed to generate.")
            else:
                my_eos_list.append(my_eos.eos)
    else:
        print("ERROR: No EOS columns found. Unable to create EOS object.")
        my_eos_list = None


#my_eos = EOSManager.EOSLALSimulation('SLy')
#eos_base = EOSManager.EOSLindblomSpectral(name=eos_name,spec_params=spec_params,use_lal_spec_eos=True)
if opts.render_eos_objects: #directly render all eos in provided range using their own axes
    initialize_eos(opts.eos_file[0])
    
    if my_eos_list is None:
        print("All provided EOS parameters failed; exiting.")
        sys.exit(0)
    print("EOS list initialized; total:",len(my_eos_list))
    
    for e in my_eos_list:
        render_eos(e.eos,'rest_mass_density', 'pressure')
    print("All EOS rendered.")
    dpi_base=200
    res_base = 4*dpi_base
    plt.savefig("test_eos_pd_plot"+fig_extension,dpi=res_base)
    print("EOS figure saved.")
    sys.exit(0)
    
elif opts.render_eos_files:
    for i in np.arange(len(opts.eos_file)):        
        initialize_eos(opts.eos_file[i])
        
        if my_eos_list is None:
            print("ERROR: no valid EOSs could be created for this file.")
            continue 
        else:
            print("EOS list initialized; total:",len(my_eos_list))
            density_grid = 10**np.linspace(14,16,200) #need to choose good range of densities
            
            fill_opts = {}
            if opts.fill_color:
                fill_opts['color'] = opts.fill_color[i]
                fill_opts['alpha'] = 0.1
            else:
                fill_opts['alpha'] = 0.0
                #fill_opts['color'] = (1.0, 1.0, 1.0) #should be white
            
            plot_opts= {}
            if opts.eos_label:
                label_here = opts.eos_label[i]
                #plot_opts['label'] = opts.eos_label[i]
            if opts.color:
                plot_opts['color'] = opts.color[i]
                
            render_eos_list_quantiles_vs(my_eos_list, quantile_bounds=[0.05,0.95], xvar='rest_mass_density', xgrid=density_grid,yvar='pressure',use_log=True,plot_kwargs=plot_opts,fill_kwargs=fill_opts,plot_label=label_here)
        
    print("All EOS rendered.")
    plt.xlabel(r"log$_{10} \rho$ [g cm$^{-3}$]")
    plt.ylabel(r"log$_{10} P$ [dyn cm$^{-2}$]")
    if opts.eos_label:
        plt.legend()
    
else:
    print(" ERROR: no valid rendering option provided. Exiting.")
    sys.exit(0)

dpi_base=200
res_base = 4*dpi_base
save_name = "EOS_PDplot"
for e in opts.eos_file:
    save_name += "_"+e.split("/")[-1].split(".")[0]

plt.savefig(save_name+fig_extension,dpi=res_base)
print("EOS figure saved.")

sys.exit(0)


###############################################################################


# store long help messages in a dictionary to avoid cluttering the parser code below
argparse_help_dict = {
    'help': textwrap.dedent('''\
    This script is used to plot the results of non-tabular EOS inference obtained with RIFT or HyperPipe code.
    The user can plot pressure vs. density and mass vs. radius plots for the EOS posterior samples.
    The user can specify the EOS posterior samples files, labels, and colors for the plots.
    The user has to provide the path to the posterior samples files.
    Basic usage: 
    plot_tabular_eos_inference.py --eos-file <path_to_eos__posterior_file> --tabular-eos-label <tabular_eos_file_label> --posterior-file <path_to_posterior_samples_file> 
    --posterior-label <posterior_samples_file_label> --plot-p-vs-rho --plot-m-vs-r --use-bgcgb-colormap'''),

    'tabular-eos-file': textwrap.dedent('''\
    Path to the tabular EOS file. If multiple files are provided only the first one will be used for EOS posteriors unless specified otherwise by the user. 
    See --posterior-tabular-map for more information.','''),

    'plot-tabular-eos-prior': textwrap.dedent('''\
    Specify which tabular EOS files to plot priors for. 
    Should be specified by the label of the tabular EOS file'''),

    'posterior-tabular-map': textwrap.dedent('''\
    Specify which tabular EOS file to use for the posterior samples. 
    Should be specified by the label of the tabular EOS files and the posterior samples files. 
    The input format should be a dictionary in the form of a string:
    '{'tabular_eos_file_1': ['posterior_file_1', 'posterior_file_2'] 'tabular_eos_file_2': 'posterior_file_3', ...}'. 
    Use either labels instead of the file names.'''),

    'posterior-histogram-range-r': textwrap.dedent('''\
    Set the range for the Radius at the posterior samples histogram plot. 
    The input format should be a list of two numbers defining a range for the Radius. (e.g. [10, 15])'''),

    'posterior-histogram-range-lambda': textwrap.dedent('''\
    Set the range for the Tidal Deformability Lambda at the posterior samples histogram plot.
    The input format should be a list of two numbers defining a range for the Tidal Deformability Lambda. (e.g. [400, 1600])''')
}

parser = argparse.ArgumentParser(description = argparse_help_dict['help'])

# basic arguments
parser.add_argument('--tabular-eos-file', action = 'append', help = argparse_help_dict['tabular-eos-file'], required = True)


# extra arguments for more advanced plotting (plot posteriors for multiple tabular EOS files, plot tabular EOS priors, etc.)
parser.add_argument('--plot-tabular-eos-prior', action = 'append', help = argparse_help_dict['plot-tabular-eos-prior'])
parser.add_argument('--posterior-tabular-map', action = 'store', help = argparse_help_dict['posterior-tabular-map'])
parser.add_argument('--plot-posterior-histogram', action = 'store_true', help = 'Plot the posterior samples histogram')
parser.add_argument('--plot-lambda-tilde-ratio', action = 'store_true', help = 'Plot the lambda_tilde vs ordering statistics S ratio')
parser.add_argument('--posterior-histogram-range-r', action = 'store', help = 'Range for Radius at the posterior samples histogram plot')
parser.add_argument('--posterior-histogram-range-lambda', action = 'store', help = 'Range for Tidal Deformability Lambda at the posterior samples histogram plot')

args = parser.parse_args()

tabular_eos_labels = [None] * len(args.tabular_eos_file)

# the code can operate with less labels than files, it will use the default labels for the rest of the files
if args.tabular_eos_label is not None:
    if len(args.tabular_eos_label) < len(args.tabular_eos_file):
        warnings.warn('Number of tabular EOS labels is less than the number of tabular EOS files. The code will use the default labels after the last provided label.')
    elif len(args.tabular_eos_label) > len(args.tabular_eos_file):
        raise ValueError('Number of tabular EOS labels is greater than the number of tabular EOS files. Please check if you have provided the correct number of labels or files.')

    tabular_eos_labels[:len(args.tabular_eos_label)] = args.tabular_eos_label # fill in the provided labels, use the default labels for the rest

    if args.verbose:
        print('\nChecked the tabular EOS file labels, proceeding with the data loading...')

# posterior histogram and lambda_tilde ratio plots require saving the EOS manager object and/or the posterior samples (not just eos_indx column)
save_eos_manager = False
save_posterior_samples = False
if args.plot_posterior_histogram or args.plot_lambda_tilde_ratio:
    if args.posterior_file is None:
        raise ValueError('You have chosen to plot the posterior samples histogram or lambda_tilde ratio without providing the posterior samples files. Please provide the posterior samples files.')
    save_eos_manager = True
if args.plot_lambda_tilde_ratio:
    save_posterior_samples = True

# load the tabular EOS files
tabular_eos_data = {}
for i, tabular_eos_file in enumerate(args.tabular_eos_file):
    eos_data_i = tabplot.EOS_data_loader(tabular_eos_file, tabular_eos_labels[i], args.verbose, save_eos_manager)
    eos_data_i_label = eos_data_i['data_label'] # if the label was not provided, the EOS_data_loader function will assign a default label
    tabular_eos_data[eos_data_i_label] = eos_data_i
    tabular_eos_labels[i] = eos_data_i_label # update the label in case the default label was assigned

if args.verbose:
    print('\nLoaded the tabular EOS data.')
    for i in range(len(tabular_eos_labels)):
        print(f'Tabular EOS file: {args.tabular_eos_file[i]} with label: {tabular_eos_labels[i]}')

if args.posterior_file is not None:
    priors_labels = []
    posterior_samples_labels = [None] * len(args.posterior_file)

    if args.posterior_label is not None:
        if len(args.posterior_label) < len(args.posterior_file):
            warnings.warn('Number of posterior samples labels is less than the number of posterior samples files. The code will use the default labels after the last provided label.')
        elif len(args.posterior_label) > len(args.posterior_file):
            raise ValueError('Number of posterior samples labels is greater than the number of posterior samples files. Please check if you have provided the correct number of labels or files.')

        posterior_samples_labels[:len(args.posterior_label)] = args.posterior_label 
    
        if args.verbose:
            print('\nChecked the posterior samples labels, proceeding with the data loading...')

    # load the posterior samples files
    posterior_samples_data = {}
    for i, posterior_file in enumerate(args.posterior_file):
        posterior_data_i = tabplot.posterior_data_loader(posterior_file, posterior_samples_labels[i], args.verbose, save_posterior_samples)
        posterior_data_i_label = posterior_data_i['data_label']
        posterior_samples_data[posterior_data_i_label] = posterior_data_i
        posterior_samples_labels[i] = posterior_data_i_label

    if args.verbose:
        print('\nLoaded the posterior samples data.')
        for i in range(len(posterior_samples_labels)):
            print(f'Posterior samples file: {args.posterior_file[i]} with label: {posterior_samples_labels[i]}')

    # make the posterior-tabular map if not provided by the user
    if args.posterior_tabular_map is None:
        # if the user did not provide the posterior-tabular map, the code will only use first tabular EOS file for the posterior samples
        posterior_tabular_map = {tabular_eos_labels[0]: posterior_samples_labels}

    # check if the map is provided in the correct format and all the labels are valid
    else:
        if args.verbose:
            print('\nChecking the posterior-tabular map...')
        posterior_tabular_map = ast.literal_eval(args.posterior_tabular_map)
        if not isinstance(posterior_tabular_map, dict):
            raise ValueError('The posterior-tabular map should be a dictionary. Please check the format of the input.')
        for tabular_label in posterior_tabular_map.keys():
            if tabular_label not in tabular_eos_labels:
                raise ValueError(f'Tabular EOS label: {tabular_label} from your posterior-tabular map is not found in the tabular EOS labels. Please check the labels you provided.')
            if isinstance(posterior_tabular_map[tabular_label], list):
                for posterior_label in posterior_tabular_map[tabular_label]:
                    if posterior_label not in posterior_samples_labels:
                        raise ValueError(f'Posterior samples label: {posterior_label} from your posterior-tabular map is not found in the posterior samples labels. Please check the labels you provided.')
            else:
                if posterior_tabular_map[tabular_label] not in posterior_samples_labels:
                    raise ValueError(f'Posterior samples label: {posterior_tabular_map[tabular_label]} from your posterior-tabular map is not found in the posterior samples labels. Please check the labels you provided.')

    if args.verbose:
        print('\nCreated the posterior-tabular map.')
        for tabular_label in posterior_tabular_map.keys():
            print(f'Tabular EOS label: {tabular_label} with posterior samples labels: {posterior_tabular_map[tabular_label]}')
        print('\nMoving on to priors...')
else:
    posterior_samples_data = None
    posterior_tabular_map = None
    priors_labels = tabular_eos_labels # plot priors for all tabular EOS files if no posterior samples are provided
    warnings.warn('No posterior samples files provided. The code will only plot the tabular EOS priors.')
    
# if the user specified the tabular EOS files to plot priors for, the code will plot priors for those files
if args.plot_tabular_eos_prior is not None:
    for tabular_label in args.plot_tabular_eos_prior:
        if tabular_label not in tabular_eos_labels:
            raise ValueError(f'Tabular EOS label for prior plot: {tabular_label} is not found in the tabular EOS labels. Please check the labels you provided.')
    priors_labels = args.plot_tabular_eos_prior
elif len(tabular_eos_labels) > 1 and posterior_samples_data is not None:
    priors_labels = tabular_eos_labels[1:] # plot priors for all tabular EOS files except the first one

if args.verbose:
    print('\nChecked the tabular EOS labels for the priors.')
    for i in range(len(priors_labels)):
        print(f'Priors will be plotted for the tabular EOS file with label: {priors_labels[i]}')
    print('\nLinking the tabular EOS data with the posterior samples data according to the posterior-tabular map...')

plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12

def EOS_plotter(args, tabular_eos_data: dict, posterior_samples_data: dict, posterior_tabular_map: dict, priors_labels: list, plot_type: str) -> None:
    """
    Function to plot the EOS inference results for the tabular EOS priors and posterior samples.
    
    Parameters:
    ----------
    args : argparse.Namespace
        The argparse namespace object with the script arguments.
    tabular_eos_data : dict
        The dictionary with the tabular EOS data generated earlier in the script. 
        Format: {tabular_label: tabular_eos_data}, tabular_eos_data should be generated by the EOS_data_loader function.
    posterior_samples_data : dict
        The dictionary with the posterior samples data generated earlier in the script. 
        Format: {posterior_label: posterior_samples_data}, posterior_samples_data should be generated by the posterior_data_loader function.
    posterior_tabular_map : dict
        The dictionary with the posterior-tabular map generated earlier in the script. 
        Format: {tabular_label: posterior_samples_label}, posterior_samples_label can be a list of labels or a single string label.
    priors_labels : list
        The list with the tabular EOS labels to plot priors for.
    plot_type : str
        The type of the plot to make. Can be 'pressure_density' or 'mass_radius'.

    Returns:
    -------
    None

    Raises:
    -------
    NOTE!
    Errors will not be raised if the script has not been changed from the initial version. 
    They are here to ensure future modifications do not break the code.

    ValueError
        If the plot type is not 'pressure_density' or 'mass_radius'. 
    ValueError
        If the posterior samples data is provided without the posterior-tabular map or vice versa.
    """
    if plot_type not in ['pressure_density', 'mass_radius']:
        raise ValueError('The plot type should be either pressure_density or mass_radius. Please check the input.')
    if (posterior_samples_data is None) != (posterior_tabular_map is None):
        raise ValueError('The posterior samples data and the posterior-tabular map should be provided together. Please check the input.')

    raw_plot_data = []

    for tabular_label in priors_labels:
        raw_plot_data.append(tabplot.link_eos_data_to_posterior(tabular_eos_data[tabular_label], plot_type))
    
    if args.verbose:
        print(f'\nCollected the data for priors for the {plot_type} plot. Moving on to posteriors...')
    
    # link the tabular EOS data with the posterior samples data according to the posterior-tabular map
    if posterior_samples_data is not None:
        for tabular_label in posterior_tabular_map.keys():
            if isinstance(posterior_tabular_map[tabular_label], list):
                for posterior_label in posterior_tabular_map[tabular_label]:
                    raw_plot_data.append(tabplot.link_eos_data_to_posterior(tabular_eos_data[tabular_label], plot_type, posterior_samples_data[posterior_label]))
            else:
                raw_plot_data.append(tabplot.link_eos_data_to_posterior(tabular_eos_data[tabular_label], plot_type, posterior_samples_data[posterior_tabular_map[tabular_label]]))

        if args.verbose:
            print(f'\nCollected the data for posterior samples for the {plot_type} plot. Processing...')
    
    # choose the colors for the plots
    colors = [None] * len(raw_plot_data)
    if args.color is not None:
        if len(args.color) < len(raw_plot_data):
            warnings.warn('Number of colors is less than the number of posterior samples files. The code will use the default colors for the rest of the files.')
            colors[:len(args.color)] = args.color
        else: 
            colors = args.color 
    if args.use_bgcgb_colormap:
        colors[:5] = ['black', 'grey', 'cyan', 'green', 'blue']
        if args.color is not None:
            warnings.warn('You have chosen to use the BlackGreyCyanGreenBlue colormap. The code will ignore the first 5 colors you provided.')
    if args.verbose and any(colors is not None for c in colors):
        print(f'\nThe colors for the plots are: {colors}. Any of the None values will be replaced with the default colors. (tab10 colormap)')
    
    plot_data = []
    if plot_type == 'pressure_density':
        for i, raw_data in enumerate(raw_plot_data):
            plot_data.append(tabplot.pressure_density_plot_data_gen(**raw_data, custom_color = colors[i]))
        
        if args.verbose:
            print(f'\nProcessed the data for the {plot_type} plot. Plotting...')

        tabplot.pressure_density_plot(*plot_data, title = args.plot_p_vs_rho_title)
    
    elif plot_type == 'mass_radius':
        for i, raw_data in enumerate(raw_plot_data):
            plot_data.append(tabplot.mass_radius_plot_data_gen(**raw_data, custom_color = colors[i]))

        if args.verbose:
            print(f'\nProcessed the data for the {plot_type} plot. Plotting...')

        tabplot.mass_radius_plot(*plot_data, title = args.plot_m_vs_r_title)

    return None
    
# plot the P vs rho EOS inference
if args.plot_p_vs_rho:
    EOS_plotter(args, tabular_eos_data, posterior_samples_data, posterior_tabular_map, priors_labels, 'pressure_density')

# plot the M vs R EOS inference
if args.plot_m_vs_r:
    EOS_plotter(args, tabular_eos_data, posterior_samples_data, posterior_tabular_map, priors_labels, 'mass_radius')

# plot the posterior samples histogram
if args.plot_posterior_histogram:
    
    posterior_hist_data = []
    for tabular_label in posterior_tabular_map.keys():
        if isinstance(posterior_tabular_map[tabular_label], list):
            for posterior_label in posterior_tabular_map[tabular_label]:
                posteriors_eos = posterior_samples_data[posterior_label]['posterior_samples_eos']
                eos_manager = tabular_eos_data[tabular_label]['eos_manager']
                posterior_hist_data.append(tabplot.posterior_hist_data_gen(posteriors_eos, eos_manager, posterior_label))
        else:
            posteriors_eos = posterior_samples_data[posterior_tabular_map[tabular_label]]['posterior_samples_eos']
            eos_manager = tabular_eos_data[tabular_label]['eos_manager']
            posterior_label = posterior_tabular_map[tabular_label]
            posterior_hist_data.append(tabplot.posterior_hist_data_gen(posteriors_eos, eos_manager, posterior_label))
    
    if args.verbose:
        print('\nProcessed the data for the posterior samples histogram. Plotting...')


    r_lim = (8, 17)
    lambda_lim = (0, 1000)
    if args.posterior_histogram_range_r is not None:
        r_lim_input = ast.literal_eval(args.posterior_histogram_range_r)
        r_lim = (r_lim_input[0], r_lim_input[1])
    if args.posterior_histogram_range_lambda is not None:
        lambda_lim_input = ast.literal_eval(args.posterior_histogram_range_lambda)
        lambda_lim = (lambda_lim_input[0], lambda_lim_input[1])

    tabplot.posterior_hist_plot(*posterior_hist_data, r_lim = r_lim, lambda_lim = lambda_lim)

# plot the lambda_tilde vs S ratio
if args.plot_lambda_tilde_ratio:
    
    lambda_tilde_ratio_data = []
    for tabular_label in posterior_tabular_map.keys():
        if isinstance(posterior_tabular_map[tabular_label], list):
            for posterior_label in posterior_tabular_map[tabular_label]:
                posteriors_samples_all = posterior_samples_data[posterior_label]['posterior_samples_all']
                eos_manager = tabular_eos_data[tabular_label]['eos_manager']
                lambda_tilde_ratio_data.append(tabplot.LambdaTilderatio_data_gen(posteriors_samples_all, eos_manager, posterior_label))
        else:
            posteriors_samples_all = posterior_samples_data[posterior_tabular_map[tabular_label]]['posterior_samples_all']
            eos_manager = tabular_eos_data[tabular_label]['eos_manager']
            posterior_label = posterior_tabular_map[tabular_label]
            lambda_tilde_ratio_data.append(tabplot.LambdaTilderatio_data_gen(posteriors_samples_all, eos_manager, posterior_label))
    
    if args.verbose:
        print('\nProcessed the data for the lambda_tilde vs S ratio plot. Plotting...')
    
    tabplot.LambdaTilderatio_plot(*lambda_tilde_ratio_data)
    tabplot.LambdaTilderatio_plot(*lambda_tilde_ratio_data, ylim = (0.8, 1.2))


