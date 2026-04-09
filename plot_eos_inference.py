# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 02:00:39 2026

@author: marce
"""

#make an eos? plz?

#! /usr/bin/env python

import numpy as np
import argparse
#import RIFT.plot_utilities.EOSPlotUtilities as eosplot

# Author: Askold Vilkha (https://github.com/askoldvilkha), av7683@rit.edu, askoldvilkha@gmail.com

# This script is used to plot the results of the tabular EOS inference. 
# Minimal input is the path to the tabular EOS file and posterior samples files the user wants to have on the plot.
# The user will have to specify which plots to make by setting the corresponding flags to True.
# The script has a functionality to choose custom labels and colors for the posterior samples.
# If multiple tabular EOS files are provided, the user will have to specify which ones to plot and to use for the posterior samples.

#import matplotlib.pyplot as plt
import RIFT.physics.EOSManager as EOSManager
#from natsort import natsorted
import warnings
import sys

import ast
import textwrap

import RIFT.plot_utilities.TabularEOSPlotUtilities as tabplot
#import RIFT.plot_utilities.EOSPlotUtilities as eosplot
import lalsimulation as lalsim

try:
    import matplotlib
    print(" Matplotlib backend ", matplotlib.get_backend())
    if matplotlib.get_backend() == 'agg':
        fig_extension = '.png'
        bNoInteractivePlots=True
    else:
        matplotlib.use('agg')
        fig_extension = '.png'
        bNoInteractivePlots =True
    from matplotlib import pyplot as plt
    bNoPlots=False
except:
    print(" Error setting backend")

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()#description = argparse_help_dict['help'])

# basic arguments
parser.add_argument('--eos-file', action = 'append')#, help = argparse_help_dict['tabular-eos-file'], required = True)

opts = parser.parse_args()

#NOTE: ONLY THESE EOS_PARAMS HANDLED CURRENTLY: spectral, cs_spectral, PP
def generate_eos(eos_line, eos_headers, eos_param="spectral"):
    print("Creating EOS object of type",eos_param,"using given data line.")
    
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
        #TODO: may need to handle re-sorting for spectral types if g1 not first, as a precaution
    
    #Better than CIP, for sure...
    spec_param_array = eos_line 
    spec_params ={}

    for i in range(len(eos_names)):
        spec_params[eos_names[i]]=spec_param_array[i]
    print("EOS data:\n",spec_params)
    
    #try: #test code
    #    import RIFT.physics.EOSManager as EOSManager
    #except:
    #    print("-- ERROR: could not import EOSManager. --") #test code, only on local machine
        #return None
    
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
        print("=====\n FAILSTATE 3: EOS CREATION FAILED. Exception:\n     ",type(e),":",e,"\n EXITING.\n=====")
        sys.exit(64) #special exit code for shell_wrapper_cip.sh to detect (hopefully)!
        #print(" WARNING: RETURNED EOS OBJECT WILL BE",type(eos_base),"!\n=====")
    
    return eos_base


my_eos = None
def initialize_one_eos():
    #This gets one line of data; it will also get the names for each column, after header:
    dat = np.genfromtxt(opts.eos_file,names=True)[0]   # Parse file for them, to reduce need for burden parsing, and avoid burden/confusion.
    #all_params = np.loadtxt(opts.eos_file,names=True)[0]
    
    param_names = dat.dtype.names #separate out the names from the data
    all_params = dat.view((float, len(param_names)))
    print(all_params)
    #args_init = {'input_line' : dat_as_array, 'param_names':param_names}#, 'cip_param_names':coord_names}  # pass the recordarray broken into parts, for convenience
    
    #dat_orig_names = param_names[2:] #Adapted from ye old example_gaussian.py
    #print("Original field names:", dat_orig_names)
    
    #supplemental_init = initialize_me #getattr(external_likelihood_module, 'initialize_me') #find initialize_me()
    #supplemental_init(**args_init) 
    eos_names = []
    eos_dat = []
    pop_params = []
    pop_params_names = [] #yes this is literally just for the one print statement
    pop_params_lib = ['m1','m2','sig'] #can be added to for other populations
    for i in param_names[2:]: #should be anything past lnL, sig_lnL
        if i in pop_params_lib:
            pop_params_names.append(i)
            pop_params.append(all_params[param_names.index(i)])
        else: #anything that isn't m1, m2, sig
            eos_names.append(i)
            eos_dat.append(all_params[param_names.index(i)])
    
    global my_eos
    #global constraint_mmax_factor
    if len(eos_names) > 0:
        my_eos = generate_eos(eos_dat, eos_names)
        #constraint_mmax_factor = mmax_constraint(eos.mMaxMsun) 
        #print("m_max constraint factor for this EOS:",constraint_mmax_factor)
    else:
        print("ERROR: Unable to create EOS object.") #Likely a no-CIP-test route only
        my_eos = None

#my_eos = EOSManager.EOSLALSimulation('SLy')
#eos_base = EOSManager.EOSLindblomSpectral(name=eos_name,spec_params=spec_params,use_lal_spec_eos=True)

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


initialize_one_eos()

if my_eos is None:
    print("EOS creation failed; exiting.")
    sys.exit(0)
print("EOS initialized.")

#fig_base= None
#fig_base = 
render_eos(my_eos.eos,'rest_mass_density', 'pressure')
print("EOS rendered.")

dpi_base=200
res_base = 4*dpi_base
plt.savefig("test_eos_plot"+fig_extension,dpi=res_base)
print("Figure saved, supposedly.")

sys.exit(0)

# store long help messages in a dictionary to avoid cluttering the parser code below
argparse_help_dict = {
    'help': textwrap.dedent('''\
    This script is used to plot the results of non-tabular EOS inference obtained with RIFT or HyperPipe code.
    The user can plot pressure vs. density and mass vs. radius plots for the EOS priors and posterior samples.
    The user can specify the EOS files (?), posterior samples files, labels, and colors for the plots.
    The user has to provide the path to the tabular EOS files and the posterior samples files.
    Basic usage: 
    plot_tabular_eos_inference.py --tabular-eos-file <path_to_tabular_eos_file> --tabular-eos-label <tabular_eos_file_label> --posterior-file <path_to_posterior_samples_file> 
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
parser.add_argument('--posterior-file', action = 'append', help = 'Path to the posterior samples file. If none are provdided, only priors will be plotted.')
parser.add_argument('--plot-p-vs-rho', action = 'store_true', help = 'Plot pressure vs. density')
parser.add_argument('--plot-m-vs-r', action = 'store_true', help = 'Plot mass vs. radius')
parser.add_argument('--tabular-eos-label', action = 'append', help = 'Label for the tabular EOS file')
parser.add_argument('--posterior-label', action = 'append', help = 'Label for the posterior samples file')
parser.add_argument('--color', action = 'append', help = 'Colors for the plot. If not provided, colors will be chosen automatically')
parser.add_argument('--use-bgcgb-colormap', action = 'store_true', help = 'Use the BlackGreyCyanGreenBlue colormap for the plots')
parser.add_argument('--verbose', action = 'store_true', help = 'Print information on the progress of the code')
parser.add_argument('--plot-p-vs-rho-title', action = 'store', help = 'Title for the pressure vs. density plot')
parser.add_argument('--plot-m-vs-r-title', action = 'store', help = 'Title for the mass vs. radius plot')

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


