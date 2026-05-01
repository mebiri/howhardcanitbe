#! /usr/bin/env python
#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
'''
originally reading_NICER_MR.py by Atul Kedia (2025)
for original hyperpipe tests as described in https://arxiv.org/pdf/2405.17326
adapted by M.E. to work with my version of (read: problems with) hyperpipe
'''
import numpy as np
import argparse
import lalsimulation as lalsim
import pyreprimand as pyr
from scipy.stats import norm, multivariate_normal

'''
imports used later:
    import lal
    from RIFT.physics import EOSManager
-----
These are the imports also in eos_handler.py:
import RIFT.lalsimutils as lalsimutils
from RIFT.physics import EOSManager as EOSManager

from scipy.integrate import nquad
#import EOS_param as ep
import os
#from EOSPlotUtilities import render_eos
import lal
from scipy.stats import multivariate_normal
'''

parser = argparse.ArgumentParser()
#exclusive here
parser.add_argument("--save-pyr",action='store_true',help="If provided, save the pyr structure files")
parser.add_argument('--causal-spectral', action='store_true', help="Enable to use new causal spectral parameterization.")
parser.add_argument("--uniform-c-density-prior",action='store_true')
parser.add_argument("--chunk-save",default=True,help="Save all output lines to one file instead of 1 file per line.")
parser.add_argument("--save-all-files",action='store_true',help="If enabled, saves data files (x2) generated for each EOS line")
#data & constraints
parser.add_argument('--obs-file', action='append', help="REQUIRED: Filenames (NOT PATHS) for observations used for likelihood calculation and plots generated here.")#Supported: j0740 j1731 j0030 j0437")
parser.add_argument('--j0437-fiducial',action='store_true',help="Include to use hardcoded J0347 vals instead of data. For replicating previous results.")
parser.add_argument('--do-max-mass',action='store_true',help="Enable to include max mass constraint factor from J0740, J0348, & J1614 in likelihood")
parser.add_argument('--do-sym-energy',action='store_true',help="Enable to include nuclear symmetry energy constraint factor in likelihood")

#filenames
parser.add_argument("--fname",type=str,help="REQUIRED: Dummy argument required by RIFT API")
parser.add_argument('--fname-output-integral', type=str, help="REQUIRED: Output eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
parser.add_argument('--fname-output-samples', type=str, help="REQUIRED: NEVER USED, but is enabled. Output eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
parser.add_argument('--observation-dir', type=str,default='', help="REQUIRED: Pathname to directory containing data files")
parser.add_argument('--recycle-reprimand-objects-from', type=str, help="Pathname to reprimand objects previously made.")

#eos management
parser.add_argument('--using-eos', type=str, help="REQUIRED: Send eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
parser.add_argument('--using-eos-index',type=int, help="REQUIRED: Line number for single calculation, starting index for multi-line calculation")
parser.add_argument('--n-events-to-analyze',type=int, default=1,help="REQUIRED: Number of EOS lines to eval at once, default 1; >1 supported.")

#output management
parser.add_argument("--conforming-output-name",action='store_true',help="Needed for hyperpipe")
parser.add_argument('--plot', action='store_true', help="Enable to plot resultant M-R and Gaussians. Only for n_events = 1")
parser.add_argument('--outdir', default=".", type=str, help="[Ignored] Output eos file directory.")
parser.add_argument('--outdir-clean', type=str, help="[Ignored] Delete CleaOutile direc before starting the runtory.")

#legacy args
#parser.add_argument('--eos_start_index',type=int, help="Line number from where to start the likelihood calculation.")
#parser.add_argument('--eos_end_index',type=int, help="Line number which needs likelihood for which needs to be evaluated.")
#parser.add_argument('--j0740', action='store_true', help="Legacy option, preferably use 'observations' instead. Enable to include J0740 max mass constraint in likelihood.")
#parser.add_argument('--j0348', action='store_true', help="Legacy option, preferably use 'observations' instead. Enable to include J0348 max mass constraint in likelihood.")
#parser.add_argument('--j1614', action='store_true', help="Legacy option, preferably use 'observations' instead. Enable to include J1614 max mass constraint in likelihood.")
#parser.add_argument('--gw170817', action='store_true', help="Legacy option, preferably use 'observations' instead. Enable to include GW170817 max mass constraint in likelihood.")
#parser.add_argument('--j0030', action='store_true', help="Legacy option, preferably use 'observations' instead. Enable to include NICER-J0030 posterior in likelihood evaluation.")
#parser.add_argument('--j1731', action='store_true', help="Legacy option, preferably use 'observations' instead. Enable to include HESS-J1731 posterior in likelihood evaluation.")

opts = parser.parse_args()

#ensure using_eos_index valid for eos file length (same check in CIP & PLE)
if opts.using_eos and opts.using_eos.startswith('file:') and not(opts.using_eos_index is None):
    fname = opts.using_eos.replace('file:', '')
    try:
        dat = np.loadtxt(fname)[opts.using_eos_index]
    except Exception as e:
        import sys
        print(" Fail: EOS index out of range:\n   ",e)
        sys.exit(0)

#Probably don't want this stuff, since outdir shared with CIP & PLE...
# =============================================================================
# if opts.outdir_clean:
#     import shutil
#     try: shutil.rmtree(opts.outdir)
#     except: pass
#     del shutil
# elif opts.outdir is None:
#     opts.outdir = "."
# 
# from pathlib import Path
# Path(opts.outdir).mkdir(parents=True, exist_ok=True)
# del Path
# =============================================================================

#FOR NOW, EXIT IF NO NICER DATA PROVIDED
#may want to run w/ no NICER, just j0437_fiducial, m_max, sym. NRG, for testing?
if opts.obs_file is None: 
    print("ERROR: no observational data provided. Exiting.")
    import sys
    sys.exit(0)
observations = opts.obs_file

#if opts.j0740: observations.append('j0740')
#if opts.j0348: observations.append('j0348')
#if opts.j1614: observations.append('j1614')
#if opts.j0030: observations.append('j0030')
#if opts.j1731: observations.append('j1731')
#if opts.gw170817: observations.append('gw170817')


def gaussian_distribution(data):
    '''
    0th Column: Radii, 1st Column: Masses
    '''
    mn = np.mean(data,axis=0)
    cov = np.cov(data.T)
    a, b = 0,1
    
    x, y = np.mgrid[min(data[:,a]):max(data[:,a]):.01, min(data[:,b]):max(data[:,b]):.01]
    #pos = np.dstack((x, y))
    #2D normal dist, axes mass vs. radius
    rv = multivariate_normal([mn[a], mn[b]], [[cov[a,a], cov[a,b]], [cov[a,b], cov[b,b]]])
    
    return mn, cov, rv


def mMax_likelihood_for_EOS(M):
    '''
    Distribution taken from DiCo Science 2020: https://arxiv.org/abs/2002.11355
    '''
    m_J0740, sigma_J0740 = 2.14, 0.1
    m_J0348, sigma_J0348 = 2.01, 0.04
    m_J1614, sigma_J1614 = 1.937, 0.014 #NOTE: doesn't match ref paper (1.908, 0.016)
    #m_GW170817, sigma_GW170817 = 2.16, 0.17
    
    L = 1
    L *= norm(loc=m_J0740, scale=sigma_J0740).cdf(M)/sigma_J0740
    L *= norm(loc=m_J0348, scale=sigma_J0348).cdf(M)/sigma_J0348
    L *= norm(loc=m_J1614, scale=sigma_J1614).cdf(M)/sigma_J1614
    #if 'gw170817' in observations : L *= (1-norm(loc=m_GW170817, scale=sigma_GW170817).cdf(M))/sigma_GW170817
    
    return L


def likelihood_symmetry_energy(param_dict):
    '''
    Distribution taken from Mroczek, Miller, Noronha-Hostler, Yunes 2023: https://arxiv.org/abs/2309.02345
    '''
    import lal #import only used here
    #L = 1 #consistent but useless
    #lal.MP_SI
    S_0, sigma_s = 32, 2
    nsat = 2.7e14
    
    E_nsat = np.interp(nsat, param_dict['rest_mass_density'], param_dict['energy_density'])
    
    S_in_MeV = (E_nsat*lal.C_SI**2/(lal.QE_SI*1e48))/(nsat/(lal.MP_SI*1e3*1e39)) - 923.6
    
    #L *= norm(loc=S_0, scale=sigma_s).pdf(S_in_MeV)
    return norm(loc=S_0, scale=sigma_s).pdf(S_in_MeV)


#eos_handler.py
#############################################################
################# Make EOS from EOSManager and RePrimAnd ####
#############################################################
def make_EOS_TOV_from_EOSManager(eos, causal = True, make_monotonic_causal = True, TOV = True, return_eos_object = False):
    '''
    Making EOS elements
    '''
    from RIFT.physics import EOSManager
    spec_params = {'gamma1': eos[0].astype(float),
                   'gamma2': eos[1].astype(float),
                   'gamma3': eos[2].astype(float),
                   'gamma4': eos[3].astype(float),
                   'gamma0': 5.3716e32,  # picked from LALSimNeutronStarEOSSpectralDecomposition.c
                   'epsilon0' : 1e14, # 1.1555e35 / c**2? ~ 0.5nsat
                   'xmax' : 50.0}
    if causal:
        neweos = EOSManager.EOSLindblomSpectralSoundSpeedVersusPressure(spec_params = spec_params)
    
    else:
        spec_params = {'gamma1': eos[0].astype(float),
                       'gamma2': eos[1].astype(float),
                       'gamma3': eos[2].astype(float),
                       'gamma4': eos[3].astype(float),
                       'gamma0': 5.3716e32,  # picked from LALSimNeutronStarEOSSpectralDecomposition.c
                       'epsilon0' : 1e14, # 1.1555e35 / c**2? ~ 0.5nsat
                       'xmax' : 50.0}
        
        neweos = EOSManager.EOSLindblomSpectral(spec_params = spec_params,use_lal_spec_eos=True)
    
    if return_eos_object: return neweos.eos
    
    qry = EOSManager.QueryLS_EOS(neweos.eos)
    min_pseudo_enthalpy = 0.005
    max_pseudo_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(neweos.eos)*0.99
    hvals = max_pseudo_enthalpy* 10**np.linspace( np.log10(min_pseudo_enthalpy/max_pseudo_enthalpy),  0,num=500)
    edens = qry.extract_param('energy_density',hvals)
    press = qry.extract_param('pressure',hvals)
    p_enthalpy = qry.extract_param('pseudo_enthalpy',hvals)
    rho = qry.extract_param('rest_mass_density',hvals)
    cs = qry.extract_param('sound_speed_over_c',hvals)
    
    param_dict = {'pseudo_enthalpy': p_enthalpy,'rest_mass_density': rho,'energy_density': edens,'pressure': press,'sound_speed_over_c': cs}
    
    # Rejecting unphysical jumps in energy_density induced by LALSim's EoS handler. energy_density value cut out at 1e20 in cgs units.
    np_where_no_problem = np.where(param_dict['energy_density']<1e20)
    param_dict = {'pseudo_enthalpy': p_enthalpy[np_where_no_problem],
                  'rest_mass_density': rho[np_where_no_problem],
                  'energy_density': edens[np_where_no_problem],
                  'pressure': press[np_where_no_problem],
                  'sound_speed_over_c': cs[np_where_no_problem]}
    
    # Removing non_monotonic and acausal sections
    if make_monotonic_causal: param_dict = EOSManager.eos_monotonic_parts_and_causal_sound_speed(param_dict)
    
    p_enthalpy = param_dict['pseudo_enthalpy']; rho = param_dict['rest_mass_density']; edens = param_dict['energy_density']; press = param_dict['pressure']; cs = param_dict['sound_speed_over_c']
    
    if TOV :
        reprimand_eos=None
        try:
            reprimand_eos = EOSManager.EOSReprimand(param_dict=param_dict)
        except Exception as e:
            print("ERROR:",e)
            raise Exception("Could not retreive Reprimand EOS from EOS Manager")
        
        return reprimand_eos, param_dict
    
    return param_dict


#############################################################
######################## Physical EOSs ######################
#############################################################
def likelihood_evaluation():    
    #Create 2D Gaussians for each provided NICER data set:
    dat_rv = [] #formerly external_ns_MR_rv
    dat_mn = []   #only used for plotting
    dat_cov = []  #only used for plotting
    dat_list = [] #only used for plotting
    stars = []    #only used for plotting
    for obs in observations:
        print("Retrieving data from file: "+obs)
        fpath = opts.observation_dir + "/" + obs
        if obs[0] == 'J':
            stars.append(obs[:5])
        else:
            stars.append(obs.split(".")[0])
        
        dat_here = np.loadtxt(fpath)
        mn, cov, rv = gaussian_distribution(dat_here)
        dat_rv.append(rv)
        if opts.plot:
            dat_mn.append(mn)
            dat_cov.append(cov)
            dat_list.append(dat_here)
        
        
    #if 'j0030' in observations:
    #    from likelihood_calculator import gaussian_distribution
    #    data_j0030 = np.loadtxt(observation_base_dir + 'J0030/J0030_2spot_RM.txt', skiprows=6)#400,000 lines
    #    mn, cov, rv = gaussian_distribution(data_j0030)
    #    external_ns_MR_rv.append(rv)
    #if 'j1731' in observations:
    #    from likelihood_calculator import gaussian_distribution
    #    data_j1731 = np.loadtxt(observation_base_dir + 'J1731/xray_only_carbatm.txt', skiprows=1)#500,000 lines, approx.
    #    mn2, cov2, rv2 = gaussian_distribution(data_j1731)
    #    external_ns_MR_rv.append(rv2)
    if opts.j0437_fiducial and 'J0437' not in stars: #use fiducial vals to replicate Atul's results
        mn3 = np.array([11.36, 1.418])
        cov3 = np.array([[0.95**2, 0.0], [0.0, 0.037**2]])
        rv3 = multivariate_normal(mn3, cov3)
        dat_rv.append(rv3)
        if opts.plot:
            dat_mn.append(mn3)
            dat_cov.append(cov3)
        #r = 11.36 +- 0.95  # https://meetings.aps.org/Meeting/APR24/Session/P06.2
        #M= 1.418+- 0.037 # both 68%ile
    
    #load EOS data:
    fname = opts.using_eos.replace('file:', '')
    eos_dat = np.genfromtxt(fname,names=True)[opts.using_eos_index:opts.using_eos_index+opts.n_events_to_analyze] #should be 1 line if n_events=1
    param_names = list(eos_dat.dtype.names)
    eoss = eos_dat.view((float, len(param_names)))
    print("EOS dat size: (",len(eoss),len(eoss[0]),")")
    dat_orig_names = param_names[2:] #ignore lnL, sig_lnL - probably don't need this
    print("Original field names ", dat_orig_names)
    
    M_dict = {}
    R_dict = {}
    likelihood_dict = {}
    #for i in np.concatenate((np.arange(0,18),np.arange(19,21),np.arange(22,27),np.arange(28,88),np.arange(89,186))):
    for i in np.arange(len(eoss)):   
        #make EOS object via reprimand & EOSManager:
        if opts.recycle_reprimand_objects_from is not None:
            try:
                print(eoss[i][2:6])
                param_dict = make_EOS_TOV_from_EOSManager(eoss[i][2:6], causal = opts.causal_spectral, TOV = False)
                import lal
                tov_seq_reprimand = pyr.load_star_branch(opts.recycle_reprimand_objects_from+"/reprimand.tov.seq_"+str(i)+".h5", pyr.units.geom_solar(msun_si=lal.MSUN_SI))
                u = tov_seq_reprimand.units_to_SI
                rggm1 = tov_seq_reprimand.range_center_gm1
                gm1 = np.linspace(rggm1.min, rggm1.max, 800)
                _pyr_mrL_dat = np.zeros((len(gm1),2))
                
                _pyr_mrL_dat[:,0] = tov_seq_reprimand.grav_mass_from_center_gm1(gm1) # Mg [Mo]
                _pyr_mrL_dat[:,1] = tov_seq_reprimand.circ_radius_from_center_gm1(gm1)*u.length/1e3 #radius [km]                
            except Exception as e:
                eoss[i,0] = -1e6 # arbitrary low likelihood value for unphysical EoSs - will impact entire hyperpipe line
                eoss[i,1] = 10
                print("line",i,"failed to construct EOS with recycled pyr object:",e)
                continue
        else:
            try:
                print(eoss[i][2:6])
                reprimand_eos, param_dict = make_EOS_TOV_from_EOSManager(eoss[i][2:6], causal = opts.causal_spectral, TOV = True)
            except Exception as e:
                eoss[i,0] = -1e6 # arbitrary low likelihood value for unphysical EoSs
                eoss[i,1] = 10
                print("line",i,"failed to construct EOS:",e)
                continue
            _pyr_mrL_dat = reprimand_eos._pyr_mrL_dat
            tov_seq_reprimand = reprimand_eos.tov_seq_reprimand
        
        #more failmodes from reprimand apparently:
        if tov_seq_reprimand is None:
            print("line",i,"failed: no RePrimAnd TOV sequence returned.")
            eoss[i,0] = -1e6 # arbitrary low likelihood value for unphysical EoSs
            eoss[i,1] = 10
            continue
        if max(_pyr_mrL_dat[:,0])<1.8: #TODO: this seems like an arbitrary restriction, let the m_max constraint handle this
            print("line",i,"failed: max M < 1.8")
            eoss[i,0] = -1e6 # arbitrary low likelihood value for unphysical EoSs
            eoss[i,1] = 10
            continue
        
        #save reprimand objects, if desired
        if opts.save_pyr and not opts.recycle_reprimand_objects_from:
            if opts.fname is None:
                pyr.save_star_branch(opts.outdir+"/"+"reprimand.tov.seq_"+str(i)+".h5", tov_seq_reprimand)
            else:
                pyr.save_star_branch(opts.fname_output_integral+"_reprimand.tov.seq_"+str(i)+".h5", tov_seq_reprimand)
        
        #save data
        #if opts.fname is not None: #always the case on the cluster...
        if opts.save_all_files:
            np.savetxt(opts.fname_output_integral+"_mass-radius_"+str(opts.using_eos_index+i)+".txt", _pyr_mrL_dat[:,:3], fmt = '%10s', header="mass     radius     tidal_deformability")
            pd_header = "pseudo_enthalpy     rest_mass_density     energy_density     pressure     sound_speed_over_c"
            np.savetxt(opts.fname_output_integral+"_pressure-density_"+str(opts.using_eos_index+i)+".txt", np.c_[param_dict['pseudo_enthalpy'], param_dict['rest_mass_density'], param_dict['energy_density'], param_dict['pressure'], param_dict['sound_speed_over_c']], fmt = '%10s', header=pd_header)
        
        M_dict[i] = _pyr_mrL_dat[:,0]
        R_dict[i] = _pyr_mrL_dat[:,1]
        
        central_gm1 = tov_seq_reprimand.center_gm1_from_grav_mass(M_dict[i])
        eos_results = {'M':M_dict[i], 'R':R_dict[i], 'gm1':central_gm1}
        
        #from likelihood_calculator import mMax_likelihood_for_EOS
        from likelihood_calculator import likelihood_for_MR
        
        likelihood_dict[i] = 1
        #from likelihood_calculator import likelihood_MR_for_eos
        #from likelihood_calculator import likelihood_symmetry_energy
        #for rv in external_ns_MR_rv: likelihood_dict[i] *= likelihood_MR_for_eos(eos_results, rv, reprimand_object = tov_seq_reprimand)
        #import pdb; pdb.set_trace()
        if opts.uniform_c_density_prior :
            for rv in dat_rv: likelihood_dict[i] *= likelihood_for_MR(eos_results, rv, uniform_in = 'Log_central_density', reprimand_object = tov_seq_reprimand, eos_object = reprimand_eos.pyr_eos)
        else:
            for rv in dat_rv: likelihood_dict[i] *= likelihood_for_MR(eos_results, rv, uniform_in = 'M_fixed_grid', reprimand_object = tov_seq_reprimand)
        
        #mmax constraint (duplicate w/ ext_prior):
        #this includes GW170817 if gw170817 in opts.observations
        if opts.do_max_mass:
            print('Mmax', max(M_dict[i]))
            likelihood_dict[i] *= mMax_likelihood_for_EOS(max(M_dict[i]))
        
        #nuclear symmetry energy
        if opts.do_sym_energy:
            likelihood_dict[i] *= likelihood_symmetry_energy(param_dict)
        
        #save results to grid
        if likelihood_dict[i] !=0:
            eoss[i,0] = np.log(likelihood_dict[i])
            eoss[i,1] = 0.001  # nominal integration error
        else: 
            eoss[i,0] = -1e6 # arbitrary low likelihood value for unphysical EoSs
            eoss[i,1] = 10
    #------end loop------    
            
    #could push this to save_CIP_output script, should be the same
    postfix = ''
    if opts.conforming_output_name:
        postfix = '+annotation.dat'
    
    # opts.fname is not None only when using RIFT as is in RIT-matters/20230623
    #if opts.fname is None: 
    #    np.savetxt(opts.outdir+"/"+opts.fname_output_integral+postfix, eoss[opts.eos_start_index: opts.eos_end_index], fmt = '%10s', header="lnL     sigma_lnL   " + ' '.join(dat_orig_names))
    #else: 
    #    np.savetxt(opts.fname_output_integral+postfix, eoss[opts.eos_start_index: opts.eos_end_index], fmt = '%10s', header="lnL     sigma_lnL   " + ' '.join(dat_orig_names))
    
    if opts.chunk_save:
        # remove invalid lines
        indx_ok = np.ones(len(eoss),dtype=bool)
        indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isnan(eoss[:,0]))) #check nans (shouldn't happen)
        indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isinf(eoss[:,0]))) #check +/-inf (can happen)
        print('   Ignoring lines with lnL = -inf : {} '.format(len(eoss)-np.sum(indx_ok)))
        eoss = eoss[indx_ok]
        
        var = eoss[:,1]/eoss[:,0] #mimics sqrt(line[1]**2)/res behavior for single line
        eoss[:,1] = var
        
        #File (2/7): MARG-0-0+annotation.dat
        lineheader = ' '.join(map(str,param_names))
        np.savetxt(opts.fname_output_integral+postfix,eoss,header=lineheader)
        print("Chunk file saved.")
    
    # truncating R, M and likelihood dictionaries to keep only 
    # for i in range(len(likelihood)): R_dict.pop('key', None); del R_dict['key']
    #R_dict = R_dict[np.where(likelihood>0)]
    #M_dict = M_dict[np.where(likelihood>0)]
    #likelihood = likelihood[np.where(likelihood>0)]
    
    #Don't need this, except for debug/validation tests, most likely
    if opts.plot and opts.n_events_to_analyze == 1:
        import matplotlib
        import matplotlib.pyplot as plt
        
        matplotlib.rcParams.update({'font.size': 12.0,  'mathtext.fontset': 'stix'})
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)
        matplotlib.rcParams['xtick.labelsize'] = 15.0
        matplotlib.rcParams['ytick.labelsize'] = 15.0
        matplotlib.rcParams['axes.labelsize'] = 25.0
        matplotlib.rcParams['lines.linewidth'] = 2.0
        plt.style.use('seaborn-v0_8-whitegrid')
        #fig,(ax1,ax2) = plt.subplots(2,1)
        
        fig = plt.figure(); ax = fig.add_subplot(111)
        from plotting import plot_data_and_gaussian
        for j in np.arange(len(dat_rv)):
            plot_data_and_gaussian(dat_mn[j],dat_cov[j],dat_rv[j],dat_list[j],ax)
        #if 'j0030' in observations : plot_data_and_gaussian(mn, cov, rv, data_j0030, ax)
        #if 'j1731' in observations : plot_data_and_gaussian(mn2, cov2, rv2, data_j1731, ax)
        #if 'j1731' in observations : pass
        
        likelihood_array = np.empty((0))
        for i in likelihood_dict: likelihood_array = np.append(likelihood_array, likelihood_dict[i])
        
        lmax = max(likelihood_array)
        lmin = min(likelihood_array)
        
        for i in likelihood_dict:
            #MRcolor = plt.cm.gist_rainbow((likelihood_dict[i]-lmin)/(lmax-lmin))
            ratio = (likelihood_dict[i]-lmin)/(lmax-lmin)
            ax.plot(R_dict[i],M_dict[i], alpha = ratio, color = 'b')
        
        ax.set_xlim(7,20)
        ax.set_xlabel('Radius [km]')
        ax.set_ylabel('Mass [M/M$_\odot$]')
        
        #plt.savefig('MR_likelihood_sample_mMax.pdf',format = 'pdf')
        plt.savefig('MR_likelihood_sample_mMax.png',format = 'png')
        
        plt.show()


if __name__ == '__main__':
    likelihood_evaluation()



'''
def diagnosing():
    #diagnosing RePrimAnd failure for H2 eos
    from RIFT.physics import EOSManager as EOSManager
    h2_by_name = EOSManager.EOSLALSimulation('H2')
    
    
    #eoss = np.genfromtxt('../RIFT_improvedPBCS_indices_physical_EOS_causal.txt', dtype='str')
    qry = EOSManager.QueryLS_EOS(h2_by_name.eos)
    min_pseudo_enthalpy = 0.005
    max_pseudo_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(h2_by_name.eos)
    hvals = max_pseudo_enthalpy* 10**np.linspace( np.log10(min_pseudo_enthalpy/max_pseudo_enthalpy),  0,num=500)
    edens = qry.extract_param('energy_density',hvals)
    press = qry.extract_param('pressure',hvals)
    p_enthalpy = qry.extract_param('pseudo_enthalpy',hvals)
    rho = qry.extract_param('rest_mass_density',hvals)
    cs = qry.extract_param('sound_speed_over_c',hvals)
    
    param_dict = {'pseudo_enthalpy': p_enthalpy,'rest_mass_density': rho,'energy_density': edens,'pressure': press,'sound_speed_over_c': cs}
    param_dict = EOSManager.eos_monotonic_parts_and_causal_sound_speed(param_dict)
    p_enthalpy = param_dict['pseudo_enthalpy']; rho = param_dict['rest_mass_density']; edens = param_dict['energy_density']; press = param_dict['pressure']; cs = param_dict['sound_speed_over_c']
    reprimand_eos = EOSManager.EOSReprimand(param_dict=param_dict)
'''


