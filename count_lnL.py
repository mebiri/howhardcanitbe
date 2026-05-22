# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
Created on Tue May 19 01:38:17 2026

@author: marce
"""


import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--using-eos', type=str, action='append',help="REQUIRED: Send eos file with [lnL, sigma_lnL, gamma0, gamma1, gamma2, gamma3, m1, m2, sig] as the parameters.")
parser.add_argument('--save-consolidation-table',action='store_true',help="Must include --sum-marg to use this")
parser.add_argument('--sum-marg',action='store_true')

opts = parser.parse_args()

fail_dict = {}
eos_data = {}
eos_names = ["lnL", "gamma0", "gamma1", "gamma2", "gamma3", "m1", "m2"]
eos_indices = None

for e in opts.using_eos:
    fname = opts.using_eos.replace('file:', '')
    
    filename=fname.split("/")[-1].split(".")[0]
    print("\nInspecting filename: "+filename)
    marg = None
    if filename.startswith("consolidated_"):
        if len(filename) == 14:
            print("Recognized consolidated_X.net_marg file for iteration.")
            marg = None
        elif len(filename) == 16:
            print("Recognized consolidated_X_Y.net_marg file for MARG process.")
            marg = int(filename[-1])
        else:
            print("ERROR: could not recognize consolidated file. Exiting.")
            import sys
            sys.exit(0)
    else:
        print("ERROR: unsupported file type. Exiting.")
        import sys
        sys.exit(0)
    
    if opts.sum_marg:
        if eos_indices is None:
            dat = np.genfromtxt(fname,names=True)
            param_names = dat.dtype.names #separate out the names from the data
            dat_as_array = dat.view((float, len(param_names)))
            eos_indices = [param_names.index(n) for n in eos_names] 
            dat_to_save = dat_as_array[:,eos_indices]
            eos_data[filename] = dat_to_save
            lnL_dat = dat_as_array[:,0]
        else:
            dat = np.genfromtxt(fname)[:,eos_indices]
            dat_to_save = dat_as_array
            eos_data[filename] = dat
            lnL_dat = dat[:,0] 
    else:
        lnL_dat = np.genfromtxt(fname)[:,0]
    
    dat_len = len(lnL_dat)
    print("Length of data:",dat_len)    
    
    if marg is not None:
        if marg == 0: #PLE
            fail_codes = [["PLE-mass",-2e6]]
        elif marg == 1: #CIP
            fail_codes = [["CIP-EOS",-1.5e6], ["CIP-mass+EOS",-2.5e6], ["CIP-nan",np.nan]]
        elif marg == 2: #NICER
            fail_codes = [["NICER-other",-1e6], ["NICER-EOS",-4e6], ["NICER-Mmax",-6e6]]
        else: 
            print("ERROR: unsupported MARG file id",marg,"encountered. Exiting.")
            import sys
            sys.exit(0)
        
        print("Results for file "+fname.split("/")[-1]+":")
        indx_ok = np.ones(dat_len,dtype=bool)
        for c in fail_codes:
            if c[0] == "CIP-nan": #alt: np.isnan(c[1]) -> more flexible
                fails = np.count_nonzero(np.isnan(lnL_dat))
                indx_ok = np.logical_and(indx_ok, np.logical_not(np.isnan(lnL_dat)))
            else:
                fails = np.count_nonzero(lnL_dat == c[1])
                #fails = lnL_dat.count(c[1])
                indx_ok = np.logical_and(indx_ok, np.logical_not(lnL_dat == c[1]))

            fail_dict[c[0]] = [c[1], fails, fails/dat_len]

            print(" "+c[0]+" fails ("+str(c[1])+"):",fails,"   ",(fails/dat_len)*100.,"%")
        good_lines = lnL_dat[indx_ok]
        print(" Good lines:",len(good_lines),"   ",(len(good_lines)/dat_len)*100.,"%")
    else:
        mf_eg_mg = 0
        mf_eg_mb = 0
        mf_ef = 0
        mg_ef = 0
        mg_eg_mb = 0
        mg_eg_mg = 0
        other_fails = 0
        good_lines = 0
        
        for i in np.arange(dat_len):
            lnL = lnL_dat[i]
            
            #fail codes:
                #PLE mass: -2
                #CIP general: -1.5 (EOS or mass)
                #CIP mass: + -1 = -2.5 total (mass+EOS or just mass)
                #NICER EOS: -4 
                #NICER mmax: -6
                #NICER other: -1
                #Possible combos: 
                    #mass fail, EOS good, mmax good: -2 PLE + -2.5 CIP + 0 NICER = -4.5
                    #mass fail, EOS good, mmax bad:  -2 PLE + -2.5 CIP + -6 NICER = -10.5
                    #mass fail, EOS fail: -2 PLE + -2.5 CIP + -4 NICER = -8.5
                    #mass good, EOS fail: 0 PLE + -1.5 CIP + -4 NICER = -5.5
                    #mass good, EOS good, mmax bad: 0 PLE + 0 CIP + -6 NICER = -6
                    #mass good, EOS good, mmax good: 0 PLE + 0 CIP + 0 NICER >= 0
                    #mass good, EOS good, mmax good, other NICER = -1
         
            indx_ok = np.ones(dat_len,dtype=bool)
            indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isnan(lnL_dat[:,0])))
    
            if lnL <= -10e6:
                #mass fail, EOS good, mmax bad:  -2 PLE + -2.5 CIP + -6 NICER = -10.5
                mf_eg_mb += 1
            elif lnL <= -8e6:
                mf_ef += 1
            elif lnL <= -5.9e6:
                mg_eg_mb += 1
            elif lnL <= -5.4e6:
                mg_ef += 1
            elif lnL <= -4.4e6:
                mf_eg_mg += 1
            elif lnL <= -0.9e6:
                other_fails += 1
            else:
                good_lines += 1
        
        
        print("Results for file "+fname.split("/")[-1]+":")
        print(" Good lines:",good_lines,"   ",(good_lines/dat_len)*100.,"%")
        print(" Mass-only fails (-4.5):  ",mf_eg_mg,"  ",(mf_eg_mg/dat_len)*100.,"%")
        print(" Mass, mmax fails (-10.5):",mf_eg_mb,"  ",(mf_eg_mb/dat_len)*100.,"%")
        print(" Mass, eos fails (-8.5):  ",mf_ef,"  ",(mf_ef/dat_len)*100.,"%")
        print(" eos-only fails (-5.5):   ",mg_ef,"  ",(mg_ef/dat_len)*100.,"%")
        print(" Mmax-only fails (-6):    ",mg_eg_mb,"  ",(mg_eg_mb/dat_len)*100.,"%")
        print(" Other fails (-1):        ",other_fails,"  ",(other_fails/dat_len)*100.,"%")

#want to save totals file:
#Total  PLE_lnL CIP_lnL NCR_lnL gamma0 gamma1 gamma2 gamma3 m1 m2
#-10.5  -2      -2.5    -6      .63 -.2 0.2 -.009 1.2 1.4


