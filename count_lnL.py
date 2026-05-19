# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
Created on Tue May 19 01:38:17 2026

@author: marce
"""


import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--using-eos', type=str, help="REQUIRED: Send eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
parser.add_argument('--using-eos-index',type=int, help="REQUIRED: Line number for single calculation, starting index for multi-line calculation")
parser.add_argument('--n-events-to-analyze',type=int, default=1,help="REQUIRED: Number of EOS lines to eval at once, default 1; >1 supported.")

opts = parser.parse_args()

opts.using_eos = "consolidated_test_2.txt"

#load EOS data:
fname = opts.using_eos.replace('file:', '')
#eos_dat = np.genfromtxt(fname,names=True)#[opts.using_eos_index:opts.using_eos_index+opts.n_events_to_analyze] #should be 1 line if n_events=1
lnL_dat = np.genfromtxt(fname)[:,0]
#param_names = list(eos_dat.dtype.names)
#eoss = eos_dat.view((float, len(param_names)))
#print("EOS dat size: (",len(lnL_dat),len(lnL_dat[0]),")")
#dat_orig_names = param_names[2:] #ignore lnL, sig_lnL - unused anywhere else
#print("Original field names ", dat_orig_names)
dat_len = len(lnL_dat)
print("Length of data:",dat_len)

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
        #NICER EOS: -4 (EOS or other)
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


print("Results:")
print(" Good lines:",good_lines,"   ",(good_lines/dat_len)*100.,"%")
print(" Mass-only fails (-4.5):  ",mf_eg_mg,"  ",(mf_eg_mg/dat_len)*100.,"%")
print(" Mass, mmax fails (-10.5):",mf_eg_mb,"  ",(mf_eg_mb/dat_len)*100.,"%")
print(" Mass, eos fails (-8.5):  ",mf_ef,"  ",(mf_ef/dat_len)*100.,"%")
print(" eos-only fails (-5.5):   ",mg_ef,"  ",(mg_ef/dat_len)*100.,"%")
print(" Mmax-only fails (-6):    ",mg_eg_mb,"  ",(mg_eg_mb/dat_len)*100.,"%")
print(" Other fails (-1):        ",other_fails,"  ",(other_fails/dat_len)*100.,"%")
