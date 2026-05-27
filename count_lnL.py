#! /usr/bin/env python
"""
Created on Tue May 19 01:38:17 2026

@author: marce
"""


import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--using-eos', type=str, action='append',help="REQUIRED: Send eos file with [lnL, sigma_lnL, gamma0, gamma1, gamma2, gamma3, m1, m2, sig] as the parameters.")
parser.add_argument('--save-consolidation-table',action='store_true',help="create & save table with all provided lnLs & EOSs, for comparison")
parser.add_argument('--sum-marg',action='store_true',help="include column manually adding provided MARG lnLs together, to compare to CON file lnL in consolidation table")
parser.add_argument('--input-grid',type=str,help="Grid file for iteration (EOS NEEDS TO MATCH) - helpful, more reliable, but not required")
parser.add_argument('--save-duplicates-report',action='store_true',help="saves files with list of duplicate EOS lines encountered when making consolidation table dict")

opts = parser.parse_args()

fail_dict = {}
eos_data = {}
eos_names = ["lnL", "gamma0", "gamma1", "gamma2", "gamma3", "m1", "m2"]
eos_indices = None
cons_file = 0
marges = {}
iteration = ""

#Process One: checking MARG files for fail codes & showing output
for e, eos in enumerate(opts.using_eos):
    fname = eos.replace('file:', '')
    
    filename=fname.split("/")[-1].split(".")[0]
    print("\nInspecting filename: "+filename)
    marg = None
    if filename.startswith("consolidated_"):
        if len(filename) == 14:
            print("Recognized consolidated_X.net_marg file for iteration.")
            marg = None
            iteration = filename[-1]
            cons_file = 1
            marges[filename] = ["CON",e]
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
    
    if opts.save_consolidation_table:
        if eos_indices is None:
            dat = np.genfromtxt(fname,names=True)
            param_names = dat.dtype.names #separate out the names from the data
            dat_as_array = dat.view((float, len(param_names)))
            eos_indices = [param_names.index(n) for n in eos_names] 
            eos_data[filename] = dat_as_array[:,eos_indices]
            lnL_dat = dat_as_array[:,0]
        else:
            dat = np.genfromtxt(fname)[:,eos_indices]
            eos_data[filename] = dat
            lnL_dat = dat[:,0] 
    else:
        lnL_dat = np.genfromtxt(fname)[:,0]
    
    dat_len = len(lnL_dat)
    print("Length of data:",dat_len)    
    
    if marg is not None:
        if marg == 0: #PLE
            fail_codes = [["PLE-mass",-2e6]]
            marges[filename] = ["PLE",e]
        elif marg == 1: #CIP
            fail_codes = [["CIP-EOS",-1.5e6], ["CIP-mass",-2.5e6], ["CIP-nan",np.nan], ["CIP-Mmax",-6.5e6], ["CIP-other",-1e6]]
            marges[filename] = ["CIP",e]
        elif marg == 2: #NICER
            fail_codes = [["NICER-other",-1e6], ["NICER-EOS",-4e6], ["NICER-Mmax",-6e6]]
            marges[filename] = ["NCR",e]
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
            
            #fail codes (in order of occurence):
                #PLE mass: -2
                #CIP EOS: -1.5 
                #CIP mmax: -6.5
                #CIP mass: -2.5 
                #CIP other: -1
                #NICER EOS: -4 
                #NICER mmax: -6
                #NICER other: -1
                #Possible combos: 
                    #mass fail, EOS good, mmax good: -2 PLE + -2.5 CIP + 0 NICER = -4.5
                    #mass fail, EOS good, mmax bad:  -2 PLE + -6.5 CIP + -6 NICER = -14.5
                    #mass fail, EOS fail: -2 PLE + -1.5 CIP + -4 NICER = -7.5
                    #mass good, EOS fail: 0 PLE + -1.5 CIP + -4 NICER = -5.5
                    #mass good, EOS good, mmax bad: 0 PLE + -6.5 CIP + -6 NICER = -12.5
                    #mass good, EOS good, mmax good: 0 PLE + 0 CIP + 0 NICER >~ 0
                    #mass good, EOS good, mmax good, other NICER/CIP = -1 or -2
         
            #indx_ok = np.ones(dat_len,dtype=bool)
            #indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isnan(lnL_dat[:,0])))
    
            if lnL <= -14e6:
                #mass fail, EOS good, mmax bad:  -2 PLE + -6.5 CIP + -6 NICER = -14.5
                mf_eg_mb += 1
            elif lnL <= -12e6:
                mg_eg_mb += 1
            elif lnL <= -7e6:
                mf_ef += 1
            elif lnL <= -5.4e6:
                mg_ef += 1
            elif lnL <= -4.4e6:
                mf_eg_mg += 1
            elif lnL <= -0.5e6:
                other_fails += 1
            else:
                good_lines += 1
        
        print("Results for file "+fname.split("/")[-1]+":")
        print(" Mass-only fails (-4.5):  ",mf_eg_mg,"  ",(mf_eg_mg/dat_len)*100.,"%")
        print(" Mass, mmax fails (-14.5):",mf_eg_mb,"  ",(mf_eg_mb/dat_len)*100.,"%")
        print(" Mass, eos fails (-7.5):  ",mf_ef,"  ",(mf_ef/dat_len)*100.,"%")
        print(" eos-only fails (-5.5):   ",mg_ef,"  ",(mg_ef/dat_len)*100.,"%")
        print(" Mmax-only fails (-12.5):    ",mg_eg_mb,"  ",(mg_eg_mb/dat_len)*100.,"%")
        print(" Other fails (-1):        ",other_fails,"  ",(other_fails/dat_len)*100.,"%")
        print(" Good lines:",good_lines,"   ",(good_lines/dat_len)*100.,"%")


#Process Two (optional): make table showing all lnLs together for each eos
if opts.save_consolidation_table:
    print("\nMaking consolidation table.")
    net_table = {}
    
    if iteration is None: iteration = "no_CON"
    
    initial_dat = None
    initial_dat_len = 0
    longest_file = ""
    if opts.input_grid:
        initial_dat = np.genfromtxt(opts.input_grid)[:,eos_indices] #these better match
        initial_dat_len = len(initial_dat)
        print(" Length of initial grid file:",initial_dat_len)
    else:
        for eos in eos_data: #find max available file length
            length = len(eos_data[eos])
            if length > initial_dat_len: 
                initial_dat_len = length
                longest_file = eos
        print(" No initial grid; found highest file length:",longest_file,initial_dat_len) #should be PLE length
        initial_dat = eos_data[longest_file]  
    
    #table cols: indx CON_lnL sum_lnL PLE_lnL CIP_lnL NCR_lnL eos_indices
    #final_table = np.zeros((initial_dat_len,num_cols+len(eos_indices[1:])))
    #final_table[:,num_cols:] = np.around(initial_dat[:,1:], decimals=7)
    #final_table[:,0] = np.arange(initial_dat_len)
    
    #initial fill of dict
    dup_lines = {} #handles duplicate eos lines by removing them
    for line in initial_dat:
        line = np.around(line, decimals=7)
        #dict cols: CON_lnL PLE CIP NCR - eos vals (x6) in keys, indx & sum_lnL added later
        if tuple(line[1:]) in net_table:
            #duplicate EOS line
            if tuple(line[1:]) in dup_lines:
                dup_lines[tuple(line[1:])] += 1
            else:
                dup_lines[tuple(line[1:])] = 2 #this is the first duplicate, so 2 total
        else:
            net_table[tuple(line[1:])] = np.zeros(len(eos_data)) 
        #if not opts.input_grid:
        #    net_table[tuple(line[1:])][init_marg] = line[0]
    print(" Match table has",len(net_table),"unique lines, out of",initial_dat_len,"initial data lines")
    
    if opts.save_duplicates_report:
        header = "Count "+" ".join(eos_names[1:])
        with open("lnL_consolidation_duplicates_"+iteration+".dat", 'w') as file_out:
            file_out.write("# " + header + "\n")
            for key in list(dup_lines.keys()):
                file_out.write(" {}   ".format(dup_lines[key]) + ' '.join(map(str,key)) +"\n")
        print(" Duplication report saved.")
    
    #add lnLs for each file to dict
    marg_col_tracker = 0
    for eos in eos_data:
        print("Tabulating eos data for:",eos)
        marg_col = -1
        if marges[eos][0] == "CON":
            marg_col = 0
        else:
            marg_col = cons_file+marg_col_tracker #will be 1, 2, 3 if cons_file
            marg_col_tracker += 1
            marges[eos][1] = marg_col
        print(" MARG column for this file is:",marg_col)
        
        #if eos == longest_file:
            #final_table[:,marg_col] = eos_data[eos][:,0]
        #    print(" This is the longest file; its data has already been filled. Continuing.")
        #    continue
        
        eos_line = 0
        for i in np.arange(len(eos_data[eos])):
            line = np.around(eos_data[eos][i], decimals=7)

            if tuple(line[1:]) in net_table:
                net_table[tuple(line[1:])][marg_col] = line[0]
                eos_line += 1
            else:
                print("WARNING: non-initial EOS line encountered!\n",i,line[1:])
                net_table[tuple(line[1:])] = np.zeros(len(eos_data)) 
                net_table[tuple(line[1:])][marg_col] = line[0]

                
            #if (line[1:] == final_table[i,6:]).all():
            #    final_table[i,marg_col] = line[0]
            #    eos_line += 1
            #else:
            #    final_table[i,marg_col] = 0.0
        
        if eos_line == len(eos_data[eos]): #equal b/c last good iteration sets eos_line+1
            print(" All lines for this eos file matched.")
        else:
            print(" Not all lines for this eos data were matched:",eos_line,"/",len(eos_data[eos]))
     
    sum_head = ""
    if opts.sum_marg:
        sum_head = "lnL_sum "
    #final_table[:,2] = np.sum(final_table[:,3:6],axis=1)
    header = "idx "
    if cons_file == 1: header += "lnL_CON "
    if opts.sum_marg: header += "lnL_sum "
    for m in marges:
        if marges[m][0] != "CON":
            header += "lnL_"+marges[m][0]+" "
    header += " ".join(eos_names[1:])
    #np.savetxt("lnL_consolidation_table.dat",final_table,header=header)
    with open("lnL_consolidation_table_"+iteration+".dat", 'w') as file_out:
        file_out.write("# " + header + "\n")

        for i, key in enumerate(net_table.keys()):
            line_out = str(i)+" "
            if cons_file == 1: line_out += "{} ".format(net_table[key][0])
            if opts.sum_marg: 
                line_out += "{} ".format(np.sum(net_table[key][cons_file:]))
            for m in marges:
                if marges[m][0] != "CON":
                    line_out += "{} ".format(net_table[key][marges[m][1]])
            line_out += " ".join(map(str,key))
            #print(" {} {} ".format(lnLnet, sigma) + ' '.join(map(str,key)) )
            file_out.write(line_out + "\n")
    
    print("Consolidation table saved.")
    
#want to save totals file:
#indx CON_lnL Total  PLE_lnL CIP_lnL NCR_lnL gamma0 gamma1 gamma2 gamma3 m1 m2
#0    -10.5   -10.5  -2      -2.5    -6      .63 -.2 0.2 -.009 1.2 1.4


