#! /usr/bin/env python
"""
Created on Tue May 19 01:38:17 2026

@author: marce
"""


import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--marg-eos', type=str, action='append',help="REQUIRED: Send eos file with [lnL, sigma_lnL, gamma0, gamma1, gamma2, gamma3, m1, m2, sig] as the parameters.")
parser.add_argument('--net-eos',type=str,default=None,help="consolidated_X.net_marg file for iteration")
parser.add_argument('--save-consolidation-table',action='store_true',help="create & save table with all provided lnLs & EOSs, for comparison")
parser.add_argument('--sum-marg',action='store_true',help="include column manually adding provided MARG lnLs together, to compare to CON file lnL in consolidation table")
parser.add_argument('--input-grid',type=str,help="Grid file for iteration (EOS NEEDS TO MATCH) - helpful, more reliable, but not required")
parser.add_argument('--save-duplicates-report',action='store_true',help="saves files with list of duplicate EOS lines encountered when making consolidation table dict")
parser.add_argument('--iteration',default=None,type=int,help="Provide iteration if no CON file, to be nice")
parser.add_argument('--marges',default="PCN",type=str,help="Initials (IN ORDER) of individual MARG processes being passed in via --using-eos: P=PLE,C=CIP,N=NICER")

opts = parser.parse_args()

if len(opts.marges) != len(opts.marg_eos):
    print("ERROR: provided MARG list (--marg-eos) must be same length as MARG IDs (--marges). Exiting.")
    import sys
    sys.exit(0)


eos_data = {}
eos_names = ["lnL", "gamma0", "gamma1", "gamma2", "gamma3", "m1", "m2"]
eos_indices = None
cons_file = 0
marges = {}
iteration = ""
fail_code_dict = {}

if not (opts.iteration is None):
    iteration = str(opts.iteration)
    fail_report_name = "lnL_fail_report_"+iteration
elif opts.input_grid is not None:
    iteration = opts.input_grid.split("/")[-1].split(".")[0][-1]
    fail_report_name = "lnL_fail_report_"+iteration
else:
    fail_report_name = "lnL_fail_report"

if eos_indices is None:
    dat_init = np.genfromtxt(opts.using_eos[0].replace('file:', ''),names=True)[0]
    param_names = dat_init.dtype.names #separate out the names from the data
    eos_indices = [param_names.index(n) for n in eos_names]


#Process 0: identify provided files
if opts.net_eos:
    cons_file = 1
    opts.marg_eos.append(opts.net_eos) #haha! retroactive shortcut!

for e, eos in enumerate(opts.marg_eos):
    fname = eos.replace('file:', '')
    
    filename=fname.split("/")[-1].split(".")[0]
    print("\nInspecting filename: "+filename)
    if filename.startswith("consolidated_"):
        if len(filename) == 14:
            print("Recognized consolidated_X.net_marg file for iteration.")
            if iteration == "": 
                iteration = filename[-1]
                fail_report_name = "lnL_fail_report_"+iteration
            marges[filename] = ["CON",e,-1]
        elif len(filename) == 16:
            if opts.marges[e] == 'P': #PLE
                fail_code_dict[filename] = [["PLE-mass",-2e6]]
                marges[filename] = ["PLE",e,int(filename[-1])]
            elif opts.marges[e] == 'C': #CIP
                fail_code_dict[filename] = [["CIP-EOS",-1.5e6], ["CIP-Mmax",-6.5e6], ["CIP-mass",-2.5e6], ["CIP-nan",np.nan], ["CIP-other",-1e6]]
                marges[filename] = ["CIP",e,int(filename[-1])]
            elif opts.marges[e] == 'N': #NICER
                fail_code_dict[filename] = [["NICER-EOS",-4e6], ["NICER-Mmax",-6e6], ["NICER-other",-1e6]]
                marges[filename] = ["NCR",e,int(filename[-1])]
            else:
                print("Warning: unsupported MARG file ID: "+opts.marges[e]+". Only P,C,N supported.")
                fail_code_dict[filename] = ["Unknown",-1e6] #generic check
                marges[filename] = ["Unknown",e,int(filename[-1])]
            print("Recognized consolidated_X_Y.net_marg file for MARG process:",marges[filename][0],marges[filename][2])
        else:
            print("ERROR: could not recognize consolidated file. Exiting.")
            import sys
            sys.exit(0)
        
        if opts.save_consolidation_table:
            dat = np.genfromtxt(fname)[:,eos_indices]
            eos_data[filename] = dat
        else:
            dat = np.genfromtxt(fname)[:,0]
            eos_data[filename] = dat
    else:
        print("ERROR: unsupported file type. Exiting.")
        import sys
        sys.exit(0)

# reference grid
initial_dat = None
initial_dat_len = 0
longest_file = ""
if opts.input_grid:
    initial_dat = np.genfromtxt(opts.input_grid)[:,eos_indices] #these better match
    grid_it = opts.input_grid.split("/")[-1][-6:]
    puff_grid = opts.input_grid[:-6]+"_puff"+grid_it
    grid_len = len(initial_dat)
    puff_len = 0
    print("Attempting to access grid_puff file: "+puff_grid.split("/")[-1])
    try:
        puff_dat = np.genfromtxt(puff_grid)[:,eos_indices]
        initial_dat = np.concatenate((initial_dat,puff_dat), axis=0)
        puff_len = len(puff_dat)
    except:
        print("   WARNING: could not find grid_puff file. RIP.")
    initial_dat_len = len(initial_dat)
    print(" Length of initial grid file(s):",grid_len,"+",puff_len,"=",initial_dat_len)
    report_line = "(from grid file)"
else:
    print("No grid file supplied.")
    for e in eos_data: 
        initial_dat_len = max(initial_dat_len,len(eos_data[e]))
    print(" No initial grid; found highest file length:",initial_dat_len)
    report_line = "(no grid file)"


#Process 1: checking MARG files for fail codes & showing output
with open(fail_report_name+".txt",'w') as file_out: #will overwrite preexisting file
    file_out.write("Total initial data length for iteration "+iteration+" "+report_line+": "+str(initial_dat_len)+"\n\n")

for eos in eos_data:
    if opts.save_consolidation_table:
        lnL_dat = eos_data[eos][:,0] 
    else:
        lnL_dat = eos_data[eos]
    
    dat_len = len(lnL_dat)
    print("Length of data:",dat_len,"  (",(dat_len/initial_dat_len)*100.,"% of longest)")    
    fail_dict = {}
    
    if marges[eos][2] >= 0:
        fail_codes = fail_code_dict[eos]
        
        print("Results for file "+eos+".net_marg:")
        indx_ok = np.ones(dat_len,dtype=bool)
        for c in fail_codes:
            if c[0] == "CIP-nan": #alt: np.isnan(c[1]) -> more flexible
                fails = np.count_nonzero(np.isnan(lnL_dat))
                indx_ok = np.logical_and(indx_ok, np.logical_not(np.isnan(lnL_dat)))
            else:
                fails = np.count_nonzero(lnL_dat == c[1])
                indx_ok = np.logical_and(indx_ok, np.logical_not(lnL_dat == c[1]))

            fail_dict[c[0]] = [c[1], fails, (fails/dat_len)*100.0]
            #marges[filename].append([c[0],c[1],fails,(fails/dat_len)*100.0])

            print(" "+c[0]+" fails ("+str(c[1])+"):",fails,"   ",(fails/dat_len)*100.,"%")
        good_lines = np.sum(indx_ok)
        #marges[filename].append(["Good lines",good_lines,(good_lines/dat_len)*100.0])
        print(" Good lines:",good_lines,"   ",(good_lines/dat_len)*100.,"%")
        
        with open(fail_report_name+".txt", 'a') as file_out:
            file_out.write("Results for file "+fname.split("/")[-1]+" ("+marges[filename][0]+"):" + "\n")
            file_out.write(" Length of data: "+str(dat_len)+"  ({} % of longest)\n".format((dat_len/initial_dat_len)*100.))
            for fail in fail_dict:
                file_out.write(" "+fail+" fails ({}): {}   {} %\n".format(fail_dict[fail][0],fail_dict[fail][1],fail_dict[fail][2]))
            file_out.write(" Good lines: {}   {} %\n\n".format(good_lines,(good_lines/dat_len)*100.))
        print("Results for this file saved.")
    else:
        mf_eg_mg = 0
        mf_eg_mb = 0
        mf_ef = 0
        mg_ef = 0
        mg_eg_mb = 0
        mg_eg_mg = 0
        other_fails = 0
        good_lines = 0
        #mf_eg_mbc = 0
        #mf_eg_mbn = 0
        #mg_eg_mbc = 0
        #mg_eg_mbn = 0
        
        #this needs to be more flexible, allow for diff orders of MARGs
        if opts.marges == 'PCN':
            check_sums = [[-14.5e6,"Mass, mmax-both",0],[-12.5e6,"Mmax-both",0],
                          [-10.5e6,"Mass, mmax-NCR",0],[-8.5e6,"Mass, mmax-CIP",0],
                          [-7.5e6,"Mass, EOS",0],[-6.5e6,"Mmax-CIP",0],[-6e6,"Mmax-NCR",0],
                          [-5.5e6,"EOS-only",0],[-4.5e6,"Mass-only",0],[-1e6,"Other",0]]
        elif opts.marges == 'CN':
            check_sums = [[-12.5e6,"Mmax-both",0],[-8.5e6,"Mass, mmax-NCR",0],
                          [-6.5e6,"Mmax-CIP",0],[-6e6,"Mmax-NCR",0],
                          [-5.5e6,"EOS-only",0],[-2.5e6,"Mass-only",0],[-1e6,"Other",0]]
        elif opts.marges == 'PC':
            check_sums = [[-1e6,0]] #not implemented
        elif opts.marges == 'PN':
            check_sums = [[-1e6,0]] #not implemented
        else:
            check_sums = [[-1e6,0]]
        
        #for i in np.arange(dat_len):
            #lnL = lnL_dat[i]
            
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
                    #mass fail, EOS good, mmax good: -2 PLE + -2.5 CIP + 0 NICER   = -4.5
                    #mass fail, EOS good, all mmax bad: -2 PLE + -6.5 CIP + -6 NCR = -14.5
                    #mass fail, EOS good, CIP mmax bad: -2 PLE + -6.5 CIP + 0  NCR = -8.5
                    #mass fail, EOS good, NCR mmax bad: -2 PLE + -2.5 CIP + -6 NCR = -10.5
                    #mass fail, EOS fail: -2 PLE + -1.5 CIP + -4 NICER             = -7.5
                    #mass good, EOS fail: 0 PLE + -1.5 CIP + -4 NICER              = -5.5
                    #mass good, EOS good, all mmax bad: 0 PLE + -6.5 CIP + -6 NCR  = -12.5
                    #mass good, EOS good, CIP mmax bad: 0 PLE + -6.5 CIP + 0 NCR   = -6.5
                    #mass good, EOS good, NCR mmax bad: 0 PLE + 0 CIP + -6 NICER   = -6
                    #mass good, EOS good, mmax good: 0 PLE + 0 CIP + 0 NICER >~ 0
                    #mass good, EOS good, mmax good, other NICER/CIP = -1 or -2
        
        print("Results for file "+fname.split("/")[-1]+":")
        indx_ok = np.ones(dat_len,dtype=bool)
        #indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isnan(lnL_dat[:,0])))
        #if len(check_sums) == 1:
        #    if lnL <= -0.5e6:
        #        check_sums[0][1] += 1
        #else:
        for c in np.arange(len(check_sums)):
            indx_f = np.ones(dat_len,dtype=bool)
            indx_f = np.logical_and(indx_f,  lnL_dat[:]<= check_sums[c][0] + 0.2e6 )
            indx_f = np.logical_and(indx_f,  lnL_dat[:]>= check_sums[c][0] - 0.2e6 )
            
            check_sums[c][2] = np.sum(indx_f)

            print(" "+check_sums[c][1]+" fails ("+str(check_sums[c][0])+"):",check_sums[c][2],"   ",(check_sums[c][2]/dat_len)*100.,"%")

            indx_ok = np.logical_and(indx_ok,  np.logical_not(indx_f))
            
            #if (lnL <= check_sums[c][0] + 0.2e6) and (lnL >= check_sums[c][0] - 0.2e6):
            #    check_sums[c][2] += 1
        
        good_lines = np.sum(indx_ok)
        print(" Good lines:",good_lines,"   ",(good_lines/dat_len)*100.,"%")
        
        with open(fail_report_name, 'a') as file_out:
            file_out.write("Results for file "+fname.split("/")[-1]+":" + "\n")
            file_out.write(" Length of data: "+str(dat_len)+"  ({} % of longest)\n".format((dat_len/initial_dat_len)*100.))
            for c in np.arange(len(check_sums)):
                file_out.write(" "+check_sums[c][1]+" fails ({}): {}   {} %\n".format(check_sums[c][0],check_sums[c][2],(check_sums[c][2]/dat_len)*100.))
            file_out.write(" Good lines: {}   {} %\n".format(good_lines,(good_lines/dat_len)*100.))
            file_out.write("          of total: {} %\n\n".format((good_lines/initial_dat_len)*100.))
        print("Results for this file saved.")


#Process 2 (optional): make table showing all lnLs together for each eos
if opts.save_consolidation_table:
    print("\nMaking consolidation table.")
    net_table = {}
    
    if iteration is None: iteration = "no_CON"
    
    #initial_dat = None
    #initial_dat_len = 0
    longest_file = ""
    if opts.input_grid:
        print(" Using input grid data; length:",initial_dat_len)
        #initial_dat = np.genfromtxt(opts.input_grid)[:,eos_indices] #these better match
        #initial_dat_len = len(initial_dat)
        #print(" Length of initial grid file:",initial_dat_len)
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
        tot_dup = 0
        with open(fail_report_name+"_duplicates.txt", 'w') as file_out:
            file_out.write("# " + header + "\n")
            for key in list(dup_lines.keys()):
                file_out.write(" {}   ".format(dup_lines[key]) + ' '.join(map(str,key)) +"\n")
                tot_dup += dup_lines[key]
            file_out.write("{} unique; {} total: {} % of grid".format(len(dup_lines),tot_dup,(tot_dup/initial_dat_len)*100.))
        print(" Duplication report saved;",len(dup_lines),"unique reoccurring lines;",tot_dup,"duplicate lines total.")
    
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
        
        eos_line = 0
        non_initial_lines = 0
        for i in np.arange(len(eos_data[eos])):
            line = np.around(eos_data[eos][i], decimals=7)

            if tuple(line[1:]) in net_table:
                net_table[tuple(line[1:])][marg_col] = line[0]
                eos_line += 1
            else:
                #print("WARNING: non-initial EOS line encountered!\n",i,line[1:])
                net_table[tuple(line[1:])] = np.zeros(len(eos_data)) 
                net_table[tuple(line[1:])][marg_col] = line[0]
                with open(fail_report_name+"_additional_lines.txt", 'a') as file_out:
                    file_out.write(marges[eos][0]+" "+str(i)+" " + ' '.join(map(str,line[1:]))+"\n")
                non_initial_lines += 1
        
        if eos_line == len(eos_data[eos]): #equal b/c last good iteration sets eos_line+1
            print(" All lines for this eos file matched.")
        else:
            print(" Not all lines for this eos data were matched:",eos_line,"/",len(eos_data[eos]))
        print(" Non-initial lines encountered:",non_initial_lines)
     
    sum_head = ""
    if opts.sum_marg:
        sum_head = "lnL_sum "
    header = "idx "
    if cons_file == 1: header += "lnL_CON "
    if opts.sum_marg: header += "lnL_sum "
    for m in marges:
        if marges[m][0] != "CON":
            header += "lnL_"+marges[m][0]+" "
    header += " ".join(eos_names[1:])

    with open(fail_report_name+"_consolidation_table.dat", 'w') as file_out:
        file_out.write("# " + header + "\n")

        for i, key in enumerate(net_table.keys()):
            line_out = str(i)+" "
            if cons_file == 1: line_out += "{} ".format(net_table[key][0])
            if opts.sum_marg: 
                line_out += "{} ".format(np.sum(net_table[key][cons_file:]))
            for m in marges:
                if marges[m][0] != "CON":
                    line_out += "{} ".format(net_table[key][marges[m][1]]) #hacky? assumes CON file is last
            line_out += " ".join(map(str,key))
            #print(" {} {} ".format(lnLnet, sigma) + ' '.join(map(str,key)) )
            file_out.write(line_out + "\n")
    
    print("Consolidation table saved.")
    
#want to save totals file:
#indx CON_lnL Total  PLE_lnL CIP_lnL NCR_lnL gamma0 gamma1 gamma2 gamma3 m1 m2
#0    -10.5   -10.5  -2      -2.5    -6      .63 -.2 0.2 -.009 1.2 1.4


