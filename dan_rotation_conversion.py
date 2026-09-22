# -*- coding: utf-8 -*-
"""
Routines to apply conversion from Wysocki et al. 2020 Appendix B
rotates gamma hypercube to new one (r) that better aligns with physical EOS
"""

import numpy as np

dan_rot = [[0.43801, -0.53573, 0.52661, -0.49379],
           [-0.76705, 0.17169, 0.31255, -0.53336],
           [0.45143, 0.67967, -0.19454, -0.54443],
           [0.12646, 0.47070, 0.76626, 0.41868]]
scaled_mean = [0.89421, 0.33878, -0.07894, 0.00393]
scaled_sig = [0.35700, 0.25769, 0.05452, 0.00312]
dan_inv = np.linalg.inv(dan_rot)

def dan_rotation(X, coord_names, **kwargs):
    #coord_names will be low_level_coord_names from CEP
    print("dat received[0]:\n",X[0])
    #get gammas' indices in X_out(= X) from coord names
    r_tilde = np.zeros((len(X),4))
    rot_cols = []
    for i in np.arange(4):
        #do one coord at a time
        indx = coord_names.index("gamma"+str(i)) 
        rot_cols.append(indx)

        #convert gammas to r_tilde using equation: r_tilde = (gamma - u)/sig
        r_tilde[:,i] = (X[:,indx] - scaled_mean[i])/scaled_sig[i]

    #apply transform: r_prime = S*r_tilde ( [4 x 4].([N x 4].T) )
    r_prime = np.matmul(dan_rot,r_tilde.T).T
    X_out = X
    X_out[:,rot_cols] = r_prime 
    return X_out


def inverse_dan_rotation(X, coord_names, **kwargs):
    #apply inverse: S-1*r_prime = S-1*S*r_tilde = r_tilde ( [4 x 4].([N x 4].T) )
    print("dat received[0]:\n",X[0])
    r_prime_out = np.zeros((len(X),4))
    rot_cols = []
    for i in np.arange(4):
        #do one coord at a time
        indx = coord_names.index("gamma"+str(i)) #coord_names = dat_orig_names
        rot_cols.append(indx)
    r_prime_out = X[:,rot_cols]
    r_tilde_post = np.matmul(dan_inv,r_prime_out.T).T
    X_out = X

    for i, col in enumerate(rot_cols): #np.arange(4):
        #do one coord at a time
        #indx = coord_names.index("gamma"+str(i))

        #r_tilde = (gamma - u)/sig  ->  gamma = r_tilde*sig + u
        X_out[:,col] = (r_tilde_post[:,i]*scaled_sig[i]) + scaled_mean[i]
    return X_out


def get_bounds(param_list, bounds_dict, **kwargs):
    if 'buffer' in kwargs:
        buffer = np.float(kwargs['buffer'])
    else:
        buffer = 0.0
        print("Warning: no buffer provided; returning bounds unchanged")
        return bounds_dict
    
    use_alternate_buffer = False
    if 'use_alternate_buffer' in kwargs:
        use_alternate_buffer = True
    
    rot_coords = {}
    rot_coords["r0"] = [-4.37722, 4.91227]
    rot_coords["r1"] = [-1.82240, 2.06387]
    rot_coords["r2"] = [-0.32445, 0.36469]
    rot_coords["r3"] = [-0.09529, 0.11046]
    
    for indx, param in enumerate(rot_coords.keys()):
        # apply hypercube buffer
        if use_alternate_buffer: #new_bound = bound +/- buffer*(width of param range) -> SYMMETRIC buffer
            ubound = rot_coords[param][1] + buffer*abs(rot_coords[param][1]-rot_coords[param][0])
            lbound = rot_coords[param][0] - buffer*abs(rot_coords[param][1]-rot_coords[param][0])
        else: #new_bound = bound +/- buffer*|bound| -> asymmetric buffer
            ubound = rot_coords[param][1] + buffer*abs(rot_coords[param][1])
            lbound = rot_coords[param][0] - buffer*abs(rot_coords[param][0])
        rot_coords[param] = [lbound,ubound]
    
    #put updated bounds into new dict (hopefully same order)
    buff_dict = {}
    i = 0
    for p in param_list: #bounds_dict.keys():
        if p == "gamma"+str(i):
            buff_dict[p] = rot_coords["r"+str(i)]
            i += 1
        else:
            buff_dict[p] = bounds_dict[p]
    if i == 0:
        print(" BOUND ERROR: could not match buffered bounds to original")
    return buff_dict


