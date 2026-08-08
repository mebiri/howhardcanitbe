# -*- coding: utf-8 -*-
"""
Routines to apply conversion to given parameter lists
rotates hyperparameter cube to new user-defined rotated space
"""

import numpy as np

def rotate(th):
    return [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]

def inv_rotate(th):
    return np.linalg.inv(rotate(th))


def demo_rotation(X, coord_names, **kwargs):
    #coord_names will be low_level_coord_names from CEP
    print("dat received[0]:\n",X[0])
    
    #example kwargs usage: 
    if 'theta' in kwargs:
        theta = np.float(kwargs['theta'])
    else:
        theta = np.pi/4.0 #45 deg rotation
    
    #get gammas' indices in X_out(= X) from coord names
    x_rot = np.zeros((len(X),2))
    rot_cols = []
    for i in np.arange(2):
        #do one coord at a time
        indx = coord_names.index("x"+str(i)) 
        rot_cols.append(indx)
        x_rot[:,i] = X[:,indx]

    #apply transform
    x_prime = np.matmul(rotate(theta),x_rot.T).T
    X_out = X
    X_out[:,rot_cols] = x_prime #fill rotated values 
    return X_out


def inverse_demo_rotation(X, coord_names, **kwargs):
    print("dat received[0]:\n",X[0])
    
    if 'theta' in kwargs:
        theta = np.float(kwargs['theta'])
    else:
        theta = np.pi/4.0 #45 deg rotation
        
    x_prime_out = np.zeros((len(X),2))
    rot_cols = []
    for i in np.arange(2):
        #do one coord at a time
        indx = coord_names.index("x"+str(i)) #coord_names = dat_orig_names
        rot_cols.append(indx)
    
    #apply inverse
    x_prime_out = X[:,rot_cols]
    x_rot_post = np.matmul(inv_rotate(theta),x_prime_out.T).T
    X_out = X

    for i, col in enumerate(rot_cols): 
        X_out[:,col] = x_rot_post[:,i]
    return X_out


def get_bounds(param_list, bounds_dict, **kwargs):
    if 'buffer' in kwargs:
        buffer = np.float(kwargs['buffer'])
    else:
        buffer = 0.0
        print(" Warning: no buffer provided")
    
    #set rotated bounds to use
    rot_coords = {}
    rot_coords["r0"] = [-1.41421, 1.41421]
    rot_coords["r1"] = [-1.41421, 1.41421]
    
    for indx, param in enumerate(rot_coords.keys()):
        # apply hypercube buffer
        ubound = rot_coords[param][1] + buffer*abs(rot_coords[param][1])
        lbound = rot_coords[param][0] - buffer*abs(rot_coords[param][0])
        rot_coords[param] = [lbound,ubound]
    
    #put updated bounds into new dict (hopefully same order)
    buff_dict = {}
    i = 0
    for p in bounds_dict.keys():
        if p == "x"+str(i):
            buff_dict[p] = rot_coords["r"+str(i)]
            i += 1
        else:
            buff_dict[p] = bounds_dict[p]
    if i == 0:
        print(" BOUND ERROR: could not match buffered bounds to original")
    return buff_dict


