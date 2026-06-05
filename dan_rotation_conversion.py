# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 03:59:35 2026

@author: marce
"""

import numpy as np

dan_rot = [[0.43801, -0.53573, 0.52661, -0.49379],
           [-0.76705, 0.17169, 0.31255, -0.53336],
           [0.45143, 0.67967, -0.19454, -0.54443],
           [0.12646, 0.47070, 0.76626, 0.41868]]
scaled_mean = [0.89421, 0.33878, -0.07894, 0.00393]
scaled_sig = [0.35700, 0.25769, 0.05452, 0.00312]
dan_inv = np.linalg.inv(dan_rot)

def dan_rotation(X, coord_names, low_level_coord_names, **kwargs):
    print("dat received[0]:\n",X[0])
    #get gammas' indices in X_out(= X) from coord names
    r_tilde = np.zeros((len(X),4))
    rot_cols = []
    for i in np.arange(4):
        #do one coord at a time
        indx = low_level_coord_names.index("gamma"+str(i))
        rot_cols.append(indx)

        #convert gammas to r_tilde using equation: r_tilde = (gamma - u)/sig
        r_tilde[:,i] = (X[:,indx] - scaled_mean[i])/scaled_sig[i]

    #apply transform: r_prime = S*r_tilde ( [4 x 4].([N x 4].T) )
    r_prime = np.matmul(dan_rot,r_tilde.T).T
    X_out = X
    X_out[:,rot_cols] = r_prime #TODO check that this works
    return X_out


def inverse_dan_rotation(X, coord_names, low_level_coord_names, **kwargs):
    #apply inverse: S-1*r_prime = S-1*S*r_tilde = r_tilde ( [4 x 4].([N x 4].T) )
    print("dat received[0]:\n",X[0])
    r_prime_out = np.zeros((len(X),4))
    rot_cols = []
    for i in np.arange(4):
        #do one coord at a time
        indx = low_level_coord_names.index("gamma"+str(i)) #low_level_coord_names = dat_orig_names
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


