# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:21:21 2019

@author: alexa
"""
import numpy as np

t = { #dictionary for getting both the target logic values and the correlated string 
    # binary labels to represent the probabilities of 1 or 0 (first column is 0, 2nd 1)
    "AND": np.array([[1, 0],\
                     [1, 0],\
                     [1, 0],\
                     [0, 1]], dtype=np.float32),
    
    "NAND": np.array([[0, 1],\
                      [0, 1],\
                      [0, 1],\
                      [1, 0]], dtype=np.float32),
    
    "OR": np.array([[1, 0],\
                    [0, 1],\
                    [0, 1],\
                    [0, 1]], dtype=np.float32),
    
    "NOR": np.array([[0, 1],\
                     [1, 0],\
                     [1, 0],\
                     [1, 0]], dtype=np.float32),
    
    "XOR": np.array([[1, 0],\
                     [0, 1],\
                     [0, 1],\
                     [1, 0]], dtype=np.float32) 
}

def stable_softmax(a, derivative=0):
#     exps = np.exp(a - np.max(a))
    exps = np.exp(a)
    if len(exps.shape) > 1:
        ans = np.zeros((exps.shape[0],2))
        numerator = exps.sum(axis=1).reshape((exps.shape[0], 1))
        numerator = np.append(numerator, numerator, axis=1)
    else:
#         ans = np.zeros((exps.shape[0],))
        ans = np.zeros((2,2))
        numerator = np.sum(exps)
        
    S = exps/numerator
    
    if derivative:
        if len(exps.shape) > 1: # NOT correct for more than one row
            for i in range(exps.shape[0]):
                ans[i, 0] = S[i, 1]*(1 - S[i, 0])
                ans[i, 1] = S[i, 0]*(1 - S[i, 1])
        else:
            kro_delta = 0
            for i in range(ans.shape[0]):
                for j in range(ans.shape[1]):                        
                    if i==j:
                        kro_delta = 1
                    else:
                        kro_delta = 0
                        
                    ans[i,j] = S[i]*(kro_delta - S[j])
        return ans
    return S