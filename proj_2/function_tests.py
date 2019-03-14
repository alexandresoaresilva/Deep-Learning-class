# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:21:21 2019

@author: alexa
"""
import numpy as np

def softmax_vector(x, derivative=0):
    x_exp = np.exp(x)
    
    if derivative: #this works only for 2-element vectors
        e = np.array([0, 0])
        e[:] = x_exp[0]*x_exp[1]/np.sum(x_exp)
        return e
    
    return np.exp(x)/np.sum(np.exp(x))