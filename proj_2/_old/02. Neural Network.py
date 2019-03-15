#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# from scipy.special import softmax
# from numpy import zeros, ones, eye
import matplotlib.pyplot as plt
# import tensorflow as tf


# In[2]:


X = np.array([[0,0],              [0,1],              [1,0],              [1,1]], dtype=np.float32)#1 appended in the last column is the bias

t = { #dictionary for getting both the target logic values and the correlated string 
    # binary labels to represent the probabilities of 1 or 0 (first column is 0, 2nd 1)
    "AND": np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32),
    "NAND": np.array([[0, 1], [0, 1], [0, 1], [1, 0]], dtype=np.float32),
    "OR": np.array([[1, 0], [0, 1], [0, 1], [0, 1]], dtype=np.float32),
    "NOR": np.array([[0, 1], [1, 0], [1, 0], [1, 0]], dtype=np.float32),
    "XOR": np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32) 
}

#learning reates
RHO = np.array([.0001,.001,.01,.01,.1,1], dtype=np.float32)


# In[3]:


def forward_prop_A(W1, b1, W2, b2, X):
    A1 = (X @ W1 + b1)[0]
    Z1 = logistic_sigmoid(A1) 
    A2 = (Z1 @ W2 + b2)[0]
    Y = softmax(A2)
    return A1, A2, Y


def logistic_sigmoid(x, derivative=0):
    sigm = 1/(1 + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def softmax(X, derivative=0):
    S = np.exp(X) / np.sum(np.exp(X))

    ans = np.zeros(X.shape)
    if derivative: 
        ans[0] = S[0]*(1 - S[1])
        ans[1] = -S[0]*S[1]
        return ans
    return S

def backprop_A(W2, A1, A2, X, Y, t):
    grad_mid_layer = (t-Y) @ -softmax(A2, derivative=1).T * logistic_sigmoid(A1, derivative=1) @ W2 @ X.T
    grad_output = (t-Y) @ -softmax(A2, derivative=1).T * logistic_sigmoid(A1, derivative=0) 
    
    grad_output = grad_output @ np.eye(grad_output.shape[0],M=2)
    return grad_mid_layer, grad_output

def forward_prop_A2(W1, b1, W2, b2, X):
    A1 = (X @ W1 + b1)[0]
#     Z1 = sigmoid(A1) 
    A2 = (A1 @ W2 + b2)[0]
    Y = softmax(A2)
    return A1, A2, Y

def backprop_A2(W2, A1, A2, X, Y, t):
    grad_mid_layer = (t-Y) @ -softmax(A2, derivative=1).T * np.eye(A1.shape[0]) @ W2 @ X.T
    grad_output = (t-Y) @ -softmax(A2, derivative=1).T * logistic_sigmoid(A1, derivative=0) 
    
    grad_output = grad_output @ np.eye(grad_output.shape[0],M=2)
    return grad_mid_layer, grad_output


# In[4]:


NO_UNITS_L1 = 2
W1 = np.random.randn(2,NO_UNITS_L1)
b1 = np.zeros((1,NO_UNITS_L1))
W2 = np.random.randn(NO_UNITS_L1,2)
b2 = np.zeros((1,2))


# In[8]:


rho = .01
batch = 1

if batch:
    for i in range(30_000):
            A1, A2, Y = forward_prop_A(W1, b1, W2, b2, X)
            grad_mid_layer, grad_output = backprop_A(W2, A1, A2, X, Y, t["AND"])

            W1 = W1 - rho*grad_mid_layer
            W2 = W2 - rho*grad_output.T
            b1 = b1 - rho*np.mean(grad_mid_layer)
            b2 = b2 - rho*np.mean(grad_output)
else:
    for i in range(10_000):
        for j in range(len(X[:,1])):
            A1, A2, Y = forward_prop_A2(W1, b1, W2, b2, X[j,:])
            grad_mid_layer, grad_output = backprop_A2(W2, A1, A2, X[j,:], Y, t["AND"][j,:])

            W1 = W1 - rho*grad_mid_layer
            W2 = W2 - rho*grad_output.T
            b1 = b1 - rho*np.mean(grad_mid_layer)
            b2 = b2 - rho*np.mean(grad_output)


# In[12]:


test = 3
A1, A2, Y = forward_prop_A(W1, b1, W2, b2, X[test,:])
# print("X: " + str(X[test,:]))
print("t: " + str(t["AND"][test,:]))
print("Y: " + str(Y))


# In[ ]:


#     (t["AND"]-Y) @ -softmax(A2, derivative=1).T * sigmoid(A1, derivative=0) 
#     grad_output = grad_output @ np.eye(grad_output.shape[0],M=2)


# In[ ]:





# In[ ]:





# In[ ]:




