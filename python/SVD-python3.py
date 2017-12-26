# -*- coding: utf-8 -*-
"""
SVD
Created on Tue Dec 26 12:26:27 2017

@author: Frank-Mia
"""
import numpy as np

# generate random data matrix
n,d = 6,4
X = np.random.randn(n,d)

# optional: give it linearly dependent columns
#X[:,2] = X[:,1]

U,sigma,V = np.linalg.svd(X, full_matrices=0) # M = 6, N = 4

np.dot(np.transpose(U),U)

np.dot(np.transpose(V),V)

sigma

# decomposition is just as good if we ignore the 0 in sigma and reduce r by 1
np.linalg.norm(X - np.dot(np.dot(U[:,0:3],np.diag(sigma[0:3])),np.transpose(V[:,0:3])))

# form data from noisy linear model
wn = np.random.randn(d)
y = np.dot(X,wn) + .1*np.random.randn(n)

# solve least squares problem to estimate w
w = np.dot(np.dot(np.dot(V,np.diag(sigma**(-1))),np.transpose(U)),y)
# w = np.dot(np.dot(np.dot(V[:,0:3],np.diag(sigma[0:3]**(-1))),np.transpose(U[:,0:3])),y)

# how good is our estimate?
np.linalg.norm(w - wn)

# compute mean square error
np.mean((y - np.dot(X,w))**2)

# let's use the shorthand
w_backslash,resid,rank,s = np.linalg.lstsq(X,y)
np.linalg.norm(w_backslash - w)








