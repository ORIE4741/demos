# -*- coding: utf-8 -*-
"""
QR
Created on Tue Dec 26 12:10:46 2017
@author: Frank-Mia fz252@cornell.edu
"""
import numpy as np

# generate random data matrix
n,d = 6,4
X = np.random.randn(n,d)

# optional: give it linearly dependent columns
# X[:,2] = X[:,1]

# Understanding the pseudoinverse
# form pseudoinverse
Xd = np.linalg.pinv(X)

# X†X ≈ I_d
np.dot(Xd,X)
np.matmul(Xd,X)

# XX† !≈ I_n
np.dot(X,Xd)

Q,R = np.linalg.qr(X)

Q

R

np.dot(np.transpose(Q),Q)


# form data from noisy linear model
wn = np.random.randn(d)
y = np.dot(X,wn) + .1*np.random.randn(n)

# solve least squares problem to estimate w
w,resid,rank,s = np.linalg.lstsq(R,np.dot(np.transpose(Q),y))

# how good is our estimate?
np.linalg.norm(w - wn)

# compute mean square error
np.mean((y - np.dot(X,w))**2)

# let's use the shorthand
w_backslash,resid,rank,s = np.linalg.lstsq(X,y)
np.linalg.norm(w_backslash - w)

w_backslash



