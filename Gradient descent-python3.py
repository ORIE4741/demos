# -*- coding: utf-8 -*-
"""
Gradient Descent

Created on Sat Dec 23 17:13:18 2017

@author: Frank-Mia
"""
import numpy as np
import matplotlib.pyplot as plt

# generate random data matrix
n,d = 6,4
X = np.random.randn(n,d)

# optional: give it linearly dependent columns
# X[:,2] = X[:,1]


# form data from noisy linear model
wn = np.random.randn(d)
y = np.matmul(X,wn) + 0.1 * np.random.randn(n)

# look at least squares objective as a function of w
def f(w):
    return (np.linalg.norm(y - np.dot(X,w),2)**2)

def deltaf(w):
     return (2*np.dot(np.dot(np.transpose(X),X),w) - 2*np.dot(np.transpose(X),y))
 
# gradient at w0 approximates f(w) near w0
w0 = np.random.randn(d)
v = np.random.randn(d)
alphas = np.linspace(-2,2,100)
delta_f = deltaf(w0)
#delta_f = 2*np.dot(np.dot(np.transpose(X),X),w0) - 2*np.dot(np.transpose(X),y)

alpha = alphas[0]
plt.plot(alphas, [f(w0 + alpha*v) for alpha in alphas], color="black", label=r"$f(w_0 + \alpha v)$")
plt.plot(alphas, [f(w0) + alpha*np.matmul(delta_f, v) for alpha in alphas], color="red",label=r"$f(w_0) + \alpha (\nabla f)^T v$")
plt.legend()
plt.xlabel(r"$\alpha$")
plt.show()

# function decreases fastest in the -∇f(w) direction
w0 = np.random.randn(d)
v = np.random.randn(d)
delta_f = delta_f = deltaf(w0)
v_normalized = v/np.linalg.norm(v)
delta_f_normalized = delta_f/np.linalg.norm(delta_f)
alphas = np.linspace(-5,5,100)
plt.figure()
plt.plot(alphas, [f(w0 + alpha*v_normalized) for alpha in alphas])
plt.plot(alphas, [f(w0 + alpha*delta_f_normalized) for alpha in alphas],color="red")
plt.legend()
plt.show()

# gradient descent

alpha = .1      # small constant step size
w = np.random.randn(d)     # start at a random w
fks = []  # a list to record all the values f(w) we see
fks.append(f(w)) # record the initial value

# start descending!
for k in range(0,100):
    w -= alpha*deltaf(w) # take a gradient step
    fks.append(f(w))            # record its value

plt.plot(fks, label="sum of square errors")
plt.semilogy(fks)
plt.loglog(fks)
plt.show()