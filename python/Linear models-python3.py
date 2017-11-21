# -*- coding: utf-8 -*-
"""
ORIE 4741 Linear models demo
Created on Mon Nov 20 07:06:55 2017

@author: Frank-Mia
"""

import matplotlib.pyplot as plt
import numpy as np

def plotline(w,b,xmin=-100.,xmax=100.,label=""):
    xsamples = np.array([xmin, xmax])
    plt.plot(xsamples, [w*x+b for x in xsamples], color="black", label=label)
    plt.legend()
    
"""plot function y=f(x)"""
def plotfunc(f, xmin=-100,xmax=100,nsamples=100,label=""):
    xsamples = np.linspace(xmin,xmax,nsamples)
    plt.plot(xsamples, [f(x) for x in xsamples], color="black", label=label)

# Generate and plot data
np.random.seed(2) # seed generator
n = 20
def sample_data(num_points):
    x = np.random.rand(num_points)*10
    y = .2 + .2*x + .1*np.sin(x) + .03*np.random.randn(num_points) - .1*(x/6)**2
    return x,y

x,y = sample_data(n)

"""function to plot the above data"""
def plotdata(x=x,y=y,margin=.05):
    plt.scatter(x,y, label="data")
    plt.xlabel("x")
    plt.ylabel("y")
    range_y = np.max(y) - np.min(y)
    range_x = np.max(x) - np.min(x)
    plt.ylim((np.min(y)-margin*range_y,np.max(y)+margin*range_y))
    plt.xlim((np.min(x)-margin*range_x,np.max(x)+margin*range_x))
    plt.legend()

plotdata()

## Approximating with the mean
# the mean solves a very simple least squares problem:
X = np.ones([n,1]) # 2-dimension
w,resid,rank,s = np.linalg.lstsq(X,y)

# check the solution to our least squares problem is the mean
np.abs(np.mean(y) - w[0])

# plot the fit
plotdata()
plotline(0, w[0], label="mean")

## Approximating with a line
X = np.vstack((x,np.ones(len(x)))).T # vertically stack and then transpose
w,resid,rank,s = np.linalg.lstsq(X,y)

# plot the fit
plotdata()
plotline(w[0], w[1], label="linear fit")

# plot fit on out of sample data
plotdata()
plotline(w[0], w[1])

xtest,ytest = sample_data(20)
plt.scatter(xtest,ytest,label="test")
plt.legend()

## Approximating with a polynomial
# first, construct a Vandermonde matrix
max_order = 10

X = np.zeros((n, max_order+1))
for k in range(max_order+1):
    X[:,k] = x**k

# solve least squares problem
w,resid,rank,s = np.linalg.lstsq(X,y)

"""computes our polynomial fit evaluated at x"""
def p(x, order = max_order, w = w):
    y = 0
    for k in range(order+1):
        y += w[k]*x**k
    return y

# plot fit
plotdata()
plotfunc(p, xmin=0, xmax=9)

# plot fit on out of sample data
plt.clf()
plotdata()
plotfunc(p, xmin=0, xmax=9)

xtest,ytest = sample_data(20)
plt.scatter(xtest,ytest,label="test")

## Choosing the best model order
max_model_order = 10
rmse = np.empty(max_model_order+1) # array to store root mean square model errors
xtest,ytest = sample_data(50) # generate test set

for model_order in range(max_model_order+1):
    # form Vandermonde matrix
    X = np.zeros((n, model_order+1))
    for k in range(model_order+1):
        X[:,k] = x**k
    
    # solve least squares problem
    w,resid,rank,s = np.linalg.lstsq(X,y)
    
    # compute test error
    ptest = [p(x, order=model_order, w=w) for x in xtest]
    rmse[model_order] = np.sqrt(np.mean((ytest - ptest)**2))

plt.plot(rmse)
plt.xlabel("model order")
plt.ylabel("rmse")
plt.legend()

##  Bootstrap estimators
# sample K data sets of n samples each and compute a model on each
# see how the models vary
n = 20
K = 10

models = np.zeros((K,2))
for k in range(K):
    xk,yk = sample_data(n)
    Xk = np.vstack((xk,np.ones(n))).T
    wk,resid,rank,s = np.linalg.lstsq(Xk,yk)
    models[k,:] = wk

# histogram of the distribution of the first coefficient
# could use to compute, eg, confidence intervals
plt.hist(models[:,0])
np.mean(models,0)
np.var(models,0)

# can sample with replacement using rand
np.random.choice(np.arange(1,15+1,1), size=5, replace=True)

# eg,
a = np.linspace(0.1, 1.5, num=15, endpoint=True)
s = np.random.choice(np.arange(1,15+1,1), size=5, replace=True)
np.vstack((s,a[s])).T

# resample K bootstrap data sets of n samples each and compute a model on each
# see how the models vary
n = 20
K = 100

x,y = sample_data(n)

models = np.zeros((K,2))
for k in range(K):
    mysample = np.random.choice(np.arange(0,n,1), size=n, replace=True)
    xk,yk = x[mysample], y[mysample]
    Xk = np.vstack((xk,np.ones(n))).T
    wk,resid,rank,s = np.linalg.lstsq(Xk,yk)
    models[k,:] = wk

plt.hist(models[:,0])
np.mean(models,0)

# as K increases, mean of the bootstrap models should converge to 
# the model fit on the original data set
X = np.vstack((x,np.ones(len(x)))).T

w,resid,rank,s = np.linalg.lstsq(X,y)
w

np.var(models,0)