# -*- coding: utf-8 -*-
"""
ORIE 4741 Demo Regularized regressions python3

@author: Frank Zhang
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def generate_data(n,w):
    if "sparse" in str(type(w)):
        w = w.toarray()
    X = np.random.randn(n,w.shape[0])
    y = np.matmul(X,w)
    return X,y

def generate_noisy_data(n,w):
    X = np.random.randn(n,len(w))
    y = np.matmul(X,w) + 0.1*np.random.randn(n)
    return X,y

## ols is invariant
d = 10
n = 20
w = np.random.randn(d)
X, y = generate_data(n, w)

w,resid,rank,s = np.linalg.lstsq(X,y)
np.matmul(X,w)

yscale = 5*y
Xscale = 3*X
wscale,resid,rank,s = np.linalg.lstsq(Xscale, yscale)
np.matmul(Xscale,wscale) / 5

## ridge regression is not scaling invariant
(np.matmul(X.T,X) + np.eye(d)).shape

w,resid,rank,s = np.linalg.lstsq((np.matmul(X.T,X) + np.eye(d)), np.matmul(X.T,y))
np.matmul(X,w)

yscale = 5*y
Xscale = 3*X
wscale,resid,rank,s = np.linalg.lstsq((np.matmul(Xscale.T,Xscale) + np.eye(d)), np.matmul(Xscale.T,yscale))
np.matmul(Xscale,wscale) / 5

# standardize
def standardize(X,y):
    X_standard = X - np.mean(X,0)
    X_standard = np.matmul(X_standard, np.diag(1/np.std(X,0)))
    
    y_standard = y - np.mean(y)
    y_standard = y_standard / np.std(y)
    
    return X_standard, y_standard

Xs, ys = standardize(X,y)
w,resid,rank,s = np.linalg.lstsq((np.matmul(Xs.T,Xs) + np.eye(d)), np.matmul(Xs.T,ys))
np.matmul(Xs,w)

yscale = 5*y + 3000
Xscale = 3*X + 200

Xss, yss = standardize(Xscale,yscale)
wscale,resid,rank,s = np.linalg.lstsq((np.matmul(Xss.T,Xss) + np.eye(d)), np.matmul(Xss.T,yss))
np.matmul(Xss, wscale)

## let's compare different kinds of regularized regression
from sklearn.linear_model import Ridge,Lasso
lamda = 1
# ridge
## training the model
def ridge_regression(X,y, lamda=1):
    ridgeReg = Ridge(alpha=lamda, normalize=True)
    ridgeReg.fit(X,y)
#    pred = ridgeReg.predict(X)
#    return ridgeReg.coef_, ridgeReg.intercept_
    return ridgeReg

# lasso
def lasso(X,y, lamda=1):
    lassoReg = Lasso(alpha=lamda)
    lassoReg.fit(X,y)
    return lassoReg

# nnls
def nnls(X,y):
    nnlsReg = sp.optimize.nnls(X, y.flatten())
    return nnlsReg

# generate data
d = 30
w_randn = np.random.randn(d)
w_sparse = sp.sparse.rand(d, 1, density=0.5)
w_pos = sp.sparse.rand(d, 1, density=0.5)

# find best model for each type of data
w = w_randn

X,y = generate_data(30, w)
ridgeReg = ridge_regression(X,y, lamda = 1)
lassoReg = lasso(X,y, lamda=1)
nnlsReg = nnls(X,y)

w_ridge = ridgeReg.coef_.T
w_lasso = lassoReg.coef_
w_nonneg = nnlsReg[0]

plt.clf()
plt.hist(w_ridge)
plt.show()

plt.clf()
plt.hist(w_lasso,bins=50)
plt.show()

plt.clf()
plt.hist(w_nonneg)
plt.show()

# which fits data best?
Xtest,ytest = generate_data(20,w)

plt.clf()
plt.scatter(ytest,ridgeReg.predict(Xtest),label="ridge")
plt.scatter(ytest,lassoReg.predict(Xtest),label="lasso")
plt.scatter(ytest,np.matmul(Xtest,w_nonneg),label="NNLS")
plt.plot(ytest,ytest,label="true model")
plt.legend()
plt.xlabel("true value")
plt.ylabel("predicted value")
plt.show()

# cross validate over lambda
w = .5*np.random.randn(40)
X,y = generate_noisy_data(30, w)
Xtest,ytest = generate_noisy_data(30, w)

ls = np.linspace(0,5,num = 5/0.1+1)
error = []
for i in ls:
    ridgeReg = ridge_regression(X,y, lamda=i)
    error.append(np.sum(ytest - ridgeReg.predict(Xtest))**2)

plt.clf()
plt.plot(ls, error)
plt.show()
