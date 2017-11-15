# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:26:30 2017

@author: Frank-Mia
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import exp

crime = pd.read_csv("crime.csv")
# clean data
counties = set(crime["County"])
reporting_counties = []
nonreporting_counties = []

for county in counties:
    if len(crime[crime["County"]==county]["County"]) == 26:
        reporting_counties.append(county)
    else:
        nonreporting_counties.append(county)
for i,county in enumerate(nonreporting_counties):
    crime = crime[crime.County != county]

# just Tompkins county
tompkins = crime[crime["County"]=="Tompkins"]
tompkins = tompkins.sort_values(by="Year")
tompkins.reset_index(inplace=True,drop=True)
n = len(tompkins.Year)

# how about just using the year?
X = pd.DataFrame(tompkins["Year"])
X["Ones"] = 1.

y = tompkins["Index Count"]
w,resid,rank,s = np.linalg.lstsq(X,y)
tompkins["pred_linear"] = np.matmul(X,w)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(tompkins.Year,tompkins["Index Count"],c="r",label="Index Count")
ax.plot(tompkins.Year, tompkins["pred_linear"],label="linear model")
ax.set_xlabel("Year")
ax.set_ylabel("Crime")
ax.legend()

plt.show()

#2 Autoregressive models
X = pd.DataFrame(tompkins["Index Count"][:-1])
X["Ones"] = 1.
y = tompkins["Index Count"][1:]
w,resid,rank,s = np.linalg.lstsq(X,y)

tompkins["pred_ar1"] = np.nan
tompkins.loc[1:,"pred_ar1"] = np.matmul(X,w)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(tompkins.Year,tompkins["Index Count"],c="r",marker="o",label="Index Count")
ax.plot(tompkins.Year, tompkins["pred_linear"],c="b",linestyle="-",label="linear model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1"][1:],c="y",linestyle="-",label="AR1 model")
ax.set_xlabel("Year")
ax.set_ylabel("Crime")
ax.legend()
plt.show()

# how about using the year *and* the level of crime last year? (called ``lagged outcome'')
X = pd.DataFrame(tompkins["Year"][1:])
X.reset_index(inplace=True,drop=True)
X["Index Count"] = tompkins["Index Count"][:-1]
X["Ones"] = 1.
y = tompkins["Index Count"][1:]
w,resid,rank,s = np.linalg.lstsq(X,y)

tompkins["pred_ar1_lin"] = np.nan
tompkins.loc[1:,"pred_ar1_lin"] = np.matmul(X,w)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(tompkins.Year,tompkins["Index Count"],c="r",marker="o",label="Index Count")
ax.plot(tompkins.Year, tompkins["pred_linear"],c="b",linestyle="-",label="linear model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1"][1:],c="y",linestyle="-",label="AR1 model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1_lin"][1:], c="g",linestyle="-",label="AR + linear model")
ax.set_xlabel("Year")
ax.set_ylabel("Crime")
ax.legend()
plt.show()

## Smoothed models
alpha = 1
n = len(tompkins.Year)
X = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        X[i,j] = exp(-(tompkins.Year[i]-tompkins.Year[j])**2)

y = tompkins["Index Count"]
w,resid,rank,s = np.linalg.lstsq(X,y)

tompkins["pred_smooth"] = np.matmul(X,w)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(tompkins.Year,tompkins["Index Count"],c="r",marker="o",label="Index Count")
ax.plot(tompkins.Year, tompkins["pred_linear"],c="b",linestyle="-",label="linear model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1"][1:],c="y",linestyle="-",label="AR1 model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1_lin"][1:], c="g",linestyle="-",label="AR + linear model")
ax.plot(tompkins.Year, tompkins["pred_smooth"], c="c",linestyle="-",label="smoothed model")
ax.set_xlabel("Year")
ax.set_ylabel("Crime")
ax.legend()
plt.show()

##
alpha = 1
n = len(tompkins.Year)
nknots = round(n/2)
X = np.zeros((n,nknots))
for i in range(n):
    for j in range(nknots):
        X[i,j] = exp(-(tompkins.Year[i]-tompkins.Year[2*j])**2)
X = np.append(X, np.ones((n,1)), 1)
y = tompkins["Index Count"]
w,resid,rank,s = np.linalg.lstsq(X,y)

tompkins["pred_smooth"] = np.matmul(X,w)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(tompkins.Year,tompkins["Index Count"],c="r",marker="o",label="Index Count")
ax.plot(tompkins.Year, tompkins["pred_linear"],c="b",linestyle="-",label="linear model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1"][1:],c="y",linestyle="-",label="AR1 model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1_lin"][1:], c="g",linestyle="-",label="AR + linear model")
ax.plot(tompkins.Year, tompkins["pred_smooth"], c="c",linestyle="-",label="smoothed model")
ax.set_xlabel("Year")
ax.set_ylabel("Crime")
ax.legend()
plt.show()

## regularize + smooth
alpha = 1
n = len(tompkins.Year)
X = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        X[i,j] = exp(-(tompkins.Year[i]-tompkins.Year[j])**2)

y = tompkins["Index Count"] - tompkins["pred_linear"]
temp = np.linalg.lstsq(np.matmul(X.T,X)+np.eye(n),X.T)
w = np.matmul(temp[0],y)

tompkins["pred_smooth_reg"] = np.matmul(X,w) + tompkins["pred_linear"]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(tompkins.Year,tompkins["Index Count"],c="r",marker="o",label="Index Count")
ax.plot(tompkins.Year, tompkins["pred_linear"],c="b",linestyle="-",label="linear model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1"][1:],c="y",linestyle="-",label="AR1 model")
ax.plot(tompkins.Year[1:], tompkins["pred_ar1_lin"][1:], c="g",linestyle="-",label="AR + linear model")
ax.plot(tompkins.Year, tompkins["pred_smooth"], c="c",linestyle="-",label="smoothed model")
ax.plot(tompkins.Year, tompkins["pred_smooth_reg"], c="m",linestyle="-",label="reg smoothed model")
ax.set_xlabel("Year")
ax.set_ylabel("Crime")
ax.legend()
plt.show()
