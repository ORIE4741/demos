# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:21:02 2017

@author: Frank-Mia
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_crime(df):
    fig, ax = plt.subplots()
    labels = []
    for key, grp in df.groupby(['County']):
        grp = grp.sort_values(by = ["Year"])
        grp.plot(ax=ax, kind='line', x='Year', y='Index Rate')
        labels.append(key)
    lines, _ = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper')
    plt.show()
    
# load data
crime = pd.read_csv("crime.csv")
plot_crime(crime)

# First, let's see what the data look like.
#plot(crime,x=:Year,y=:Index_Rate,color=:County,Geom.line)

counties = set(crime["County"])
nonreporting_counties = set(crime[crime["Index Count"] > 1e5]["County"])
print(nonreporting_counties)

reporting_counties = []
nonreporting_counties = []

for county in counties:
    if len(crime[crime["County"]==county]["County"]) == 26:
        reporting_counties.append(county)
    else:
        nonreporting_counties.append(county)
print(reporting_counties)
print(nonreporting_counties)

for i,county in enumerate(nonreporting_counties):
    crime = crime[crime.County != county]

plot_crime(crime)

# just Tompkins county
tompkins = crime[crime["County"]=="Tompkins"]
tompkins.plot.scatter("Year","Index Rate",label="Tompkins")