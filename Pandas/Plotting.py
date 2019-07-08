# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 21:50:36 2019

@author: sarth
"""

import pandas as pd

df = pd.read_pickle("C:\\Users\\sarth\\Documents\\Pandas\\Dataset\\collection-master\\data_frame1.pickle")

#Simple plot
acq_years = df.groupby("acquisitionYear").size()

acq_years.head()

acq_years.plot()

#Using Matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"figure.autolayout":True,"axes.titlepad": 20})

fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
acq_years.plot(ax=subplot)
fig.show()

#Add axis labels
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
acq_years.plot(ax=subplot)
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
fig.show()

#Increase granularity 
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
acq_years.plot(ax=subplot, rot=45) #Rotate the tick values by 45^ for better view
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
subplot.locator_params(nbins=40, axis="x") # add 40 scale values to the x-axes
fig.show()

#Add log scale and grid
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
acq_years.plot(ax=subplot, rot=45, logy=True, grid=True) #Rotate the tick values by 45^ for better view
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
subplot.locator_params(nbins=40, axis="x") # add 40 scale values to the x-axes
fig.show()

#font setting
title_font = {"family":"Source sans pro",
              "color":"darkblue",
              "weight":"normal",
              "size":20,
              }
labels_font = {"family":"cosolas",
              "color":"darkred",
              "weight":"normal",
              "size":16,
              }

#Set font
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
acq_years.plot(ax=subplot, rot=45, logy=True, grid=True) #Rotate the tick values by 45^ for better view
subplot.set_xlabel("Acquisition Year", fontdict=labels_font, labelpad=10)
subplot.set_ylabel("Artworks Acquired",fontdict=labels_font)
subplot.locator_params(nbins=30, axis="x") # add 40 scale values to the x-axes
subplot.set_title("Tate Gallery Acquisitions",fontdict = title_font)
fig.show()

#Saving the plot
fig.savefig("C:\\Users\\sarth\\Documents\\Pandas\\Plots\\plot.png")
#Save as internet format file
fig.savefig("C:\\Users\\sarth\\Documents\\Pandas\\Plots\\plot.svg",format="svg")
