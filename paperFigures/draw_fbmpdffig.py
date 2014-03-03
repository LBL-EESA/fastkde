#!/usr/bin/env python
from numpy import *
from numpy.random import randn
import time
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
#import pylab as P
import matplotlib.pyplot as P
from increments import \
                  incrementPDF as inc, \
                  bernacchiaDensityEstimate as be

class Timer():
   def __enter__(self): 
     self.start = time.time()
     return self
   def __exit__(self, *args): 
     self.end = time.time()
     self.delta = self.end - self.start

#Define a gaussian function for evaluation purposes
def mygaus(x,mu=0.,sig=1.):
  return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))

#*******************************************************************************
#*******************************************************************************
#**************** Generate a random sample of data and BP11 Estimate ***********
#*******************************************************************************
#*******************************************************************************

#Set the hurst parameter (times 10)
h10 = 6

distVals = 2**asarray(range(9))
hval = h10/10.
#Load data from a validation sample
datafile = "./increments/codeValidation/0{}.dat".format(h10)
ydata = genfromtxt(datafile)[:]

#Calculate the increment histogram
myInc = inc.incrementPDF( \
            data = ydata,\
            isPeriodic=False,\
            distanceValues = distVals, \
            fillValue = 0.,\
            doMomentEstimation=True,\
            moments = [])


#*******************************************************************************
#*******************************************************************************
#******************* Plot the Increment PDFS ***********************************
#*******************************************************************************
#*******************************************************************************
#Set the plot font
#TODO: Put this an included file
font = { 'family' : 'serif', \
         'size' : '18' }
matplotlib.rc('font', **font)

#Create a figure
figc = P.figure()

#Generate the main plot of the absolute difference between the two ECF methods
pdfplot = figc.add_subplot(111)

#Plot an actual gaussian
pdfplot.plot(myInc.bePDF[0].x,mygaus(myInc.bePDF[0].x), \
              color = "gray", \
              linewidth= 10)

for ix in range(myInc.num_distance_values):
  #Find good bins
  igood = nonzero(myInc.bePDF[ix].fSC[:] > 0)[0]
  #Plot the normalized PDF
  pdfplot.plot(myInc.bePDF[ix].x[igood],myInc.bePDF[ix].fSC[igood])


drawThresholdLines = False
if(drawThresholdLines):
    #Find the 1-item density
    #oneItemDensity = 1./(myInc.bePDF[0].numDataPoints*myInc.bePDF[0].deltaX)
    oneItemDensity = myInc.bePDF[0].distributionThreshold
    #plot the 1-item density
    pdfplot.plot(myInc.bePDF[0].x,oneItemDensity*ones([myInc.bePDF[0].numXPoints]),\
                color="gray",\
                linewidth=3, \
                linestyle='--')
    #plot the 10-item density
    pdfplot.plot(myInc.bePDF[0].x,100*oneItemDensity*ones([myInc.bePDF[0].numXPoints]),\
                color="gray",\
                linewidth=3,\
                linestyle='--')


#Set the axis labels
pdfplot.set_xlabel("Increment Value, $\Delta_x F$")
pdfplot.set_ylabel("Probability Density Estimate, $\hat{f}(\Delta_x F)$")

#Set the y axis properties
pdfplot.set_yscale('log') #log scale
pdfplot.set_xlim([-4,12])
pdfplot.set_ylim([1e-7,1])

cutBelowThreshold = True
if(cutBelowThreshold):
    pdfplot.set_ylim([myInc.bePDF[0].distributionThreshold,1])

#Add the sub-plot label
pdfplot.text(\
        0.9,0.9,"(c)", \
        bbox = {'facecolor':'white', 'color':'white', 'alpha':0.5, 'pad':10}, \
        fontsize = 18, \
        transform = pdfplot.transAxes)

#Add a subplot with the structure functions
pdfsub = P.axes([0.61,0.56,0.25,0.32],xscale="log",yscale="log")
for im in range(myInc.num_moments)[:3]:
  pdfsub.plot(myInc.distanceValues, \
              myInc.structureFunctions[im,:], \
              linewidth=2)
  #Add curve labels
  pdfsub.text(myInc.distanceValues[1],myInc.structureFunctions[im,1]/3., \
              "m={}".format(myInc.moments[im]), \
              fontsize = 12, \
              color = "gray")

#Limit the x-axis
pdfsub.set_xlim([1,5e2])
#pdfsub.set_ylim([1e-6,1e-1])
#Set the axis labels
pdfsub.set_xlabel("Increment Dist., $\Delta x$")
pdfsub.set_ylabel('$\langle|\Delta_x F|^m \\rangle$')

P.tight_layout()
#Save an eps version of the figure
figc.savefig("fbmpdffig.eps")

#*******************************************************************************
#*******************************************************************************
#******************* Plot the structure function exponents *********************
#*******************************************************************************
#*******************************************************************************

figd =  P.figure()
#Generate the main plot of the absolute difference between the two ECF methods
momentplot = figd.add_subplot(111)

momentplot.plot(myInc.moments,hval*myInc.moments,\
                color="gray",
                linewidth=2)
momentplot.plot(myInc.moments,myInc.structureFunctionsExponent,'ko')

#Add the sub-plot label
momentplot.text(\
        0.9,0.9,"(d)", \
        bbox = {'facecolor':'white', 'color':'white', 'alpha':0.5, 'pad':10}, \
        fontsize = 18, \
        transform = momentplot.transAxes)

#Set the axis labels
momentplot.set_xlabel("Structure Function Order, $m$")
momentplot.set_ylabel("Structure Function Exponent, $H_m$")

P.tight_layout()
#Save an eps version of the figure
figd.savefig("fbmmomentfig.eps")
