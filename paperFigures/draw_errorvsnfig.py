#!/usr/bin/env python
from numpy import *
from numpy.random import randn
from increments import  bernacchiaDensityEstimate as be, \
                        Bin
import time
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
#import pylab as P
import matplotlib.pyplot as P

import pdb

#Define a gaussian function for evaluation purposes
def mygaus(x,mu=0.,sig=1.):
  return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))

#*******************************************************************************
#*******************************************************************************
#**************** Generate a random sample of data and BP11 Estimate ***********
#*******************************************************************************
#*******************************************************************************
#Set the number of draws
numDraws = 200
#Set the size of the sample to calculate
powmax = 13
npow = asarray(range(powmax)) + 1.0
#Set the maximum sample size
nmax = 2**powmax

numPerBin = Bin.Bin(2,nmax,nperdecade = 30)

relativeErrorsum = zeros([numPerBin.count])
relativeError = zeros([numPerBin.count])
countPerBin = zeros([numPerBin.count])

for i in range(numDraws):
  #Create a random normal sample of this size
  randsample = random.normal(size = nmax)

  #Do the BP11 estimate
  bkernel = be.bernacchiaDensityEstimate(randsample,countThreshold=1)
  #Find parts of the estimate that have at least one data point per bin
  iAboveOne = bkernel.findGoodDistributionInds()

  #Calculate the relative error per bin
  tmprelativeError = abs(mygaus(bkernel.x[iAboveOne])-bkernel.fSC[iAboveOne])/mygaus(bkernel.x[iAboveOne])
  tmpones = tmprelativeError/tmprelativeError
  #Calculate the number of data points per bin
  #tmpNumPerBin = bkernel.numDataPoints*bkernel.deltaX*bkernel.fSC[iAboveOne]
  tmpNumPerBin = bkernel.fSC[iAboveOne]/bkernel.distributionThreshold
  binInds = asarray([numPerBin.getIndex(num) for num in tmpNumPerBin])
  relativeErrorsum[binInds] += tmprelativeError
  countPerBin[binInds] += tmpones


igood = nonzero(countPerBin > 0)[0]
relativeError[igood] = 100*relativeErrorsum[igood]/countPerBin[igood]

pointThreshold = 50.
iAbovePointThreshold = nonzero(numPerBin.center > pointThreshold)[0]
minusOneConvergence = numPerBin.center**(-1.)
minusOneConvergence *= average((relativeError[iAbovePointThreshold]))/average((minusOneConvergence[iAbovePointThreshold]))

minusOneHalfConvergence = numPerBin.center**(-1./2.)
minusOneHalfConvergence *= average((relativeError[iAbovePointThreshold]))/average((minusOneHalfConvergence[iAbovePointThreshold]))

#Normalize the number of points to a relative fraction of the total
numPerBin.center /= float(nmax)

#*******************************************************************************
#*******************************************************************************
#******************* Plot the FFT Method error and the ECF *********************
#*******************************************************************************
#*******************************************************************************

#Create a figure
fig = P.figure()

#Set the plot font
#TODO: Put this an included file
font = { 'family' : 'serif', \
         'size' : '18' }
matplotlib.rc('font', **font)

#Generate the main plot of the absolute difference between the two ECF methods
superp = fig.add_subplot(111,xscale = "log",yscale="log")
#Plot the error convergence
#superp.plot(  bkernel.x[iAboveOne], numPerBin, \
superp.plot(  numPerBin.center[igood], relativeError[igood], \
              color = 'black', \
              linewidth = 2)
#Plot the -1 convergence line
superp.plot( numPerBin.center, minusOneConvergence, \
        color = 'gray', \
        linestyle = '--', \
        linewidth = 2)
#superp.plot( numPerBin.center, minusOneHalfConvergence, \
#        color = 'gray', \
#        linewidth = 2)
superp.set_xlabel("Relative number of kernel contributions, $\hat{n}/N$")
superp.set_ylabel("Relative error in BP11 estimate [%]")

superp.set_ylim([1e-2,1e6])

P.tight_layout()
P.savefig("errorvsnfig.eps")
