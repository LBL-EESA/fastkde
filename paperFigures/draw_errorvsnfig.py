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
powmax = 17
npow = asarray(range(powmax)) + 1.0
#Set the maximum sample size
nmax = 2**powmax

numPerBin = Bin.Bin(1,1000,nperdecade = 30)

esqsum = zeros([numPerBin.count])
esq = zeros([numPerBin.count])
countPerBin = zeros([numPerBin.count])

for i in range(numDraws):
  #Create a random normal sample of this size
  randsample = random.normal(size = nmax)

  #Do the BP11 estimate
  bkernel = be.bernacchiaDensityEstimate(randsample,countThreshold=1)
  #Find parts of the estimate that have at least one data point per bin
  iAboveOne = bkernel.findGoodDistributionInds()

  #Calculate the squared error per bin
  tmpesq = abs(mygaus(bkernel.x[iAboveOne])-bkernel.fSC[iAboveOne])/mygaus(bkernel.x[iAboveOne])
  tmpones = tmpesq/tmpesq
  #Calculate the number of data points per bin
  #tmpNumPerBin = bkernel.numDataPoints*bkernel.deltaX*bkernel.fSC[iAboveOne]
  tmpNumPerBin = bkernel.fSC[iAboveOne]/bkernel.distributionThreshold
  binInds = asarray([numPerBin.getIndex(num) for num in tmpNumPerBin])
  esqsum[binInds] += tmpesq
  countPerBin[binInds] += tmpones


igood = nonzero(countPerBin > 0)[0]
esq[igood] = 100*esqsum[igood]/countPerBin[igood]

minusOneConvergence = numPerBin.center**(-1.)
minusOneConvergence *= average((esq))/average((minusOneConvergence))

minusOneHalfConvergence = numPerBin.center**(-1./2.)
minusOneHalfConvergence *= average((esq))/average((minusOneHalfConvergence))

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
superp.plot(  numPerBin.center[igood], esq[igood], \
              color = 'black', \
              linewidth = 2)
#Plot the -1 convergence line
superp.plot( numPerBin.center, minusOneConvergence, \
        color = 'gray', \
        linestyle = '--', \
        linewidth = 2)
superp.plot( numPerBin.center, minusOneHalfConvergence, \
        color = 'gray', \
        linewidth = 2)
superp.set_xlabel("Number of data points per interval, $\hat{n}$")
superp.set_ylabel("Relative error in BP11 estimate [%]")

P.tight_layout()
P.savefig("errorvsnfig.eps")
