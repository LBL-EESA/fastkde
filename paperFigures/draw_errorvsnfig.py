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
numDraws = 30

powMaxArray = range(11,20)
nmaxArray = [ 2**pp for pp in powMaxArray ]

masterNumKernels = []
masterRelativeError = []
masterMinusOneConvergence = []
masterIGood = []

for nmax,powmax in zip(nmaxArray,powMaxArray):

    numKernels = Bin.Bin(2,nmax,nperdecade = 10)

    relativeErrorsum = zeros([numKernels.count])
    relativeError = zeros([numKernels.count])
    countKernels = zeros([numKernels.count])

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
      print amin(mygaus(bkernel.x[iAboveOne]))
      print amin(bkernel.fSC[iAboveOne])

      #Estimate the number of kernels contributions per point
      tmpNumKernels = bkernel.fSC[iAboveOne]/bkernel.distributionThreshold
      binInds = asarray([numKernels.getIndex(num) for num in tmpNumKernels])
      relativeErrorsum[binInds] += tmprelativeError
      countKernels[binInds] += tmpones


    igood = nonzero(countKernels > 0)[0]
    relativeError[igood] = 100*relativeErrorsum[igood]/countKernels[igood]

    #Normalize the number of points to a relative fraction of the total
    #numKernels.center /= float(nmax)

    pointThreshold = 1e-2*float(nmax)
    iAbovePointThreshold = nonzero(numKernels.center > pointThreshold)[0]
    minusOneConvergence = numKernels.center**(-1.)
    minusOneConvergence *= average((relativeError[iAbovePointThreshold]))/average((minusOneConvergence[iAbovePointThreshold]))

    minusOneHalfConvergence = numKernels.center**(-1./2.)
    minusOneHalfConvergence *= average((relativeError[iAbovePointThreshold]))/average((minusOneHalfConvergence[iAbovePointThreshold]))


    masterNumKernels.append(numKernels)
    masterRelativeError.append(relativeError)
    masterMinusOneConvergence.append(minusOneConvergence)
    masterIGood.append(igood)

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

for numKernels,relativeError,minusOneConvergence,igood in \
        zip(masterNumKernels,masterRelativeError,masterMinusOneConvergence,masterIGood):
    #Plot the error convergence
    superp.plot(  numKernels.center[igood], relativeError[igood], \
                  color = 'black', \
                  linewidth = 2)
    #Plot the -1 convergence line
    superp.plot( numKernels.center, minusOneConvergence, \
            color = 'gray', \
            linestyle = '--', \
            linewidth = 2)
    #superp.plot( numKernels.center, minusOneHalfConvergence, \
    #        color = 'gray', \
    #        linewidth = 2)

superp.set_xlabel("Approximate number of kernel contributions, $\hat{n}$")
superp.set_ylabel("Relative error in BP11 estimate [%]")

superp.set_ylim([1e-2,1e6])

P.tight_layout()
P.savefig("errorvsnfig.eps")
