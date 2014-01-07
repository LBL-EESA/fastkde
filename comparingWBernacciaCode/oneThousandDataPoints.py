#!/usr/bin/env python
from numpy import *
import numpy.random
import bernacchiaDensityEstimate as bp11
import pylab as P

#Define a gaussian function for evaluation purposes
def myStdGaus(x,mu=0.,sig=1.):
    return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))


randSample = random.randn(1000)

beEstimate = bp11.bernacchiaDensityEstimate(randSample)

xEstimate = beEstimate.deStandardizeX()
yEstimate = beEstimate.deStandardizePDF()

igood = beEstimate.findGoodDistributionInds()

myfig = P.figure()
myplot = myfig.add_subplot(111,yscale="log")
myplot.plot(xEstimate[igood],yEstimate[igood],'r-')
myplot.plot(xEstimate[igood],myStdGaus(xEstimate[igood]),'b-')

myplot.set_ylim([1e-4,10])

P.show()
