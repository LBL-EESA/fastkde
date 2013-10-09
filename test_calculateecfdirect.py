#!/usr/bin/env python
from numpy import *
import ftnbp11helper as ftn
import pylab as P

random.seed(0)

print ftn.calculateecfdirect.__doc__

def mySTDGaus1D(x):
  return 1./sqrt(2*pi) * exp(-x**2/2)

#Set the frequency points (Hermitian FFT-friendly)
numXPoints = 4097
xpoints = linspace(-20,20,numXPoints)
deltaX = xpoints[1] - xpoints[0]
dum = fft.fftfreq(numXPoints,deltaX/(2*pi))
tpoints = dum[0:len(dum)/2 + 1]

#First, a one-dimensional test
ndatapoints = 2**15
xrand = random.normal(loc=0.0,scale=1.0,size=[ndatapoints])
xrand = reshape(xrand,[1,ndatapoints])

ecf = ftn.calculateecfdirect( datapoints = xrand,  \
                              dataaverage = asarray([0.0]), \
                              datastd = asarray([1.0]), \
                              tpoints = tpoints,  \
                              freqspacesize = len(tpoints)  \
                            )


gausFunc = mySTDGaus1D(xpoints)
#Transform the gaussian function to fourier space -- calculate its
#actual characteristic function
gausFuncCF = fft.ihfft(fft.ifftshift(gausFunc),len(gausFunc))
gausFuncCF /= gausFuncCF[0]

print ecf[0],gausFuncCF[0]

P.subplot(211,xscale = "log",yscale="log")
P.plot(tpoints,abs(ecf)**2,'r-')
P.plot(tpoints,abs(gausFuncCF)**2,'b-')
P.subplot(212,xscale = "log",yscale="log")
P.plot(tpoints,abs(ecf-gausFuncCF)**2,'k-')
P.ylim([1e-10,10])
P.show()
