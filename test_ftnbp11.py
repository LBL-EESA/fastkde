#!/usr/bin/env python
from numpy import *
import ftnbp11 as ftn

print ftn.lowesthypervolumefilter.__doc__

#Create an array that starts at 128 and decreases
#linearly as indices increase
testVec = range(8,1,-1)
t1,t2 = meshgrid(testVec,testVec)
testArray = t1**2 + t2**2 
#Set the last index to the first; the standard test will set this to 1
# but lowesthypervolumefilter() shouldn't
testArray.ravel()[-1] = testArray.ravel()[0]

thresh = 50

#Generate a standard
maskStandard = zeros(shape(testArray))
iAboveThresh = nonzero(testArray.ravel() >= thresh)[0]
maskStandard.ravel()[iAboveThresh] = 1
print ""
print "maskStandard: "
print ""
print maskStandard

icalcphi,imax = ftn.lowesthypervolumefilter( ecfsq = testArray, \
                                        ecfthreshold = thresh, \
                                        nvariables = 2, \
                                        ntpoints = 7)
maskTest = zeros(shape(testArray))
maskTest.ravel()[icalcphi[:imax]] = 1
print ""
print "maskTest: "
print ""
print maskTest
