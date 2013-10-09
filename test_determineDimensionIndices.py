#!/usr/bin/env python
from numpy import *
import ftnbp11helper as ftn

print ftn.determinedimensionindices.__doc__
dimsize = 4
ndims = 3

testVals = range(1,dimsize**ndims + 1)
testArray = reshape(testVals,ndims*[dimsize])

print testArray

for i in range(1,dimsize**ndims + 1):
  funcInds =  ftn.determinedimensionindices(i=i,dimsize=dimsize,ndims=ndims)
  funcInds -= 1 
  
  print funcInds + 1, \
        i,testArray[funcInds[0],funcInds[1],funcInds[2]]
