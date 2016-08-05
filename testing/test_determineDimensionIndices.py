#!/usr/bin/env python
from numpy import *
import ftnecf as ftn

print(ftn.determineflattenedindex.__doc__)
dimsize = 4
ndims = 3
npoints = dimsize**ndims

testVals = asarray(range(1,dimsize**ndims + 1))
testArray = reshape(testVals,ndims*[dimsize])

print(testArray)

funcInds =  ftn.mapdimensionindices(npoints=npoints,dimsize=dimsize,ndims=ndims)
for i in range(npoints):
  
  k = ftn.determineflattenedindex(idimcounters=funcInds[i,:],dimsize=dimsize)

  
  indtuple = tuple(funcInds[i,:]-1)
  print(funcInds[i,:], \
        i+1,k,testArray[indtuple])
