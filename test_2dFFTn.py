#!/usr/bin/env python
from numpy import *

#Create an array that starts at 128 and decreases
#linearly as indices increase
testVec = range(8,1,-1)
t1,t2 = meshgrid(testVec,testVec)
testArray = t1**2 + t2**2 
veclenhalf = len(testVec)/2 + 1

testFFT = fft.ifftn(fft.ifftshift(testArray))
firstHalfSlice = tuple(2*[slice(0,veclenhalf)])
lastHalfSlice = tuple(2*[slice(veclenhalf,None)])
reverserSlice = tuple(2*[slice(-1,None,-1)])
#Extract the positive frequencies
testFFTPosFreq = testFFT[firstHalfSlice]
testFFTNegFreq = testFFT[lastHalfSlice][reverserSlice]

#Check for Hermetian symmetry
#print array_equiv(testFFTPosFreq,testFFTNegFreq)
#print testFFTPosFreq-testFFTNegFreq

#Attempt to regenerate the original array
testFFTNew = (0.0+0.0j)*zeros(2*[2*veclenhalf-1])
#print shape(testFFTNew)
testFFTNew[firstHalfSlice] = testFFTPosFreq
reverserSlice = tuple(2*[slice(-1,0,-1)])
testFFTNew[lastHalfSlice] = testFFTPosFreq[reverserSlice]

#print testFFTNew

firstHalfSlice = tuple(2*[slice(0,3)])
lastHalfSlice = tuple(2*[slice(3,None)])
a = reshape(asarray(range(25)),[5,5])
b = reshape(a.ravel()[::-1],[5,5])
#print a
#print b

numDims = 3

a = [-3,-2,-1,0,1,2,3]
lenhalf = len(a)/2 + 1
aa = asarray(meshgrid(*tuple(numDims*[a])))
c = sum(aa**2,axis=0)
firstHalfSlice = tuple(numDims*[slice(0,lenhalf)])
d = fft.ifftshift(c)[firstHalfSlice]
print c
print d

padSequence = numDims*[tuple([lenhalf-1,0])]
cRemade = pad(d,padSequence,'reflect')
print array_equiv(cRemade,c)

