#!/usr/bin/env python
from numpy import *
import ftnbp11helper as ftn

random.seed(0)

print ftn.calculateecfdirect.__doc__

def mySTDGaus1D(x):
  return 1./sqrt(2*pi) * exp(-x**2/2)

def mySTDGaus2D(x,y):
  return 1./(2*pi) * exp(-(x**2 + y**2)/2)

#Set the frequency points (Hermitian FFT-friendly)
numXPoints = 513
xpoints = linspace(-20,20,numXPoints)
deltaX = xpoints[1] - xpoints[0]
dum = fft.fftfreq(numXPoints,deltaX/(2*pi))
tpoints = dum[0:len(dum)/2 + 1]

doOneDimensionalTest = False
if(doOneDimensionalTest):
  import pylab as P
  #First, a one-dimensional test
  ndatapoints = 2**10
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
  P.ylim([1e-10,1])
  P.subplot(212,xscale = "log",yscale="log")
  P.plot(tpoints,abs(ecf-gausFuncCF)**2,'k-')
  P.ylim([1e-10,10])
  P.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

xp2d,yp2d = meshgrid(xpoints,xpoints)
mygaus2d = mySTDGaus2D(xp2d,yp2d)

#mygauscf = fft.ihfft(fft.ifftshift(mygaus2d),len(tpoints))
tmpcf = fft.ifft(fft.ifftshift(mygaus2d))
mygauscf = tmpcf[:len(tpoints),:len(tpoints)]
mygauscf /= mygauscf[0,0]


doPlotBivariateNormal = False
if(doPlotBivariateNormal):
  fig = plt.figure()
  ax = fig.add_subplot(111,projection='3d')
  ax.plot_surface(xp2d,yp2d,mygaus2d)
  plt.show()
  quit()

ndatapoints = 2**10
nvariables = 2
xyrand = random.normal(loc=0.0,scale=1.0,size=[nvariables,ndatapoints])
ecf = ftn.calculateecfdirect( datapoints = xyrand,  \
                              dataaverage = asarray([0.0,0.0]), \
                              datastd = asarray([1.0,1.0]), \
                              tpoints = tpoints,  \
                              freqspacesize = len(tpoints)**nvariables  \
                            )
ecf = reshape(ecf,nvariables*[len(tpoints)])


tp2d,wp2d = meshgrid(tpoints,tpoints)

print ecf[0,0],mygauscf[0,0]

fig = plt.figure()
ax = fig.add_subplot(211,projection='3d')
#ax.plot_surface(tp2d[::4,::4],wp2d[::4,::4],(abs(ecf-mygauscf)**2/abs(mygauscf)**2)[::4,::4])
ax.plot_wireframe(tp2d[::4,::4],wp2d[::4,::4],(abs(ecf)**2)[::4,::4],color='r')
ax.plot_wireframe(tp2d[::4,::4],wp2d[::4,::4],(abs(mygauscf)**2)[::4,::4],color='b')
#plt.xlim([0,5])
#plt.ylim([0,5])
#ax = fig.add_subplot(212,projection='3d')
plt.show()
