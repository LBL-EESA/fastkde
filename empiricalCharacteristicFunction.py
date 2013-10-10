#!/usr/bin/env python
from numpy import *
import ftnecf as ftn

#*****************************************************************************
#*****************************************************************************
#******************* Frequency/real-space conversions ************************
#*****************************************************************************
#*****************************************************************************
def calcXfromT(tpoints):
  """Calculates real space points given a set of hermetian frequency points (the >0 half). """
  #Use fftfreq to produce a set of frequencies that correspond to the fourier
  #transform of a signal on the tpoints points
  deltaT = tpoints[1] - tpoints[0]
  return  fft.fftshift(fft.fftfreq(2*len(tpoints)- 1,deltaT/(2*pi)))

def calcTfromX(xpoints):
  """Calculates frequency points given a signal in real space. """
  #Use fftfreq to produce a set of frequencies that correspond to the fourier
  #transform of a signal on the x points
  deltaX = xpoints[1] - xpoints[0]
  dum =  fft.fftfreq(len(xpoints),deltaX/(2*pi))

  #Since we will only be transforming hermitian signals (the ECF is hermitian
  #by construction), only do calculations for 0-or-positive frequencies
  return dum[0:len(dum)/2+1]

class ECF:

  def __init__( self,\
                inputData, \
                tpoints, \
                dataAverage = None, \
                dataStandardDeviation = None, \
                useFFTApproximation = True):

    #TODO: check for regularity
    self.tpoints = tpoints

    if(dataAverage != None):
      #TODO: Check for proper dimensioning
      self.dataAverage = dataAverage
    else:
      self.dataAverage = average(inputData,1)


    if(dataStandardDeviation != None):
      #TODO: Check for proper dimensioning
      self.dataStandardDeviation = dataStandardDeviation
    else:
      self.dataStandardDeviation = std(inputData,1)

    #Get the data shape (nvariables,ndatapoints)
    #TODO: Check for proper dimensioning
    dshape = shape(inputData)
    self.nvariables = dshape[0]
    self.ndatapoints = dshape[1]


    self.useFFTApproximation = useFFTApproximation


    if(self.useFFTApproximation):
      self.ECF = self.__calculateECFbyFFT__(  inputData,\
                                              self.dataAverage, \
                                              self.dataStandardDeviation)
    else:
      self.ECF = self.__calculateECFDirect__( inputData,\
                                              self.dataAverage, \
                                              self.dataStandardDeviation)

    return

  #*****************************************************************************
  #*****************************************************************************
  #******************* __calculateECFDirect__() ********************************
  #*****************************************************************************
  #*****************************************************************************
  def __calculateECFDirect__(self,mydata,myaverage,mystddev):
    """Directly calculate the fourier-space representation of the input data"""

    #Call a Fortran routine to do an efficient calculation of the density in
    #fourier space.

    ecf = ftn.calculateecfdirect( datapoints = mydata,  \
                                  dataaverage = myaverage, \
                                  datastd = mystddev, \
                                  tpoints = self.tpoints,  \
                                  freqspacesize = len(self.tpoints)**self.nvariables  \
                              )
    ecf = reshape(ecf,self.nvariables*[len(tpoints)])
    return ecf

  #*****************************************************************************
  #*****************************************************************************
  #******************* __calculateECFbyFFT__() *********************************
  #*****************************************************************************
  #*****************************************************************************
  def __calculateECFbyFFT__(self,mydata,myaverage,mystddev):
    """Use the non-uniform FFT method to estimate the fourier representation of
    the input data."""

    xpoints = calcXfromT(self.tpoints)

    #Calculate details of the gaussian kernel
    tau = 1.5629
    nspread = 30
    nspreadhalf = nspread/2
    fourTau = 4*tau

    kde = ftn.calculatekerneldensityestimate( datapoints = mydata,  \
                                              dataaverage = myaverage, \
                                              datastd = mystddev, \
                                              xpoints = xpoints,  \
                                              nspreadhalf = nspreadhalf, \
                                              fourtau = fourTau, \
                                              realspacesize = len(xpoints)**self.nvariables  \
                                            )

    kde = reshape(kde,self.nvariables*[len(xpoints)])

    tmpfft = fft.ifft(fft.ifftshift(kde))
    kdeFFT = tmpfft[:len(self.tpoints),:len(self.tpoints)]

    tpointgrids = meshgrid(*self.nvariables*[self.tpoints])

    #Deconvolve FFT(kde) (divide by the FFT of the gaussian) to obtain the ECF estimate
    deltaX = xpoints[1] - xpoints[0]
    ecf = kdeFFT*exp(tau*(sum((tpointgrids*deltaX**2),axis=0)))/kdeFFT[0,0]

    return ecf



if(__name__ == "__main__"):
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt


  random.seed(0)


  def mySTDGaus2D(x,y):
    return 1./(2*pi) * exp(-(x**2 + y**2)/2)

  #Set the frequency points (Hermitian FFT-friendly)
  numXPoints = 513
  xpoints = linspace(-20,20,numXPoints)
  deltaX = xpoints[1] - xpoints[0]
  dum = fft.fftfreq(numXPoints,deltaX/(2*pi))
  tpoints = dum[0:len(dum)/2 + 1]

  xp2d,yp2d = meshgrid(xpoints,xpoints)
  mygaus2d = mySTDGaus2D(xp2d,yp2d)
  tmpcf = fft.ifft(fft.ifftshift(mygaus2d))
  mygauscf = tmpcf[:len(tpoints),:len(tpoints)]
  mygauscf /= mygauscf[0,0]


  ndatapoints = 2**10
  nvariables = 2
  xyrand = random.normal(loc=0.0,scale=1.0,size=[nvariables,ndatapoints])

  ecfFFT = ECF(xyrand,tpoints,useFFTApproximation=True).ECF
  ecfDFT = ECF(xyrand,tpoints,useFFTApproximation=False).ECF

  tp2d,wp2d = meshgrid(tpoints,tpoints)

  fig = plt.figure()
  ax = fig.add_subplot(211,projection='3d')

  ax.plot_wireframe(tp2d[::4,::4],wp2d[::4,::4],(abs(ecfFFT)**2)[::4,::4],color='r')
  ax.plot_wireframe(tp2d[::4,::4],wp2d[::4,::4],(abs(ecfDFT)**2)[::4,::4],color='b')

  ax2 = fig.add_subplot(212,xscale="log",yscale="log")

  errorK = average(abs(ecfFFT-ecfDFT),0)

  print ecfFFT[0,0],ecfDFT[0,0],mygauscf[0,0]
  ax2.plot(tpoints,abs(average(ecfFFT,0))**2,'r-')
  ax2.plot(tpoints,abs(average(ecfDFT,0))**2,'b-')
  ax2.plot(tpoints,abs(average(mygauscf,0))**2,'k-')

  plt.show()


