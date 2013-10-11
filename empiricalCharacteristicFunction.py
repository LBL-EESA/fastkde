#!/usr/bin/env python
from numpy import *
import ftnecf as ftn

#*****************************************************************************
#*****************************************************************************
#******************* Frequency/real-space conversions ************************
#*****************************************************************************
#*****************************************************************************
def calcXfromT(tpoints):
  """Calculates real space points given a set of hermetian frequency points. """
  #Use fftfreq to produce a set of frequencies that correspond to the fourier
  #transform of a signal on the tpoints points
  deltaT = tpoints[1] - tpoints[0]
  return  fft.fftshift(fft.fftfreq(len(tpoints),deltaT/(2*pi)))

def calcTfromX(xpoints):
  """Calculates frequency points given a signal in real space. """
  #Use fftfreq to produce a set of frequencies that correspond to the fourier
  #transform of a signal on the x points
  deltaX = xpoints[1] - xpoints[0]
  return fft.fftshift(fft.fftfreq(len(xpoints),deltaX/(2*pi)))

class ECF:

  def __init__( self,\
                inputData, \
                tpoints, \
                dataAverage = None, \
                dataStandardDeviation = None, \
                doStoreConvolution = False, \
                useFFTApproximation = True):
    """
    Calculates the empirical characteristic function of arbitrary sets of
    variables.

    Uses either the direct Fourier transform or nuFFT method (described by
    O'Brien et al. (2014, J. Roy. Stat. Soc. C) to calculate the Fourier transform
    of the data to yield the ECF.

    """

    #Set whether we use the nuFFT approximation
    self.useFFTApproximation = useFFTApproximation

    #Get the data shape (nvariables,ndatapoints)
    dshape = shape(inputData)
    rank = len(dshape)
    if(rank != 2):
      raise ValueError,"inputData must be a rank-2 array of shape [nvariables,ndatapoints]"
    #Extract the number of variables
    self.nvariables = dshape[0]
    #Extract the number of data points
    self.ndatapoints = dshape[1]


    #Set the frequency points
    self.tpoints = tpoints
    #Check for regularity if we are doing nuFFT
    if(self.useFFTApproximation):
      #Get the spacing of the first two points
      dt = tpoints[1]-tpoints[0]
      #Get the spacing of all points
      deltaT = tpoints[1:] - tpoints[:-1]
      #Get the difference between these spacings
      deltaTdiff = deltaT - dt
      fTolerance = dt/1e6
      #Check that all these differences are less than 1/1e6
      if(not all(abs(deltaTdiff < fTolerance))):
        raise ValueError,"tpoints must be regularly spaced if useFFTApproximation is True"

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


    #Set whether we store the convolved version
    #of the data used in the FFT estimate; mainly
    #used for diagnostic purposes
    self.doStoreConvolution = doStoreConvolution
    #Initialize the convolvedData member to nothing
    self.convolvedData = None



    if(self.useFFTApproximation):
      #Calculate the ECF using the fast method
      self.ECF = self.__calculateECFbyFFT__(  inputData,\
                                              self.dataAverage, \
                                              self.dataStandardDeviation)
    else:
      #Calculate the ECF using the slow (direct, but exact) method
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
    #Reshape the flattened array into a proper multidimensional array
    ecf = reshape(ecf,self.nvariables*[len(self.tpoints)])


    return ecf

  #*****************************************************************************
  #*****************************************************************************
  #******************* __calculateECFbyFFT__() *********************************
  #*****************************************************************************
  #*****************************************************************************
  def __calculateECFbyFFT__(self,mydata,myaverage,mystddev):
    """Use the non-uniform FFT method to estimate the fourier representation of
    the input data."""

    #Determine the points corresponding to the x-grid
    xpoints = calcXfromT(self.tpoints)
    #Calculate the grid spacing
    deltaX = xpoints[1] - xpoints[0]

    #Calculate details of the gaussian kernel
    tau = 1.5629
    nspread = 30
    nspreadhalf = nspread/2
    fourTau = 4*tau

    if(self.nvariables > 1):
      #If this is a multidimensional ECF, use meshgrid to create a multidimensional grid
      #of all the frequency points
      tpointgrids = asarray(meshgrid(*self.nvariables*[self.tpoints]))
    else:
      #Otherwise, just create a [1,len(tpoints)] array of the frequency points
      tpointgrids = asarray([self.tpoints])

    #Do a fast kernel density estimate, using the Greengard and Lee (2004, SIAM)
    #kernel parameters
    kde = ftn.calculatekerneldensityestimate( datapoints = mydata,  \
                                              dataaverage = myaverage, \
                                              datastd = mystddev, \
                                              xpoints = xpoints,  \
                                              nspreadhalf = nspreadhalf, \
                                              fourtau = fourTau, \
                                              realspacesize = len(xpoints)**self.nvariables  \
                                            )

    #Reshape the KDE estimate into a multidimensional array
    kde = reshape(kde,self.nvariables*[len(xpoints)])
    if(self.doStoreConvolution):
      self.convolvedData = kde

    #Take the FFT of the KDE estimate (ifftshift is used to reorder the kde
    #array such that 0 is the lowest corner [first index] of the array, as
    #required by ifft)
    #And fftshift is used to put the zero-frequency in the center of the array
    kdeFFT = fft.fftshift(fft.ifftn(fft.ifftshift(kde)))

    #Deconvolve FFT(kde) (divide by the FFT of the gaussian) to obtain the ECF estimate
    midPointAccessor = tuple(self.nvariables*[(len(self.tpoints)-1)/2])
    ecf = kdeFFT*exp(tau*sum((tpointgrids*deltaX)**2,axis=0))/kdeFFT[midPointAccessor]
    #ecf = kdeFFT/kdeFFT[midPointAccessor]

    return ecf



if(__name__ == "__main__"):


  #Set the random seed to 0 so the results are repetable
  random.seed(0)

  #Flag whether to do the 1-D test
  doOneDimensionalTest = False
  if(doOneDimensionalTest):
    import pylab as P
    def mySTDGaus1D(x):
      return 1./sqrt(2*pi) * exp(-x**2/2)

    #Set the real-space/frequency points (Hermitian FFT-friendly)
    numXPoints = 513
    xpoints = linspace(-20,20,numXPoints)
    tpoints = calcTfromX(xpoints)

    #Calculate the FFT of an actual gaussian; use
    #this as the empirical characteristic function standard
    mygaus1d = mySTDGaus1D(xpoints)
    mygauscf = fft.fftshift(fft.ifftn(fft.ifftshift(mygaus1d)))
    nh = (len(tpoints)-1)/2
    mygauscf /= mygauscf[nh]


    #Set the number of data points
    ndatapoints = 2**10
    #Set the number of variables
    nvariables = 1
    #Randomly sample from a normal distribution
    xyrand = random.normal(loc=0.0,scale=1.0,size=[nvariables,ndatapoints])

    #Calculat the ECF using the fast method
    ecfFFT = ECF(xyrand,tpoints,useFFTApproximation=True).ECF
    #Calculat the ECF using the slow method
    ecfDFT = ECF(xyrand,tpoints,useFFTApproximation=False).ECF

    #Print the 0-frequencies (should be 1 for all)
    print ecfFFT[nh],ecfDFT[nh],mygauscf[nh]

    P.subplot(111,xscale="log",yscale="log")
    #Plot the magnitude of the fast and slow ECFs
    #(these should overlap for all but the highest half of the frequencies)
    P.plot(tpoints,abs(ecfFFT),'r-')
    P.plot(tpoints,abs(ecfDFT),'b-')
    #Plot the gaussian characteristic function standard
    P.plot(tpoints,abs(mygauscf),'k-')
    P.show()
    
  doTwoDimensionalTest = True #Flag whether to do 2D tests
  if(doTwoDimensionalTest):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    def mySTDGaus2D(x,y):
      return 1./(2*pi) * exp(-(x**2 + y**2)/2)

    #Set the frequency points (Hermitian FFT-friendly)
    numXPoints = 513
    xpoints = linspace(-20,20,numXPoints)
    tpoints = calcTfromX(xpoints)

    #Calculate points from a 2D gaussian, and take their 2D FFT
    #to estimate the characteristic function standard
    xp2d,yp2d = meshgrid(xpoints,xpoints)
    mygaus2d = mySTDGaus2D(xp2d,yp2d)
    mygauscf = fft.fftshift(fft.ifftn(fft.ifftshift(mygaus2d)))
    nh = (len(tpoints)-1)/2
    midPointAccessor = tuple(2*[nh])
    mygauscf /= mygauscf[midPointAccessor]


    #Sample points from a gaussian distribution
    ndatapoints = 2**10
    nvariables = 2
    xyrand = random.normal(loc=0.0,scale=1.0,size=[nvariables,ndatapoints])

    #Calculate the ECF using the fast method
    CecfFFT = ECF(xyrand,tpoints,useFFTApproximation=True)
    ecfFFT = CecfFFT.ECF
    #Calculate the ECF using the slow method
    CecfDFT = ECF(xyrand,tpoints,useFFTApproximation=False)
    ecfDFT = CecfDFT.ECF

    #Use meshgrid to generate 2D arrays of the frequency points
    tp2d,wp2d = meshgrid(tpoints,tpoints)

    #Create a figure
    fig = plt.figure()

    #Create a 3D set of axes
    ax = fig.add_subplot(221,projection='3d')

    #ax.plot_wireframe(tp2d[::4,::4],wp2d[::4,::4],(abs(mygauscf)**2)[::4,::4],color='k')
    #plot the fast and slow ECFs using a wireframe (they should overlap to the eye)
    ax.plot_wireframe(tp2d[::4,::4],wp2d[::4,::4],(abs(ecfFFT)**2)[::4,::4],color='r')
    ax.plot_wireframe(tp2d[::4,::4],wp2d[::4,::4],(abs(ecfDFT)**2)[::4,::4],color='b')

    #Create a 2D set of axes
    ax2 = fig.add_subplot(222,xscale="log",yscale="log")


    #Print the normalization constants (should be 1)
    print ecfFFT[midPointAccessor],ecfDFT[midPointAccessor],mygauscf[midPointAccessor]

    #plot the magnitudes of the fast and slow ECFs along
    #an aribtrary slice (they should overlap except in the high frequency range)
    ax2.plot(tpoints,abs(ecfFFT[nh+5,:]),'r-')
    ax2.plot(tpoints,abs(ecfDFT[nh+5,:]),'b-')
    #Plot the magnitude of the gaussian characteristic function
    ax2.plot(tpoints,abs(mygauscf[nh+5,:]),'k-')

    #Plot the average difference between the slow and fast ECFs
    #(will be relatively high because I use a coarse X grid, so that
    # the slow calculation will finish in my lifetime)
    errorK = average(abs(ecfFFT-ecfDFT),0)
    ax3 = fig.add_subplot(223,xscale="log",yscale="log")
    ax3.plot(tpoints[len(tpoints)/2:3*len(tpoints)/4],errorK[len(tpoints)/2:3*len(tpoints)/4],'k-')

    plt.show()


