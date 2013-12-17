#!/usr/bin/env python
from numpy import *
from numpy.random import randn
import knuthAverage as kn
import ftnphisc
import copy
from types import *
from scipy.interpolate import interp1d
import pdb
import time

class Timer():
   def __init__(self,n=None): self.n = n
   def __enter__(self): self.start = time.time()
   def __exit__(self, *args): print "N = {}, t = {} seconds".format(self.n,time.time() - self.start)

def calcTfromX(xpoints):
  """Calculates frequency points given a signal in real space. """
  #Use fftfreq to produce a set of frequencies that correspond to the fourier
  #transform of a signal on the self.x points
  deltaX = xpoints[1] - xpoints[0]
  dum =  fft.fftfreq(len(xpoints),deltaX/(2*pi))

  #Since we will only be transforming hermitian signals (the ECF is hermitian
  #by construction), only do calculations for 0-or-positive frequencies
  return dum[0:len(dum)/2+1]

def reStandardizeECF(origFreq,origECF,origAvg,origStd,newAvg,newStd):
  """Given a new average and standard deviation, return an approximation of the
     ECF that would have been calculated had newAvg and newStd been used to standardize
     the data in the first place.
     
       origFreq     : The frequencies of the original ECF

       origECF      : The original ECF (a complex array)
       
       origAvg      : The average originally used to standardize the ECF

       origStd      : The standard deviation originally used to standardize the ECF
       
       newAvg       : The new average

       newStd       : The new standard deviation

       Returns:
        A restandardized ECF sampled on the origFreq grid
       
  """

  #Scale the frequencies accordingly
  newFreq = origFreq*(newStd/origStd)

  #Swap the newAvg with origAvg
  newECF = origECF*exp(1j*(origAvg-newAvg)*origFreq/origStd)


  #Generate a linear spline representation of the ECF
  interpolationType='linear'
  ecfSplineReal = interp1d(newFreq,real(newECF),interpolationType,bounds_error=False,fill_value=0.0)
  ecfSplineImag = interp1d(newFreq,imag(newECF),interpolationType,bounds_error=False,fill_value=0.0)



  newECF = zeros([len(origFreq)]) + 0.0j*zeros([len(origFreq)])
  #Sample the new ECF at the original frequencies
  newECF = ecfSplineReal(origFreq) + 1.0j*ecfSplineImag(origFreq)
  #newECF /= newECF[0]
 
  #Return the restandardized ECF
  return newECF

class bernacchiaDensityEstimate:

  def __init__( self,\
                data = None,\
                x = None, \
                numPoints = 4097, \
                numSigma = 20, \
                deltaX = None, \
                dataAverage = None, \
                dataStandardDeviation = None, \
                dataMin = None, \
                dataMax = None, \
                countThreshold = 1, \
                doApproximateECF = True, \
                compareECF = False, \
                doFFT = True, \
                doStoreData = False\
              ):
    """ 

    Estimates the density function of a given dataset using the self-consistent
    method of Bernacchia and Pigolotti (2011, J. R. Statistic Soc. B.).  Prior
    to estimating the PDF, the data are standardized to have a mean of 0 and a
    variance of 1.  
    
    Standardization is done so that PDFs of varying widths can be calculated on
    a unified grid; the original PDF can be re-obtained by scaling, offsetting,
    and renormalizing the calculated PDF.  Assuming the PDF is reasonably
    narrow, then most of the information in the PDF should be contained in the
    returned domain.  The width of the domain is set in terms of multiples of
    unit standard deviations of the data; the default is 20-sigma.

    usage: bdensity = bernacchiaDensityEstimate(  data, \
                                                  x = None, \
                                                  numPoints = 8197,\
                                                  numSigma = 20, \
                                                  deltaX = None, \
                                                  dataAverage = None, \
                                                  dataStandardDeviation = None, \
                                                  doApproximateECF = True, \
                                                  doFFT = True \
                                                )

      data (array_like)   : the data from which to estimate the PDF. If data
                            is multidimensional, the data are flattened first. 

      x                   : the x-values of the estimated PDF.  They must be evenly
                            spaced and they should have an odd length

      numPoints           : the number of points in the domain.  If deltaX is
                            given, this value is overwritten.

      numSigma            : the number of unit standard deviations that the PDF
                            domain should span.

      deltaX              : if given, this specifies the spacing between domain
                            values.

      dataAverage         : if given, this specifies the average of the data, to be
                            subtracted

      dataStandardDeviation : if given, this specifies the standard deviation
                              of the data

      doApproximateECF    : flags whether to approximate the ECF using a (much faster)
                            FFT.  In tests, this is accurate to ~1e-14 over low 
                            frequencies, but is inaccurate to ~1e-2 for the highest ~5% 
                            of frequencies.

      compareECF          : flags whether to calculate the ECF directly and via FFT

      doFFT               : flags whether to calculate phiSC and its FFT to obtain
                            fSC

      doStoreData         : flags whether to store the incoming data

    Returns: a bernacchiaDensityEstimate object

    """
    
    if(data != None):
      #Calculate and/or save the standard deviation/average of the data
      if(dataAverage == None):
        self.dataAverage = average(data)
      else:
        self.dataAverage = dataAverage
      #Standard deviation
      if(dataStandardDeviation == None):
        self.dataStandardDeviation = std(data)
      else:
        self.dataStandardDeviation = dataStandardDeviation

      #Minimum
      if(dataMin == None):
        self.dataMin = amin(data)
      else:
        self.dataMin = dataMin
      #Maximum
      if(dataMax == None):
        self.dataMax = amax(data)
      else:
        self.dataMax = dataMax

      #Set the number of data points
      self.numDataPoints = len(data)

    else:
      self.numDataPoints = 0

    self.doStoreData = doStoreData
    if(self.doStoreData):
      self.data = data

    #Store the doFFT flag
    self.doFFT = doFFT

    #Set the number of points
    self.numPoints = numPoints

    #Set whether to approximate the ECF using the FFT method
    self.doApproximateECF = doApproximateECF

    #Set whether to calculate the ECF via both methods
    self.compareECF = compareECF
    if(self.compareECF):
      #Force the calculation of the FFT-based method if comparison is on
      self.doApproximateECF = True
      #Turn of real-space transformation
      self.doFFT = False

    if(x == None):
      #Determine the x-points of the estimated PDF
      if(deltaX == None): 
        assert numPoints > 1, "numPoints < 2: {}".format(numPoints)
        assert type(numPoints) is IntType, "numPoints is not an integer: {}".formate(numPoints)
        self.x = linspace(-numSigma,numSigma,numPoints)
        self.deltaX = self.x[1] - self.x[0]
        self.numXPoints = numPoints
      else:
        self.x = arange(-numSigma,numSigma+deltaX,deltaX)
        self.deltaX = deltaX
        self.numXPoints = len(self.x)
        
      self.xMin = -numSigma
      self.xMax = numSigma
    else:
      #If the x-points are specified, then use the specified points
      #TODO: this FFT-based implementation of the BP11 density estimate requires that
      #       x be evenly spaced (and it should have an odd number of points). This
      #       should be checked
      self.x = x
      self.deltaX = x[1] - x[0]
      self.numXPoints = len(self.x)
      self.xMin = amin(x)
      self.xMax = amax(x)

    #Calculate the frequency points
    self.t = calcTfromX(self.x)
    self.numTPoints = len(self.t)
    self.deltaT = self.t[2] - self.t[1]

    self.ithresh = 3  #Set the number of consecutive frequency points, that are
                      #below the stability threshold, that will cause the phiSC
                      #loop to stop


    self.phiSC = (0.0+0.0j)*zeros([self.numTPoints])
    self.ECF = (0.0+0.0j)*zeros([self.numTPoints])

    #Initialize the good distribution index
    self.goodDistributionInds = []

    #Set the threshold for the `number of kernels contributions' above which
    #the PDF will be considered valid
    self.countThreshold = countThreshold

    if(data != None):
      if(self.doApproximateECF):
        #Note that this routine also standardizes the data on-the-fly
        self.__calculateECFbyFFT__(data)

        if(self.compareECF):
          #Calculate the optimal distribution (in Fourier space) based on the ECF
          #Note that this routine also standardizes the data on-the-fly
          self.__calculatephiSC__(data)
        else:
          self.ECF = self.fftECF
      else:
        #Calculate the optimal distribution (in Fourier space) based on the ECF
        #Note that this routine also standardizes the data on-the-fly
        self.__calculatephiSC__(data)


      if(self.doFFT):
        #Transform the optimal distribution to real space
        if(self.doApproximateECF):
          self.calculateDensityFromECF()
        else:
          self.__transformphiSC__()

        self.goodDistributionInds = self.findGoodDistributionInds()

    return

  def __calculatephiSC__(self,mydata):
    """Estimate the optimum distribution (in Fourier space) using the BP11 method"""

    #Call a Fortran routine to do an efficient calculation of the density in
    #fourier space.
    [self.phiSC,self.ECF] = ftnphisc.calculatephisc( 
                                          datapoints = mydata, \
                                          dataaverage = self.dataAverage, \
                                          datastd = self.dataStandardDeviation, \
                                          tpoints = self.t, \
                                          ithresh = self.ithresh, \
                                         )
    
  def __transformphiSC__(self):
    """ Transform the self-consistent estimate of the distribution from frequency to real
        space 
    """
    #Since phiSC is Hermitian, use the numpy hfft routine to guarantee a transform
    #that is real
    self.fSC = fft.fftshift(fft.hfft(self.phiSC,self.numXPoints))*self.deltaT/(2*pi)

  def __add__(self,rhs):
    #Check for proper typing
    if(not isinstance(rhs,bernacchiaDensityEstimate)):
      raise TypeError, "unsupported operand type(s) for +: {} and {}".format(type(self),type(rhs))

    retObj = copy.deepcopy(self)
    retObj.phiSC = (0+0j)*zeros([retObj.numTPoints])

    if( (self.dataAverage == rhs.dataAverage) and (self.dataStandardDeviation == rhs.dataStandardDeviation) ):
      #If the dataAverage and standardDeviation for both objects are the same, then the avg/stddev don't need
      #to be recalculated, and the ECFs don't need to be resampled
      retObj.numDataPoints = self.numDataPoints + retObj.numDataPoints
      selfECFReStandardized = self.ECF
      rhsECFReStandardized = rhs.ECF
    else:
      #Update the data average, standard deviation, and count
      [ retObj.dataAverage, \
        retObj.dataStandardDeviation, \
        retObj.numDataPoints ] = \
                                      kn.knuthCombine( \
                                                      self.dataAverage,\
                                                      self.dataStandardDeviation**2,\
                                                      self.numDataPoints,\
                                                      rhs.dataAverage,\
                                                      rhs.dataStandardDeviation**2,\
                                                      rhs.numDataPoints)

      #Convert the returned variance back into standard deviation
      retObj.dataStandardDeviation = sqrt(retObj.dataStandardDeviation)

      #********************************************************************
      # Average the Empirical Characteristic Function of the two objects
      #********************************************************************

      #Re-standardize the LHS ECF value
      selfECFReStandardized = reStandardizeECF(origFreq = self.t, \
                                               origECF  = self.ECF, \
                                               origAvg  = self.dataAverage, \
                                               origStd  = self.dataStandardDeviation, \
                                               newAvg   = retObj.dataAverage, \
                                               newStd   = retObj.dataStandardDeviation)

      #Re-standardize the RHS ECF value
      rhsECFReStandardized  = reStandardizeECF(origFreq = rhs.t, \
                                               origECF  = rhs.ECF, \
                                               origAvg  = rhs.dataAverage, \
                                               origStd  = rhs.dataStandardDeviation, \
                                               newAvg   = retObj.dataAverage, \
                                               newStd   = retObj.dataStandardDeviation)


    #Average the restandardized ECF values (restandardization is necessary
    #so that the corresponding frequencies are equivalent)
    retObj.ECF = (self.numDataPoints*selfECFReStandardized + rhs.numDataPoints*rhsECFReStandardized) \
                /retObj.numDataPoints

    if(retObj.doFFT):
      retObj.calculateDensityFromECF()


    #If we are storing data, then join the new data onto the data store
    if(retObj.doStoreData):
      retObj.data = concatenate((self.data,rhs.data))

    #Return the new object
    return retObj

  def calculateDensityFromECF(self,doFlushArrays=False):
    """ Given an ECF, calculate the self-consistent density in fourier-space
        and transform that density to real space to obtain fSC. """
    #Make an easy-to-read and float version of self.numDataPoints
    N = float(self.numDataPoints)

    #Calculate the stability threshold for the ECF
    ecfThresh = 4.*(N-1.)/(N*N)

    #Calculate the squared magnitude of the ECF signal
    ecfSq = abs(self.ECF)**2

    #Find indices above the ECF thresold
    iAboveThresh = nonzero(ecfSq >= ecfThresh)[0]

    #Determine the indices over which to calculate phiSC; these will be
    #values where ECF is above the ECF threshold, and below the frequency
    #where the ECF threshold has been below-threshold ithresh consecutive times
    iDiff = diff(iAboveThresh)
    iCutOff = nonzero(iDiff > (self.ithresh+1))[0]
    if(len(iCutOff) > 0):
      #Cut off indices above the point where ithresh or more ECF
      #values are below the threshold (if there are any)
      iCalcPhi = iAboveThresh[:(iCutOff[0]+1)]
    else:
      #Otherwise just return the above-threshold indices
      iCalcPhi = iAboveThresh
    
    if(doFlushArrays):
      #self.phiSC = (0.0+0.0j)*zeros([self.numTPoints])
      self.phiSC[:] = (0.0+0.0j)

    #Calculate the transform kernel (only at unfiltered points)
    kappaSC = (0.0+0.0j)*zeros(shape(self.ECF))
    kappaSC[iCalcPhi] = (N/(2*(N-1)))\
                            *(1+sqrt(1-ecfThresh/ecfSq[iCalcPhi]))

    #Do the phiSC calculation only for the necessary points
    self.phiSC[iCalcPhi] =  self.ECF[iCalcPhi]*kappaSC[iCalcPhi]

    #Calculate the magnitude of the transformed kernel at the 0-point
    #(this is used for thresholding the real-space distribution to avoid
    # points with very little kernel contribution)
    self.kSCMax = real(kappaSC[iCalcPhi[0]] + 2*sum(kappaSC[iCalcPhi[1:]]))*(self.deltaT/(2*pi))

    #Calculate the distribution threshold as a multiple of an individual kernelet
    self.distributionThreshold = self.countThreshold*(self.kSCMax/self.numDataPoints)

    #Transform phiSC into the real-space distribution
    self.__transformphiSC__()


  def __calculateECFbyFFT__(self,data):
    #Calculate details of the gaussian kernel
    tau = 1.5629
    nspread = 30
    nspreadhalf = nspread/2
    fourTau = 4*tau

    self.fkde = zeros([self.numXPoints])

    self.fkde = ftnphisc.calculatekerneldensityestimate(  \
                                                        datapoints = data, \
                                                        dataaverage = self.dataAverage, \
                                                        datastd = self.dataStandardDeviation, \
                                                        xpoints = self.x,\
                                                        nspreadhalf = nspreadhalf,\
                                                        fourtau = fourTau)


    #Calculate the FFT of the kernel density estimate
    kdeFFT = fft.ihfft(fft.ifftshift(self.fkde),len(self.fkde))

    #pdb.set_trace()
    #Deconvolve the transformed kde to obtain the empirical characteristic 
    #function
    tprime = self.t*self.deltaX
    self.fftECF = kdeFFT*exp(tau*tprime**2)/kdeFFT[0]

  def findGoodDistributionInds(self):
    return nonzero(self.fSC >= self.distributionThreshold)[0]

  def deStandardizeX(self):
    return self.x*self.dataStandardDeviation + self.dataAverage

  def deStandardizePDF(self):
    return self.fSC/self.dataStandardDeviation
#*******************************************************************************
#*******************************************************************************
#***************************** Unit testing code *******************************
#*******************************************************************************
#*******************************************************************************
# Test this implementation of the BP11 density estimate against a normal 
# distribution.  Calculate the estimate for a variety of sample sizes and show
# how the distribution error decreases as sample size increases.  As of revision
# 9 of the code, this unit testing shows that this implementation of the BP11
# estimate converges on the true normal distribution like N**-1, which agrees
# the theoretical and empirical convergence rate given in BP11.
if(__name__ == "__main__"):
  import pylab as P
  import scipy.stats as stats

  #print ftnphisc.__doc__

  #Define a gaussian function for evaluation purposes
  def mygaus(x,mu=0.,sig=1.):
    return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))
  
  #Set the size of the sample to calculate
  powmax = 19
  npow = asarray(range(7,powmax)) + 1.0

  #Set the maximum sample size
  nmax = 2**powmax
  #Create a random normal sample of this size
  randsample = 3*random.normal(size = nmax)


  bTestFFTMethod = False
  if(bTestFFTMethod):
    bkernel = bernacchiaDensityEstimate(randsample,compareECF=True,numPoints=16385)

    #Plot the ECF
    P.subplot(2,2,1)
    P.title("Real(ECF)")
    P.plot(bkernel.t,real(bkernel.ECF),'b-')
    P.plot(bkernel.t,real(bkernel.fftECF),'r-')

    P.subplot(2,2,2)
    P.title("Imag(ECF)")
    P.plot(bkernel.t,imag(bkernel.ECF),'b-')
    P.plot(bkernel.t,imag(bkernel.fftECF),'r-')

    P.subplot(2,2,3,xscale="log",yscale="log")
    P.title("|ECF|^2")
    P.plot(bkernel.t[1:],abs(bkernel.ECF[1:])**2,'b-')
    P.plot(bkernel.t[1:],abs(bkernel.fftECF[1:])**2,'r-')

    P.subplot(2,2,4,yscale="log")
    P.title("|ECF-fftECF]")
    #P.plot(bkernel.t[1:],abs(bkernel.ECF[1:]-bkernel.fftECF[1:]))
    P.plot(asarray(range(1,bkernel.numTPoints)),abs(bkernel.ECF[1:]-bkernel.fftECF[1:]))
    #P.plot(bkernel.x,bkernel.fkde)

    P.show()
    quit()


  #Pre-define sample size and error-squared arrays
  nsample = zeros([len(npow)])
  esq = zeros([len(npow)])
  epct = zeros([len(npow)])

  #Do the optimal calculation on a number of different random draws
  for i,n in zip(range(len(npow)),npow):
    #Extract a sample of length 2**n + 1 from the previously-created
    #random sample
    randgauss = randsample[:(2**n + 1)]
    #Set the sample size
    nsample[i] = len(randgauss)

    with Timer(nsample[i]):
      #Do the BP11 density estimate
      bkernel = bernacchiaDensityEstimate(randgauss,countThreshold=100)

    #Calculate the mean squared error between the estimated density
    #And the gaussian
    #esq[i] = average(abs(mygaus(bkernel.x)-bkernel.fSC)**2 *bkernel.deltaX)
    igood = bkernel.goodDistributionInds
    #esq[i] = average(abs(mygaus(bkernel.x)-bkernel.fSC)**2 *bkernel.deltaX)
    esq[i] = average(abs(mygaus(bkernel.x[igood])-bkernel.fSC[igood])**2 *bkernel.deltaX)
    epct[i] = 100*sum(abs(mygaus(bkernel.x)-bkernel.fSC)*bkernel.deltaX)
    #Print the sample size and the error to show that the code is proceeeding
    #print "{}, {}%".format(nsample[i],epct[i])

    #Plot the optimal distribution
    P.subplot(2,2,1)
    P.plot(bkernel.x[bkernel.goodDistributionInds],bkernel.fSC[bkernel.goodDistributionInds],'b-')

    #Plot the empirical characteristic function
    P.subplot(2,2,2,xscale="log",yscale="log")
    P.plot(bkernel.t[1:],abs(bkernel.ECF[1:])**2,'b-')


  #Plot the sample gaussian
  P.subplot(2,2,1)
  P.plot(bkernel.x,mygaus(bkernel.x),'r-')


  #Do a simple power law fit to the scaling
  [m,b,_,_,_] = stats.linregress(log(nsample),log(esq))
  #Print the error scaling (following BP11, this is expected to be m ~ -1)
  print "Error scales ~ N**{}".format(m)

  #Plot the error vs sample size on a log-log curve
  P.subplot(2,2,3)
  P.loglog(nsample,esq)
  P.plot(nsample,exp(b)*nsample**m,'r-')

  print ""

  bDemoSum = False
  if(not bDemoSum):
    P.show() 
  else:
    #*********************************************************************
    # Demonstrate the capability to sum bernacchiaDensityEstimate objects
    #*********************************************************************

    nsamp = 512
    nloop = nmax/nsamp


    #Pre-define sample size and error-squared arrays
    nsample2 = zeros([nloop])
    esq2 = zeros([nloop])

    for i in range(nloop):
      randgauss = randsample[i*nsamp:(i+1)*nsamp]
      if(i == 0):
        bkernel2 = bernacchiaDensityEstimate(randgauss)
        nsample2[i] = len(randgauss)
      else:
        bkernel2 += bernacchiaDensityEstimate(randgauss)
        nsample2[i] = nsample2[i-1] + len(randgauss)

      #Calculate the mean squared error between the estimated density
      #And the gaussian
      esq2[i] = average(abs(mygaus(bkernel2.x)-bkernel2.fSC)**2 * bkernel2.deltaX)
      #Print the sample size and the error to show that the code is proceeeding
      print "{}, {}".format(nsample2[i],esq2[i])

    #Plot the distribution
    P.subplot(2,2,1)
    P.plot(bkernel2.x,bkernel2.fSC,'g-')

    #Plot the ECF
    P.subplot(2,2,2,xscale="log",yscale="log")
    P.plot(bkernel2.t[1:],abs(bkernel2.ECF[1:])**2,'b-')

    #Plot the error-rate change
    P.subplot(2,2,3)
    P.loglog(nsample2,esq2,'g-')

    #Plot the difference between the two distributions
    P.subplot(2,2,4)
    P.plot(bkernel2.x, abs(bkernel.fSC - bkernel2.fSC)*bkernel.deltaX)


    #Show the plots
    P.show()
