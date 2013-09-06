#!/usr/bin/env python
from numpy import *
from numpy.random import randn
import knuthAverage as kn
import ftnbp11helper
import copy
from types import *
import pdb
import time

#A simple timer for comparing ECF calculation methods
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

class bernacchiaDensityEstimate:

  def __init__( self,\
                data = [],\
                x = [], \
                numPoints = 4097, \
                numSigma = 20, \
                deltaX = [], \
                dataAverage = [], \
                dataStandardDeviation = [], \
                dataMin = [], \
                dataMax = [], \
                countThreshold = 1, \
                doApproximateECF = True, \
                compareECF = False, \
                doFFT = True \
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
                                                  x = [], \
                                                  numPoints = 4097,\
                                                  numSigma = 20, \
                                                  deltaX = [], \
                                                  dataAverage = [], \
                                                  dataStandardDeviation = [], \
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


    Returns: a bernacchiaDensityEstimate object

    """
    
    if(data != []):

      #First check the rank of the data
      dataRank = len(shape(data))
      #If the data are a vector, promote the data to a rank-1 array with only 1 column
      if(dataRank == 1):
        data = reshape(data,[1,len(data)])
      if(dataRank > 2):
        raise ValueError

      #Set the number of variables
      self.numVariables = shape(data)[0]
      #Set the number of data points
      self.numDataPoints = shape(data)[1]

      #Calculate and/or save the standard deviation/average of the data
      if(dataAverage == [] or len(dataAverage) != self.numVariables):
        self.dataAverage = zeros([self.numVariables])
        for v in range(self.numVariables):
          self.dataAverage[v] = average(data[v,:])
      else:
        self.dataAverage = dataAverage
      #Standard deviation
      if(dataStandardDeviation == [] or len(dataStandardDeviation) != self.numVariables):
        self.dataStandardDeviation = zeros([self.numVariables])
        for v in range(self.numVariables):
          self.dataStandardDeviation[v] = std(data[v,:])
      else:
        self.dataStandardDeviation = dataStandardDeviation

      #Minimum
      if(dataMin == [] or len(dataMin) != self.numVariables):
        self.dataMin = zeros([self.numVariables])
        for v in range(self.numVariables):
          self.dataMin[v] = amin(data[v,:])
      else:
        self.dataMin = dataMin
      #Maximum
      if(dataMax == [] or len(dataMin) != self.numVariables):
        self.dataMax = zeros([self.numVariables])
        for v in range(self.numVariables):
          self.dataMax[v] = amax(data[v,:])
      else:
        self.dataMax = dataMax


    else:
      self.numDataPoints = 0

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

    if(x == []):
      #Determine the x-points of the estimated PDF
      if(deltaX == []): 
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


    self.phiSC = (0.0+0.0j)*zeros([self.numVariables,self.numTPoints])
    self.ECF = (0.0+0.0j)*zeros([self.numVariables,self.numTPoints])

    #Initialize the good distribution index
    self.goodDistributionInds = []

    #Calculate the distribution frequency corresponding to the given count threshold
    self.distributionThreshold = float(countThreshold)/(self.numDataPoints*self.deltaX)

    if(data != []):

      #*************************************************
      # Calculate the Empirical Characteristic Function
      #*************************************************
      if(self.doApproximateECF):
        #Note that this routine also standardizes the data on-the-fly
        self.__calculateECFbyFFT__(data,self.dataAverage,self.dataStandardDeviation)
        self.ECF = self.fftECF
      else:
        #Calculate the optimal distribution (in Fourier space) based on the ECF
        #Note that this routine also standardizes the data on-the-fly
        self.__calculateECFDirect__(data,self.dataAverage,self.dataStandardDeviation)


      if(self.doFFT):
        #*************************************************
        # Apply the filter
        #*************************************************
        #Apply the Bernacchia filter to the ECF to obtain
        #the fourier representation of the self-consistent density
        self.applyBernacchiaFilter()

        #*************************************************
        # Transform to real space
        #*************************************************
        #Transform the optimal distribution to real space
        self.__transformphiSC__()

        self.goodDistributionInds = self.findGoodDistributionInds()

    return


  #*****************************************************************************
  #** bernacchiaDensityEstimate: ***********************************************
  #******************* applyBernacchiaFilter() *********************************
  #*****************************************************************************
  #*****************************************************************************
  def applyBernacchiaFilter(self,doFlushArrays=False):
    """ Given an ECF, calculate the self-consistent density in fourier-space by
    applying the BP11 filter."""

    #Make an easy-to-read and float version of self.numDataPoints
    N = float(self.numDataPoints)

    #Calculate the stability threshold for the ECF
    ecfThresh = 4.*(N-1.)/(N*N)

    #Calculate the squared magnitude of the ECF signal
    ecfSq = abs(self.ECF)**2

    #Find indices above the ECF thresold
    iAboveThresh = nonzero(ecfSq[0,:] >= ecfThresh)[0]

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
      self.phiSC[:,:] = (0.0+0.0j)

    #Do the phiSC calculation only for the necessary points
    self.phiSC[0,iCalcPhi] = (N*self.ECF[0,iCalcPhi]/(2*(N-1)))\
                              *(1+sqrt(1-ecfThresh/ecfSq[0,iCalcPhi]))


  #*****************************************************************************
  #** bernacchiaDensityEstimate: ***********************************************
  #******************* __calculateECFDirect__() ********************************
  #*****************************************************************************
  #*****************************************************************************
  def __calculateECFDirect__(self,mydata,myaverage,mystddev):
    """Directly calculate the fourier-space representation of the input data"""

    #Call a Fortran routine to do an efficient calculation of the density in
    #fourier space.
    self.ECF = ftnbp11helper.calculateecf( 
                                        datapoints = mydata, \
                                        dataaverage = myaverage, \
                                        datastd = mystddev, \
                                        tpoints = self.t
                                       )
    
  #*****************************************************************************
  #** bernacchiaDensityEstimate: ***********************************************
  #******************* __calculateECFbyFFT__() *********************************
  #*****************************************************************************
  #*****************************************************************************
  def __calculateECFbyFFT__(self,data,myaverage,mystddev):
    """Use the non-uniform FFT method to estimate the fourier representation of
    the input data."""

    #Calculate details of the gaussian kernel
    tau = 1.5629
    nspread = 30
    nspreadhalf = nspread/2
    fourTau = 4*tau

    fkde = zeros([self.numXPoints])

    fkde = ftnbp11helper.calculatekerneldensityestimate(  \
                                                        datapoints = data, \
                                                        dataaverage = myaverage, \
                                                        datastd = mystddev, \
                                                        xpoints = self.x,\
                                                        nspreadhalf = nspreadhalf,\
                                                        fourtau = fourTau)


    #Calculate the FFT of the kernel density estimate
    kdeFFT = fft.ihfft(fft.ifftshift(fkde),len(fkde))

    #pdb.set_trace()
    #Deconvolve the transformed kde to obtain the empirical characteristic 
    #function
    tprime = self.t*self.deltaX
    self.fftECF = kdeFFT*exp(tau*tprime**2)/kdeFFT[0]

  #*****************************************************************************
  #** bernacchiaDensityEstimate: ***********************************************
  #******************* findGoodDistributionInds() ******************************
  #*****************************************************************************
  #*****************************************************************************
  def findGoodDistributionInds(self):
    """Find indices of the optimal distribution that are above a specificed threshold"""
    #TODO: Remove the 0-index 
    return nonzero(self.fSC[0,:] >= self.distributionThreshold)[0]

  #*****************************************************************************
  #** bernacchiaDensityEstimate: ***********************************************
  #******************* __transformphiSC__() ************************************
  #*****************************************************************************
  #*****************************************************************************
  def __transformphiSC__(self):
    """ Transform the self-consistent estimate of the distribution from
    frequency space to real space"""

    #Since phiSC is Hermitian, use the numpy hfft routine to guarantee a transform
    #that is real
    self.fSC = fft.fftshift(fft.hfft(self.phiSC,self.numXPoints))*self.deltaT/(2*pi)

  #*****************************************************************************
  #** bernacchiaDensityEstimate: ***********************************************
  #******************* Addition operator __add__ *******************************
  #*****************************************************************************
  #*****************************************************************************
  def __add__(self,rhs):
    """ Addition operator for the bernacchiaDensityEstimate object.  Adds the
        empirical characteristic functions of the two estimates, reapplies
        the BP11 filter, and transforms back to real space.  This is useful
        for parallelized calculation of densities. """
    #Check for proper typing
    if(not isinstance(rhs,bernacchiaDensityEstimate)):
      raise TypeError, "unsupported operand type(s) for +: {} and {}".format(type(self),type(rhs))

    retObj = copy.deepcopy(self)
    retObj.phiSC = (0+0j)*zeros([retObj.numTPoints])

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

    #Average the Empirical Characteristic Function of the two objects
    retObj.ECF = (self.numDataPoints*self.ECF + rhs.numDataPoints*rhs.ECF) \
                /retObj.numDataPoints

    if(retObj.doFFT):
      retObj.applyBernacchiaFilter()
      retObj.__transformphiSC__()

    #Return the new object
    return retObj


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

  #print ftnbp11helper.__doc__

  #Define a gaussian function for evaluation purposes
  def mygaus(x,mu=0.,sig=1.):
    return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))
  
  #Set the size of the sample to calculate
  powmax = 15
  npow = asarray(range(powmax)) + 1.0

  #Set the maximum sample size
  nmax = 2**powmax
  #Create a random normal sample of this size
  randsample = 3*random.normal(size = nmax)


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
      bkernel = bernacchiaDensityEstimate(randgauss,doApproximateECF=False)

    #Calculate the mean squared error between the estimated density
    #And the gaussian
    #esq[i] = average(abs(mygaus(bkernel.x)-bkernel.fSC)**2 *bkernel.deltaX)
    igood = bkernel.goodDistributionInds
    esq[i] = average(abs(mygaus(bkernel.x)-bkernel.fSC[0,:])**2 *bkernel.deltaX)
    epct[i] = 100*sum(abs(mygaus(bkernel.x)-bkernel.fSC[0,:])*bkernel.deltaX)
    #Print the sample size and the error to show that the code is proceeeding
    #print "{}, {}%".format(nsample[i],epct[i])

    #Plot the optimal distribution
    P.subplot(2,2,1)
    P.plot(bkernel.x[bkernel.goodDistributionInds],bkernel.fSC[0,bkernel.goodDistributionInds],'b-')

    #Plot the empirical characteristic function
    P.subplot(2,2,2,xscale="log",yscale="log")
    P.plot(bkernel.t[1:],abs(bkernel.ECF[0,1:])**2,'b-')


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
    P.plot(bkernel2.t[1:],abs(bkernel2.ECF[0,1:])**2,'b-')

    #Plot the error-rate change
    P.subplot(2,2,3)
    P.loglog(nsample2,esq2,'g-')

    #Plot the difference between the two distributions
    P.subplot(2,2,4)
    P.plot(bkernel2.x, abs(bkernel.fSC - bkernel2.fSC)*bkernel.deltaX)


    #Show the plots
    P.show()
