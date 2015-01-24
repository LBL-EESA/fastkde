#!/usr/bin/env python
from numpy import *
from numpy.random import randn
import knuthAverage as kn
import empiricalCharacteristicFunction as ecf
import ftnbp11 as ftn
#If numpy's version is less than 1.7, then use the version of arraypad
#supplied with this code, since pad() doesn't exist in lower numpy versions
if(float(".".join(__version__.split(".")[:2])) < 1.7):
  from arraypad import pad
import copy
from types import *
import pdb
import time
import sys
from nufft import calcTfromX
import floodFillSearchC as flood

#A simple timer for comparing ECF calculation methods
class Timer():
   def __init__(self,n=None): self.n = n
   def __enter__(self): self.start = time.time()
   def __exit__(self, *args): print "N = {}, t = {} seconds".format(self.n,time.time() - self.start)

def nextHighestPowerOfTwo(number):
    """Returns the nearest power of two that is greater than or equal to number"""
    return int(2**(ceil(log2(number))))

class selfConsistentDensityEstimate:

  def __init__( self,\
                data = None,\
                xgrids = None, \
                numPointsPerSigma = 10, \
                numPoints=None, \
                countThreshold = 1, \
                doApproximateECF = True, \
                ecfPrecision = 1, \
                doSaveTransformedKernel = False, \
                doFFT = True, \
                doSaveMarginals = True, \
                beVerbose = False, \
                fracContiguousHyperVolumes = 0.01, \
                numContiguousHyperVolumes = None, \
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

    usage: sdensity = selfConsistentDensityEstimate(  data, \
                                                  x = None, \
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

      numSigma            : the number of unit standard deviations that the PDF
                            domain should span.

      numPointsPerSigma   : the number of points on the data grid per standard
                            deviation; this influences the total size of the x-grid that is
                            automatically calculated if no aspects of the grid are specified.

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

      ecfPrecision        : sets the precision of the approximate ECF.  If set to 2, it uses
                            double precision accuracy; 1 otherwise

      doFFT               : flags whether to calculate phiSC and its FFT to obtain
                            pdf

      doSaveMarginals     : flags whether to calculate and save the marginal distributions

      fracContiguousHyperVolumes :  the fraction of contiguous hypervolumes of the ECF, that 
                                    are above the ECF threshold, to use in the density estimate

      numContiguousHyperVolumes : like fracContiguousHyperVolumes, but specify an integer number
                                  to use.  fracContiguousHyperVolumes will be ignored if this
                                  is provided as an argument.


    Returns: a selfConsistentDensityEstimate object

    """

    def vprint(msg):
        """Only print if beVerbose is True"""
        if beVerbose:
            print(msg)

    addOne = True #Force x grids to be (2**n) + 1
    
    if(data is not None):

      #First check the rank of the data
      dataRank = len(shape(data))
      #If the data are a vector, promote the data to a rank-1 array with only 1 column
      if(dataRank == 1):
          data = array(data[newaxis,:])
      else:
          data = array(data)
      if(dataRank > 2):
          raise ValueError,"data must be a rank-2 array of shape [numVariables,numDataPoints]"

      #Set the rank of the data
      self.dataRank = dataRank

      #Set the number of variables
      self.numVariables = shape(data)[0]
      #Set the number of data points
      self.numDataPoints = shape(data)[1]

      self.dataAverage = average(data,axis=1)
      self.dataStandardDeviation = std(data,axis=1)

      self.fracContiguousHyperVolumes = fracContiguousHyperVolumes

      if numContiguousHyperVolumes is not None:
          self.fracContiguousHyperVolumes = numContiguousHyperVolumes

      vprint("Operating on data with numVariables = {}, numDataPoints = {}".format(self.numVariables,self.numDataPoints))

    else:
      self.numDataPoints = 0

    #Store the doFFT flag
    self.doFFT = doFFT

    #Save the marginals flag
    self.doSaveMarginals = doSaveMarginals
    if dataRank == 1:
        self.doSaveMarginals = False

    #Set whether to approximate the ECF using the FFT method
    self.doApproximateECF = doApproximateECF

    #Set the approximate ECF precision
    self.ecfPrecision = ecfPrecision

    #Preinitialize the ecf threshold
    self.ecfThreshold = None

    #Flag whether to save the transformed kernel
    self.doSaveTransformedKernel = doSaveTransformedKernel
    #initialize the kernel and its transform
    self.kappaSC = None
    self.kSC = None

    #***********************
    # Calculate the x grids
    #***********************
    if(xgrids is None):

        #Get the range of the data 
        self.xMin = amin(data,1)
        self.xMax = amax(data,1)

        #Get the grid mid-points
        midPoint = 0.5*(self.xMax + self.xMin)

        vprint("Difference between midpoint and average: {}".format(midPoint - self.dataAverage))

        forceZeroDataAverage = True
        if forceZeroDataAverage:
            #Shift the edges of the range so that the mid point is also the data average
            for v in range(self.numVariables):
                distance = midPoint[v] - self.dataAverage[v]
                if distance > 0:
                    self.xMin[v] -= 2*distance
                else:
                    self.xMax[v] -= 2*distance 

                assert isclose(0.5*(self.xMax[v] + self.xMin[v]), self.dataAverage[v])

            self.midPoint = self.dataAverage
        else:
            self.midPoint = midPoint

        #inflate the range by 5% to ensure that the data all fit within the range
        self.xMin -= 0.05*(self.xMin-self.dataAverage)
        self.xMax -= 0.05*(self.xMax-self.dataAverage)


        if numPoints is None:
            #Calculate the number of standard deviations there
            # are in the data range
            dataRange = self.xMax - self.xMin
            numSigma = dataRange/self.dataStandardDeviation
            
            #Set the number of points for each dimensions
            self.numXPoints = array([nextHighestPowerOfTwo(ns * numPointsPerSigma) + int(addOne) for ns in numSigma])
        else:
            self.numXPoints = array(self.numVariables*(numPoints,))


        #Set the grids for each dimension
        self.xgrids = [ linspace(xmin,xmax,np) for xmin,xmax,np in zip(self.xMin,self.xMax,self.numXPoints)]

        vprint("Grids created with xmin: {}, xmax: {}, npoints: {}".format(self.xMin,self.xMax,self.numXPoints))
    else:
        #Set the xgrid from the function argument
        self.xgrids = xgrids
        self.xMin = array([amin(xg) for xg in xgrids])
        self.xMax = array([amax(xg) for xg in xgrids])
        self.numXPoints = array([len(xg) for xg in xgrids])
        #Get the grid mid-points
        self.midPoint = 0.5*(self.xMax + self.xMin)


    #Get the grid spacings
    self.deltaX = array([ xg[1] - xg[0] for xg in self.xgrids])

    #Check that the xgrids are regular and proper powers of two
    for v in range(self.numVariables):
        xg = self.xgrids[v]
        dx = (xg[1:]-self.dataAverage[v])/self.dataStandardDeviation[v] - (xg[:-1] - self.dataAverage[v])/self.dataStandardDeviation[v]
        dxdiff = dx - self.deltaX[v]/self.dataStandardDeviation[v]
        fTolerance = self.deltaX[v]/(1e4*self.dataStandardDeviation[v])
        #Check that these differences are less than 1/1e6
        if(not all(abs(dxdiff) < fTolerance)):
            raise ValueError,"All grids in xgrids must be regularly spaced"

        log2size = log2(len(xg) - addOne)
        if log2size != floor(log2size):
            if addOne:
                extraStr = " + 1"
            else:
                extraStr = ""

            raise ValueError,"All grids in xgrids must be powers of 2" + extraStr + ", but got {}".format(len(xg))

    #Calculate the frequency point grids (for 0-centered data)
    self.tgrids = [ calcTfromX((xg-av)/sd) for xg,av,sd in zip(self.xgrids,self.dataAverage,self.dataStandardDeviation) ]
    self.numTPoints = array([len(tg) for tg in self.tgrids])
    self.deltaT = array([tg[2] - tg[1] for tg in self.tgrids])

    self.phiSC = (0.0+0.0j)*zeros(self.numTPoints)
    self.ECF = (0.0+0.0j)*zeros(self.numTPoints)

    #Initialize the good distribution index
    self.goodDistributionInds = []

    #Set the verbosity flag
    self.beVerbose = beVerbose

    self.convolvedData = None

    #Calculate the distribution frequency corresponding to the given count threshold
    self.countThreshold = countThreshold

    #Initialize the marginals
    self.marginalObjects = None

    if(data is not None):

      #*************************************************
      # Calculate the Empirical Characteristic Function
      #*************************************************
      #Note that this routine also standardizes the data on-the-fly
      vprint("Calculating the ECF")
      sys.stdout.flush()

      #Transfrom the data to 0-centered coordinates
      for v in range(self.numVariables):
          data[v,:] = (data[v,:] - self.dataAverage[v])/self.dataStandardDeviation[v]
          
      #Calculate the ECF (see empiricalCharacteristicFunction.py)
      ecfObj = ecf.ECF( inputData = data, \
                        tgrids = self.tgrids, \
                        useFFTApproximation = self.doApproximateECF, \
                        precision = self.ecfPrecision, \
                        beVerbose = self.beVerbose)

      #Extract the ECF from the ECF object
      self.ECF = ecfObj.ECF

      if(self.doFFT):
        #*************************************************
        # Apply the filter
        #*************************************************
        #Apply the Bernacchia and Pigolotti (2011) filter to the ECF to obtain
        #the fourier representation of the self-consistent density
        vprint("Applying the filter")
        self.applyBernacchiaFilter()

        #*************************************************
        # Transform to real space
        #*************************************************
        #Transform the optimal distribution to real space
        vprint("Transforming to real space")
        sys.stdout.flush()
        self.__transformphiSC__()

        #Calculate and save the marginal distribution objects
        if(self.doSaveMarginals):
          self.marginalObjects = []
          for i in xrange(self.dataRank):
            self.marginalObjects.append(selfConsistentDensityEstimate(data[i,:], \
                                          xgrids = self.xgrids, \
                                          countThreshold = self.countThreshold, \
                                          doSaveMarginals = False) )
                                                                  
    return


  #*****************************************************************************
  #** selfConsistentDensityEstimate: ***********************************************
  #******************* applyBernacchiaFilter() *********************************
  #*****************************************************************************
  #*****************************************************************************
  def applyBernacchiaFilter(self,doFlushArrays=True):
    """ Given an ECF, calculate the self-consistent density in fourier-space by
    applying the BP11 filter."""

    #Make an easy-to-read and float version of self.numDataPoints
    N = float(self.numDataPoints)

    #Calculate the stability threshold for the ECF
    ecfThresh = 4.*(N-1.)/(N*N)
    self.ecfThreshold = ecfThresh

    #Calculate the squared magnitude of the ECF 
    ecfSq = abs(self.ECF)**2

    #Find all hypervolumes where ecfSq is greater than the stability threshold
    contiguousInds = flood.floodFillSearch(ecfSq,searchThreshold = self.ecfThreshold)

    if contiguousInds == []:
        raise RuntimeError,"No ECF values found above the ECF threshold.  max(ecfSq) = {}, ecfThresh = {}".format(amax(ecfSq),ecfThresh)

    #Sort them by distance from the center
    sortedInds = flood.sortByDistanceFromCenter(contiguousInds,shape(ecfSq))

    numVolumes = len(sortedInds)
    if self.fracContiguousHyperVolumes >= 1:
        numVolumesToUse = int(self.fracContiguousHyperVolumes)
    else:
        numVolumesToUse = int(self.fracContiguousHyperVolumes*numVolumes)
    if numVolumesToUse < 1:
        numVolumesToUse = 1

    #Initialize the filtered value list
    iCalcPhi = self.numVariables*[array([],dtype='int')]

    #Pull out fracContiguousHyperVolumes of contiguous hyper volumes, in order of distance from
    #the origin
    for i in xrange(numVolumesToUse):
        for n in xrange(self.numVariables):
            iCalcPhi[n] = concatenate( (iCalcPhi[n],sortedInds[i][n]) )

    #Convert iCalcPhi to a list of tuples, such that it is compatible with the output of where()
    iCalcPhi = [ tuple(ii) for ii in iCalcPhi ]
   
    #If flagged, clear the phiSC array.  This is needed if the same selfConsistentDensityEstimate object
    #is reused for multiple data.
    if(doFlushArrays):
      self.phiSC[:] = (0.0+0.0j)

    #Calculate the transform of the self-consistent Kernel (and only calculate it at
    # points where ecfSq is above ecfThresh)
    kappaSC = (1.0+0.0j)*zeros(shape(self.ECF))
    kappaSC[iCalcPhi] = (N/(2*(N-1)))\
                              *(1+sqrt(1-ecfThresh/ecfSq[iCalcPhi]))

    #Store the fourier kernel if we are going to save the transformed kernel
    if(self.doSaveTransformedKernel):
      self.kappaSC = kappaSC

    #Calculate the transform of the self-consistent density estimate
    self.phiSC[iCalcPhi] = self.ECF[iCalcPhi]*kappaSC[iCalcPhi]

    #Calculate the magnitude of the transformed kernel at the (0,0,0....) point
    #in real space.  It is assumed that this is the peak of the kernel; this is used in
    # findGoodDistributionInds() to estimate the number of kernels contributing to a given
    # point on the self-consistent density estimate.
    self.kSCMax = real(sum(kappaSC[iCalcPhi])*prod(self.deltaT)*(1./(2*pi))**self.numVariables)

    #Calculate the distribution threshold as a multiple of an individual kernelet
    self.distributionThreshold = self.countThreshold*(self.kSCMax/self.numDataPoints)

  #*****************************************************************************
  #** selfConsistentDensityEstimate: ***********************************************
  #******************* findGoodDistributionInds() ******************************
  #*****************************************************************************
  #*****************************************************************************
  def findGoodDistributionInds(self):
    """Find indices of the optimal distribution that are above a specificed threshold"""
    return where(self.pdf >= self.distributionThreshold)

  #*****************************************************************************
  #** selfConsistentDensityEstimate: ***********************************************
  #******************* findBadDistributionInds() *******************************
  #*****************************************************************************
  #*****************************************************************************
  def findBadDistributionInds(self):
    """Find indices of the optimal distribution that are below a specificed threshold"""
    return where(self.pdf < self.distributionThreshold)

  #*****************************************************************************
  #** selfConsistentDensityEstimate: ***********************************************
  #******************* __transformphiSC__() ************************************
  #*****************************************************************************
  #*****************************************************************************
  def __transformphiSC__(self):
    """ Transform the self-consistent estimate of the distribution from
    frequency space to real space"""

    #Transform the PDF estimate to real space
    pdf = fft.fftshift(real(fft.fftn(fft.ifftshift(self.phiSC))))*prod(self.deltaT)*(1./(2*pi))**self.numVariables

    #Unnormalize it
    pdf /= prod(self.dataStandardDeviation)
    
    if(self.beVerbose):
      normConst = sum(pdf*prod(self.deltaX))
      midPointAccessor = tuple([(tp-1)/2 for tp in self.numTPoints])
      print "Normalization of pdf = {}. phiSC[0] = {}".format(normConst,self.phiSC[midPointAccessor])

    #transpose the self-consistent density estimate
    self.pdf = pdf.transpose()

    #Set self.fSC for backward compatibility
    self.fSC = self.pdf

    #Take the transform of the self-consistent kernel if flagged
    if(self.doSaveTransformedKernel):
      kSC = fft.fftshift(real(fft.fftn(fft.ifftshift(self.kappaSC))))*prod(self.deltaT)*(1./(2*pi))**self.numVariables
      self.kSC = kSC.transpose()

  def getTransformedPDF():
      """Returns a copy of the PDF.  This function exists for backward compatibility"""
      return array(self.pdf)

  def getTransformedAxes():
      """Returns a copy of the axes.  This function exists for backward compatibility"""
      return tuple([array(xg) for xg in self.xgrids])

  #*****************************************************************************
  #** selfConsistentDensityEstimate: *******************************************
  #******************* getCopula      ******************************************
  #*****************************************************************************
  #*****************************************************************************
  def getCopula(self,data=None):
    """Estimates the copula of the underlying PDF"""

    #If the data are univariate, simply return the PDF itself
    if(self.dataRank == 1):
      return self.pdf

    #Check if we need to calculate the marginal distributions
    if(not self.doSaveMarginals):
      if(data is None):
        raise ValueError,"the data must be provided as argument 'data', if doSaveMarginals=False when the original PDF was calculated"
      else:
        #Estimate the marginal distributions
        marginalObjects = []
        for i in xrange(self.dataRank):
          marginalObjects.append(selfConsistentDensityEstimate(data[i,:], \
                                      xgrids = self.xgrids, \
                                      countThreshold = self.countThreshold, \
                                      doSaveMarginals = False))
    else:
      #If not, just use the saved marginals
      marginalObjects = self.marginalObjects

    #Calculate the marginal distributions and mask bad (or zero) values
    marginals = []
    for obj in marginalObjects:
      #Get a masked version of the PDF
      m = ma.array(obj.pdf)
      #Mask bad values
      m[obj.findBadDistributionInds()] = ma.masked
      #Add the marginal to the list
      marginals.append(m)

    #Calculate the PDF assuming independent marginals
    independencePDF = ma.prod(meshgrid(*tuple(marginals)),axis=0)
    #Divide off the indepdencnce PDF to calculate the copula
    actualPDF = ma.array(self.pdf)
    actualPDF[self.findBadDistributionInds()] = ma.masked
    copulaPDF = actualPDF/independencePDF

    return copulaPDF


  #*****************************************************************************
  #** selfConsistentDensityEstimate: ***********************************************
  #******************* Addition operator __add__ *******************************
  #*****************************************************************************
  #*****************************************************************************
  def __add__(self,rhs):
    """ Addition operator for the selfConsistentDensityEstimate object.  Adds the
        empirical characteristic functions of the two estimates, reapplies
        the BP11 filter, and transforms back to real space.  This is useful
        for parallelized calculation of densities.  Note that this only works
        if the xgrids are the same for both operands."""
    #Check for proper typing
    if(not isinstance(rhs,selfConsistentDensityEstimate)):
      raise TypeError, "unsupported operand type(s) for +: {} and {}".format(type(self),type(rhs))

    #Check that the xgrids are the same for both objects
    for sxg,rxg in zip(self.xgrids,rhs.xgrids):
        if not all(isclose(sxg,rxg)):
            raise NotImplementedError,"addition for operands with different xgrids is not yet implemented."

    retObj = copy.deepcopy(self)
    retObj.phiSC = (0.0+0.0j)*zeros(self.numTPoints)

    retObj.numDataPoints += rhs.numDataPoints

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

  #set a seed so that results are repeatable
  random.seed(0)

  doOneDimensionalTests = False
  if(doOneDimensionalTests):
    import pylab as P
    import scipy.stats as stats

    #print ftnbp11helper.__doc__

    mu = -1e3
    sig = 1e2
    #Define a gaussian function for evaluation purposes
    def mygaus(x):
      return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))
    
    #Set the size of the sample to calculate
    powmax = 19
    npow = asarray(range(powmax)) + 1.0

    #Set the maximum sample size
    nmax = 2**powmax
    #Create a random normal sample of this size
    randsample = sig*random.normal(size = nmax) + mu


    #Pre-define sample size and error-squared arrays
    nsample = zeros([len(npow)])
    esq = zeros([len(npow)])
    epct = zeros([len(npow)])

    evaluateError = True
    if evaluateError:
        #Do the optimal calculation on a number of different random draws
        for i,n in zip(range(len(npow)),npow):
          #Extract a sample of length 2**n + 1 from the previously-created
          #random sample
          randgauss = randsample[:(2**n + 1)]
          #Set the sample size
          nsample[i] = len(randgauss)

          with Timer(nsample[i]):
            #Do the BP11 density estimate
            bkernel = selfConsistentDensityEstimate(randgauss,doApproximateECF=True,numPoints=2049)

          #Calculate the mean squared error between the estimated density
          #And the gaussian
          #esq[i] = average(abs(mygaus(bkernel.x)-bkernel.pdf)**2 *bkernel.deltaX)
          esq[i] = average(abs(mygaus(bkernel.xgrids[0])-bkernel.pdf[:])**2 *bkernel.deltaX[0])
          epct[i] = 100*sum(abs(mygaus(bkernel.xgrids[0])-bkernel.pdf[:])*bkernel.deltaX[0])
          #Print the sample size and the error to show that the code is proceeeding
          #print "{}, {}%".format(nsample[i],epct[i])

          #Plot the optimal distribution
          P.subplot(2,2,1)
          pdfmask = ma.masked_less(bkernel.pdf,bkernel.distributionThreshold)
          P.plot(bkernel.xgrids[0],pdfmask,'b-')

          #Plot the empirical characteristic function
          P.subplot(2,2,2,xscale="log",yscale="log")
          P.plot(bkernel.tgrids[0][1:],abs(bkernel.ECF[1:])**2,'b-')

        #Plot the sample gaussian
        P.subplot(2,2,1)
        P.plot(bkernel.xgrids[0],mygaus(bkernel.xgrids[0]),'r-')


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
          # Demonstrate the capability to sum selfConsistentDensityEstimate objects
          #*********************************************************************

          nsamp = 512
          nloop = nmax/nsamp


          #Pre-define sample size and error-squared arrays
          nsample2 = zeros([nloop])
          esq2 = zeros([nloop])

          for i in range(nloop):
            randgauss = randsample[i*nsamp:(i+1)*nsamp]
            if(i == 0):
              bkernel2 = selfConsistentDensityEstimate(randgauss)
              nsample2[i] = len(randgauss)
            else:
              bkernel2 += selfConsistentDensityEstimate(randgauss)
              nsample2[i] = nsample2[i-1] + len(randgauss)

            #Calculate the mean squared error between the estimated density
            #And the gaussian
            esq2[i] = average(abs(mygaus(bkernel2.xgrids[0])-bkernel2.pdf)**2 * bkernel2.deltaX[0])
            #Print the sample size and the error to show that the code is proceeeding
            print "{}, {}".format(nsample2[i],esq2[i])

          #Plot the distribution
          P.subplot(2,2,1)
          P.plot(bkernel2.xgrids[0],bkernel2.pdf,'g-')

          #Plot the ECF
          P.subplot(2,2,2,xscale="log",yscale="log")
          P.plot(bkernel2.tgrids[0][1:],abs(bkernel2.ECF[0,1:])**2,'b-')

          #Plot the error-rate change
          P.subplot(2,2,3)
          P.loglog(nsample2,esq2,'g-')

          #Plot the difference between the two distributions
          P.subplot(2,2,4)
          P.plot(bkernel2.xgrids[0], abs(bkernel.pdf - bkernel2.pdf)*bkernel.deltaX[0])


          #Show the plots
          P.show()
    else:
        #Simply do the BP11 density estimate and plot it
        bkernel = selfConsistentDensityEstimate(randsample,\
                                                doApproximateECF=True, \
                                                beVerbose = True, \
                                                numPoints = 1025)
        #Plot the optimal distribution
        P.subplot(2,1,1)
        pdfmask = ma.masked_less(bkernel.pdf,bkernel.distributionThreshold)
        P.plot(bkernel.xgrids[0],pdfmask,'b-')
        #Plot the sample gaussian
        P.plot(bkernel.xgrids[0],mygaus(bkernel.xgrids[0]),'r-')

        #for d in randsample:
        #    P.plot([d,d],[0,1./len(randsample)],'k-',alpha=0.5)

        #Plot the transforms
        P.subplot(2,1,2)
        P.plot(bkernel.tgrids[0],abs(bkernel.phiSC),'b-')
        ecfStandard = fft.ifft(mygaus(bkernel.xgrids[0]))
        ecfStandard /= ecfStandard[0]
        ecfStandard = fft.fftshift(ecfStandard)
        P.plot(bkernel.tgrids[0],abs(ecfStandard),'r-')

        mean = sum(bkernel.xgrids[0]*bkernel.pdf*bkernel.deltaX[0])
        print bkernel.deltaX[0]
        print mean - bkernel.dataAverage[0]

        P.show()







  doTwoDimensionalTests = True
  if(doTwoDimensionalTests):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    nvariables = 2
    #Seed with 0 so results are reproducable
    random.seed(0)

    #Define a bivariate normal function
    def norm2d(x,y,mux=0,muy=0,sx=1,sy=1,r=0):
      coef = 1./(2*pi*sx*sy*sqrt(1.-r**2))
      expArg = -(1./(2*(1-r**2)))*( (x-mux)**2/sx**2 + (y-muy)**2/sy**2 - 2*r*(x-mux)*(y-muy)/(sx*sy))
      return coef*exp(expArg)
    
    #Set the size of the sample to calculate
    powmax = 14
    npow = asarray(range(3,powmax))

    #Set the maximum sample size
    nmax = 2**powmax

    def covMat(sx,sy,r):
      return [[sx**2,r*sx*sy],[r*sx*sy,sy**2]]

    gausParams = []
    gausParams.append([0.0,0.0,1.0,1.0,0.0]) #Standard, uncorrelated bivariate
    gausParams.append([2.0,0.0,1.0,1.0,0.7]) #correlation 0.7, mean x+2
    gausParams.append([0.0,2.0,1.0,0.5,0.0]) #Flat in y-direction, mean y+2
    gausParams.append([2.0,2.0,0.5,1.0,0.0]) #Flat in x-direction, mean xy+2

    #Define the corresponding standard function
    def pdfStandard(x,y):
      pdfStandard = zeros(shape(x))
      for gg in gausParams:
        pdfStandard += norm2d(x2d,y2d,*tuple(gg))*(1./ngg)

      return pdfStandard


    #Generate samples from this distribution
    randsamples = []
    ngg = len(gausParams)
    for gg in gausParams:
      mu = gg[:2]
      gCovMat = covMat(*tuple(gg[2:]))
      size = tuple([2,nmax/ngg])
      #Append a 2D gaussian to the list
      randsamples.append(random.multivariate_normal(mu,gCovMat,(nmax/ngg,)).transpose())

    #Concatenate the gaussian samples
    randsample = concatenate(tuple(randsamples),axis=1)

    #Shuffle the samples along the long axis so that we
    #can draw successively larger samples
    ishuffle = asarray(range(nmax))
    random.shuffle(ishuffle)
    randsample = randsample[:,ishuffle]

    doSaveCSV = False
    if(doSaveCSV):
        savetxt("bp11_2d_samples.csv",randsample.transpose(),delimiter=",")

    #Pre-define sample size and error-squared arrays
    nsample = zeros([len(npow)])
    esq = zeros([len(npow)])
    epct = zeros([len(npow)])

    evaluateError = True
    if(evaluateError):
      #Do the optimal calculation on a number of different random draws
      for z,n in zip(range(len(npow)),npow):
        #Extract a sample of length 2**n + 1 from the previously-created
        #random sample
        randsub = randsample[:,:(2**n)]
        #Set the sample size
        nsample[z] = shape(randsub)[1]

        with Timer(nsample[z]):
            #Do the BP11 density estimate
            bkernel = selfConsistentDensityEstimate(  randsub,  \
                                                beVerbose=False, \
                                                countThreshold = 1, \
                                                doSaveMarginals = False, \
                                                numPoints=129)

        x,y = tuple(bkernel.xgrids)
        x2d,y2d = meshgrid(x,y)

        #Calculate the mean squared error between the estimated density
        #And the gaussian
        #esq[z] = average(abs(mygaus(bkernel.x)-bkernel.pdf)**2 *bkernel.deltaX)
        #esq[z] = average(abs(pdfStandard(x2d,y2d)-bkernel.getTransformedPDF())**2 *bkernel.deltaX**2)
        absdiffsq = abs(pdfStandard(x2d,y2d)-bkernel.pdf)**2
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        esq[z] = sum(dy*sum(absdiffsq*dx,axis=0))/(len(x)*len(y))
        #Print the sample size and the error to show that the code is proceeeding
        #print "{}: {}, {}".format(n,nsample[z],esq[z])

      #Do a simple power law fit to the scaling
      [m,b,_,_,_] = stats.linregress(log(nsample),log(esq))
      #Print the error scaling (following BP11, this is expected to be m ~ -1)
      print "Error scales ~ N**{}".format(m)
    else:
      with Timer(shape(randsample)[1]):
        bkernel = selfConsistentDensityEstimate(  randsample,  \
                                              beVerbose=True, \
                                              countThreshold = 1, \
                                              doSaveMarginals=False, \
                                              numPoints = 129)




    doPlot = True
    if(doPlot):

      x,y = tuple(bkernel.xgrids)
      x2d,y2d = meshgrid(x,y)

      fig = plt.figure()
      ax1 = fig.add_subplot(121)
      clevs = asarray(range(2,10))/100.
      ax1.contour(x2d,y2d,bkernel.pdf,levels = clevs)
      ax1.contour(x2d,y2d,pdfStandard(x2d,y2d),levels=clevs,colors='k')
      #ax1.plot(randsample[0,:],randsample[1,:],'k.',markersize=1)
      plt.xlim([-4,6])
      plt.ylim([-4,6])

      if(evaluateError):
        #Plot the error vs sample size on a log-log curve
        ax3 = fig.add_subplot(122,xscale="log",yscale="log")
        ax3.plot(nsample,esq)
        ax3.plot(nsample,exp(b)*nsample**m,'r-')
        #ax3 = fig.add_subplot(223)
        #ax3.plot(randsample[0,::16],randsample[1,::16],'k.',markersize=1)
        #plt.xlim([-4,6])
        #plt.ylim([-4,6])
      else:
        ax3 = fig.add_subplot(122)
        errorStandardSum= sum(abs(pdfStandard(x2d,y2d)-bkernel.pdf)**2,axis=0)
        ax3.plot(x,errorStandardSum)



      plt.show()
