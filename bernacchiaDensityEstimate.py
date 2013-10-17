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

#A simple timer for comparing ECF calculation methods
class Timer():
   def __init__(self,n=None): self.n = n
   def __enter__(self): self.start = time.time()
   def __exit__(self, *args): print "N = {}, t = {} seconds".format(self.n,time.time() - self.start)

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
                ecfPrecision = 2, \
                doStoreConvolution = False, \
                doFFT = True, \
                beVerbose = False \
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
                                                  doStoreConvolution = False, \
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

      ecfPrecision        : sets the precision of the approximate ECF.  If set to 2, it uses
                            double precision accuracy; 1 otherwise

      doStoreConvolution  : flags whether to store the KDE used in the nuFFT

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
        raise ValueError,"data must be a rank-2 array of shape [numVariables,numDataPoints]"

      #Set the number of variables
      self.numVariables = shape(data)[0]
      #Set the number of data points
      self.numDataPoints = shape(data)[1]

      if(beVerbose):
        print "Operating on data with numVariables = {}, numDataPoints = {}".format(self.numVariables,self.numDataPoints)

      #Calculate and/or save the standard deviation/average of the data
      if(dataAverage == [] or len(dataAverage) != self.numVariables):
        self.dataAverage = average(data,1)
      else:
        self.dataAverage = dataAverage
      #Standard deviation
      if(dataStandardDeviation == [] or len(dataStandardDeviation) != self.numVariables):
        self.dataStandardDeviation = std(data,1)
      else:
        self.dataStandardDeviation = dataStandardDeviation

      if(beVerbose):
        print "Data has average: {}".format(self.dataAverage)
        print "Data has standard deviation: {}".format(self.dataStandardDeviation)

      #Minimum
      if(dataMin == [] or len(dataMin) != self.numVariables):
        self.dataMin = amin(data,1)
      else:
        self.dataMin = dataMin
      #Maximum
      if(dataMax == [] or len(dataMin) != self.numVariables):
        self.dataMax = amax(data,1)
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

    #Set the approximate ECF precision
    self.ecfPrecision = ecfPrecision

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
    self.t = ecf.calcTfromX(self.x)
    self.numTPoints = len(self.t)
    self.deltaT = self.t[2] - self.t[1]

    self.ithresh = 1  #Set the number of consecutive frequency points, that are
                      #below the stability threshold, that will cause the phiSC
                      #loop to stop


    self.phiSC = (0.0+0.0j)*zeros(self.numVariables*[self.numTPoints])
    self.ECF = (0.0+0.0j)*zeros(self.numVariables*[self.numTPoints])

    #Initialize the good distribution index
    self.goodDistributionInds = []

    #Set the verbosity flag
    self.beVerbose = beVerbose

    self.doStoreConvolution = doStoreConvolution
    self.convolvedData = None

    #Calculate the distribution frequency corresponding to the given count threshold
    self.distributionThreshold = float(countThreshold)/(self.numDataPoints*self.deltaX**self.numVariables)

    if(data != []):

      #*************************************************
      # Calculate the Empirical Characteristic Function
      #*************************************************
      #Note that this routine also standardizes the data on-the-fly
      if(self.beVerbose):
        print "Calculating the ECF"
        sys.stdout.flush()

      ecfObj = ecf.ECF( inputData = data, \
                        tpoints = self.t, \
                        dataAverage = self.dataAverage, \
                        dataStandardDeviation = self.dataStandardDeviation, \
                        useFFTApproximation = self.doApproximateECF, \
                        doStoreConvolution = self.doStoreConvolution, \
                        precision = self.ecfPrecision, \
                        beVerbose = self.beVerbose)

      #Extract the ECF
      self.ECF = ecfObj.ECF
      if(self.doStoreConvolution):
        self.convolvedData = ecfObj.convolvedData

      if(self.doFFT):
        #*************************************************
        # Apply the filter
        #*************************************************
        #Apply the Bernacchia filter to the ECF to obtain
        #the fourier representation of the self-consistent density
        if(self.beVerbose):
          print "Applying the filter"
        self.applyBernacchiaFilter()

        #*************************************************
        # Transform to real space
        #*************************************************
        #Transform the optimal distribution to real space
        if(self.beVerbose):
          print "Transforming to real space"
          sys.stdout.flush()
        self.__transformphiSC__()

        #if(self.beVerbose):
        #  print "Finding good distribution indices"
        #  sys.stdout.flush()
        #self.goodDistributionInds = self.findGoodDistributionInds()

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
#
#    #Find indices above the ECF thresold
#    iAboveThresh = nonzero(ecfSq >= ecfThresh)[0]
#
#    #Determine the indices over which to calculate phiSC; these will be
#    #values where ECF is above the ECF threshold, and below the frequency
#    #where the ECF threshold has been below-threshold ithresh consecutive times
#    iDiff = diff(iAboveThresh)
#    iCutOff = nonzero(iDiff > (self.ithresh+1))[0]
#    if(len(iCutOff) > 0):
#      #Cut off indices above the point where ithresh or more ECF
#      #values are below the threshold (if there are any)
#      iCalcPhi = iAboveThresh[:(iCutOff[0]+1)]
#    else:
#      #Otherwise just return the above-threshold indices
#      iCalcPhi = iAboveThresh

    iCalcPhitmp,imax = ftn.lowesthypervolumefilter( ecfsq = ecfSq.ravel(), \
                                        ecfthreshold = ecfThresh, \
                                        nvariables = self.numVariables, \
                                        ntpoints = self.numTPoints)
    
    iCalcPhi = iCalcPhitmp[:imax]
    
    if(doFlushArrays):
      self.phiSC[:] = (0.0+0.0j)

    #Do the phiSC calculation only for the necessary points
    self.phiSC.ravel()[iCalcPhi] = (N*self.ECF.ravel()[iCalcPhi]/(2*(N-1)))\
                              *(1+sqrt(1-ecfThresh/ecfSq.ravel()[iCalcPhi]))

  #*****************************************************************************
  #** bernacchiaDensityEstimate: ***********************************************
  #******************* findGoodDistributionInds() ******************************
  #*****************************************************************************
  #*****************************************************************************
  def findGoodDistributionInds(self):
    """Find indices of the optimal distribution that are above a specificed threshold"""
    #TODO: Remove the 0-index 
    return nonzero(self.fSC >= self.distributionThreshold)[0]

  #*****************************************************************************
  #** bernacchiaDensityEstimate: ***********************************************
  #******************* __transformphiSC__() ************************************
  #*****************************************************************************
  #*****************************************************************************
  def __transformphiSC__(self):
    """ Transform the self-consistent estimate of the distribution from
    frequency space to real space"""

#    #Generate a set of array slices to access the lower half of the array
#    firstHalfSlice = tuple(self.numVariables*[slice(0,self.numTPoints)])
#    #Create a temporary array to hold both the negative and positive frequency
#    #values of phiSC (it is Hermitian symmetric) in a way that conforms with fftn()
#    padSequence = self.numVariables*[tuple([self.numTPoints-1,0])]
#    #phiSCSymmetric = fft.ifftshift(pad(self.phiSC,padSequence,'reflect'))
#    phiSCSymmetric = pad(self.phiSC,padSequence,'reflect')
#
#    #Transform the PDF estimate to real space
#    fSC = real(fft.fftshift(fft.fftn(phiSCSymmetric)))*(self.deltaT/(2*pi))**self.numVariables
    fSC = fft.fftshift(real(fft.fftn(fft.ifftshift(self.phiSC))))*(self.deltaT/(2*pi))**self.numVariables
      
    
    if(self.beVerbose):
      normConst = sum(fSC*self.deltaX**self.numVariables)
      midPointAccessor = tuple(self.numVariables*[(self.numTPoints-1)/2])
      print "Normalization of fSC = {}. phiSC[0] = {}".format(normConst,self.phiSC[midPointAccessor])

    self.fSC = fSC.transpose()
    #self.fSC = ma.masked_less(fSC.transpose(),self.distributionThreshold)
    #self.fSC = ma.masked_less(fSC.transpose(),0.0)

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

  def getTransformedAxes(self):
    return tuple([ (self.x*self.dataStandardDeviation[i] + self.dataAverage[i]) for i in range(self.numVariables) ])

  def getTransformedPDF(self):
    return self.fSC/prod(self.dataStandardDeviation)

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

  doOneDimensionalTests = False
  if(doOneDimensionalTests):
    import pylab as P
    import scipy.stats as stats

    #print ftnbp11helper.__doc__

    #Define a gaussian function for evaluation purposes
    def mygaus(x,mu=0.,sig=1.):
      return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))
    
    #Set the size of the sample to calculate
    powmax = 19
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
        bkernel = bernacchiaDensityEstimate(randgauss,doApproximateECF=True)

      #Calculate the mean squared error between the estimated density
      #And the gaussian
      #esq[i] = average(abs(mygaus(bkernel.x)-bkernel.fSC)**2 *bkernel.deltaX)
      esq[i] = average(abs(mygaus(bkernel.x)-bkernel.fSC[:])**2 *bkernel.deltaX)
      epct[i] = 100*sum(abs(mygaus(bkernel.x)-bkernel.fSC[:])*bkernel.deltaX)
      #Print the sample size and the error to show that the code is proceeeding
      #print "{}, {}%".format(nsample[i],epct[i])

      #Plot the optimal distribution
      P.subplot(2,2,1)
      fSCmask = ma.masked_less(bkernel.fSC,bkernel.distributionThreshold)
      P.plot(bkernel.x,fSCmask,'b-')

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
      P.plot(bkernel2.t[1:],abs(bkernel2.ECF[0,1:])**2,'b-')

      #Plot the error-rate change
      P.subplot(2,2,3)
      P.loglog(nsample2,esq2,'g-')

      #Plot the difference between the two distributions
      P.subplot(2,2,4)
      P.plot(bkernel2.x, abs(bkernel.fSC - bkernel2.fSC)*bkernel.deltaX)


      #Show the plots
      P.show()

  doTwoDimensionalTests = True
  if(doTwoDimensionalTests):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    nvariables = 2
    #Seed with 0 so results are reproducable
    #random.seed(0)

    #Define a bivariate normal function
    def norm2d(x,y,mux=0,muy=0,sx=1,sy=1,r=0):
      coef = 1./(2*pi*sx*sy*sqrt(1.-r**2))
      expArg = -(1./(2*(1-r**2)))*( (x-mux)**2/sx**2 + (y-muy)**2/sy**2 - 2*r*(x-mux)*(y-muy)/(sx*sy))
      return coef*exp(expArg)
    
    #Set the size of the sample to calculate
    powmax = 17
    npow = asarray(range(3,powmax)) + 1.0

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
      cov = covMat(*tuple(gg[2:]))
      size = tuple([2,nmax/ngg])
      #Append a 2D gaussian to the list
      randsamples.append(random.multivariate_normal(mu,cov,(nmax/ngg,)).transpose())

    #Concatenate the gaussian samples
    randsample = concatenate(tuple(randsamples),axis=1)

    #Shuffle the samples along the long axis so that we
    #can draw successively larger samples
    ishuffle = asarray(range(nmax))
    random.shuffle(ishuffle)
    randsample = randsample[:,ishuffle]

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
          bkernel = bernacchiaDensityEstimate(  randsub,  \
                                                beVerbose=False, \
                                                numPoints=1025, \
                                                numSigma=10, \
                                                doStoreConvolution=False, \
                                                countThreshold = 1)



        x,y = bkernel.getTransformedAxes()
        x2d,y2d = meshgrid(x,y)

        #Calculate the mean squared error between the estimated density
        #And the gaussian
        #esq[z] = average(abs(mygaus(bkernel.x)-bkernel.fSC)**2 *bkernel.deltaX)
        #esq[z] = average(abs(pdfStandard(x2d,y2d)-bkernel.getTransformedPDF())**2 *bkernel.deltaX**2)
        absdiffsq = abs(pdfStandard(x2d,y2d)-bkernel.getTransformedPDF())**2
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        esq[z] = sum(dy*sum(absdiffsq*dx,axis=0))/(len(x)*len(y))
        #Print the sample size and the error to show that the code is proceeeding
        print "{}: {}, {}".format(n,nsample[z],esq[z])

      #Do a simple power law fit to the scaling
      [m,b,_,_,_] = stats.linregress(log(nsample),log(esq))
      #Print the error scaling (following BP11, this is expected to be m ~ -1)
      print "Error scales ~ N**{}".format(m)
    else:
      with Timer(shape(randsample)[1]):
        bkernel = bernacchiaDensityEstimate(  randsample,  \
                                              beVerbose=False, \
                                              numPoints=1025, \
                                              numSigma=10, \
                                              doStoreConvolution=False, \
                                              countThreshold = 1)




    doPlot = True
    if(doPlot):

      x,y = bkernel.getTransformedAxes()
      x2d,y2d = meshgrid(x,y)

      fig = plt.figure()
      ax1 = fig.add_subplot(121)
      clevs = asarray(range(2,10))/100.
      ax1.contour(x2d,y2d,bkernel.getTransformedPDF(),levels = clevs)
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
        errorStandardSum= sum(abs(pdfStandard(x2d,y2d)-bkernel.getTransformedPDF())**2,axis=1)
        ax3.plot(x,errorStandardSum)



      plt.show()
