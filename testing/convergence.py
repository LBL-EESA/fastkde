#!/usr/bin/env python

try:
    from builtins import range  # Python 2.7/3.x compatibility
except:
    from builtins import range


from numpy import *
import numpy as npy
from numpy.random import randn
import time
import sys
from fastkde.fastKDE import fastKDE

#A simple timer for comparing ECF calculation methods
class Timer():
   def __init__(self,n=None): self.n = n
   def __enter__(self): self.start = time.time()
   def __exit__(self, *args): print(("N = {}, t = {} seconds".format(self.n,time.time() - self.start)))

def nextHighestPowerOfTwo(number):
    """Returns the nearest power of two that is greater than or equal to number"""
    return int(2**(ceil(log2(number))))

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

  doOneDimensionalTests = True
  if(doOneDimensionalTests):
    import pylab as P
    import scipy.stats as stats

    mu = -1e3
    sig = 1e3
    #Define a gaussian function for evaluation purposes
    def mygaus(x):
      return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))
    
    #Set the size of the sample to calculate
    powmax = 19
    npow = asarray(list(range(powmax))) + 1.0

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
        for i,n in zip(list(range(len(npow))),npow):
          #Extract a sample of length 2**n + 1 from the previously-created
          #random sample
          randgauss = randsample[:int(2**n + 1)]
          #Set the sample size
          nsample[i] = len(randgauss)

          with Timer(nsample[i]):
            #Do the BP11 density estimate
            bkernel = fastKDE(randgauss,doApproximateECF=True,numPoints=513)

          #Calculate the mean squared error between the estimated density
          #And the gaussian
          #esq[i] = average(abs(mygaus(bkernel.x)-bkernel.pdf)**2 *bkernel.deltaX)
          esq[i] = average(abs(mygaus(bkernel.axes[0])-bkernel.pdf[:])**2 *bkernel.deltaX[0])
          epct[i] = 100*sum(abs(mygaus(bkernel.axes[0])-bkernel.pdf[:])*bkernel.deltaX[0])
          #Print the sample size and the error to show that the code is proceeeding
          #print "{}, {}%".format(nsample[i],epct[i])

          #Plot the optimal distribution
          P.subplot(2,2,1)#,yscale="log")
          #pdfmask = ma.masked_less(bkernel.pdf,bkernel.distributionThreshold)
          pdfmask = bkernel.pdf
          P.plot(bkernel.axes[0],pdfmask,'b-')

          #Plot the empirical characteristic function
          P.subplot(2,2,2,xscale="log",yscale="log")
          P.plot(bkernel.tgrids[0][1:],abs(bkernel.ECF[1:])**2,'b-')

        #Plot the sample gaussian
        P.subplot(2,2,1)#,yscale="log")
        P.plot(bkernel.axes[0],mygaus(bkernel.axes[0]),'r-')


        #Do a simple power law fit to the scaling
        [m,b,_,_,_] = stats.linregress(log(nsample),log(esq))
        #Print the error scaling (following BP11, this is expected to be m ~ -1)
        print(("Error scales ~ N**{}".format(m)))

        #Plot the error vs sample size on a log-log curve
        P.subplot(2,2,3)
        P.loglog(nsample,esq)
        P.plot(nsample,exp(b)*nsample**m,'r-')

        print("")

        bDemoSum = False
        if(not bDemoSum):
          P.show() 
        else:
          #*********************************************************************
          # Demonstrate the capability to sum fastKDE objects
          #*********************************************************************

          nsamp = 512
          nloop = nmax/nsamp


          #Pre-define sample size and error-squared arrays
          nsample2 = zeros([nloop])
          esq2 = zeros([nloop])

          for i in range(nloop):
            randgauss = randsample[i*nsamp:(i+1)*nsamp]
            if(i == 0):
              bkernel2 = fastKDE(randgauss)
              nsample2[i] = len(randgauss)
            else:
              bkernel2 += fastKDE(randgauss)
              nsample2[i] = nsample2[i-1] + len(randgauss)

            #Calculate the mean squared error between the estimated density
            #And the gaussian
            esq2[i] = average(abs(mygaus(bkernel2.axes[0])-bkernel2.pdf)**2 * bkernel2.deltaX[0])
            #Print the sample size and the error to show that the code is proceeeding
            print(("{}, {}".format(nsample2[i],esq2[i])))

          #Plot the distribution
          P.subplot(2,2,1)
          P.plot(bkernel2.axes[0],bkernel2.pdf,'g-')

          #Plot the ECF
          P.subplot(2,2,2,xscale="log",yscale="log")
          P.plot(bkernel2.tgrids[0][1:],abs(bkernel2.ECF[0,1:])**2,'b-')

          #Plot the error-rate change
          P.subplot(2,2,3)
          P.loglog(nsample2,esq2,'g-')

          #Plot the difference between the two distributions
          P.subplot(2,2,4)
          P.plot(bkernel2.axes[0], abs(bkernel.pdf - bkernel2.pdf)*bkernel.deltaX[0])


          #Show the plots
          P.show()
    else:
        print(randsample)
        #Simply do the BP11 density estimate and plot it
        bkernel = fastKDE(randsample,\
                                                doApproximateECF=True, \
                                                beVerbose = True, \
                                                numPoints = 513)
        #Plot the optimal distribution
        P.subplot(2,1,1)
        #pdfmask = ma.masked_less(bkernel.pdf,bkernel.distributionThreshold)
        pdfmask = bkernel.pdf
        P.plot(bkernel.axes[0],pdfmask,'b-')
        #Plot the sample gaussian
        P.plot(bkernel.axes[0],mygaus(bkernel.axes[0]),'r-')

        #for d in randsample:
        #    P.plot([d,d],[0,1./len(randsample)],'k-',alpha=0.5)

        #Plot the transforms
        P.subplot(2,1,2)
        P.plot(bkernel.tgrids[0],abs(bkernel.phiSC),'b-')
        ecfStandard = fft.ifft(mygaus(bkernel.axes[0]))
        ecfStandard /= ecfStandard[0]
        ecfStandard = fft.fftshift(ecfStandard)
        P.plot(bkernel.tgrids[0],abs(ecfStandard),'r-')

        mean = sum(bkernel.axes[0]*bkernel.pdf*bkernel.deltaX[0])

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
    powmax = 16
    npow = asarray(list(range(1,powmax))) + 1.0

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
      randsamples.append(random.multivariate_normal(mu,gCovMat,(int(nmax/ngg),)).transpose())

    #Concatenate the gaussian samples
    randsample = concatenate(tuple(randsamples),axis=1)

    #Shuffle the samples along the long axis so that we
    #can draw successively larger samples
    ishuffle = asarray(list(range(nmax)))
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
      for z,n in zip(list(range(len(npow))),npow):
        #Extract a sample of length 2**n + 1 from the previously-created
        #random sample
        randsub = randsample[:,:int(2**n)]
        #Set the sample size
        nsample[z] = shape(randsub)[1]

        with Timer(nsample[z]):
            #Do the BP11 density estimate
            bkernel = fastKDE(  randsub,  \
                                                beVerbose=False, \
                                                doSaveMarginals = False, \
                                                numPoints=129)

        x,y = tuple(bkernel.axes)
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
      print(("Error scales ~ N**{}".format(m)))
    else:
      with Timer(shape(randsample)[1]):
        bkernel = fastKDE(  randsample,  \
                            beVerbose=True, \
                            doSaveMarginals=False, \
                            numPoints = 129)




    doPlot = True
    if(doPlot):

      x,y = tuple(bkernel.axes)
      x2d,y2d = meshgrid(x,y)

      fig = plt.figure()
      ax1 = fig.add_subplot(121)
      clevs = asarray(list(range(2,10)))/100.
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
