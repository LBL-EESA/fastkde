#!/usr/bin/env python
from numpy import *
from numpy.random import randn
from increments import bernacchiaDensityEstimate as be
import time
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
#import pylab as P
import matplotlib.pyplot as P

class Timer():
   def __enter__(self): 
     self.start = time.time()
     return self
   def __exit__(self, *args): 
     self.end = time.time()
     self.delta = self.end - self.start

#Define a gaussian function for evaluation purposes
def mygaus(x,mu=0.,sig=1.):
  return (1./(sig*sqrt(2*pi)))*exp(-(x-mu)**2/(2.*sig**2))

#*******************************************************************************
#*******************************************************************************
#**************** Generate a random sample of data and BP11 Estimate ***********
#*******************************************************************************
#*******************************************************************************
#Set the size of the sample to calculate
powmax = 19
npow = asarray(range(powmax)) + 1.0

#Set the maximum sample size
nmax = 2**powmax
#Create a random normal sample of this size
randsample = 3*random.normal(size = nmax)

#Pre-define sample size and error-squared arrays
nsample = zeros([len(npow)])
e2fft = zeros([len(npow)])
e2dft = zeros([len(npow)])
einf = zeros([len(npow)])
timeFFT = zeros([len(npow)])
timeDFT = zeros([len(npow)])

#Do the optimal calculation on a number of different random draws
for i,n in zip(range(len(npow)),npow):
  #Extract a sample of length 2**n from the previously-created
  #random sample
  randgauss = randsample[:(2**n)]
  #Set the sample size
  nsample[i] = len(randgauss)

  with Timer() as t1:
    #Do the BP11 density estimate
    bkernel = be.bernacchiaDensityEstimate(randgauss)

  timeFFT[i] = t1.delta

  with Timer() as t2:
    #Do the BP11 density estimate
    bkernel2 = be.bernacchiaDensityEstimate(randgauss,doApproximateECF=False)

  timeDFT[i] = t2.delta
  print "N = {}, tfft = {}, tdft = {} seconds".format(nsample[i],timeFFT[i],timeDFT[i])

  #Calculate the mean squared error between the estimated density
  #And the gaussian
  e2fft[i] = (sum(abs(mygaus(bkernel.x)-bkernel.fSC)**2 ) / sum(abs(mygaus(bkernel.x))**2))
  e2dft[i] = (sum(abs(mygaus(bkernel2.x)-bkernel2.fSC)**2 ) / sum(abs(mygaus(bkernel2.x))**2))

#Do a simple power law fit to the scaling
[m,b,_,_,_] = stats.linregress(log(nsample),log(e2fft))

minusOneConvergence = nsample**(-1.)
minusOneConvergence *= average((e2fft))/average((minusOneConvergence))

#*******************************************************************************
#*******************************************************************************
#******************* Plot the FFT Method error and the ECF *********************
#*******************************************************************************
#*******************************************************************************

#Create a figure
fig = P.figure()

#Set the plot font
#TODO: Put this an included file
font = { 'family' : 'serif', \
         'size' : '18' }
matplotlib.rc('font', **font)

#Generate the main plot of the absolute difference between the two ECF methods
superp = fig.add_subplot(111,xscale = "log",yscale="log")
#Plot the error convergence
superp.plot(  nsample, e2fft, \
              color = 'black', \
              linewidth = 2)
superp.plot( nsample, e2dft, \
        color = 'black', \
        linestyle = "--", \
        linewidth = 2)
#Plot the -1 convergence line
superp.plot( nsample, minusOneConvergence, \
        color = 'gray', \
        linewidth = 2)
#Set the axis lables
superp.set_xlabel("Sample Size, $N$")
superp.set_ylabel("Density Estimate Error, $E_2$")

#On a second y-axis, plot the time
superp_y2 = superp.twinx()
superp_y2.set_yscale("log")
for item in ([superp_y2.yaxis.label ] + superp_y2.get_yticklabels()):
  item.set_color("red")
superp_y2.plot(nsample, timeFFT, \
                color = "red",  \
                linewidth = 2)
superp_y2.plot(nsample, timeDFT, \
                color = "red", \
                linestyle = "--", \
                linewidth = 2)
superp_y2.set_ylabel("Execution Time [s]")

superp_y2.set_yticks(10.**asarray([-2, -1, 0, 1, 2]))

fig.subplots_adjust(right=0.15)


#Add the sub-plot label
superp.text(\
        0.9,0.9,"(b)", \
        bbox = {'facecolor':'white', 'color':'white', 'alpha':0.5, 'pad':10}, \
        fontsize = 18, \
        transform = superp.transAxes)

P.tight_layout()
#Save an eps version of the figure
P.savefig("convergencefig.eps")
