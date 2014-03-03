#!/usr/bin/env python
from numpy import *
from numpy.random import randn
from increments import bernacchiaDensityEstimate as be
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
matplotlib.use('Agg')
#import pylab as P
import matplotlib.pyplot as P



#*******************************************************************************
#*******************************************************************************
#**************** Generate a random sample of data and BP11 Estimate ***********
#*******************************************************************************
#*******************************************************************************
#Set the size of the sample to calculate
nsamp = 1
powmax = 12
#npow = [64, 256, 1024, 4096,]
npow = log2(asarray([128, 512, 2048, 8192])/2.)
#npow = asarray([7,9,11,13,15,17])

#Set the maximum sample size
nmax = 2**powmax
#Create a random normal sample of this size
randsample = random.normal(size = nmax)

#Do the Bernacchia Density Estimate with both the DFT and FFT
bkernel = be.bernacchiaDensityEstimate(randsample,compareECF=True,numPoints=4097)

isFirstLoop = True

for p,pp in zip(range(len(npow)),npow):
  for n in range(nsamp):
    npts = int(2**pp)
    print "sample {}, npts = {}".format(n,npts)
    randsample = random.normal(size=npts)
    bksamp = be.bernacchiaDensityEstimate(randsample,compareECF=True,numPoints=4097)
    if(isFirstLoop):
      #Estimate the FFT-method error from nsamp realizations of a gaussian field
      ffterr = zeros([nsamp,powmax,bksamp.numTPoints])
      isFirstLoop = False

    ffterr[n,p,:] = abs(bksamp.ECF - bksamp.fftECF)

ffterravg = average(ffterr,axis=0)

#*******************************************************************************
#*******************************************************************************
#******************* Plot the FFT Method error and the ECF *********************
#*******************************************************************************
#*******************************************************************************

#Create a figure
fig = P.figure()

#Set the colormap
myColorMap = P.get_cmap('gist_rainbow')
cnorm = colors.Normalize(vmin = amin(npow),vmax = amax(npow))
cmapper = cmx.ScalarMappable(norm=cnorm,cmap = myColorMap)

#Set the plot font
#TODO: Put this an included file
font = { 'family' : 'serif', \
         'size' : '18' }
matplotlib.rc('font', **font)

#Generate the main plot of the absolute difference between the two ECF methods
superp = fig.add_subplot(111,yscale="log")
for p in range(len(npow)) : 
  superp.plot( \
          bksamp.t[1:len(bksamp.t)/2],  \
          ffterravg[p,1:len(bksamp.t)/2], \
          color = cmapper.to_rgba(npow[p]), \
          label = "N = {}".format(int(2**npow[p])), \
          linewidth = 2 \
          )

#Add the legend
hand,lab = superp.get_legend_handles_labels()
superp.legend(hand,lab, \
              loc='upper left',\
              prop = { 'size' : 14 })
#Set the axis lables
superp.set_xlabel(r"Fourier Frequency, $\tau$")
superp.set_ylabel("FFT Method Error, $|\mathcal{C}_{FFT}$ $-$ $\mathcal{C}_{DFT}|$")

#Add the sub-plot label
superp.text(\
        0.9,0.9,"(a)", \
        bbox = {'facecolor':'white', 'color':'white', 'alpha':0.5, 'pad':10}, \
        fontsize = 18, \
        transform = superp.transAxes)

#Draw the inset plot of fftECF
subp = P.axes([0.62,0.27,0.3,0.25],xscale = "log",yscale = "log")
subp.plot( bkernel.t[1:len(bkernel.t)/2], \
        abs(bkernel.fftECF[1:len(bkernel.t)/2])**2,  \
        color = 'black')
#Draw the threshold line
subp.plot( bkernel.t[1:len(bkernel.t)/2], \
        bkernel.ecfThreshold*ones([len(bkernel.t[1:len(bkernel.t)/2])]), \
        color = 'gray')
#Set the subaxis labels
subp.set_xlabel(r"Fourier Frequency, $\tau$")
subp.set_ylabel("$|\mathcal{C}_{FFT}|^2$")
subp.set_xlim([1e-1,2e2])
subp.set_yticks(10.**asarray(range(-7,0,2)))

P.tight_layout()

#Save an eps version of the figure
P.savefig("ffterrorfig.eps")
