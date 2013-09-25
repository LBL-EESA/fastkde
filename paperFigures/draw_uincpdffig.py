#!/usr/bin/env python
from numpy import *
from numpy.random import randn
import time
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
#import pylab as P
import matplotlib.pyplot as P
from increments import \
                      incrementPDF as inc, \
                      bernacchiaDensityEstimate as be, \
                      earthPhysics as phys
import netCDF4 as nc
import copy

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

def mysech(x,mu=0.,sig=1.):
  return (1./(2.*sig*cosh(pi*(x-mu)/(2.*sig))))

def mysechsq(x,mu=0.,sig=sqrt(3.)/pi):
  return (1./(4.*sig*cosh((x-mu)/(2.*sig))**2))

def niceLatString(lat):
  if(sign(lat) < 0):
    northSouth = "S"
  else:
    northSouth = "N"

  return "{:2.0f}$^o${}".format(abs(lat),northSouth)

def niceLevString(lev):
  return "{} hPa".format(int(round(lev/10.)*10))

def niceDistString(dist):
  return "{} km".format(int(round(dist/10.)*10))

class figInfo:
  def __init__(self,ilat,ilev,figName):
    self.ilat = ilat
    self.ilev = ilev
    self.figName = figName


figLetters = ["a","b","c","d","e","f","g","h","i"]
iLetter = 0

figAthroughC = figInfo(ilat=215,ilev=20,figName="uincpdfabc.eps")
figDthroughF = figInfo(ilat=384,ilev=18,figName="uincpdfdef.eps")
figGthroughI = figInfo(ilat=510,ilev=24,figName="uincpdfghi.eps")

#*******************************************************************************
#*******************************************************************************
#**************** Read/load a U increment **************************************
#*******************************************************************************
#*******************************************************************************


#Load PDF data
infile = "/projects/regional/APE/updraftAnalysis/firststep.inc.mpas_655362.nc"

fin = nc.Dataset(infile,"r")

Vpdf = fin.variables["incrementPDF"]
Vpdfcount = fin.variables["incrementCount"]
Vpdfavg = fin.variables["incrementAverage"]
Vpdfstd = fin.variables["incrementStandardDeviation"]
Vlat = fin.variables["lat"]
Vlev = fin.variables["lev"]
Vdist = fin.variables["dist"]
Vinc = fin.variables["inc"]
Vfreq = fin.variables["freq"]


nlat = len(Vlat[:])
nlev = len(Vlev[:])
ndist = len(Vdist[:])
ninc = len(Vinc[:])
nfreq = len(Vfreq[:])

#Set the maximum distance for which to calculate the structure function
rmax = 500. * 1000. #m

for myFig in [figAthroughC,figDthroughF,figGthroughI]:
  #Create an empty increment histogram
  myInc = inc.incrementPDF( \
              data = [],\
              npts = 1, \
              distanceValues = Vdist[:], \
              incrementValues = Vinc[:], \
              frequencyValues = Vfreq[:], \
              isPeriodic=True,\
              doMomentEstimation=True)

  #Create an empty BP11 density estimate
  bepdf = be.bernacchiaDensityEstimate(data = [], x = Vinc[:])

  nmoments = len(myInc.moments)

  ilat = myFig.ilat
  ilev = myFig.ilev

  #Set the grid spacing
  dXEquator = phys.rearth*phys.degtorad*(Vlat[nlat/2] - Vlat[nlat/2-1])
  dX=dXEquator*cos(phys.degtorad*Vlat[ilat])


  for idist in range(ndist):
    if(not Vpdfcount[0,ilev,ilat,idist] == 0):
      bepdf.fSC = Vpdf[0,ilev,ilat,idist,:]
      bepdf.numDataPoints = Vpdfcount[0,ilev,ilat,idist]
      bepdf.dataAverage = Vpdfavg[0,ilev,ilat,idist]
      bepdf.dataStandardDeviation = Vpdfstd[0,ilev,ilat,idist]
      bepdf.distributionThreshold = 1./(bepdf.numDataPoints*bepdf.deltaX)
      bepdf.goodDistributionInds = bepdf.findGoodDistributionInds()
    else:
      bepdf.fSC = zeros([len(bepdf.x)])
      bepdf.dataAverage = 0.0
      bepdf.dataStandardDeviation = 0.0
      bepdf.numDataPoints = 0.0

    #Store the bePDF in the given increment
    myInc.bePDF[idist] = copy.copy(bepdf)

  #Set the upper/lower bound of the fit
  myInc.fitUpperBound = rmax/dX
  myInc.fitLowerBound = 3


  myInc.estimateMoments()
  myInc.estimatePowerLawSlopes()

#*******************************************************************************
#*******************************************************************************
#******************* Plot the Increment PDFS ***********************************
#*******************************************************************************
#*******************************************************************************
  #Set the plot font
  #TODO: Put this an included file
  font = { 'family' : 'serif', \
           'size' : '18' }
  matplotlib.rc('font', **font)

  #Create a figure
  figc = P.figure()

  #Generate the main plot of the absolute difference between the two ECF methods
  pdfplot = figc.add_subplot(111)

  #Plot an actual gaussian
  pdfplot.plot(myInc.bePDF[0].x,mygaus(myInc.bePDF[0].x), \
                color = "gray", \
                linewidth= 3, \
                linestyle = ':')

  #Plot an actual gaussian
  pdfplot.plot(myInc.bePDF[0].x,mysechsq(myInc.bePDF[0].x), \
                color = "gray", \
                linewidth= 8, \
                linestyle = '-')

  plottedCurveList = []
  plottedCurveLabels = []
  for ix in range(1,myInc.num_distance_values):
    #Find good bins
    igood = nonzero(myInc.bePDF[ix].fSC[:] >= myInc.bePDF[ix].distributionThreshold)[0]
    #Plot the normalized PDF
    #if(myInc.distanceValues[ix] <= rmax/1000. and ix >= 3):
    if(myInc.distanceValues[ix-1] <= myInc.fitUpperBound and myInc.distanceValues[ix] >= myInc.fitLowerBound):
      plotTmp, = pdfplot.plot(myInc.bePDF[ix].x[igood],myInc.bePDF[ix].fSC[igood])
      plottedCurveList.append(plotTmp)
      plottedCurveLabels.append(niceDistString(myInc.distanceValues[ix]*dX/1000.))


  lg = pdfplot.legend( \
                    plottedCurveList,\
                    plottedCurveLabels,\
                    bbox_to_anchor = (0.2,0.1,0.2,0.2),\
                    loc=10,\
                    prop={'size':16})
  #lg.get_frame().set_linewidth(0)
  lg.draw_frame(False)


  #Set the x-axis range
  pdfplot.set_yscale('log')
  pdfplot.set_xlim([-6,15])
  pdfplot.set_ylim([myInc.bePDF[0].distributionThreshold,1])

  #Set the axis labels
  pdfplot.set_xlabel("Increment Value, $\Delta_x U$")
  pdfplot.set_ylabel("Probability Density Estimate, $\hat{f}(\Delta_x U)$")

  pdfplot.set_title("lat = {}, level = {}".format(niceLatString(Vlat[ilat]),niceLevString(Vlev[ilev])))

  #Add the sub-plot label
  pdfplot.text(\
          0.02,0.91,"({})".format(figLetters[iLetter]), \
          bbox = {'facecolor':'white', 'color':'white', 'alpha':0.5, 'pad':10}, \
          fontsize = 18, \
          transform = pdfplot.transAxes)
  iLetter += 1

#*******************************************************************************
#*******************************************************************************
#******************* Plot the structure function *******************************
#*******************************************************************************
#*******************************************************************************

  #Add a subplot with the structure functions
  sfsub = P.axes([0.65,0.67,0.25,0.20],xscale="log",yscale="log")

  #Rescale the distances by the grid spacing for plotting purposes
  myInc.distanceValues *= dX/1000.
  im = 0 #Select the first moment (first order structure function)
  sfsub.plot(myInc.distanceValues, \
              myInc.structureFunctions[im,:], \
              'bo')

  sfsub.plot(myInc.distanceValues, \
              myInc.powerLawFits[im,:], \
              linewidth=2, \
              linestyle = '--', \
              color = "gray")

  #Limit the x-axis
  #sfsub.set_xlim([0,5e2])
  sfsub.set_xlim([20,6e3])
  #sfsub.set_ylim([1e-6,1e-1])
  #Set the axis labels
  sfsub.set_xlabel("Inc. Dist., $\Delta x$ [km]")
  sfsub.set_ylabel('$\langle|\Delta_x U|\\rangle$')

  sfsub.text(\
          0.1,0.75,"({})".format(figLetters[iLetter]), \
          bbox = {'facecolor':'white', 'color':'white', 'alpha':0.5, 'pad':10}, \
          fontsize = 18, \
          transform = sfsub.transAxes)
  iLetter += 1
#*******************************************************************************
#*******************************************************************************
#******************* Plot the structure function exponents *********************
#*******************************************************************************
#*******************************************************************************

  #figd =  P.figure()
  #Generate the main plot of the absolute difference between the two ECF methods
  #momentplot = figd.add_subplot(111)

  momentplot = P.axes([0.70,0.28,0.20,0.24])

  momentplot.plot(myInc.moments,myInc.structureFunctionsExponent[0]*myInc.moments,\
                  color="gray",
                  linewidth=2)
  momentplot.plot(myInc.moments,myInc.structureFunctionsExponent,'ko')

  ##Add the sub-plot label
  momentplot.text(\
          0.1,0.8,"({})".format(figLetters[iLetter]), \
          bbox = {'facecolor':'white', 'color':'white', 'alpha':0.5, 'pad':10}, \
          fontsize = 18, \
          transform = momentplot.transAxes)

  #Set the axis labels
  momentplot.set_xlabel("SF Order, $m$")
  momentplot.set_ylabel("$H_m$")
  momentplot.set_ylim([0.0,3.0])

  iLetter += 1

  P.tight_layout()
  #Save an eps version of the figure
  figc.savefig(myFig.figName)
