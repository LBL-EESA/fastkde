#!/usr/bin/env python
from numpy import *
import nufft

class ECF:

  def __init__( self,\
                inputData, \
                tgrids, \
                precision = 2, \
                useFFTApproximation = True, \
                beVerbose = False):
    """
    Calculates the empirical characteristic function of arbitrary sets of
    variables.

    Uses either the direct Fourier transform or nuFFT method (described by
    O'Brien et al. (2014, J. Roy. Stat. Soc. C) to calculate the Fourier transform
    of the data to yield the ECF.


        input:
        ------

            inputData   : The input data. 
                          Array like with shape = (nvariables,npoints).

            tgrids      : The frequency-space grids to which to transform the data
                          A list of frequency arrays for each variable dimension.

            useFFTApproximation : Flag whether to use the nuFFT approximation to the DFT

            beVerbose : Flags whether to be verbose


        output:
        -------

            An ECF object.  The ECF itself is stored in self.ECF

    """

    #Set whether we use the nuFFT approximation
    self.useFFTApproximation = useFFTApproximation

    #Get the data shape (nvariables,ndatapoints)
    dshape = shape(inputData)
    rank = len(dshape)
    if(rank != 2):
      raise ValueError,"inputData must be a rank-2 array of shape [nvariables,ndatapoints]; got rank = {}".format(rank)
    #Extract the number of variables
    self.nvariables = dshape[0]
    #Extract the number of data points
    self.ndatapoints = dshape[1]


    #Set the frequency points
    self.tgrids = list(tgrids)

    try:
        gridRank = len(self.tgrids)
    except:
        raise ValueError,"Could not determine the number of tgrids"

    if  gridRank != self.nvariables:
        raise ValueError,"The rank of tgrids should be {}.  It is {}".format(gridRank,self.nvariables)

    #Check for regularity if we are doing nuFFT
    if(self.useFFTApproximation):
      
      for n in range(self.nvariables):
          tpoints = tgrids[n]

          #Get the spacing of the first two points
          dt = tpoints[1]-tpoints[0]
          #Get the spacing of all points
          deltaT = tpoints[1:] - tpoints[:-1]
          #Get the difference between these spacings
          deltaTdiff = deltaT - dt
          fTolerance = dt/1e6
          #Check that all these differences are less than 1/1e6
          if(not all(abs(deltaTdiff < fTolerance))):
            raise ValueError,"All grids in tgrids must be regularly spaced if useFFTApproximation is True"

    #Set verbosity
    self.beVerbose = beVerbose

    #Set the precision
    self.precision = precision

    #Set the fill value for the frequency grids
    fillValue = -1e20

    #Get the maximum frequency grid length
    ntmax = amax([len(tgrid) for tgrid in tgrids])
    #Create the frequency grids array
    frequencyGrids = fillValue*ones([nvariables,ntmax])
    #Fill the frequency grids array
    for v in range(nvariables):
        frequencyGrids[v,:len(tgrids[v])] = tgrids[v]

    #Simply pass in the input data as provided
    preparedInputData = inputData

    #Calculate the ECF
    if(self.useFFTApproximation):
      #Calculate the ECF using the fast method
      ECF = nufft.nuifft( \
                        abscissas = inputData, \
                        ordinates = ones([inputData.shape[1]],dtype=complex128), \
                        frequencyGrids = frequencyGrids, \
                        missingFreqVal = fillValue, \
                        precision = precision, \
                        beVerbose = int(beVerbose))

    else:
      #Calculate the ECF using the slow (direct, but exact) method
      ECF = nufft.idft( \
                        abscissas = inputData, \
                        ordinates = ones([inputData.shape[1]],dtype=complex128), \
                        frequencyGrids = frequencyGrids, \
                        missingFreqVal = fillValue)

    #Ensure that the ECF is normalized
    midPointAccessor = tuple( [ (len(tgrid) - 1)/2 for tgrid in tgrids ])
    #Save the ECF in the object
    self.ECF = ECF/ECF[midPointAccessor]


    return

#*******************************************************************************
#*******************************************************************************
#******************** Unit testing code ****************************************
#*******************************************************************************
#*******************************************************************************
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
    tpoints = nufft.calcTfromX(xpoints)

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
    ecfFFT = ECF(xyrand,tpoints[newaxis,:],useFFTApproximation=True).ECF
    #Calculat the ECF using the slow method
    ecfDFT = ECF(xyrand,tpoints[newaxis,:],useFFTApproximation=False).ECF

    #Print the 0-frequencies (should be 1 for all)
    print ecfFFT[nh],ecfDFT[nh],mygauscf[nh]

    P.subplot(121,xscale="log",yscale="log")
    #Plot the magnitude of the fast and slow ECFs
    #(these should overlap for all but the highest half of the frequencies)
    P.plot(tpoints,abs(ecfFFT),'r-')
    P.plot(tpoints,abs(ecfDFT),'b-')
    #Plot the gaussian characteristic function standard
    P.plot(tpoints,abs(mygauscf),'k-')

    P.subplot(122,xscale="log",yscale="log")

    ihalf = len(ecfDFT)/2
    ithreequarters = ihalf + ihalf/2
    sh = slice(ihalf,ithreequarters)
    P.plot(tpoints[sh],abs(ecfDFT[sh]-ecfFFT[sh]),'k-')
    P.show()
    
  doTwoDimensionalTest = True #Flag whether to do 2D tests
  if(doTwoDimensionalTest):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    def mySTDGaus2D(x,y):
      return 1./(2*pi) * exp(-(x**2 + y**2)/2)

    #Set the frequency points (Hermitian FFT-friendly)
    numXPoints = 127
    xpoints = linspace(-20,20,numXPoints)
    tpoints = nufft.calcTfromX(xpoints)

    #Calculate points from a 2D gaussian, and take their 2D FFT
    #to estimate the characteristic function standard
    xp2d,yp2d = meshgrid(xpoints,xpoints)
    mygaus2d = mySTDGaus2D(xp2d,yp2d)
    mygauscf = fft.fftshift(fft.ifftn(fft.ifftshift(mygaus2d)))
    nh = (len(tpoints)-1)/2
    midPointAccessor = tuple(2*[nh])
    mygauscf /= mygauscf[midPointAccessor]


    #Sample points from a gaussian distribution
    ndatapoints = 2**5
    nvariables = 2
    xyrand = random.normal(loc=0.0,scale=1.0,size=[nvariables,ndatapoints])

    tpointgrids = concatenate(2*(tpoints[newaxis,:],),axis=0)
    #Calculate the ECF using the fast method
    CecfFFT = ECF(xyrand,tpointgrids,useFFTApproximation=True)
    ecfFFT = CecfFFT.ECF
    #Calculate the ECF using the slow method
    CecfDFT = ECF(xyrand,tpointgrids,useFFTApproximation=False)
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


