import numpy as np
cimport numpy as np
cimport cython
import numpy.fft as fft

cdef extern from "math.h" nogil:
    double floor(double x)
    double exp(double x)

#*******************************************************************************
#*******************************************************************************
#************************** nufft **********************************************
#*******************************************************************************
#*******************************************************************************
cpdef np.ndarray[double complex] nufft( \
                            np.float_t[:,:] abscissas, \
                            double complex [:] ordinates, \
                            np.ndarray[np.float_t,ndim=2] frequencyGrids, \
                            np.float_t missingFreqVal = -1e20, \
                            int precision = 2):
    """Approximate the direct Fourier transform of abscissa, ordinate pairs
       using the non-uniform FFT method.
        
        input:
        ------

            abscissas   : abscissa values.
                          A numpy array of shape (ndimensions,npoints)
                          (assumed to be real)

            ordinates   : ordinates values.
                          A numpy array of shape (npoints)

            frequencyGrids : The frequency grids on which to calcualte the DFT
                             A masked numpy array of shape (ndimensions,ntmax), where ntmax
                             is the length of the longest frequency grid.  

            missingFreqVal : A value indicating missing frequency values.  This is used to 
                             allow each dimension to have different sized frequency spaces;
                             dimensions with smaller frequency spaces (than ntmax) should be 
                             padded at the end with missingFreqVal.

            precision   : Sets the precision of the approximation.  1 for floating point
                          and 2 for double precision accuracy.

        output:
        -------

            The DFT of abscissa/ordinate pairs.  Calculated in one dimension as:

            DFT[t] = sum( [ a[i] * exp(-j * x[i] * t ) for i in range(N) ] ) 

            for each of the t points in frequencyGrids (and where j = sqrt(-1)).
    
    """

    cdef int numDimensions
    cdef int numDataPoints
    cdef int ntMax

    #*******************************
    # Get variable dimensionalities
    # and do consistency checks
    #*******************************
    #Get the shape of abscissas 
    try:
        numDimensions = abscissas.shape[0]
        numDataPoints = abscissas.shape[1]
    except:
        raise ValueError,"Could not determine shape of abscissas"

    #Check ordinates
    try:
        ordShape = ordinates.shape[0]
    except:
        raise ValueError,"Could not determine shape of ordinates"
        
    if ordShape != numDataPoints:
            raise ValueError, "Incompatible shapes for ordinates and abscissas"

    #Check frequencyGrids
    try:
        freqShape = frequencyGrids.shape
    except:
        raise ValueError,"Could not determine shape of ordinates"

    if freqShape[0] != numDimensions:
            raise ValueError, "Incompatible shapes for abscissas and frequencyGrids"

    #Get the max number of frequency points
    ntMax = freqShape[1]


    #********************************************
    # Calculate the size of the frequency spaces
    #********************************************
    cdef int n
    cdef np.ndarray[np.int_t,ndim=1] frequencySizes = np.zeros([numDimensions],dtype=np.int)

    for n in range(numDimensions):
        iNotMissing = np.nonzero(frequencyGrids[n,:] != missingFreqVal)[0]
        if len(iNotMissing) != 0:
            frequencySizes[n] = len(iNotMissing)
        else:
            raise ValueError,"Some frequencies in frequencyGrids have no valid points"

    #Get the total size of the frequency space
    cdef np.int_t freqSpaceSize = np.prod(frequencySizes)



    cdef np.ndarray[np.float_t,ndim=2] \
            abscissaGrids = missingFreqVal*np.ones([numDimensions,ntMax])
    #******************************
    # Calculate the abscissa grids
    #******************************
    for n in range(numDimensions):
        abscissaGrids[n,:frequencySizes[n]] = calcXfromT(frequencyGrids[n,:frequencySizes[n]])


    #*******************************
    # Define convolution parameters 
    #*******************************
    cdef np.float_t tau,fourTau
    cdef long nspread
    cdef long nspreadhalf, hyperSlabSize

    #Set the parameters that control the precision of the nuFFT
    if precision == 1:
        tau = 0.5993
        nspread = 10
    else:
        tau = 1.5629
        nspread = 28
    nspreadhalf = nspread/2
    fourTau = 4*tau
    hyperSlabSize = nspread**numDimensions

    #get the shape of a hyperslab
    cdef np.ndarray[np.int_t,ndim=1] hyperSlabShape = np.array(numDimensions*(nspread,),dtype=np.int)

    #Calculate the quantities necessary for estimating x-indices
    cdef np.ndarray[np.float_t,ndim=1]  \
            xmins = np.array([ abscissaGrids[n,0] for n in range(numDimensions) ])
    cdef np.ndarray[np.float_t,ndim=1]  \
            deltaxs = np.array([ abscissaGrids[n,1] - abscissaGrids[n,0] for n in range(numDimensions) ])

    #Inialize worker terms for the convolution
    cdef np.ndarray[long,ndim=1] mvec = np.zeros([numDimensions])
    cdef np.ndarray[long,ndim=1] m0vec = np.zeros([numDimensions])
    cdef np.ndarray[np.float_t,ndim=1] mprimevec = np.zeros([numDimensions])
    cdef np.float_t mprime = 0.0
    cdef double complex gaussTerm = 0.0
    cdef np.float_t gaussArg = 0.0
    cdef int j, v, i, k

    #Initialize the (raveled) data convolution
    cdef np.ndarray[double complex,ndim=1] convolvedData = np.zeros([freqSpaceSize],dtype=np.complex128)


    #***********************************
    # Convolve the data with a gaussian
    #***********************************
    for j in range(numDataPoints):
        
        for v in range(numDimensions):
            #Set the vecctor of indicies of the grid point nearest to the current
            #data point
            mprime = (abscissas[v,j] - xmins[v])/deltaxs[v]
            m0vec[v] = <long> floor(mprime)
            mprimevec[v] = mprime - m0vec[v] + nspreadhalf

        for i in range(hyperSlabSize):
            unravelIndex(i,hyperSlabShape,mvec)
            #Calculate the distance between the data point and the current
            #hyperslab point
            gaussArg = 0.0
            for v in range(numDimensions):
                gaussArg += (<np.float_t>mvec[v] - mprimevec[v])**2
            #Calculate the gaussian of this distance
            gaussTerm = ordinates[j]*(<double complex> exp(-gaussArg/fourTau))

            #Transform from hyperslab coordinates back to full space coordinates
            mvec = (mvec + m0vec) - nspreadhalf

            #Calculate the flattened array index of the current point
            k = ravelIndex(frequencySizes,mvec)

            #Add the gaussian term to this point (only if it is within our domain
            if k >= 0 and k < freqSpaceSize:
                convolvedData[k] = convolvedData[k] + gaussTerm


    #Pack the unraveled convolvedData array into an n-dimensional array
    cdef np.ndarray[double complex] \
            convolvedRaveled = np.reshape(convolvedData,frequencySizes)

    #Take its FFT
    #( ifftshift is used to reorder the kde array such that 0 is the lowest corner 
    #  first index] of the array, as required by ifft)
    #And fftshift is used to put the zero-frequency in the center of the array
    cdef np.ndarray[double complex] \
            convolvedFFT = fft.fftshift(fft.ifftn(fft.ifftshift(convolvedRaveled)))

    #Define a grid of frequeny points for use in the deconvolution
    tpointgrids = np.array(np.meshgrid( * [ frequencyGrids[v,:frequencySizes[v]] for v in range(numDimensions)  ] ))

    #Get the height of the DFT at the 0-frequency point (used for normalization
    midPointAccessor = tuple([ (tsize - 1)/2 for tsize in frequencySizes])
    cdef np.float_t convolvedFFTMidPoint = convolvedFFT[midPointAccessor]

    #Pre-declare and allocate a raveled form of the DFT
    cdef np.ndarray[double complex,ndim=1] DFT = np.zeros([freqSpaceSize],dtype=np.complex128)
    #dft = convolvedFFT * \
    #        np.exp(tau*np.sum((tpointgrids*deltaX)**2,axis=0)) / \
    #        convolvedFFT[midPointAccessor]

    cdef np.ndarray[np.int_t,ndim=1] dimInds = np.zeros([numDimensions],dtype=np.int)

    cdef np.ndarray[double complex,ndim=1] convolvedFFTRaveled = convolvedFFT.ravel()

    #Deconvolve the FFT (divide by the FFT of the gaussian) to obtain the DFT estimate
    for i in range(freqSpaceSize):
        unravelIndex(i,frequencySizes,dimInds)
        gaussArg = 0.0
        for v in range(numDimensions):
           gaussArg += (frequencyGrids[v,dimInds[v]]*deltaxs[v])**2
        DFT[i] = convolvedFFT.ravel() * np.exp(tau*gaussArg) / convolvedFFTMidPoint


    #Reshape the DFT to an array
    return np.reshape(DFT,tuple(frequencySizes))


#*******************************************************************************
#*******************************************************************************
#************************** dft ************************************************
#*******************************************************************************
#*******************************************************************************
cpdef np.ndarray[double complex] dft( \
                            np.float_t[:,:] abscissas, \
                            double complex [:] ordinates, \
                            np.ndarray[np.float_t,ndim=2] frequencyGrids, \
                            np.float_t missingFreqVal = -1e20):
    """Calculates the direct Fourier transform of abscissa, ordinate pairs
        
        input:
        ------

            abscissas   : abscissa values.
                          A numpy array of shape (ndimensions,npoints)
                          (assumed to be real)

            ordinates   : ordinates values.
                          A numpy array of shape (npoints)

            frequencyGrids : The frequency grids on which to calcualte the DFT
                             A masked numpy array of shape (ndimensions,ntmax), where ntmax
                             is the length of the longest frequency grid.  

            missingFreqVal : A value indicating missing frequency values.  This is used to 
                             allow each dimension to have different sized frequency spaces;
                             dimensions with smaller frequency spaces (than ntmax) should be 
                             padded at the end with missingFreqVal.

        output:
        -------

            The DFT of abscissa/ordinate pairs.  Calculated in one dimension as:

            DFT[t] = sum( [ a[i] * exp(-j * x[i] * t ) for i in range(N) ] ) 

            for each of the t points in frequencyGrids (and where j = sqrt(-1)).
    
    """

    cdef int numDimensions
    cdef int numDataPoints
    cdef int ntMax

    #*******************************
    # Get variable dimensionalities
    # and do consistency checks
    #*******************************
    #Get the shape of abscissas 
    try:
        numDimensions = abscissas.shape[0]
        numDataPoints = abscissas.shape[1]
    except:
        raise ValueError,"Could not determine shape of abscissas"

    #Check ordinates
    try:
        ordShape = ordinates.shape[0]
    except:
        raise ValueError,"Could not determine shape of ordinates"
        
    if ordShape != numDataPoints:
            raise ValueError, "Incompatible shapes for ordinates and abscissas"

    #Check frequencyGrids
    try:
        freqShape = frequencyGrids.shape
    except:
        raise ValueError,"Could not determine shape of ordinates"

    if freqShape[0] != numDimensions:
            raise ValueError, "Incompatible shapes for abscissas and frequencyGrids"

    #Get the max number of frequency points
    ntMax = freqShape[1]


    #********************************************
    # Calculate the size of the frequency spaces
    #********************************************
    cdef int n
    cdef np.ndarray[np.int_t,ndim=1] frequencySizes = np.zeros([numDimensions],dtype=np.int)

    for n in range(numDimensions):
        iNotMissing = np.nonzero(frequencyGrids[n,:] != missingFreqVal)[0]
        if len(iNotMissing) != 0:
            frequencySizes[n] = len(iNotMissing)
        else:
            raise ValueError,"Some frequencies in frequencyGrids have no valid points"

    #Get the total size of the frequency space
    cdef np.int_t freqSpaceSize = np.prod(frequencySizes)

    #Pre-declare and allocate a raveled form of the DFT
    cdef np.ndarray[double complex,ndim=1] DFT = np.zeros([freqSpaceSize],dtype=np.complex128)

    cdef int t,i

    cdef np.ndarray[np.int_t,ndim=1] dimInds = np.zeros([numDimensions],dtype=np.int)

    cdef double complex myDFT
    cdef double ecfArg

    #cdef double complex dftConst = -2.0j * <double complex> np.pi
    cdef double complex dftConst = -1.0j #* <double complex> np.pi

    #cdef double complex normConstant = <double complex> (1./(numDataPoints * (2 * np.pi)**((<float> numDimensions)/2.0)))
    cdef double complex normConstant = 1.0j


    for i in range(freqSpaceSize):
        unravelIndex(i,frequencySizes,dimInds)

        myDFT = 0.0 + 0.0j 

        for j in range(numDataPoints):
            ecfArg = 0.0

            for k in range(numDimensions):
                ecfArg += (abscissas[k,j] * frequencyGrids[k,dimInds[k]])#/frequencySizes[k]

            myDFT += ordinates[j]*np.exp(dftConst * <double complex> ecfArg)

        DFT[i] = normConstant * myDFT

    return np.reshape(DFT,tuple(frequencySizes))
                
                


#*******************************************************************************
#*******************************************************************************
#********************* unravelIndex() ******************************************
#*******************************************************************************
#*******************************************************************************
@cython.boundscheck(False)
cpdef int unravelIndex( \
                   np.int_t i, \
                   np.int_t [:] frequencySizes, \
                   np.int_t [:] dimInds) :
    """Takes the 1D index i of a raveled variable of shape frequencySizes and returns
    an array of the unraveled indices."""

    cdef np.int_t ndims
    cdef np.int_t n,nd,iDum
    cdef np.int_t hyperSize

    ndims = frequencySizes.shape[0]

    hyperSize = 1
    for n in range(1,ndims):
        hyperSize *= frequencySizes[n]
        
    iDum = i
    for n in range(ndims):
        dimInds[n] = <np.int_t> floor((<double> iDum)/(<double> hyperSize))
        iDum -= dimInds[n]*hyperSize
        if n < ndims:
            hyperSize /= frequencySizes[n+1]

    return 0

#*******************************************************************************
#*******************************************************************************
#********************* ravelIndex() ********************************************
#*******************************************************************************
#*******************************************************************************
@cython.boundscheck(False)
cpdef int ravelIndex( \
                   np.int_t [:] frequencySizes, \
                   np.int_t [:] dimInds) :
    """Calculates the 1D index i of a raveled variable of shape frequencySizes, given
    an array of the unraveled indices."""

    cdef np.int_t ndims
    cdef np.int_t n
    cdef np.int_t hyperSize
    cdef np.int_t i

    ndims = frequencySizes.shape[0]

    hyperSize = 1
    i = 0
    for n in range(ndims-1,-1,-1):
        i += hyperSize*dimInds[n]
        hyperSize *= frequencySizes[n]
    return i

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
  return  fft.fftshift(fft.fftfreq(len(tpoints),deltaT/(2*np.pi)))

def calcTfromX(xpoints):
  """Calculates frequency points given a signal in real space. """
  #Use fftfreq to produce a set of frequencies that correspond to the fourier
  #transform of a signal on the x points
  deltaX = xpoints[1] - xpoints[0]
  return fft.fftshift(fft.fftfreq(len(xpoints),deltaX/(2*np.pi)))

