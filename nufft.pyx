import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h" nogil:
    double floor(double x)

cdef extern from "complex.h" nogil:
    complex exp ( complex x)

cpdef np.ndarray[double complex] DFT( \
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

            DFT[t] = sum( [ a[i] * exp(-j * x[i] * t * 2 * pi / N) for i in range(N) ] ) 

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

    #Pre-declare and allocate a raveled form of the ECF
    cdef np.ndarray[double complex,ndim=1] ECF = np.zeros([freqSpaceSize],dtype=np.complex128)

    cdef int t,i

    cdef np.ndarray[np.int_t,ndim=1] dimInds = np.zeros([numDimensions],dtype=np.int)

    cdef double complex myECF
    cdef double ecfArg

    #cdef double complex dftConst = -2.0j * <double complex> np.pi
    cdef double complex dftConst = -1.0j #* <double complex> np.pi

    #cdef double complex normConstant = <double complex> (1./(numDataPoints * (2 * np.pi)**((<float> numDimensions)/2.0)))
    cdef double complex normConstant = 1.0j


    for i in range(freqSpaceSize):
        unravelIndex(i,frequencySizes,dimInds)

        myECF = 0.0 + 0.0j 

        for j in range(numDataPoints):
            ecfArg = 0.0

            for k in range(numDimensions):
                ecfArg += (abscissas[k,j] * frequencyGrids[k,dimInds[k]])#/frequencySizes[k]

            myECF += ordinates[j]*np.exp(dftConst * <double complex> ecfArg)

        ECF[i] = normConstant * myECF

    return np.reshape(ECF,tuple(frequencySizes))
                
                


#*******************************************************************************
#*******************************************************************************
#********************* uravelIndex() *******************************************
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

