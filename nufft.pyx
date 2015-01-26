import numpy as np
cimport numpy as np
cimport cython
import numpy.fft as fft
import cython.parallel as cpy

cdef extern from "math.h" nogil:
    double floor(double x)
    double exp(double x)

cdef extern from "complex.h" nogil:
    double complex cexp(double complex x)

#*******************************************************************************
#*******************************************************************************
#************************** nuifft *********************************************
#*******************************************************************************
#*******************************************************************************
@cython.boundscheck(False)
cpdef np.ndarray[double complex] nuifft( \
                            np.float_t [:,:] abscissas, \
                            np.complex128_t [:] ordinates, \
                            np.float_t [:,:] frequencyGrids, \
                            np.float_t missingFreqVal = -1e20, \
                            int precision = 2, \
                            int beVerbose = 0):
    """Approximates the unnormalized direct Fourier transform of abscissa, ordinate pairs
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

            beVerbose   : Flags whether to print to STDOUT as the method progresses
                          (int 0=don't print 1=print)

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
    vprint("Checking dimensionalities and arguments",beVerbose)
    #Get the shape of abscissas 
    try:
        numDimensions = np.shape(abscissas)[0]
        numDataPoints = np.shape(abscissas)[1]
    except:
        raise ValueError,"Could not determine shape of abscissas"

    #Check ordinates
    try:
        ordShape = np.shape(ordinates)[0]
    except:
        raise ValueError,"Could not determine shape of ordinates"
        
    if ordShape != numDataPoints:
            raise ValueError, "Incompatible shapes for ordinates and abscissas"

    #Check frequencyGrids
    try:
        freqShape = np.shape(frequencyGrids)
    except:
        raise ValueError,"Could not determine shape of ordinates"

    if freqShape[0] != numDimensions:
            raise ValueError, "Incompatible shapes for abscissas and frequencyGrids"

    #Get the max number of frequency points
    ntMax = freqShape[1]


    #********************************************
    # Calculate the size of the frequency spaces
    #********************************************
    vprint("Getting the size of the frequency spaces",beVerbose)
    cdef int n,t,iNotMissing
    cdef np.int_t [:] frequencySizes = np.zeros([numDimensions],dtype=np.int)

    for n in range(numDimensions):
        iNotMissing = 0
        for t in range(ntMax):
            if frequencyGrids[n,t] != missingFreqVal:
                iNotMissing += 1
        if iNotMissing != 0:
            frequencySizes[n] = <np.int_t>iNotMissing
        else:
            raise ValueError,"Some frequencies in frequencyGrids have no valid points"

    #Get the total size of the frequency space
    cdef np.int_t freqSpaceSize = np.prod(frequencySizes)



    cdef np.float_t [:,:] \
            abscissaGrids = missingFreqVal*np.ones([numDimensions,ntMax])
    #******************************
    # Calculate the abscissa grids
    #******************************
    vprint("Creating the convolution grid",beVerbose)
    for n in range(numDimensions):
        xdum = calcXfromT(frequencyGrids[n,:frequencySizes[n]])
        for t in range(frequencySizes[n]):
            abscissaGrids[n,t] = xdum[t]


    #*******************************
    # Define convolution parameters 
    #*******************************
    vprint("Initializing the convolution",beVerbose)
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
    #cdef np.ndarray[np.int_t,ndim=1] hyperSlabShape = nspread*np.ones([numDimensions],dtype=np.int)
    cdef np.int_t [:] hyperSlabShape = nspread*np.ones([numDimensions],dtype=np.int)
    vprint("\tconvolution hyperslab shape: {}".format(hyperSlabShape),beVerbose)

    #Calculate the quantities necessary for estimating x-indices
    cdef np.float_t [:]  \
            xmins = np.array([ abscissaGrids[n,0] for n in range(numDimensions) ])
    cdef np.float_t [:]  \
            deltaxs = np.array([ abscissaGrids[n,1] - abscissaGrids[n,0] for n in range(numDimensions) ])

    #Inialize worker terms for the convolution
    #cdef np.ndarray[long,ndim=1] mvec = np.zeros([numDimensions],dtype=np.int)
    cdef np.int_t [:] mvec = np.zeros([numDimensions],dtype=np.int)
    cdef np.int_t [:] m0vec = np.zeros([numDimensions],dtype=np.int)
    cdef np.float_t [:] mprimevec = np.zeros([numDimensions])
    cdef np.float_t mprime = 0.0
    cdef double complex gaussTerm = 0.0
    cdef np.float_t gaussArg = 0.0
    cdef int j, v, i, k
    i = j = v = k = 0

    #Initialize the (raveled) data convolution
    cdef double complex [:] convolvedData = np.zeros([freqSpaceSize],dtype=np.complex128)


    #***********************************
    # Convolve the data with a gaussian
    #***********************************
    vprint("Performing the convolution",beVerbose)
    with nogil:
        for j in range(numDataPoints):
            
            #Transform the coordinates of our current abscissas value
            #into the coordinate system of a hyperslab centered on the nearest
            #point (nearest in the floor() sense)
            for v in range(numDimensions):
                #Transform the current abscissa value into an approximate
                #grid value
                mprime = (abscissas[v,j] - xmins[v])/deltaxs[v]
                #Calculate the nearest (in the floor sense) grid abscissa
                m0vec[v] = <long> floor(mprime)
                #Finish transforming the absicssa value to hyperslab coordinates
                mprimevec[v] = mprime - m0vec[v] + nspreadhalf

            #Cycle over all the hyperslab points and calculate the gaussian convolution
            #of the current data point only on points in the hyperslab
            for i in range(hyperSlabSize):
                #Get the indices (mvec) of the current point (i) in our hyperslab
                unravelIndex(i,hyperSlabShape,mvec,numDimensions)

                gaussArg = 0.0
                for v in range(numDimensions):
                    #Calculate the distance between the data point and the current
                    #hyperslab point
                    gaussArg = gaussArg + (<np.float_t>mvec[v] - mprimevec[v])**2
                    #Transform from hyperslab coordinates back to full space coordinates
                    mvec[v] = (mvec[v] + m0vec[v]) - nspreadhalf
                #Calculate the gaussian of this distance
                gaussTerm = ordinates[j]*(<double complex> exp(-gaussArg/fourTau))

                #Calculate the flattened array index of the current point
                k = ravelIndex(frequencySizes,mvec,numDimensions)

                #Add the gaussian term to this point (only if it is within our domain
                if k >= 0 and k < freqSpaceSize:
                    convolvedData[k] = convolvedData[k] + gaussTerm

    #Take its FFT
    #( ifftshift is used to reorder the kde array such that 0 is the lowest corner 
    #  first index] of the array, as required by ifft)
    #And fftshift is used to put the zero-frequency in the center of the array
    #reshape is used to form a proper n-dimensional array from the vector of convolved data
    vprint("Raveling and doing FFT on convolved data",beVerbose)
    convolvedFFT = fft.fftshift(fft.ifftn(fft.ifftshift(np.reshape(convolvedData,frequencySizes))))
    vprint("\tshape(convolvedFFT) = {}".format(np.shape(convolvedFFT)),beVerbose)

    vprint("Initializing the deconvolution",beVerbose)

    #Ravel the convolved data
    cdef double complex [:] convolvedFFTRaveled = convolvedFFT.ravel()

    #Pre-declare and allocate a raveled form of the DFT
    cdef double complex [:] DFT = np.zeros([freqSpaceSize],dtype=np.complex128)
    #Pre declare a dimension index vector
    cdef np.int_t [:] dimInds = np.zeros([numDimensions],dtype=np.int)

    #Deconvolve the FFT (divide by the FFT of the gaussian) to obtain the DFT estimate
    vprint("Deconvolving the Fourier transformed data",beVerbose)
    with nogil:
        for i in range(freqSpaceSize):
            unravelIndex(i,frequencySizes,dimInds,numDimensions)
            gaussArg = 0.0
            for v in range(numDimensions):
               gaussArg = gaussArg + (frequencyGrids[v,dimInds[v]]*deltaxs[v])**2
            DFT[i] = convolvedFFTRaveled[i] * cexp(tau*gaussArg)


    #Reshape the DFT to an array
    vprint("Reshaping and returning.",beVerbose)
    return np.reshape(DFT,tuple(frequencySizes))


#*******************************************************************************
#*******************************************************************************
#************************** idft ***********************************************
#*******************************************************************************
#*******************************************************************************
@cython.boundscheck(False)
cpdef np.ndarray[double complex] idft( \
                            np.float_t [:,:] abscissas, \
                            np.complex128_t [:] ordinates, \
                            np.float_t [:,:] frequencyGrids, \
                            np.float_t missingFreqVal = -1e20, \
                            beVerbose = False):
    """Calculates the unnormalized direct Fourier transform of abscissa, ordinate pairs
        
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

            beVerbose   : Flags whether to print to STDOUT as the method progresses
                          (int 0=don't print 1=print)
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
    vprint("Checking dimensionalities and arguments",beVerbose)
    try:
        numDimensions = np.shape(abscissas)[0]
        numDataPoints = np.shape(abscissas)[1]
    except:
        raise ValueError,"Could not determine shape of abscissas"

    #Check ordinates
    try:
        ordShape = np.shape(ordinates)[0]
    except:
        raise ValueError,"Could not determine shape of ordinates"
        
    if ordShape != numDataPoints:
            raise ValueError, "Incompatible shapes for ordinates and abscissas"

    #Check frequencyGrids
    try:
        freqShape = np.shape(frequencyGrids)
    except:
        raise ValueError,"Could not determine shape of ordinates"

    if freqShape[0] != numDimensions:
            raise ValueError, "Incompatible shapes for abscissas and frequencyGrids"

    #Get the max number of frequency points
    ntMax = freqShape[1]


    #********************************************
    # Calculate the size of the frequency spaces
    #********************************************
    vprint("Getting the size of the frequency spaces",beVerbose)
    cdef int n,t,iNotMissing
    cdef np.int_t [:] frequencySizes = np.zeros([numDimensions],dtype=np.int)

    for n in range(numDimensions):
        iNotMissing = 0
        for t in range(ntMax):
            if frequencyGrids[n,t] != missingFreqVal:
                iNotMissing += 1
        if iNotMissing != 0:
            frequencySizes[n] = <np.int_t>iNotMissing
        else:
            raise ValueError,"Some frequencies in frequencyGrids have no valid points"

    #Get the total size of the frequency space
    cdef np.int_t freqSpaceSize = np.prod(frequencySizes)

    #Pre-declare and allocate a raveled form of the DFT
    cdef double complex [:] DFT = np.zeros([freqSpaceSize],dtype=np.complex128)

    cdef int i,k

    cdef np.int_t [:] dimInds = np.zeros([numDimensions],dtype=np.int)

    cdef double complex myDFT
    cdef double expArg

    #cdef double complex dftConst = -2.0j * <double complex> np.pi
    cdef double complex dftConst = 1.0j #* <double complex> np.pi


    vprint("Calculting the DFT",beVerbose)
    with nogil:
        for i in range(freqSpaceSize):
            unravelIndex(i,frequencySizes,dimInds,numDimensions)

            myDFT = 0.0 + 0.0j 

            for j in range(numDataPoints):
                expArg = 0.0

                for k in range(numDimensions):
                    expArg = expArg +(abscissas[k,j] * frequencyGrids[k,dimInds[k]])

                myDFT = myDFT + ordinates[j]*cexp(dftConst * <double complex> expArg)

            DFT[i] = myDFT

    return np.reshape(DFT,tuple(frequencySizes))

#*******************************************************************************
#*******************************************************************************
#********************* unravelIndex() ******************************************
#*******************************************************************************
#*******************************************************************************
@cython.boundscheck(False)
cdef inline int unravelIndex( \
                   np.int_t i, \
                   np.int_t [:] frequencySizes, \
                   np.int_t [:] dimInds, \
                   np.int_t ndims) nogil:
    """Takes the 1D index i of a raveled variable of shape frequencySizes and returns
    an array of the unraveled indices."""

    cdef np.int_t n,nd,iDum
    cdef np.int_t hyperSize

    hyperSize = 1
    for n in range(1,ndims):
        hyperSize *= frequencySizes[n]
        
    iDum = i
    for n in range(ndims):
        dimInds[n] = <np.int_t> floor((<double> iDum)/(<double> hyperSize))
        iDum -= dimInds[n]*hyperSize
        if n < (ndims-1):
            hyperSize /= frequencySizes[n+1]
        else:
            hyperSize = 1

    return 0

#*******************************************************************************
#*******************************************************************************
#********************* ravelIndex() ********************************************
#*******************************************************************************
#*******************************************************************************
@cython.boundscheck(False)
cdef inline int ravelIndex( \
                   np.int_t [:] frequencySizes, \
                   np.int_t [:] dimInds, \
                   np.int_t ndims) nogil:
    """Calculates the 1D index i of a raveled variable of shape frequencySizes, given
    an array of the unraveled indices."""

    cdef np.int_t n
    cdef np.int_t hyperSize
    cdef np.int_t i

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

def vprint(msg,beVerbose):
    """Prints only if beVerbose is True"""
    if beVerbose:
        print(msg)


