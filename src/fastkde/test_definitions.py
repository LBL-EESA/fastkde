import fastkde.fastKDE as fastKDE
import time
import scipy.optimize as opt
import scipy.stats as stat
import pylab as PP
from multiprocessing import Pool
import numpy as npy

berkeleyLabText = """                                                           
.++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++ +++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++';   :+++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++;+++++   ++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++ ++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++':.. ..:;++++++++++++++++++++++++++++++,+++++++++++
++++++++++++++++;     `....       ;+++++++++++++++++++++++++.;++++++++++
+++++++++++++;  :+++++++++++++++:    :+++++++++++++++++++++++ ++++++++++
+++++++++++`,++++++++++++++++++++++;    +++++++++++++++++++ +,++++++++++
+++++++++`++++++++++++++++++++++++++++    ++++++++++++++++++++ +++++++++
+++++++;++++++++++++++++++++++++++++++++   ,++++++++++++++:+++ +++++++++
++++++++++++++++++++++++++++++++++++++++++   ++++++++++++:+++++:++++++++
+++++++++++++++++++++++++++++++++++++++++++`  ;++++++++++ ++++++`+++++++
+++++++++++++++++++++++++++++++++++++++++++++  `+++++++++ '+:+:; +++++++
++++++++++++++++++++++++++++++++++++++++++++++   ++++++++ .; + ; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++   ...++++ .; + ; +++++++
++++++++++++++++++++++++++++++++++++++++++++++++     ++++ .; + ; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ .; + ; +++++++
+++++++.  ,++:   +++. `+++,  `+++   +++   ,++'  `++; ++++ .' + ; +++++++
++++++  ;+++.  ++++  ++++  .++++  ++++  ;+++:  ++++; ++++ +++++; +++++++
++++++  ++++  `++++  ++++  ;++++  ++++  ++++:  ++++; ++++ +++++; +++++++
++++++  ++++  .++++  ++++  ;++++  ++++  ++++.  ++++; ++++ +++++; +++++++
++++++ .++++ .'++++ .++++ .+++++ :++++ :++++  :++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++ +++++; +++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; ++++:+++++':+++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++; +++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++    +    +    +  +  +    + +++;    ; ++ `+++ ;++++  .++    +++++++
++++++ ++ . :+++ ++ :  ` ++ ++++ +++; ++++ .' ++++ ;++++ : ++ ++ ;++++++
++++++ .  + `..+ :. :   ;++ ..,+ +++; ..+++  :++++ ;++++ + ++ .  +++++++
++++++ :, '    +    +    ++   .+ +++;   +++  +++++ ;+++` + ;+ :` +++++++
++++++ ++ : :+++ +  +  + ,+ ++++ +++; +++++. +++++ ;+++     + ++ .++++++
++++++    ;    ; ++ :  +. +    +    ;    ++. +++++    + ++; +    +++++++
++++++:::++::::':++::::++:;::::+::::'::::++;:+++++::::':+++:+:::++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
,++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++,
"""  # Adapted from http://picascii.com/


def underlyingFunction(x, x0=305, y0=200, yrange=4):
    """A sigmoid with a transition centered on x0"""
    return (yrange / 2) * npy.tanh(x - x0) + y0


def stochasticSample(
    x0=305, y0=200, yrange=4, numSamples=1e3, xmid=305, xp1=5, xp2=2, yp1=0, yp2=None
):
    if yp2 is None:
        # By default, have the range of the noise be larger than the range in y
        yp2 = 3 * yrange
    # Generate random samples of X from the gamma distribution
    # Note: I flip the gamma distribution around here so that the upper range has a short tail
    x = -(npy.random.gamma(xp1, xp2, int(numSamples)) - xp1 * xp2) + xmid
    # Generate random samples of y from x and add normally distributed noise
    y = underlyingFunction(x, x0, y0, yrange) + npy.random.normal(
        loc=yp1, scale=yp2, size=numSamples
    )

    # Concatenate the paired samples together
    xy = npy.concatenate((x[npy.newaxis, :], y[npy.newaxis, :]), axis=0)
    return xy


def conditionalPDF(
    y, x, x0=305, y0=200, yrange=4, xmid=305, xp1=5, xp2=2, yp1=0, yp2=None
):
    if yp2 is None:
        # By default, have the range of the noise be larger than the range in y
        yp2 = 3 * yrange

    mu = underlyingFunction(x, x0, y0, yrange)

    return stat.norm.pdf(y, loc=mu, scale=yp2)


def marginalX(y, x, x0=305, y0=200, yrange=4, xmid=305, xp1=5, xp2=2, yp1=0, yp2=None):
    xbar = xp1 * xp2
    return stat.gamnpy.ma.pdf(x0 - x + xbar, a=xp1, scale=xp2)


def jointXY(y, x, x0=305, y0=200, yrange=4, xmid=305, xp1=5, xp2=2, yp1=0, yp2=None):
    # Return the product of the conditional and the marginal
    return marginalX(y, x, x0, y0, yrange, xmid, xp1, xp2, yp1, yp2) * conditionalPDF(
        y, x, x0, y0, yrange, xmid, xp1, xp2, yp1, yp2
    )


def getModeCurve(conditional, axes):
    """Extract the mode curve from a 2D conditional (assumes conditioning on axis 0)"""
    modeCurve = npy.array(
        [axes[1][conditional[:, i].argmax()] for i in range(conditional.shape[1])]
    )
    # Remove points whre all of the PDF is missing
    allMising = npy.prod((conditional.mask), axis=0)
    modeCurve = npy.ma.masked_where(allMising == 1, modeCurve)
    return modeCurve


def asciiToPoints(text, convertCharacter=" "):
    """Given ascii art text, convert the whitespace into a list of point"""

    # Get the lines from the text
    textLines = text.split("\n")[1:-1]

    # Convert the text into a point array
    pointArray = npy.array(
        [[1 if char == convertCharacter else 0 for char in line] for line in textLines]
    ).T[:, ::-1]

    return npy.where(pointArray == 1)


def powLaw(x, a, c):
    return c * x**a


# A simple timer for comparing ECF calculation methods
class Timer:
    """A simple timer class

    Example:
        ```python
        myTimer = Timer()
        with myTimer:
            doSomething()

        print myTimer.duration
        ```
    """

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.duration = time.time() - self.start


def genCovMat(varianceList, correlationDict):
    """Generates a covariance matrix, given a list of variances for all variables and a dict of correlation
    coefficients for all pairs of variables.
    
    The dict's keys must be tuples that describe the pair of points for the correlation coefficient.
    
    For example, the following would be valid input for for a 3 variable distribution:
    
        ```python
        varianceList = [0.1, 10.0, 100.0]
        correlationDict = { (0,1) :  0.9, \
                            (0,2) :  0.0, \
                            (1,2) : -0.4 }

        covMat = genCovMat(varianceList,correlationDict)
        ```
    """

    # Generate a list of all possible unique variable pairs
    numVars = len(varianceList)
    pairList = correlationDict.keys()
    for i, j in pairList:
        # Check that the indices are different from one another for this pair
        assert i != j, "The (i,j) tuple keys for correlationDict must all be different"
        assert i >= 0 and j >= 0, "i and j must be >= 0"
        assert i < numVars and j < numVars, "i and j must be < {}".format(numVars)
        # Check that the correlation coefficients are valid
        assert (
            abs(correlationDict[(i, j)]) <= 1
        ), "All correlation coefficients must have a magnitude less than or equal to 1"

    # Make sure we have enough correlation pairs
    numPairsNeeded = (numVars**2 - numVars) / 2
    assert (
        len(pairList) == numPairsNeeded
    ), "{} correlation pairs were provided; {} are needed".format(
        len(pairList), numPairsNeeded
    )

    # Initialize the covariance matrix
    covMat = npy.zeros([numVars, numVars])

    # Insert the variances
    for i in range(numVars):
        covMat[i, i] = varianceList[i]

    # Insert the covariances
    for i, j in pairList:
        covMat[i, j] = covMat[j, i] = (
            correlationDict[(i, j)]
            * npy.sqrt(varianceList[i])
            * npy.sqrt(varianceList[j])
        )

    return covMat


def estimatePDFWrapper(arg, **kwargs):
    """A timed wrapper for doing a PDF estimate and evaluating its error"""

    j, numSamplesMax, sampleList, scKWArgs, conditionVar, pdfTestObj = arg

    # Seed the random number generator
    npy.random.seed(j)

    # Pull samples from the distribution
    mySamples = pdfTestObj.sampleFromDistribution(numSamplesMax)

    ISEVals = []
    timingVals = []
    for numSamples in sampleList:
        # Extract the current sample set
        mySample = mySamples[..., :numSamples]

        # Initialize the timer
        myTimer = Timer()

        # Do the KDE estimate and time it
        with myTimer:
            _pdf = fastKDE.fastKDE(mySample, **scKWArgs)
            if conditionVar is not None:
                cpdf = _pdf.estimateConditionals(conditionVar, mySample, peakFrac=0.01)

        # Calculate the integrated squared error
        deltaXs = [npy.diff(ax) for ax in _pdf.axes]
        deltaXs = [npy.concatenate((dx, [dx[-1]])) for dx in deltaXs]
        deltaXProducts = npy.prod(npy.meshgrid(*tuple(deltaXs)), axis=0)

        if conditionVar is None:
            ISE = npy.ma.sum(
                deltaXProducts * (pdfTestObj.pdfStandard(_pdf.axes) - _pdf.pdf) ** 2
            )
        else:
            ISE = npy.ma.sum(
                deltaXProducts * (pdfTestObj.pdfStandard(_pdf.axes) - cpdf) ** 2
            ) / len(cpdf.compressed())

        ISEVals.append(ISE)
        timingVals.append(myTimer.duration)

    return timingVals, ISEVals, mySamples


class testDistribution(object):
    """A generic distribution--meant to be overridden--for testing the fast KDE method"""

    def __init__(self, **kwargs):
        # Save all incoming keyword arguments
        for key, val in kwargs.items():
            self.__setitem__(key, val)

        self.pdfName = "None"

    def sampleFromDistribution(self, numSamples=2097152):
        pass

    def pdfStandard(self, axes):
        pass

    def doTesting(
        self,
        numSamplesMax=2097152,
        numRepetitions=30,
        scKWArgs={
            "fracContiguousHyperVolumes": 1,
            "numPoints": 257,
            "positiveShift": False,
        },
        numProcs=1,
    ):
        self.numProcs = int(numProcs)

        try:
            conditionVar = self.conditionVar
        except AttributeError:
            conditionVar = None

        # Set the number of test samples to be a power of 2
        powmax = int(npy.floor(npy.log2(numSamplesMax)))
        numSamplesMax = 2**powmax

        # Set the list of numbers-of-samples
        sampleList = 2 ** (npy.arange(4, powmax + 1)) + 1

        # Save the sampling list
        self.sampleList = sampleList

        if self.numProcs == 1:
            returnList = [
                estimatePDFWrapper(
                    (j, numSamplesMax, sampleList, scKWArgs, conditionVar, self)
                )
                for j in range(numRepetitions)
            ]

        else:
            # Do the PDF estimation using multiple processors
            estimationPool = Pool(self.numProcs)
            returnList = estimationPool.map(
                estimatePDFWrapper,
                zip(
                    range(numRepetitions),
                    [numSamplesMax] * numRepetitions,
                    [sampleList] * numRepetitions,
                    [scKWArgs] * numRepetitions,
                    [conditionVar] * numRepetitions,
                    [self] * numRepetitions,
                ),
            )

        # Save the longest sample of each repetition
        # self.sampleObjs = [ ra[2] for ra in returnList]
        # Save the timing values
        self.masterTiming = npy.array([ra[0] for ra in returnList])
        # Save the ISE values
        self.masterISE = npy.array([ra[1] for ra in returnList])

        self.meanISE = npy.average(self.masterISE, axis=0)
        self.meanTiming = npy.average(self.masterTiming, axis=0)
        self.stdISE = npy.std(self.masterISE, axis=0)
        self.stdTiming = npy.std(self.masterTiming, axis=0)

        popt, pcov = opt.curve_fit(
            powLaw, self.sampleList, self.meanISE, p0=(1, -1), sigma=self.stdISE
        )

        self._popt = popt
        self._pcov = pcov
        self.ISESlope = popt[0]
        self.ISENorm = popt[1]
        self.ISEFit = popt[1] * npy.array(self.sampleList) ** popt[0]

    def generatePlots(self, saveType=None, show=True):
        """Generate plots of the error rate and the timing"""

        fig = PP.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, xscale="log", yscale="log")

        ax.errorbar(self.sampleList, self.meanISE, yerr=self.stdISE)
        ax.plot(self.sampleList, self.ISEFit, linewidth=3, alpha=0.5, color="gray")

        ax.set_xlabel("# of Samples")
        ax.set_ylabel("I.S.E.")

        ax.legend(
            [
                "{} ISE".format(self.pdfName),
                r"N$^{" + "{:0.02f}".format(self.ISESlope) + "}$",
            ]
        )

        if show:
            PP.show()


class testNormal1D(testDistribution):
    def __init__(self, **kwargs):
        # Call the class constructor
        super(testNormal1D, self).__init__(**kwargs)

        try:
            self.mu
        except AttributeError:
            self.mu = 0.0

        try:
            self.sig
        except AttributeError:
            self.sig = 1.0

        self.pdfName = "Normal"

    def sampleFromDistribution(self, numSamples=2097152):
        """Samples from a random normal distribution"""
        sig = self.sig
        mu = self.mu
        randsample = sig * npy.random.normal(size=numSamples) + mu
        return randsample[npy.newaxis, :]

    def pdfStandard(self, axes):
        """Returns the value of the normal distribution at location `axes`"""
        x = axes[0]
        sig = self.sig
        mu = self.mu
        pdfVal = (1.0 / (sig * npy.sqrt(2 * npy.pi))) * npy.exp(
            -((x - mu) ** 2) / (2.0 * sig**2)
        )

        return pdfVal


class testNormal2D(testDistribution):
    def __init__(self, **kwargs):
        # Call the class constructor
        super(testNormal2D, self).__init__(**kwargs)

        try:
            self.mu
        except AttributeError:
            self.mu = npy.array([0.0, 0.0])

        try:
            self.sig
        except AttributeError:
            self.sig = npy.array([1.0, 3.0])

        try:
            self.rho
        except AttributeError:
            self.rho = 0.7

        self.pdfName = "Bivariate Normal"

    def sampleFromDistribution(self, numSamples=2097152):
        """Samples from a bivariate normal distribution"""
        sx, sy = self.sig
        mu = self.mu
        r = self.rho
        covMat = [[sx**2, r * sx * sy], [r * sx * sy, sy**2]]

        randsample = npy.random.multivariate_normal(mu, covMat, (numSamples,)).T
        return randsample

    def pdfStandard(self, axes):
        """Returns the value of the bivariate normal distribution at location `axes`"""
        sx, sy = self.sig
        mu = self.mu
        r = self.rho

        C = [[sx**2, r * sx * sy], [r * sx * sy, sy**2]]

        xarrays = npy.array(npy.meshgrid(axes[0], axes[1])).T

        return stat.multivariate_normal.pdf(xarrays, mu, C).T


class testNormalND(testDistribution):
    def __init__(self, **kwargs):
        # Call the class constructor
        super(testNormalND, self).__init__(**kwargs)

        try:
            self.mu
        except AttributeError:
            self.mu = [0.0, 50.0, -50.0]

        try:
            self.covMat
        except AttributeError:
            varianceList = [0.1, 10.0, 100.0]
            correlationDict = {(0, 1): 0.9, (0, 2): 0.0, (1, 2): -0.4}
            self.covMat = genCovMat(varianceList, correlationDict)

        self.rank = len(self.mu)

        self.pdfName = "Multivariate Normal"

    def sampleFromDistribution(self, numSamples=2097152):
        """Samples from a multivariate normal distribution"""
        mu = self.mu
        covMat = self.covMat

        # randsample = npy.random.multivariate_normal(mu,covMat,(numSamples,)).T
        randsample = stat.multivariate_normal.rvs(mu, covMat, (numSamples,)).T
        return randsample

    def pdfStandard(self, axes):
        """Returns the value of the multivariate normal distribution at location `axes`"""

        mu = self.mu
        C = self.covMat
        k = self.rank

        lastFirst = npy.zeros(k + 1, dtype=int)
        lastFirst[-1] = 0
        # lastFirst[:-1] = roll(npy.arange(k) + 1,1)
        lastFirst[:-1] = npy.arange(k) + 1
        xarrays = npy.transpose(
            npy.array(npy.meshgrid(*axes, indexing="ij")), axes=lastFirst
        )

        return stat.multivariate_normal.pdf(xarrays, mu, C).T


class testMixtureModel(testDistribution):
    def __init__(self, **kwargs):
        # Call the class constructor
        super(testMixtureModel, self).__init__(**kwargs)

        try:
            self.muList
        except AttributeError:
            xp, yp = asciiToPoints(berkeleyLabText)
            xbar = 0.9 * npy.median(xp)
            ybar = npy.median(yp)

            # Transform to center
            xp = xp - xbar
            yp = yp - ybar

            # Rotate the image
            theta = 37.0 * npy.pi / 180.0
            xpd = xp * npy.cos(theta) - yp * npy.sin(theta)
            ypd = xp * npy.sin(theta) + yp * npy.cos(theta)
            xp = xpd
            yp = ypd

            # Calculate the variance
            varx = npy.var(xp)
            vary = npy.var(yp)

            self.muList = list(zip(xp, yp))
            self.covList = list(0.3 * npy.ones(len(self.muList)))

            for i in range(int(len(xp))):
                self.muList.append((0.0, 0.0))
                self.covList.append(genCovMat([varx, vary], {(0, 1): 0.0}))

        self.pdfName = "Mixture of Standard Normals"

    def sampleFromDistribution(self, numSamples=2097152):
        """Samples from a multivariate normal distribution"""
        numNormals = len(self.muList)

        # Generate a sequence of random draws, representing randomly (uniformly) choosing
        # from among the given normal distributions
        iDraws = npy.random.randint(0, numNormals, size=numSamples)

        # From these draws, construct the lengths of samples for each multivariate normal
        numDraws = [len(npy.nonzero(iDraws == i)[0]) for i in range(numNormals)]

        # randsample = npy.random.multivariate_normal(mu,covMat,(numSamples,)).T
        sampleList = []
        for mu, cov, N in zip(self.muList, self.covList, numDraws):
            if N > 0:
                sample = stat.multivariate_normal.rvs(mu, cov, N)
                if len(npy.shape(sample)) == 1:
                    sample = sample[npy.newaxis, :]
                sampleList.append(sample)

        # Concatenate the samples
        randsample = npy.concatenate(sampleList, axis=0)

        # Shuffle the samples in-place
        npy.random.shuffle(randsample)

        # Return the transpose [var,sample]
        return randsample.T

    def pdfStandard(self, axes):
        """Returns the value of the multivariate normal distribution at location `axes`"""

        k = len(axes)

        # Create the x grid to feed to multivariate_normal
        lastFirst = npy.zeros(k + 1, dtype=int)
        lastFirst[-1] = 0
        lastFirst[:-1] = npy.arange(k) + 1
        xarrays = npy.transpose(
            npy.array(npy.meshgrid(*axes, indexing="ij")), axes=lastFirst
        )

        pdfStandard = npy.zeros(npy.shape(xarrays[..., 0]))

        # Go through the distribution centers
        numNormals = float(len(self.muList))
        for mu, cov in zip(self.muList, self.covList):
            pdfStandard += stat.multivariate_normal.pdf(xarrays, mu, cov) / numNormals

        return pdfStandard.T


#    def doTesting(self, \
#                  scKWArgs = {'fracContiguousHyperVolumes' : 1, \
#                              'numPoints' : 513}, \
#                  **kwargs):
#        """Default to 513 points for the mixture model"""
#        if 'scKWArgs' in kwargs:
#            del(kwargs['scKWArgs'])
#        super(testMixtureModel,self).doTesting(scKWArgs=scKWArgs,**kwargs)


class transitionPDF(testDistribution):
    """A test for transition PDF convergence"""

    def __init__(self, **kwargs):
        # Save all incoming keyword arguments
        for key, val in kwargs.items():
            self.__setitem__(key, val)

        self.pdfName = "Transition"

    def sampleFromDistribution(self, numSamples=2097152):
        return stochasticSample(numSamples=numSamples)

    def pdfStandard(self, axes):
        xx, yy = npy.meshgrid(*axes)
        return jointXY(yy, xx)


class testConditional(transitionPDF):
    """A test for transition PDF convergence of the conditional"""

    def __init__(self, **kwargs):
        # Save all incoming keyword arguments
        for key, val in kwargs.items():
            self.__setitem__(key, val)

        self.pdfName = "Conditional"
        self.conditionVar = 0

    def pdfStandard(self, axes):
        xx, yy = npy.meshgrid(*axes)
        return conditionalPDF(yy, xx)
