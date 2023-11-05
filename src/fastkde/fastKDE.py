# !/usr/bin/env python

import numpy as npy
import scipy.optimize
import fastkde.empiricalCharacteristicFunction as ecf
import copy
import time
import warnings
from fastkde.nufft import calc_t_from_x, dft_points

import fastkde.floodFillSearch as flood


# A simple timer for comparing ECF calculation methods
class Timer:
    def __init__(self, n=None):
        self.n = n

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print("N = {}, t = {} seconds".format(self.n, time.time() - self.start))


def next_highest_power_of_two(number):
    """Returns the nearest power of two that is greater than or equal to number"""
    return int(2 ** (npy.ceil(npy.log2(number))))


class fastKDE:
    def __init__(
        self,
        data=None,
        axes=None,
        log_axes=False,
        num_points_per_sigma=10,
        num_points=None,
        do_approximate_ecf=True,
        ecf_precision=1,
        do_save_transformed_kernel=False,
        do_fft=True,
        do_save_marginals=True,
        be_verbose=False,
        frac_contiguous_hyper_volumes=1,
        num_contiguous_hyper_volumes=None,
        positive_shift=True,
        count_threshold=None,
        axis_expansion_factor=1.0,
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

        input:
        ------

          data (array_like)   : the data from which to estimate the PDF.  Should be 1-
                                or 2-dimensional. If 2-dimensional, this flags calculation
                                of an N-dimensional PDF.  The first index
                                should refer to each variable and the second index the
                                observations of the varibles.

          axes                : the axis-values of the estimated PDF.  They must be evenly
                                spaced and they should have a length that is a power of two
                                plus one (e.g., 33).

          log_axes            : Flags whether axes should be log spaced (i.e., the
                                PDF is calculated based on log(data) and then
                                transformed back to sample space).  Should be a
                                logical value (True or False) or a list of logical
                                values with an item for each variable (i.e,
                                len(log_axes) == shape(data)[0]) specifying which
                                axes should use log spacing.  If only True or False
                                is given, that value is used for all variables.

          num_points_per_sigma   : the number of points on the data grid per
                                   standard deviation; this influences the total
                                   size of the axes that are automatically
                                   calculated if no other aspects of the grid are
                                   specified.

          num_points           : the number of points to use for the pdf grid. If
                                 provided as a scalar, each axis will have the same
                                 number of points.  Otherwise, it should be an
                                 iterable with a value for each axis length.  Axis
                                 lengths should be a power of two plus one (e.g.,
                                 33)

          deltaX              : if given, this specifies the spacing between domain
                                values.

          do_approximate_ecf    : flags whether to approximate the ECF using a (much
                                  faster) FFT.  In tests, this is accurate to ~1e-14
                                  over low frequencies, but is inaccurate to ~1e-2
                                  for the highest ~5% of frequencies.

          ecf_precision        : sets the precision of the approximate ECF.  If set
                                 to 2, it uses double precision accuracy; 1
                                 otherwise

          do_fft               : flags whether to calculate phiSC and its FFT to
                                 obtain pdf

          do_save_marginals     : flags whether to calculate and save the marginal
                                  distributions

          frac_contiguous_hyper_volumes : the fraction of contiguous hypervolumes of
                                          the ECF, that are above the ECF threshold,
                                          to use in the density estimate

          num_contiguous_hyper_volumes : like frac_contiguous_hyper_volumes, but
                                         specify an integer number to use.
                                         frac_contiguous_hyper_volumes will be
                                         ignored if this is provided as an argument.

          positive_shift     : translate the PDF vertically such that the estimate
                               is positive or 0 everywhere

          count_threshold    : this argument does nothing; it has been deprecated.
                               It is kept as an argument for backward compatibility.

          axis_expansion_factor : sets the amount by which the KDE grid will be
                                  expanded relative to the original min-max spread
                                  for each variable: 1.0 means a 100% (2x) expansion
                                  in the range.  Such an expansion is necessary to
                                  avoid kernel power from one end of the grid
                                  leaking into the opposite end due to the
                                  perioidicity of the Fourier transform.

        Returns: a fastKDE object

        """

        def vprint(msg):
            """Only print if be_verbose is True"""
            if be_verbose:
                print(msg)

        addOne = True  # Force x grids to be (2**n) + 1

        if data is not None:
            # Save the original data for the marginal calculation
            original_data = npy.array(data)
            data = npy.array(data)

            # First check the rank of the data
            data_rank = len(npy.shape(data))
            # If the data are a vector, promote the data to a rank-1 array with only 1 column
            if data_rank == 1:
                data = npy.array(original_data[npy.newaxis, :], dtype=npy.float_)
            else:
                data = npy.array(original_data, dtype=npy.float_)
            if data_rank > 2:
                raise ValueError(
                    "data must be a rank-2 array of shape [num_variables,num_data_points]"
                )

            # Set the rank of the data
            self.data_rank = data_rank

            # Set the number of variables
            self.num_variables = npy.shape(data)[0]
            # Set the number of data points
            self.num_data_points = npy.shape(data)[1]

            # Check if we need log axes for any variables
            try:
                # Check if an iterable was provided for log_axes
                log_axes[0]
            except TypeError:
                # Otherwise set the array to be a list filled with the same value
                log_axes = self.num_variables * [log_axes]

            # Save the log_axes variable
            self.log_axes = log_axes

            # Loop over variables and take the logarithm of any with log axes
            for v in range(self.num_variables):
                # Take the logarithm of the given variable
                if log_axes[v]:
                    # Check wheter all the data are positive
                    if npy.amin(data[v, :]) <= 0:
                        raise ValueError(
                            (
                                "logarithmic axes were specified for variable {}, but"
                                + "that variable contains values less than 0: min = {}"
                            ).format(v, npy.amin(data[v, :]))
                        )
                    data[v, :] = npy.log(data[v, :])

            self.frac_contiguous_hyper_volumes = frac_contiguous_hyper_volumes

            if num_contiguous_hyper_volumes is not None:
                self.frac_contiguous_hyper_volumes = num_contiguous_hyper_volumes

            vprint(
                (
                    "Operating on data with num_variables = {}, "
                    + "num_data_points = {}"
                ).format(self.num_variables, self.num_data_points)
            )

        else:
            self.num_data_points = 0

        # Create a variable to hold the original PDF and axes
        self.originalPDF = None
        self.originalAxes = None

        # Store the do_fft flag
        self.do_fft = do_fft

        # Save the marginals flag
        self.do_save_marginals = do_save_marginals
        if self.num_variables == 1:
            self.do_save_marginals = False

        # Set whether to approximate the ECF using the FFT method
        self.do_approximate_ecf = do_approximate_ecf

        # Set the approximate ECF precision
        self.ecf_precision = ecf_precision

        # Preinitialize the ecf threshold
        self.ecfThreshold = None

        # Flag whether to save the transformed kernel
        self.do_save_transformed_kernel = do_save_transformed_kernel
        # initialize the kernel and its transform
        self.kappaSC = None
        self.kSC = None

        self.positive_shift = positive_shift

        # ***********************
        #  Calculate the x grids
        # ***********************
        if axes is None:
            # Get the range of the data
            self.xMin = npy.amin(data, 1)
            self.xMax = npy.amax(data, 1)

            vprint("Data stats:")
            vprint("\tminima: {}".format(self.xMin))
            vprint("\tmaxima: {}".format(self.xMax))

            #  Get the grid mid-points
            midPoint = 0.5 * (self.xMax + self.xMin)

            #  inflate the range by axis_expansion_factor to ensure that KDE power
            #  doesn't leak through the periodic axis boundary
            self.xMin += axis_expansion_factor * (self.xMin - midPoint)
            self.xMax += axis_expansion_factor * (self.xMax - midPoint)

            if num_points is None:
                # Calculate the number of standard deviations there
                #  are in the data range
                dataRange = self.xMax - self.xMin
                numSigma = dataRange / npy.std(data, axis=1)

                # Set the number of points for each dimensions
                self.numXPoints = npy.array(
                    [
                        next_highest_power_of_two(ns * num_points_per_sigma)
                        + int(addOne)
                        for ns in numSigma
                    ]
                )
            else:
                # If we can iterate through
                try:
                    lenNum = len(num_points)
                    isIterable = True
                except TypeError:
                    isIterable = False
                    lenNum = 1

                if isIterable:
                    if lenNum == self.num_variables:
                        self.numXPoints = num_points
                    else:
                        raise ValueError(
                            (
                                "len(num_points) = {}, but it should match"
                                + "num_variables = {}"
                            ).format(lenNum, self.num_variables)
                        )
                else:
                    self.numXPoints = npy.array(self.num_variables * (num_points,))

            # Set the grids for each dimension
            self.axes = [
                npy.linspace(xmin, xmax, np)
                for xmin, xmax, np in zip(self.xMin, self.xMax, self.numXPoints)
            ]

            vprint(
                ("Grids created with xmin: {}, xmax: {}, " + "npoints: {}").format(
                    self.xMin, self.xMax, self.numXPoints
                )
            )
        else:
            # Set the xgrid from the function argument
            self.axes = axes
            self.xMin = npy.array([npy.amin(xg) for xg in axes])
            self.xMax = npy.array([npy.amax(xg) for xg in axes])
            self.numXPoints = npy.array([len(xg) for xg in axes])
            # Get the grid mid-points
            self.midPoint = 0.5 * (self.xMax + self.xMin)

        # Set the midpoint of the incoming grid
        self.dataMid = 0.5 * (self.xMax + self.xMin)
        # Set the range to be +/- pi
        self.dataNorm = (self.xMax - self.xMin) / (2.0 * npy.pi)

        # Get the grid spacings
        self.deltaX = npy.array([xg[1] - xg[0] for xg in self.axes])

        # Save xgrids as axes for backward compatibility
        self.xgrids = self.axes

        # Check that the axes are regular and proper powers of two
        for v in range(self.num_variables):
            xg = self.axes[v]
            dx = (xg[1:] - self.dataMid[v]) / self.dataNorm[v] - (
                xg[:-1] - self.dataMid[v]
            ) / self.dataNorm[v]
            dxdiff = dx - self.deltaX[v] / self.dataNorm[v]
            fTolerance = self.deltaX[v] / (1e4 * self.dataNorm[v])
            # Check that these differences are less than 1/1e6
            if not all(npy.abs(dxdiff) < fTolerance):
                raise ValueError("All grids in axes must be regularly spaced")

            log2size = npy.log2(len(xg) - addOne)
            if log2size != npy.floor(log2size):
                if addOne:
                    extraStr = " + 1"
                else:
                    extraStr = ""

                raise ValueError(
                    "All grids in axes must be powers of 2"
                    + extraStr
                    + ", but got {}".format(len(xg))
                )

        # Calculate the frequency point grids (for 0-centered data)
        self.tgrids = [
            calc_t_from_x((xg - av) / sd)
            for xg, av, sd in zip(self.axes, self.dataMid, self.dataNorm)
        ]
        self.numTPoints = npy.array([len(tg) for tg in self.tgrids])
        self.deltaT = npy.array([tg[2] - tg[1] for tg in self.tgrids])

        self.phiSC = (0.0 + 0.0j) * npy.zeros(self.numTPoints)
        self.ECF = (0.0 + 0.0j) * npy.zeros(self.numTPoints)

        # Initialize the good distribution index
        self.goodDistributionInds = []

        # Set the verbosity flag
        self.be_verbose = be_verbose

        self.convolvedData = None

        # Initialize the marginals
        self.marginalObjects = None

        if data is not None:
            # *************************************************
            #  Calculate the Empirical Characteristic Function
            # *************************************************
            # Note that this routine also standardizes the data on-the-fly
            vprint("Calculating the ECF")

            # Transfrom the data to 0-centered coordinates
            for v in range(self.num_variables):
                data[v, :] = (data[v, :] - self.dataMid[v]) / self.dataNorm[v]

            # Calculate the ECF (see empiricalCharacteristicFunction.py)
            ecfObj = ecf.ECF(
                input_data=data,
                tgrids=self.tgrids,
                use_fft_approximation=self.do_approximate_ecf,
                precision=self.ecf_precision,
                be_verbose=self.be_verbose,
            )

            # Extract the ECF from the ECF object
            self.ECF = ecfObj.ECF

            if self.do_fft:
                # *************************************************
                #  Apply the filter
                # *************************************************
                # Apply the Bernacchia and Pigolotti (2011) filter to the ECF to obtain
                # the fourier representation of the self-consistent density
                vprint("Applying the filter")
                self.applyBernacchiaFilter()

                # *************************************************
                #  Transform to real space
                # *************************************************
                # Transform the optimal distribution to real space
                vprint("Transforming to real space")
                self.__transformphiSC__()

                # Calculate and save the marginal distribution objects
                if self.do_save_marginals:
                    self.marginalObjects = []
                    for i in range(self.num_variables):
                        self.marginalObjects.append(
                            fastKDE(
                                original_data[i, :],
                                axes=[self.originalAxes[i]],
                                positive_shift=self.positive_shift,
                                frac_contiguous_hyper_volumes=self.frac_contiguous_hyper_volumes,
                                log_axes=self.log_axes[i],
                                do_save_marginals=False,
                            )
                        )

        return

    # *****************************************************************************
    # ** fastKDE: ***********************************************
    # ******************* applyBernacchiaFilter() *********************************
    # *****************************************************************************
    # *****************************************************************************
    def applyBernacchiaFilter(self, doFlushArrays=True):
        """Given an ECF, calculate the self-consistent density in fourier-space by
        applying the BP11 filter."""

        # Make an easy-to-read and float version of self.num_data_points
        N = float(self.num_data_points)

        # Calculate the stability threshold for the ECF
        ecfThresh = 4.0 * (N - 1.0) / (N * N)
        self.ecfThreshold = ecfThresh

        # Calculate the squared magnitude of the ECF
        ecfSq = abs(self.ECF) ** 2

        # Find all hypervolumes where ecfSq is greater than the stability threshold
        contiguousInds = flood.flood_fill_search(
            ecfSq, search_threshold=self.ecfThreshold
        )

        if contiguousInds == []:
            raise RuntimeError(
                (
                    "No ECF values found above the ECF threshold.  "
                    + "max(ecfSq) = {}, ecfThresh = {}"
                ).format(npy.amax(ecfSq), ecfThresh)
            )

        # Sort them by distance from the center
        sortedInds = flood.sort_by_distance_from_center(
            contiguousInds, npy.shape(ecfSq)
        )

        # Convert the fraction of hypervolumes to a number if needbe
        numVolumes = len(sortedInds)
        if self.frac_contiguous_hyper_volumes >= 1:
            numVolumesToUse = int(self.frac_contiguous_hyper_volumes)
        else:
            numVolumesToUse = int(self.frac_contiguous_hyper_volumes * numVolumes)
        if numVolumesToUse < 1:
            numVolumesToUse = 1

        # Check that we don't exceed the number of provided volumes
        if numVolumesToUse > numVolumes:
            numVolumesToUse = numVolumes

        # Initialize the filtered value list
        iCalcPhi = self.num_variables * [npy.array([], dtype="int")]

        #  Pull out frac_contiguous_hyper_volumes of contiguous hyper volumes, in
        #  order of distance from the origin
        for i in range(numVolumesToUse):
            for n in range(self.num_variables):
                iCalcPhi[n] = npy.concatenate((iCalcPhi[n], sortedInds[i][n]))

        # Convert iCalcPhi to a list of tuples, such that it is compatible with the
        # output of where()
        if self.num_variables != 1:
            iCalcPhi = [tuple(ii) for ii in iCalcPhi]

        #  convert to a tuple, to avoid a numpy warning
        iCalcPhi = tuple(iCalcPhi)

        # Save the filter
        self.iCalcPhi = iCalcPhi

        # If flagged, clear the phiSC array.  This is needed if the same fastKDE object
        # is reused for multiple data.
        if doFlushArrays:
            self.phiSC[:] = 0.0 + 0.0j

        # Calculate the transform of the self-consistent Kernel (and only calculate it at
        #  points where ecfSq is above ecfThresh)
        kappaSC = (1.0 + 0.0j) * npy.zeros(npy.shape(self.ECF))
        kappaSC[iCalcPhi] = (N / (2 * (N - 1))) * (
            1 + npy.sqrt(1 - ecfThresh / ecfSq[iCalcPhi])
        )

        # Store the fourier kernel if we are going to save the transformed kernel
        if self.do_save_transformed_kernel:
            self.kappaSC = kappaSC

        midPointAccessor = tuple([(tp - 1) // 2 for tp in self.numTPoints])
        # Calculate the transform of the self-consistent density estimate
        self.phiSC[iCalcPhi] = self.ECF[iCalcPhi] * kappaSC[iCalcPhi]

        if self.be_verbose:
            print(
                ("Normalization of kappaSC, ECF, and phiSC: " + "{}, {}, {}").format(
                    kappaSC[midPointAccessor],
                    self.ECF[midPointAccessor],
                    self.phiSC[midPointAccessor],
                )
            )

    # *****************************************************************************
    # ** fastKDE: ***********************************************
    # ******************* findGoodDistributionInds() ******************************
    # *****************************************************************************
    # *****************************************************************************
    def findGoodDistributionInds(self):
        """Find indices of the optimal distribution that are above 0.0"""
        return npy.where(self.pdf >= 0.0)

    # *****************************************************************************
    # ** fastKDE: ***********************************************
    # ******************* findBadDistributionInds() *******************************
    # *****************************************************************************
    # *****************************************************************************
    def findBadDistributionInds(self):
        """Find indices of the optimal distribution that are below 0.0"""
        return npy.where(self.pdf < 0.0)

    # *****************************************************************************
    # ** fastKDE: ***********************************************
    # ******************* __transformphiSC__() ************************************
    # *****************************************************************************
    # *****************************************************************************
    def __transformphiSC__(self):
        """Transform the self-consistent estimate of the distribution from
        frequency space to real space"""

        # Transform the PDF estimate to real space
        pdf = (
            npy.fft.fftshift(npy.real(npy.fft.fftn(npy.fft.ifftshift(self.phiSC))))
            * npy.prod(self.deltaT)
            * (1.0 / (2 * npy.pi)) ** self.num_variables
        )

        # Unnormalize it
        pdf /= npy.prod(self.dataNorm)

        # transpose the self-consistent density estimate
        self.pdf = pdf.transpose()

        # Shift the PDF such that the negative areas can be set to 0, while the
        # positive area is still normalized to 1
        if self.positive_shift:
            if len(npy.where(self.pdf < 0)[0]) != 0:
                # Define a function f(delta), such that f(delta) is how far off
                # self.pdf-delta is from being normalized; hence, we want to find
                # the zero of this function
                def normFunc(delta):
                    """Calculate how far off from normal is the shifted PDF"""
                    ipos = npy.where((self.pdf - delta) >= 0.0)
                    return 1 - sum((self.pdf[ipos] - delta) * npy.prod(self.deltaX))

                # Set the initial guess for the newton-raphson search
                # a = -normFunc(0)
                a = 0.0

                # Find the zero of the above function; i.e., find delta, such that
                # the shifted PDF is normalized
                try:
                    delta = scipy.optimize.newton(normFunc, a, maxiter=10000)
                except RuntimeError:
                    delta = 0.0

                # Check if the positive shift method failed
                if not npy.isfinite(delta) or delta < 0 or delta >= npy.amax(self.pdf):
                    if self.be_verbose:
                        print(
                            "positive_shift algorithm failure: defaulting to no shift"
                        )
                    delta = 0.0

                # If a shift is provided, do the shift
                if delta != 0.0:
                    # Shift the PDF
                    self.pdf -= delta
                    # And set the negative values to 0
                    self.pdf[npy.where(self.pdf < 0)] = 0.0

        if self.be_verbose:
            normConst = sum(pdf * npy.prod(self.deltaX))
            midPointAccessor = tuple([(tp - 1) // 2 for tp in self.numTPoints])
            print(
                ("Normalization of pdf = {}. " + "phiSC[0] = {}").format(
                    normConst, self.phiSC[midPointAccessor]
                )
            )

        # Save the original PDF and axes
        self.originalPDF = npy.array(self.pdf)
        self.originalAxes = list(self.axes)

        # Check if any variables need to be transformed due to use
        # of logarithmic axes
        for v in range(self.num_variables):
            if self.log_axes[v]:
                # Transform the axis back to data space
                self.axes[v] = npy.exp(self.axes[v])
                # Generate a slice to help the axis conform in shape to the PDF
                conformSlice = self.num_variables * [npy.newaxis]
                conformSlice[v] = slice(None, None, None)
                conformSlice = tuple(conformSlice)
                # Transform the PDF
                self.pdf /= self.axes[v][conformSlice[::-1]]

        # Set self.fSC for backward compatibility
        self.fSC = self.pdf

        # Take the transform of the self-consistent kernel if flagged
        if self.do_save_transformed_kernel:
            kSC = (
                npy.fft.fftshift(
                    npy.real(npy.fft.fftn(npy.fft.ifftshift(self.kappaSC)))
                )
                * npy.prod(self.deltaT)
                * (1.0 / (2 * npy.pi)) ** self.num_variables
            )
            kSC /= npy.prod(self.dataNorm)
            self.kSC = kSC.transpose()

    # *****************************************************************************
    # ** fastKDE: ***********************************************
    # ******************* __transformphiSC_points__() ************************************
    # *****************************************************************************
    # *****************************************************************************
    def __transformphiSC_points__(self, list_of_points):
        """Transform the self-consistent estimate of the distribution from
        frequency space to a set of points in real space"""

        # Transfrom the point to 0-centered coordinates
        list_of_points = npy.array(list_of_points, copy=True)
        for v in range(self.num_variables):
            list_of_points[v, :] = (
                list_of_points[v, :] - self.dataMid[v]
            ) / self.dataNorm[v]

        # Set the fill value for the frequency grids
        fillValue = -1e20

        # Get the maximum frequency grid length
        tgrids = self.tgrids
        ntmax = npy.amax([len(tgrid) for tgrid in tgrids])

        # Create the frequency grids array
        frequencyGrids = fillValue * npy.ones([self.num_variables, ntmax])
        # Fill the frequency grids array
        for v in range(self.num_variables):
            frequencyGrids[v, : len(tgrids[v])] = tgrids[v]

        #  do the inverse direct Fourier transform
        pdf = (
            dft_points(
                frequencyGrids,
                self.phiSC.ravel(),
                list_of_points,
                missingFreqVal=fillValue,
            )
            * npy.prod(self.deltaT)
            * (1.0 / (2 * npy.pi)) ** self.num_variables
        )

        #  unstandarize the PDF
        pdf /= npy.prod(self.dataNorm)

        #  TODO: implement positive_shift for point-based PDF estimates
        #  TODO: implement log_axes for point-based PDF estimates

        return pdf

    def getTransformedPDF(self):
        """Returns a copy of the PDF.  This function exists for backward compatibility"""
        return npy.array(self.pdf)

    def getTransformedAxes(self):
        """Returns a copy of the axes.  This function exists for backward compatibility"""
        return tuple([npy.array(xg) for xg in self.axes])

    def getTransformedCopula(self, data=None):
        """A wrapper for getCopula; this function is deprecated."""
        return self.getCopula(data)

    def estimateConditionals(self, variables, data, peakFrac=0.0, reApplyFilter=False):
        """For a multidimensional PDF, estimates the conditional P(x_i | x_j).

        input:
        ------

            variables   : A integer or tuple of array indicies indicating the
                          variables on which to condition e.g., For a 2D PDF,

                          obj.estimateConditionals(1) estimates P(x_0 | x_1)
                          from the joint PDF P(x_0,x_1) that is the result of
                          the self-consistent density estimate.

                          For a 3D PDF:

                          obj.estimateConditionals( (0,2) ) estimates P( x_0,
                          x_2 | x_1) from P(x_0,x_1,x_2)

                          If all possible variables are listed, the copula is
                          returned instead.

                          If negative values are provided, variables are wrapped
                          (i.e., index -1 indicates the last variable)

            data        : The data original used to create the fastKDE object.
                          This is needed to calculated the various marginals
                          required in the conditional computation.

            peakFrac    : The fractional threshold below which to truncate the
                          marginal PDF (to avoid divding by small numbers); this
                          is the fraction of the height of the mode.

            reapplyFilter : Flags whether to reapply the ECF filter to the conditional

        output:
        -------


            Returns P( x_i | x_j )

        """

        data = npy.array(data)

        # If the data are univariate, simply return the PDF itself
        if self.num_variables == 1:
            return self.pdf

        # Check that we can interpret the variables tuple
        try:
            len(variables)
        except TypeError:
            try:
                range(variables)
                variables = (variables,)
            except TypeError:
                raise ValueError(
                    "variables appears to be neither a tuple or an integer"
                )

        # Check that the variable indices are sane
        rightSideVariableIndices = []
        for ind in tuple(variables):
            if ind > self.num_variables - 1:
                raise ValueError("out-of-bounds positive index found in 'variables'")
            if ind < 0:
                dum = self.num_variables + ind
                if dum < 0:
                    raise ValueError(
                        "out-of-bounds negative index found in 'variables'"
                    )
            else:
                dum = ind
            rightSideVariableIndices.append(dum)

        # Pull the unique indices and make sure they are sorted
        rightSideVariableIndices = tuple(sorted(list(set(rightSideVariableIndices))))
        if len(rightSideVariableIndices) > self.num_variables:
            raise ValueError(
                "More indices were provided in 'variables' than there are " "variables."
            )

        # Check if all variables were provided
        if len(rightSideVariableIndices) == self.num_variables:
            return self.getCopula(data)

        # If there are no right side variables, return the PDF
        if len(rightSideVariableIndices) == 0:
            return self.pdf

        # Create the list of left-side variable indices
        leftSideVariableIndices = list(range(self.num_variables))
        for ind in sorted(rightSideVariableIndices)[::-1]:
            leftSideVariableIndices.pop(ind)

        # Calculate the marginal PDF
        marginalObject = fastKDE(
            data[rightSideVariableIndices, :],
            axes=[self.originalAxes[i] for i in rightSideVariableIndices],
            positive_shift=self.positive_shift,
            frac_contiguous_hyper_volumes=self.frac_contiguous_hyper_volumes,
            log_axes=[self.log_axes[i] for i in rightSideVariableIndices],
            do_save_marginals=False,
        )

        # Make the shape of the new marginal object match that of the original PDF
        # (using the magic of the numpy newaxis)
        conformantSlice = list(self.num_variables * (slice(None, None, None),))
        # Insert a newaxis for each of the left-side indices
        sumAxes = []
        for ind in leftSideVariableIndices:
            # The PDF object has var0 in its rightmost axis, so transform ind
            # accordingly (it references as though var0 is the leftmost axis)
            ip = self.num_variables - ind - 1
            conformantSlice[ip] = npy.newaxis
            # Add this index to the list of axes over which to sum for normalization
            sumAxes.append(ip)
        conformantSlice = tuple(conformantSlice)

        marginalThreshold = peakFrac * npy.amax(marginalObject.pdf)
        # Create and mask the marginal PDF
        marginalPDF = npy.ma.masked_less_equal(
            marginalObject.pdf[conformantSlice], marginalThreshold
        )

        # Calculate the conditional PDF
        conditionalPDF = npy.ma.array(self.pdf) / marginalPDF

        # Refilter the conditional
        if reApplyFilter:
            conditionalPDF = npy.ma.masked_less_equal(
                self.reApplyFilter(conditionalPDF), 0.0
            )

        # Calculate the normalization matrix
        dxs = [npy.diff(self.axes[i]) for i in leftSideVariableIndices]
        dxs = [npy.concatenate((dx, [dx[-1]])) for dx in dxs]
        if len(dxs) == 1:
            dxProd = npy.array(dxs[0])
        else:
            dxProd = npy.prod(npy.meshgrid(*dxs), axis=0)
        cslice = self.num_variables * [npy.newaxis]
        for i in leftSideVariableIndices:
            cslice[i] = slice(None, None, None)
        cslice = tuple(cslice)
        dxProd = dxProd[cslice[::-1]]

        normFactor = npy.ma.masked_equal(
            sum(conditionalPDF * dxProd, axis=tuple(sumAxes)), 0.0
        )

        # Normalize the conditional PDF for the leftside variables
        conditionalPDF /= normFactor[conformantSlice]

        return conditionalPDF

    # *****************************************************************************
    # ** fastKDE: *******************************************
    # ******************* getCopula      ******************************************
    # *****************************************************************************
    # *****************************************************************************
    def getCopula(self, data=None, peakFrac=0.0):
        """Estimates the copula of the underlying PDF"""

        # If the data are univariate, simply return the PDF itself
        if self.num_variables == 1:
            return self.pdf

        # Check if we need to calculate the marginal distributions
        if not self.do_save_marginals:
            if data is None:
                raise ValueError(
                    "the data must be provided as argument 'data', if "
                    "do_save_marginals=False when the original PDF was calculated"
                )
            else:
                # Estimate the marginal distributions
                marginalObjects = []
                for i in range(self.num_variables):
                    marginalObjects.append(
                        fastKDE(
                            data[i, :],
                            axes=[self.originalAxes[i]],
                            positive_shift=self.positive_shift,
                            frac_contiguous_hyper_volumes=self.frac_contiguous_hyper_volumes,
                            log_axes=self.log_axes[i],
                            do_save_marginals=False,
                        )
                    )
        else:
            # If not, just use the saved marginals
            marginalObjects = self.marginalObjects

        # Calculate the marginal distributions and mask bad (or zero) values
        marginals = []
        for obj in marginalObjects:
            # Add the marginal to the list while masking <0 values
            marginalThreshold = peakFrac * npy.amax(obj.pdf)
            # Create and mask the marginal PDF
            marginals.append(npy.ma.masked_less_equal(obj.pdf, marginalThreshold))

        # Calculate the PDF assuming independent marginals
        independencePDF = npy.ma.prod(npy.meshgrid(*tuple(marginals)), axis=0)
        # Divide off the indepdencnce PDF to calculate the copula
        # actualPDF = ma.array(self.pdf)
        # actualPDF[self.findBadDistributionInds()] = ma.masked
        actualPDF = npy.ma.array(self.pdf)
        copulaPDF = actualPDF / independencePDF

        return copulaPDF

    def reApplyFilter(self, pdf):
        """Reapplies the filter to a PDF estimate.

        This is used, e.g., to remove high-frequency noise that results from
        calculting the conditionals.
        """

        # Transform the PDF to fourier space
        phiTilde_tmp = npy.fft.fftshift(
            npy.fft.ifftn(npy.fft.ifftshift(npy.ma.filled(pdf, 0.0)))
        )
        # Normalize the transform
        midPointAccessor = tuple([(tp - 1) // 2 for tp in self.numTPoints])
        phiTilde_tmp /= phiTilde_tmp[midPointAccessor]

        # Reapply the filter
        phiTilde = (0.0 + 0.0j) * npy.zeros(npy.shape(phiTilde_tmp))
        phiTilde[self.iCalcPhi] = phiTilde_tmp[self.iCalcPhi]

        # Transform back to real space
        # Transform the PDF estimate to real space
        pdf = (
            npy.fft.fftshift(npy.real(npy.fft.fftn(npy.fft.ifftshift(phiTilde))))
            * npy.prod(self.deltaT)
            * (1.0 / (2 * npy.pi)) ** self.num_variables
        )

        # Return the transpose of the PDF
        return pdf.transpose()

    # *****************************************************************************
    # ** fastKDE: ***********************************************
    # ******************* Addition operator __add__ *******************************
    # *****************************************************************************
    # *****************************************************************************
    def __add__(self, rhs):
        """Addition operator for the fastKDE object.  Adds the
        empirical characteristic functions of the two estimates, reapplies
        the BP11 filter, and transforms back to real space.  This is useful
        for parallelized calculation of densities.  Note that this only works
        if the axes are the same for both operands."""
        # Check for proper typing
        if not isinstance(rhs, fastKDE):
            raise TypeError(
                "unsupported operand type(s) for +: "
                "{} and {}".format(type(self), type(rhs))
            )

        # Check that the axes are the same for both objects
        for sxg, rxg in zip(self.axes, rhs.axes):
            if not all(npy.isclose(sxg, rxg)):
                raise NotImplementedError(
                    "addition for operands with different axes is not yet "
                    "implemented."
                )

        retObj = copy.deepcopy(self)
        retObj.phiSC = (0.0 + 0.0j) * npy.zeros(self.numTPoints)

        retObj.num_data_points += rhs.num_data_points

        # Convert the returned variance back into standard deviation
        retObj.dataStandardDeviation = npy.sqrt(retObj.dataStandardDeviation)

        # Average the Empirical Characteristic Function of the two objects
        retObj.ECF = (
            self.num_data_points * self.ECF + rhs.num_data_points * rhs.ECF
        ) / retObj.num_data_points

        if retObj.do_fft:
            retObj.applyBernacchiaFilter()
            retObj.__transformphiSC__()

        # Return the new object
        return retObj


def pdf(*args, **kwargs):
    """Estimate the self-consistent kernel density estimate of the input data

    input:
    ------

        var1            : An input variable.

        var2, var3...   : Additional input varibles whose length corresponds
                          to the length of var1.  As input variables are
                          added, the dimensionality of the resulting PDF
                          increases (e.g., supplying var1 and var2 results
                          in a 2D PDF).

        **kwargs        : Any additional keyword arguments get passed
                          directly to fastKDE.fastKDE();  see the docstring
                          of fastKDE.fastKDE() for details of kwargs.

    returns:
    --------

        pdf,axes    :       The pdf and the axes of the PDF (i.e., this is
                            analogous to hist,bins for a histogram).

                            If there are multiple input variables, the axes
                            variable is a list of the axes, with each axis
                            corresponding to an input variable.


    NOTE: The computational expense and the memory requirement of this
    method grows exponentially with the number of input variables.
    """

    # Try to get var1 from the args or kwargs
    try:
        var1 = args[0]
    except IndexError:
        try:
            var1 = kwargs["var1"]
        except KeyError:
            raise ValueError("No input data were provided.")

    # Check that var1 is arraylike
    try:
        var1Shape = npy.shape(var1)
        assert len(var1Shape) != 0, "var1 is not an array"
    except TypeError or AssertionError:
        raise ValueError(
            "Could not get shape of var1; it does not appear to be array-like."
        )

    # Check that var1 is a vector
    if len(var1Shape) != 1:
        raise ValueError(
            "var1 should be a vector.  If multiple variables are combined in a "
            "single array, please use the fastKDE class interface instead."
        )

    # Get the length of var1
    N = var1Shape[0]

    # Check for input varibles provided as key word arguments
    varArgs = []
    varKeys = sorted([v for v in kwargs if "var" in v])
    for key in varKeys:
        # Ignore var1 since this was either provided as an argument
        # or was read as a keyword argument above
        if key != "var1":
            try:
                int(key[3:])
            except ValueError:
                raise ValueError(
                    "Incomprehensible variable-like keyword provided: " "{}".format(key)
                )

            # Append this variable
            varArgs.append(kwargs[key])

    # Check if a mixture of keyword and arguments were provided for additional variables
    if len(varArgs) != 0 and len(args) > 1:
        raise ValueError(
            "additional variables were provided as a mixture of arguments and "
            "keyword arguments. They all must be one or the other."
        )

    # Set the additional variables to be the rest of the input arguments
    # if none were provided as key word arguments
    if len(args) > 1:
        varArgs = args[1:]

    # Remove the variables from kwargs
    for key in list(varKeys):
        del kwargs[key]

    # Start preparing the input data for
    # concatenation
    inputVariables = npy.array(var1[npy.newaxis, :])

    # Attempt to read additional variables
    # and concatenate them to the input variable
    for i in range(len(varArgs)):
        try:
            varn = npy.array(varArgs[i][npy.newaxis, :])
        except BaseException as e:
            print(e)
            raise ValueError(
                "Could not convert var{} into a numpy arrray".format(i + 1)
            )

        lenN = npy.shape(varn)[1]
        if lenN != N:
            raise ValueError(
                (
                    "len(var{}) is {}, but it should be the same of len(var1) " + "= {}"
                ).format(i + 1, lenN, N)
            )

        inputVariables = npy.concatenate((inputVariables, varn))

    # Remove the do_save_marginals keyword argument
    try:
        kwargs["do_save_marginals"]
        del kwargs["do_save_marginals"]
    except KeyError:
        pass

    # Calculate the PDF
    _pdfobj = fastKDE(inputVariables, do_save_marginals=False, **kwargs)
    if len(_pdfobj.axes) == 1:
        return _pdfobj.pdf, _pdfobj.axes[0]
    else:
        return _pdfobj.pdf, _pdfobj.axes


def conditional(inputVars, conditioningVars, **kwargs):
    """Estimates the conditional PDF of `inputVars` given `conditioningVars`

        inputVars           : A vector of input values, or a list of such vectors

        conditioningVars    : A vector of conditioning values, or a list of such vectors

        **kwargs            : Any additional keyword arguments get passed
                              directly to fastKDE.fastKDE() or
                              fastKDE.estimateConditionals();  see the
                              docstrings of fastKDE.fastKDE() and
                              fastKDE.estimateConditionals() for details of
                              kwargs.
                              
                              Note the following two arguments have different
                              default values here:
                                  positive_shift=True by default, and
                                  peakFrac = 0.01 by default.

        :returns: (cPDF, axes) \
                where cPDF is the PDF(inputVars | conditioningVars), and axes is a list
                of axis vectors giving the points at which cPDF is defined.

                If N conditioningVars were provided, then axes[0:N-1]
                corresponds to the variables provided in conditioningVars, in
                the order they were provided; axes[N:M-1] corresponds to the M
                inputVars provided, in the order provided.

                Ex:

        ```python

        import pylab as PP
        from numpy import *

        # ***************************
        #  Generate random samples
        # ***************************
        #  Stochastically sample from the function underlyingFunction() (a sigmoid):
        #  sample the absicissa values from a gamma distribution
        #  relate the ordinate values to the sample absicissa values and add
        #  noise from a normal distribution

        # Set the number of samples
        numSamples = int(1e6)

        # Define a sigmoid function
        def underlyingFunction(x,x0=305,y0=200,yrange=4):
            return (yrange/2)*tanh(x-x0) + y0

        xp1,xp2,xmid = 5,2,305  # Set gamma distribution parameters
        yp1,yp2 = 0,12          # Set normal distribution parameters (mean and std)

        # Generate random samples of X from the gamma distribution
        x = -(random.gamma(xp1,xp2,int(numSamples))-xp1*xp2) + xmid
        # Generate random samples of y from x and add normally distributed noise
        y = underlyingFunction(x) + random.normal(loc=yp1,scale=yp2,size=numSamples)

        # ***************************
        #  Calculate the conditional
        # ***************************
        pOfYGivenX,axes = fastKDE.conditional(y,x)

        ```

    """

    # Check whether inputVars is an iterable of vectors or a single vector;
    # ensure it is an iterable
    try:
        ivarLengths = [len(v) for v in inputVars]
    except TypeError:
        inputVars = [inputVars]
        ivarLengths = [len(v) for v in inputVars]

    # Check whether conditioningVars is an iterable of vectors or a single vector;
    # ensure it is an iterable
    try:
        cvarLengths = [len(v) for v in conditioningVars]
    except TypeError:
        conditioningVars = [conditioningVars]
        cvarLengths = [len(v) for v in conditioningVars]

    # Create a list of all variables
    fullVarList = conditioningVars + inputVars

    # Check that all input variable lengths are the same
    if not all(npy.array([len(v) for v in fullVarList]) == ivarLengths[0]):
        raise ValueError(
            (
                "inputVars and conditioningVars all must be the same length. "
                + "Got {} for inputVars and {} for conditioningVars"
            ).format(ivarLengths, cvarLengths)
        )

    # Extract the peakFrac argument
    if "peakFrac" in kwargs:
        peakFrac = kwargs["peakFrac"]
        del kwargs["peakFrac"]
    else:
        peakFrac = 0.01

    # Default to positive_shift=True
    if "positive_shift" in kwargs:
        positive_shift = kwargs["positive_shift"]
        del kwargs["positive_shift"]
    else:
        positive_shift = True

    # Estimate the full joint PDF
    _pdf = fastKDE(npy.array(fullVarList), positive_shift=positive_shift, **kwargs)

    # Set the indices of the conditional variables
    cvarInds = list(range(len(conditioningVars)))

    # Estimate the conditional
    cpdf = _pdf.estimateConditionals(
        cvarInds, npy.array(fullVarList), peakFrac=peakFrac
    )

    # Return the conditional and the axes
    return cpdf, _pdf.axes


def pdf_at_points(*args, **kwargs):
    """Estimate the self-consistent kernel density estimate of the input data
    at a fixed set of points.

    input:
    ------

        var1            : An input variable.

        var2, var3...   : Additional input varibles whose length
                          corresponds to the length of var1.  As input
                          variables are added, the dimensionality of the
                          resulting PDF increases (e.g., supplying var1
                          and var2 results in a 2D PDF).

        list_of_points  : Points at which the PDF should be estimated.
                          Points should be provided as a list of tuples,
                          where each tuple contains a point at which the
                          PDF should be estimated.  If not provided, the
                          input data points will be used by default.

        **kwargs        : Any additional keyword arguments get passed
                          directly to fastKDE.fastKDE();  see the docstring
                          of fastKDE.fastKDE() for details of kwargs.

    returns:
    --------

        pdf             :   The pdf evaluated at list_of_points (or at the
                            input data points if list_of_points was not
                            provided)



    NOTE: The computational expense and the memory requirement of this
    method grows exponentially with the number of input variables.

    NOTE: `pdf_at_points()` is potentially slow relative to `pdf()` becuase
    it does not take advantage of the inverse FFT for transforming from
    Fourier space to data space.  However, if few input points are
    requested, it may actually be faster.

    """

    # Try to get var1 from the args or kwargs
    try:
        var1 = args[0]
    except IndexError:
        try:
            var1 = kwargs["var1"]
        except KeyError:
            raise ValueError("No input data were provided.")

    # Check that var1 is arraylike
    try:
        var1Shape = npy.shape(var1)
        assert len(var1Shape) != 0, "var1 is not an array"
    except ValueError or AssertionError:
        raise ValueError(
            "Could not get shape of var1; it does not appear to be array-like."
        )

    # Check that var1 is a vector
    if len(var1Shape) != 1:
        raise ValueError(
            "var1 should be a vector.  If multiple variables are combined in a "
            "single array, please use the fastKDE class interface instead."
        )

    # Get the length of var1
    N = var1Shape[0]

    # Check for input varibles provided as key word arguments
    varArgs = []
    varKeys = sorted([v for v in kwargs if "var" in v])
    for key in varKeys:
        # Ignore var1 since this was either provided as an argument
        # or was read as a keyword argument above
        if key != "var1":
            try:
                int(key[3:])
            except ValueError:
                raise ValueError(
                    "Incomprehensible variable-like keyword provided: " "{}".format(key)
                )

            # Append this variable
            varArgs.append(kwargs[key])

    # Check if a mixture of keyword and arguments were provided for additional variables
    if len(varArgs) != 0 and len(args) > 1:
        raise ValueError(
            "additional variables were provided as a mixture of arguments and "
            "keyword arguments.  They all must be one or the other."
        )

    # Set the additional variables to be the rest of the input arguments
    # if none were provided as key word arguments
    if len(args) > 1:
        varArgs = args[1:]

    # Remove the variables from kwargs
    for key in list(varKeys):
        del kwargs[key]

    # Start preparing the input data for
    # concatenation
    inputVariables = npy.array(var1[npy.newaxis, :])

    # Attempt to read additional variables
    # and concatenate them to the input variable
    for i in range(len(varArgs)):
        try:
            varn = npy.array(varArgs[i][npy.newaxis, :])
        except BaseException as e:
            print(e)
            raise ValueError(
                "Could not convert var{} into a numpy arrray".format(i + 1)
            )

        lenN = npy.shape(varn)[1]
        if lenN != N:
            raise ValueError(
                (
                    "len(var{}) is {}, but it should be the same of len(var1) " + "= {}"
                ).format(i + 1, lenN, N)
            )

        inputVariables = npy.concatenate((inputVariables, varn))

    #  check if list_of_points was provided
    list_of_points_provided_in_kwargs = False
    try:
        #  extract the list of points
        list_of_points = kwargs["list_of_points"]
        #  delete it from the keyword argument dictionary
        del kwargs["list_of_points"]
        #  flag that this argument was provided
        list_of_points_provided_in_kwargs = True
    except KeyError:
        #  default to using the input points as the list of points
        list_of_points = inputVariables

    #  make sure list_of_points is in the expected format
    if list_of_points_provided_in_kwargs:
        try:
            list_of_points = npy.array(list_of_points, copy=True, dtype=npy.float_).T
        except ValueError:
            raise RuntimeError("Could not convert list_of_points to a numpy array.")

        #  check the rank of the input data points
        data_rank = len(npy.shape(list_of_points))

        # If the data are a vector, promote the data to a rank-1 array with only
        # 1 column
        if data_rank == 1:
            list_of_points = npy.array(list_of_points[npy.newaxis, :], dtype=npy.float_)

        if data_rank > 2:
            #  raise an error indicating the proper shape for list_of_points
            raise ValueError(
                "list_of_points must be able to be broadcast to a rank-2 array "
                "of shape [num_data_points,num_variables]"
            )

    # Remove the do_save_marginals keyword argument
    try:
        _ = kwargs["do_save_marginals"]
        del kwargs["do_save_marginals"]
    except KeyError:
        pass

    # Remove the log_axes argument
    try:
        _ = kwargs["log_axes"]
        del kwargs["log_axes"]
        warnings.warn(
            "fastKDE.pdf_at_points() does not currently support the log_axes "
            "option; it will be ignored."
        )
    except KeyError:
        pass

    # Remove the positive_shift argument
    try:
        _ = kwargs["positive_shift"]
        del kwargs["positive_shift"]
        warnings.warn(
            "fastKDE.pdf_at_points() does not currently support the "
            "positive_shift option; it will be ignored."
        )
    except KeyError:
        pass

    # Calculate the PDF in Fourier space
    _pdfobj = fastKDE(
        inputVariables,
        do_save_marginals=False,
        do_fft=False,
        positive_shift=False,
        log_axes=False,
        **kwargs,
    )

    #  complete the Fourier-space calculation of the PDF
    _pdfobj.applyBernacchiaFilter()

    #  calculate the PDF at the requested points
    pdf = _pdfobj.__transformphiSC_points__(list_of_points)

    #  return the pdf
    return pdf


# *******************************************************************************
# *******************************************************************************
# ***************************** Unit testing code *******************************
# *******************************************************************************
# *******************************************************************************
#  Test this implementation of the BP11 density estimate against a normal
#  distribution.  Calculate the estimate for a variety of sample sizes and show
#  how the distribution error decreases as sample size increases.  As of revision
#  9 of the code, this unit testing shows that this implementation of the BP11
#  estimate converges on the true normal distribution like N**-1, which agrees
#  the theoretical and empirical convergence rate given in BP11.
if __name__ == "__main__":
    # set a seed so that results are repeatable
    npy.random.seed(0)

    doOneDimensionalTests = True
    if doOneDimensionalTests:
        import matplotlib.pyplot as plt
        import scipy.stats as stats

        mu = -1e3
        sig = 1e3

        # Define a gaussian function for evaluation purposes
        def mygaus(x):
            return (1.0 / (sig * npy.sqrt(2 * npy.pi))) * npy.exp(
                -((x - mu) ** 2) / (2.0 * sig**2)
            )

        # Set the size of the sample to calculate
        powmax = 19
        npow = npy.asarray(range(powmax)) + 1.0

        # Set the maximum sample size
        nmax = 2**powmax
        # Create a random normal sample of this size
        randsample = sig * npy.random.normal(size=nmax) + mu

        # Pre-define sample size and error-squared arrays
        nsample = npy.zeros([len(npow)])
        esq = npy.zeros([len(npow)])
        epct = npy.zeros([len(npow)])

        evaluateError = True
        if evaluateError:
            # Do the optimal calculation on a number of different random draws
            for i, n in zip(range(len(npow)), npow):
                # Extract a sample of length 2**n + 1 from the previously-created
                # random sample
                randgauss = randsample[: (2**n + 1)]
                # Set the sample size
                nsample[i] = len(randgauss)

                with Timer(nsample[i]):
                    # Do the BP11 density estimate
                    bkernel = fastKDE(
                        randgauss, do_approximate_ecf=True, num_points=513
                    )

                # Calculate the mean squared error between the estimated density and
                # the gaussian esq[i] = average(abs(mygaus(bkernel.x)-bkernel.pdf)**2
                # *bkernel.deltaX)
                esq[i] = npy.average(
                    abs(mygaus(bkernel.axes[0]) - bkernel.pdf[:]) ** 2
                    * bkernel.deltaX[0]
                )
                epct[i] = 100 * sum(
                    abs(mygaus(bkernel.axes[0]) - bkernel.pdf[:]) * bkernel.deltaX[0]
                )

                # Plot the optimal distribution
                plt.subplot(2, 2, 1)  # ,yscale="log")
                # pdfmask = ma.masked_less(bkernel.pdf,bkernel.distributionThreshold)
                pdfmask = bkernel.pdf
                plt.plot(bkernel.axes[0], pdfmask, "b-")

                # Plot the empirical characteristic function
                plt.subplot(2, 2, 2, xscale="log", yscale="log")
                plt.plot(bkernel.tgrids[0][1:], abs(bkernel.ECF[1:]) ** 2, "b-")

            # Plot the sample gaussian
            plt.subplot(2, 2, 1)  # ,yscale="log")
            plt.plot(bkernel.axes[0], mygaus(bkernel.axes[0]), "r-")

            # Do a simple power law fit to the scaling
            [m, b, _, _, _] = stats.linregress(npy.log(nsample), npy.log(esq))
            # Print the error scaling (following BP11, this is expected to be m ~ -1)
            print("Error scales ~ N**{}".format(m))

            # Plot the error vs sample size on a log-log curve
            plt.subplot(2, 2, 3)
            plt.loglog(nsample, esq)
            plt.plot(nsample, npy.exp(b) * nsample**m, "r-")

            print("")

            bDemoSum = False
            if not bDemoSum:
                plt.show()
            else:
                # *********************************************************************
                #  Demonstrate the capability to sum fastKDE objects
                # *********************************************************************

                nsamp = 512
                nloop = nmax / nsamp

                # Pre-define sample size and error-squared arrays
                nsample2 = npy.zeros([nloop])
                esq2 = npy.zeros([nloop])

                for i in range(nloop):
                    randgauss = randsample[i * nsamp : (i + 1) * nsamp]
                    if i == 0:
                        bkernel2 = fastKDE(randgauss)
                        nsample2[i] = len(randgauss)
                    else:
                        bkernel2 += fastKDE(randgauss)
                        nsample2[i] = nsample2[i - 1] + len(randgauss)

                    # Calculate the mean squared error between the estimated density
                    # And the gaussian
                    esq2[i] = npy.average(
                        npy.abs(mygaus(bkernel2.axes[0]) - bkernel2.pdf) ** 2
                        * bkernel2.deltaX[0]
                    )
                    # Print the sample size and the error to show that the code is proceeeding
                    print("{}, {}".format(nsample2[i], esq2[i]))

                # Plot the distribution
                plt.subplot(2, 2, 1)
                plt.plot(bkernel2.axes[0], bkernel2.pdf, "g-")

                # Plot the ECF
                plt.subplot(2, 2, 2, xscale="log", yscale="log")
                plt.plot(bkernel2.tgrids[0][1:], abs(bkernel2.ECF[0, 1:]) ** 2, "b-")

                # Plot the error-rate change
                plt.subplot(2, 2, 3)
                plt.loglog(nsample2, esq2, "g-")

                # Plot the difference between the two distributions
                plt.subplot(2, 2, 4)
                plt.plot(
                    bkernel2.axes[0],
                    abs(bkernel.pdf - bkernel2.pdf) * bkernel.deltaX[0],
                )

                # Show the plots
                plt.show()
        else:
            print(randsample)
            # Simply do the BP11 density estimate and plot it
            bkernel = fastKDE(
                randsample, do_approximate_ecf=True, be_verbose=True, num_points=513
            )
            # Plot the optimal distribution
            plt.subplot(2, 1, 1)
            # pdfmask = ma.masked_less(bkernel.pdf,bkernel.distributionThreshold)
            pdfmask = bkernel.pdf
            plt.plot(bkernel.axes[0], pdfmask, "b-")
            # Plot the sample gaussian
            plt.plot(bkernel.axes[0], mygaus(bkernel.axes[0]), "r-")

            # for d in randsample:
            #     plt.plot([d,d],[0,1./len(randsample)],'k-',alpha=0.5)

            # Plot the transforms
            plt.subplot(2, 1, 2)
            plt.plot(bkernel.tgrids[0], abs(bkernel.phiSC), "b-")
            ecfStandard = npy.fft.ifft(mygaus(bkernel.axes[0]))
            ecfStandard /= ecfStandard[0]
            ecfStandard = npy.fft.fftshift(ecfStandard)
            plt.plot(bkernel.tgrids[0], abs(ecfStandard), "r-")

            mean = sum(bkernel.axes[0] * bkernel.pdf * bkernel.deltaX[0])

            plt.show()

    doTwoDimensionalTests = True
    if doTwoDimensionalTests:
        import matplotlib.pyplot as plt
        import scipy.stats as stats

        nvariables = 2
        # Seed with 0 so results are reproducable
        npy.random.seed(0)

        # Define a bivariate normal function
        def norm2d(x, y, mux=0, muy=0, sx=1, sy=1, r=0):
            coef = 1.0 / (2 * npy.pi * sx * sy * npy.sqrt(1.0 - r**2))
            expArg = -(1.0 / (2 * (1 - r**2))) * (
                (x - mux) ** 2 / sx**2
                + (y - muy) ** 2 / sy**2
                - 2 * r * (x - mux) * (y - muy) / (sx * sy)
            )
            return coef * npy.exp(expArg)

        # Set the size of the sample to calculate
        powmax = 16
        npow = npy.asarray(range(1, powmax)) + 1.0

        # Set the maximum sample size
        nmax = 2**powmax

        def covMat(sx, sy, r):
            return [[sx**2, r * sx * sy], [r * sx * sy, sy**2]]

        gausParams = []
        gausParams.append([0.0, 0.0, 1.0, 1.0, 0.0])  # Standard, uncorrelated bivariate
        gausParams.append([2.0, 0.0, 1.0, 1.0, 0.7])  # correlation 0.7, mean x+2
        gausParams.append([0.0, 2.0, 1.0, 0.5, 0.0])  # Flat in y-direction, mean y+2
        gausParams.append([2.0, 2.0, 0.5, 1.0, 0.0])  # Flat in x-direction, mean xy+2

        # Define the corresponding standard function
        def pdfStandard(x, y):
            pdfStandard = npy.zeros(npy.shape(x))
            for gg in gausParams:
                pdfStandard += norm2d(x2d, y2d, *tuple(gg)) * (1.0 / ngg)

            return pdfStandard

        # Generate samples from this distribution
        randsamples = []
        ngg = len(gausParams)
        for gg in gausParams:
            mu = gg[:2]
            gCovMat = covMat(*tuple(gg[2:]))
            size = tuple([2, nmax / ngg])
            # Append a 2D gaussian to the list
            randsamples.append(
                npy.random.multivariate_normal(mu, gCovMat, (nmax / ngg,)).transpose()
            )

        # Concatenate the gaussian samples
        randsample = npy.concatenate(tuple(randsamples), axis=1)

        # Shuffle the samples along the long axis so that we
        # can draw successively larger samples
        ishuffle = npy.asarray(range(nmax))
        npy.random.shuffle(ishuffle)
        randsample = randsample[:, ishuffle]

        doSaveCSV = False
        if doSaveCSV:
            npy.savetxt("bp11_2d_samples.csv", randsample.transpose(), delimiter=",")

        # Pre-define sample size and error-squared arrays
        nsample = npy.zeros([len(npow)])
        esq = npy.zeros([len(npow)])
        epct = npy.zeros([len(npow)])

        evaluateError = True
        if evaluateError:
            # Do the optimal calculation on a number of different random draws
            for z, n in zip(range(len(npow)), npow):
                # Extract a sample of length 2**n + 1 from the previously-created
                # random sample
                randsub = randsample[:, : (2**n)]
                # Set the sample size
                nsample[z] = npy.shape(randsub)[1]

                with Timer(nsample[z]):
                    # Do the BP11 density estimate
                    bkernel = fastKDE(
                        randsub,
                        be_verbose=False,
                        do_save_marginals=False,
                        num_points=129,
                    )

                x, y = tuple(bkernel.axes)
                x2d, y2d = npy.meshgrid(x, y)

                # Calculate the mean squared error between the estimated density
                # and the gaussian
                absdiffsq = abs(pdfStandard(x2d, y2d) - bkernel.pdf) ** 2
                dx = x[1] - x[0]
                dy = y[1] - y[0]
                esq[z] = sum(dy * sum(absdiffsq * dx, axis=0)) / (len(x) * len(y))
                # Print the sample size and the error to show that the code is proceeeding
                # print "{}: {}, {}".format(n,nsample[z],esq[z])

            # Do a simple power law fit to the scaling
            [m, b, _, _, _] = stats.linregress(npy.log(nsample), npy.log(esq))
            # Print the error scaling (following BP11, this is expected to be m ~ -1)
            print("Error scales ~ N**{}".format(m))
        else:
            with Timer(npy.shape(randsample)[1]):
                bkernel = fastKDE(
                    randsample, be_verbose=True, do_save_marginals=False, num_points=129
                )

        doPlot = True
        if doPlot:
            x, y = tuple(bkernel.axes)
            x2d, y2d = npy.meshgrid(x, y)

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            clevs = npy.asarray(range(2, 10)) / 100.0
            ax1.contour(x2d, y2d, bkernel.pdf, levels=clevs)
            ax1.contour(x2d, y2d, pdfStandard(x2d, y2d), levels=clevs, colors="k")
            # ax1.plot(randsample[0,:],randsample[1,:],'k.',markersize=1)
            plt.xlim([-4, 6])
            plt.ylim([-4, 6])

            if evaluateError:
                # Plot the error vs sample size on a log-log curve
                ax3 = fig.add_subplot(122, xscale="log", yscale="log")
                ax3.plot(nsample, esq)
                ax3.plot(nsample, npy.exp(b) * nsample**m, "r-")
                # ax3 = fig.add_subplot(223)
                # ax3.plot(randsample[0,::16],randsample[1,::16],'k.',markersize=1)
                # plt.xlim([-4,6])
                # plt.ylim([-4,6])
            else:
                ax3 = fig.add_subplot(122)
                errorStandardSum = sum(
                    abs(pdfStandard(x2d, y2d) - bkernel.pdf) ** 2, axis=0
                )
                ax3.plot(x, errorStandardSum)

            plt.show()
