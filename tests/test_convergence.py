import numpy as npy
from fastkde.fastKDE import fastKDE
import scipy.stats as stats

def test_convergence_1d():
    # set a seed so that results are repeatable
    npy.random.seed(49885)

    mu = -1e3
    sig = 1e3

    # Define a gaussian function for evaluation purposes
    def mygaus(x):
        return (1.0 / (sig * npy.sqrt(2 * npy.pi))) * npy.exp(
            -((x - mu) ** 2) / (2.0 * sig**2)
        )

    # Set the size of the sample to calculate
    powmax = 19
    npow = npy.asarray(list(range(powmax))) + 1.0

    # Set the maximum sample size
    nmax = 2**powmax
    # Create a random normal sample of this size
    randsample = sig * npy.random.normal(size=nmax) + mu

    # Pre-define sample size and error-squared arrays
    nsample = npy.zeros([len(npow)])
    esq = npy.zeros([len(npow)])

    # Do the optimal calculation on a number of different random draws
    for i, n in enumerate(npow):
        # Extract a sample of length 2**n + 1 from the previously-created
        # random sample
        randgauss = randsample[: int(2**n + 1)]
        # Set the sample size
        nsample[i] = len(randgauss)

        # Do the BP11 density estimate
        bkernel = fastKDE(randgauss, num_points=513)

        # Calculate the mean squared error between the estimated density
        # And the gaussian
        esq[i] = npy.average(
            abs(mygaus(bkernel.axes[0]) - bkernel.pdf[:]) ** 2
            * bkernel.deltaX[0]
        )
        
    # Do a simple power law fit to the scaling
    [m, b, _, _, _] = stats.linregress(npy.log(nsample), npy.log(esq))

    # Print the error scaling (following BP11, this is npy.expected to be m ~ -1)
    print(("1D error scales ~ N**{}".format(m)))

    # assert that the error decays reasonably (this varies substantially from run to run)
    assert npy.abs(m) >= 0.5, f"1D error scaling is not close to ~ N**-1: {m}"

def test_convergence_2d():
    # Seed the rng so results are reproducable
    npy.random.seed(49885)

    # Define a bivariate normal function
    def norm2d(x, y, mux=0, muy=0, sx=1, sy=1, r=0):
        coef = 1.0 / (2 * npy.pi * sx * sy * npy.sqrt(1.0 - r**2))
        npy.expArg = -(1.0 / (2 * (1 - r**2))) * (
            (x - mux) ** 2 / sx**2
            + (y - muy) ** 2 / sy**2
            - 2 * r * (x - mux) * (y - muy) / (sx * sy)
        )
        return coef * npy.exp(npy.expArg)

    # Set the size of the sample to calculate
    powmax = 16
    npow = npy.asarray(list(range(1, powmax))) + 1.0

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
        # Append a 2D gaussian to the list
        randsamples.append(
            npy.random.multivariate_normal(mu, gCovMat, (int(nmax / ngg),)).transpose()
        )

    # Concatenate the gaussian samples
    randsample = npy.concatenate(tuple(randsamples), axis=1)

    # Shuffle the samples along the long axis so that we
    # can draw successively larger samples
    ishuffle = npy.asarray(list(range(nmax)))
    npy.random.shuffle(ishuffle)
    randsample = randsample[:, ishuffle]

    # Pre-define sample size and error-squared arrays
    nsample = npy.zeros([len(npow)])
    esq = npy.zeros([len(npow)])

    # Do the optimal calculation on a number of different random draws
    for z, n in enumerate(npow):
        # Extract a sample of length 2**n + 1 from the previously-created
        # random sample
        randsub = randsample[:, : int(2**n)]
        # Set the sample size
        nsample[z] = npy.shape(randsub)[1]

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
        # And the gaussian
        absdiffsq = abs(pdfStandard(x2d, y2d) - bkernel.pdf) ** 2
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        esq[z] = npy.sum(dy * npy.sum(absdiffsq * dx, axis=0)) / (len(x) * len(y))

    # Do a simple power law fit to the scaling
    [m, b, _, _, _] = stats.linregress(npy.log(nsample), npy.log(esq))

    # Print the error scaling (following BP11, this is npy.expected to be m ~ -1)
    print(("2D error scales ~ N**{}".format(m)))

    # assert that the error decays reasonably (this varies substantially from run to run)
    assert npy.abs(m) >= 0.5, f"2D error scaling is not close to ~ N**-1: {m}"

