import numpy as np

from scipy import stats

from fastkde import fastKDE


def test_simple_2D():
    np.random.seed(42)
    N = int(2e5)
    var1 = 50 * np.random.normal(size=N) + 0.1
    var2 = 0.01 * np.random.normal(size=N) - 300

    # Do the self-consistent density estimate
    myPDF, axes = fastKDE.pdf(var1, var2)

    # Extract the axes from the axis list
    v1, v2 = axes


def test_simple_3D():
    np.random.seed(42)
    N = int(1e3)  # number of points

    # generate 3 independent samples from 3 different distributions
    x_1 = stats.norm.rvs(size=N)
    x_2 = stats.gamma.rvs(2, size=N)
    x_3 = stats.betaprime.rvs(5, 6, size=N)

    # calculate the 3D PDF
    pdf, values = fastKDE.pdf(
        x_1, x_2, x_3, num_points=[65, 65, 65]
    )  # simply add more variables to the argument list for higher dimensions
    # note though that memory quickly becomes an issue
    # the numPoints argument results in a coarser PDF--but one that is calculated
    # faster (and with less memory)

    # calculate the index of the mode of the distribution
    # (we'll plot 2D slices through the mode)
    i_mode_ravel = np.argmax(pdf.ravel())
    nmode = np.unravel_index(i_mode_ravel, np.shape(pdf))
