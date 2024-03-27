import numpy as np
from scipy import stats
import fastkde


def test_simple_2D():
    np.random.seed(42)
    N = int(2e5)
    var1 = 50 * np.random.normal(size=N) + 0.1
    var2 = 0.01 * np.random.normal(size=N) - 300

    # Do the self-consistent density estimate
    myPDF, axes = fastkde.pdf(var1, var2, use_xarray=False)

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
    pdf, values = fastkde.pdf(
        x_1, x_2, x_3, num_points=[65, 65, 65],
        use_xarray=False
    )  # simply add more variables to the argument list for higher dimensions
    # note though that memory quickly becomes an issue
    # the numPoints argument results in a coarser PDF--but one that is calculated
    # faster (and with less memory)

def test_simple_1D_with_xarray():
    np.random.seed(42)
    N = int(2e5)
    var1 = 50 * np.random.normal(size=N) + 0.1

    # Do the self-consistent density estimate
    myPDF = fastkde.pdf(var1, use_xarray=True)


def test_simple_2D_with_xarray():
    np.random.seed(42)
    N = int(2e5)
    var1 = 50 * np.random.normal(size=N) + 0.1
    var2 = 0.01 * np.random.normal(size=N) - 300

    # Do the self-consistent density estimate
    myPDF = fastkde.pdf(var1, var2, use_xarray=True, num_points = [129, 257])

def test_simple_conditional_2D_with_xarray():
    np.random.seed(42)
    N = int(2e5)
    var1 = 50 * np.random.normal(size=N) + 0.1
    var2 = 0.01 * np.random.normal(size=N) - 300

    # Do the self-consistent density estimate
    myPDF = fastkde.conditional(var1, var2, use_xarray=True, num_points = [129, 257])

def test_pdf_at_points():
    """ Demonstrate using the pdf_at_points function. """""
    train_x = 50*np.random.normal(size=100) + 0.1
    train_y = 0.01*np.random.normal(size=100) - 300

    test_x = 50*np.random.normal(size=100) + 0.1
    test_y = 0.01*np.random.normal(size=100) - 300

    test_points = list(zip(test_x, test_y))
    test_point_pdf_values = fastkde.pdf_at_points(train_x, train_y, list_of_points = test_points)