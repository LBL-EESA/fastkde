from fastkde import fastKDE
from scipy import stats
import pylab as PP
import matplotlib as mpl
import numpy as np

# set plot default fonts (fonts that are generally nice figures
font = {"family": "serif", "size": "15", "weight": "bold"}
mpl.rc("font", **font)
mpl.rc(
    "axes", labelweight="bold"
)  # needed for bold axis labels in more recent version of matplotlib

# Generate two random variables dataset (representing 100000 pairs of datapoints)
N = int(2e5)
var1 = 50 * np.random.normal(size=N) + 0.1
var2 = 0.01 * np.random.normal(size=N) - 300

# Do the self-consistent density estimate
myPDF, axes = fastKDE.pdf(var1, var2)

# Extract the axes from the axis list
v1, v2 = axes

# Plot contours of the PDF should be a set of concentric ellipsoids centered on
# (0.1, -300) Comparitively, the y axis range should be tiny and the x axis range
# should be large
PP.contour(v1, v2, myPDF)
PP.show()
